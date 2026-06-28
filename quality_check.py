from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from isac_dqn.environment import ISACPowerAllocationEnv, make_env_config
from isac_dqn.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check ISAC-DQN result artifacts and environment behavior.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"), help="Path to YAML config.")
    parser.add_argument("--results-dir", default=None, help="Results directory to check. Defaults to results/final if present.")
    parser.add_argument("--models-dir", default=None, help="Model directory to check. Defaults to config model_dir if present.")
    parser.add_argument("--skip-models", action="store_true", help="Skip trained model file checks.")
    return parser.parse_args()


def assert_file(path: Path, min_size: int = 1) -> None:
    if not path.exists():
        raise AssertionError(f"Missing file: {path}")
    if path.stat().st_size < min_size:
        raise AssertionError(f"File too small: {path}")


def check_environment(cfg: dict) -> dict:
    env = ISACPowerAllocationEnv(make_env_config(cfg["environment"]), alpha=0.5, seed=int(cfg["seed"]))
    state = env.reset()
    action = env.random_action()
    next_state, reward, done, info = env.step(action)
    if state.shape[0] != env.state_dim:
        raise AssertionError("State dimension mismatch")
    if next_state.shape[0] != env.state_dim:
        raise AssertionError("Next state dimension mismatch")
    for key in ["throughput_mbps", "sensing_score", "total_power_w", "avg_sensing_split", "reward"]:
        if key not in info:
            raise AssertionError(f"Missing metric from environment: {key}")
        if not np.isfinite(info[key]):
            raise AssertionError(f"Non-finite environment metric: {key}={info[key]}")
    return {
        "state_dim": env.state_dim,
        "num_cell_actions": env.num_cell_actions,
        "global_action_outputs": env.action_dim,
        "sample_reward": float(reward),
    }


def check_csv_outputs(results_dir: Path) -> dict:
    required = {
        "training_history.csv": ["alpha", "episode", "episode_reward", "epsilon", "loss"],
        "evaluation_results.csv": [
            "alpha",
            "policy",
            "throughput_mbps",
            "sensing_score",
            "total_power_w",
            "avg_sinr_db",
            "avg_cqi",
            "avg_sensing_split",
            "reward",
        ],
        "robustness_results.csv": ["alpha", "perturbation_db", "throughput_mbps", "sensing_score", "reward"],
        "summary_by_policy.csv": ["alpha", "policy", "throughput_mbps", "sensing_score", "avg_sensing_split", "reward"],
    }
    stats = {}
    for file_name, columns in required.items():
        path = results_dir / file_name
        assert_file(path, min_size=100)
        df = pd.read_csv(path)
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise AssertionError(f"{file_name} missing columns: {missing}")
        numeric = df.select_dtypes(include=[np.number])
        if numeric.replace([np.inf, -np.inf], np.nan).isna().all(axis=None):
            raise AssertionError(f"{file_name} has no valid numeric values")
        stats[file_name] = {"rows": int(len(df)), "columns": list(df.columns)}
    return stats


def check_figures(figures_dir: Path) -> dict:
    figure_names = [
        "training_reward.png",
        "training_loss.png",
        "tradeoff_throughput_mbps.png",
        "tradeoff_sensing_score.png",
        "tradeoff_total_power_w.png",
        "tradeoff_avg_sinr_db.png",
        "tradeoff_avg_cqi.png",
        "tradeoff_avg_sensing_split.png",
        "tradeoff_reward.png",
        "dqn_comm_sense_tradeoff.png",
        "dqn_sensing_split_vs_alpha.png",
        "robustness_throughput_mbps.png",
        "robustness_sensing_score.png",
        "robustness_reward.png",
    ]
    for name in figure_names:
        assert_file(figures_dir / name, min_size=5000)
    return {"figures": figure_names}


def check_models(cfg: dict, models_dir: Path) -> dict:
    alphas = [float(a) for a in cfg["experiment"]["alphas"]]
    model_files = []
    for alpha in alphas:
        path = models_dir / f"dqn_alpha_{alpha:.2f}.pt"
        assert_file(path, min_size=10000)
        payload = torch.load(path, map_location="cpu")
        if "state_dict" not in payload:
            raise AssertionError(f"Model file missing state_dict: {path}")
        model_files.append(path.name)
    return {"models": model_files}


def check_tradeoff(results_dir: Path) -> dict:
    summary = pd.read_csv(results_dir / "summary_by_policy.csv")
    ga = summary[summary["policy"] == "ga_isac"].sort_values("alpha")
    dqn = summary[summary["policy"] == "dqn"].sort_values("alpha")
    report = {}
    if len(ga) >= 2:
        report["ga_isac_low_alpha_sensing_split"] = float(ga.iloc[0]["avg_sensing_split"])
        report["ga_isac_high_alpha_sensing_split"] = float(ga.iloc[-1]["avg_sensing_split"])
        report["ga_isac_low_alpha_sensing_score"] = float(ga.iloc[0]["sensing_score"])
        report["ga_isac_high_alpha_sensing_score"] = float(ga.iloc[-1]["sensing_score"])
        report["ga_isac_high_alpha_throughput"] = float(ga.iloc[-1]["throughput_mbps"])
        report["ga_isac_low_alpha_throughput"] = float(ga.iloc[0]["throughput_mbps"])
    if len(dqn) >= 2:
        report["dqn_low_alpha_sensing_split"] = float(dqn.iloc[0]["avg_sensing_split"])
        report["dqn_high_alpha_sensing_split"] = float(dqn.iloc[-1]["avg_sensing_split"])
        report["dqn_low_alpha_reward"] = float(dqn.iloc[0]["reward"])
        report["dqn_high_alpha_reward"] = float(dqn.iloc[-1]["reward"])
    return report


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    default_results_dir = "results/final" if (ROOT / "results" / "final").exists() else cfg.get("output_dir", "results")
    results_dir = ROOT / (args.results_dir or default_results_dir)
    configured_models_dir = ROOT / (args.models_dir or cfg.get("model_dir", "models"))
    models_dir = configured_models_dir if configured_models_dir.exists() else None
    figures_dir = results_dir / "figures"
    artifact_report = check_figures(figures_dir)
    if args.skip_models or models_dir is None:
        artifact_report["models"] = "skipped"
    else:
        artifact_report.update(check_models(cfg, models_dir))

    checked_results_dir = results_dir.relative_to(ROOT).as_posix() if results_dir.is_relative_to(ROOT) else str(results_dir)
    checked_models_dir = None
    if models_dir is not None:
        checked_models_dir = models_dir.relative_to(ROOT).as_posix() if models_dir.is_relative_to(ROOT) else str(models_dir)

    report = {
        "python_version": sys.version.split()[0],
        "torch": torch.__version__,
        "checked_results_dir": checked_results_dir,
        "checked_models_dir": checked_models_dir,
        "environment": check_environment(cfg),
        "csv_outputs": check_csv_outputs(results_dir),
        "artifacts": artifact_report,
        "tradeoff": check_tradeoff(results_dir),
        "status": "PASS",
    }
    out = results_dir / "quality_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
