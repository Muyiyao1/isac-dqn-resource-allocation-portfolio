from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import trange


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from isac_dqn.baselines import fixed_action, genetic_search_action, max_power_action
from isac_dqn.dqn_agent import DQNConfig, MultiCellDQNAgent
from isac_dqn.environment import ISACPowerAllocationEnv, make_env_config
from isac_dqn.plotting import plot_robustness, plot_tradeoff, plot_training
from isac_dqn.utils import ensure_dir, load_config, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate ISAC-DQN power allocation.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"), help="Path to YAML config.")
    parser.add_argument("--quick", action="store_true", help="Run a short smoke test.")
    parser.add_argument("--no-ga", action="store_true", help="Skip GA baselines to save time.")
    parser.add_argument("--output-dir", default=None, help="Override the output directory from config.yaml.")
    parser.add_argument("--model-dir", default=None, help="Override the model directory from config.yaml.")
    return parser.parse_args()


def apply_quick_overrides(cfg: Dict) -> Dict:
    cfg = dict(cfg)
    cfg["experiment"] = dict(cfg["experiment"])
    cfg["dqn"] = dict(cfg["dqn"])
    cfg["experiment"]["alphas"] = [0.5]
    cfg["experiment"]["train_episodes"] = 10
    cfg["experiment"]["eval_episodes"] = 4
    cfg["experiment"]["baseline_episodes"] = 4
    cfg["experiment"]["ga_population"] = 10
    cfg["experiment"]["ga_generations"] = 5
    cfg["experiment"]["robustness_perturbation_db"] = [0.0, 4.0]
    cfg["dqn"]["min_replay_size"] = 32
    cfg["dqn"]["batch_size"] = 16
    cfg["dqn"]["target_update_interval"] = 20
    return cfg


def make_env(cfg: Dict, alpha: float, seed: int, perturbation_db: float = 0.0) -> ISACPowerAllocationEnv:
    env_cfg = make_env_config(cfg["environment"])
    return ISACPowerAllocationEnv(env_cfg, alpha=alpha, seed=seed, channel_perturbation_std_db=perturbation_db)


def make_agent(env: ISACPowerAllocationEnv, cfg: Dict, seed: int) -> MultiCellDQNAgent:
    dqn_cfg = DQNConfig(**cfg["dqn"])
    return MultiCellDQNAgent(
        state_dim=env.state_dim,
        num_cells=env.cfg.num_cells,
        num_cell_actions=env.num_cell_actions,
        cfg=dqn_cfg,
        seed=seed,
    )


def train_agent(alpha: float, cfg: Dict, model_dir: Path, seed: int) -> tuple[MultiCellDQNAgent, List[Dict]]:
    env = make_env(cfg, alpha=alpha, seed=seed)
    agent = make_agent(env, cfg, seed)
    rows: List[Dict] = []
    train_episodes = int(cfg["experiment"]["train_episodes"])

    iterator = trange(train_episodes, desc=f"train alpha={alpha}", leave=False)
    for episode in iterator:
        state = env.reset()
        total_reward = 0.0
        losses = []
        last_info = {}

        for _ in range(env.cfg.max_steps_per_episode):
            action = agent.act(state, explore=True)
            next_state, reward, done, info = env.step(action)
            agent.observe(state, action, info.get("per_cell_reward", reward), next_state, done)
            update_info = agent.update()
            if update_info is not None:
                losses.append(update_info["loss"])
            total_reward += reward
            state = next_state
            last_info = info
            if done:
                break

        loss_mean = float(np.mean(losses)) if losses else np.nan
        row = {
            "alpha": alpha,
            "episode": episode,
            "episode_reward": total_reward,
            "epsilon": agent.epsilon(),
            "loss": loss_mean,
        }
        for key in ["throughput_mbps", "sensing_score", "total_power_w", "avg_sinr_db", "avg_cqi", "avg_sensing_split", "reward"]:
            row[key] = float(last_info.get(key, np.nan))
        rows.append(row)
        iterator.set_postfix(reward=f"{total_reward:.3f}", eps=f"{agent.epsilon():.3f}", loss=f"{loss_mean:.4f}")

    agent.save(model_dir / f"dqn_alpha_{alpha:.2f}.pt", meta={"alpha": alpha, "seed": seed})
    return agent, rows


def rollout_agent(agent: MultiCellDQNAgent, env: ISACPowerAllocationEnv) -> Dict[str, float]:
    state = env.reset()
    last_info = {}
    for _ in range(env.cfg.max_steps_per_episode):
        action = agent.act(state, explore=False)
        state, _, done, info = env.step(action)
        last_info = info
        if done:
            break
    return {k: float(v) for k, v in last_info.items() if np.isscalar(v)}


def evaluate_all(
    trained_agents: Dict[float, MultiCellDQNAgent],
    cfg: Dict,
    seed: int,
    skip_ga: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eval_rows: List[Dict] = []
    robust_rows: List[Dict] = []
    exp_cfg = cfg["experiment"]

    for alpha, agent in trained_agents.items():
        for episode in range(int(exp_cfg["eval_episodes"])):
            env = make_env(cfg, alpha=alpha, seed=seed + 10000 + episode)
            row = rollout_agent(agent, env)
            row.update({"alpha": alpha, "episode": episode, "policy": "dqn"})
            eval_rows.append(row)

        baseline_specs = [
            ("random", lambda env: env.random_action()),
            ("fixed_0.8w", lambda env: fixed_action(env, 0.8)),
            ("max_power", lambda env: max_power_action(env)),
        ]
        if not skip_ga:
            baseline_specs.extend(
                [
                    (
                        "ga_comm",
                        lambda env: genetic_search_action(
                            env,
                            population_size=int(exp_cfg["ga_population"]),
                            generations=int(exp_cfg["ga_generations"]),
                            objective="throughput",
                        ),
                    ),
                    (
                        "ga_isac",
                        lambda env: genetic_search_action(
                            env,
                            population_size=int(exp_cfg["ga_population"]),
                            generations=int(exp_cfg["ga_generations"]),
                            objective="reward",
                        ),
                    ),
                ]
            )

        for policy_name, action_fn in baseline_specs:
            for episode in range(int(exp_cfg["baseline_episodes"])):
                env = make_env(cfg, alpha=alpha, seed=seed + 20000 + episode)
                env.reset()
                action = action_fn(env)
                row = env.evaluate_action(action)
                row = {k: float(v) for k, v in row.items() if np.isscalar(v)}
                row.update({"alpha": alpha, "episode": episode, "policy": policy_name})
                eval_rows.append(row)

        for perturb in exp_cfg["robustness_perturbation_db"]:
            for episode in range(int(exp_cfg["eval_episodes"])):
                env = make_env(cfg, alpha=alpha, seed=seed + 30000 + episode, perturbation_db=float(perturb))
                row = rollout_agent(agent, env)
                row.update({"alpha": alpha, "episode": episode, "policy": "dqn", "perturbation_db": float(perturb)})
                robust_rows.append(row)

    return pd.DataFrame(eval_rows), pd.DataFrame(robust_rows)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.quick:
        cfg = apply_quick_overrides(cfg)
        # Keep smoke-test outputs separate so quick runs cannot overwrite the
        # included final figures and CSV files.
        cfg["output_dir"] = "results/quick"
        cfg["model_dir"] = "models/quick"

    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    if args.model_dir is not None:
        cfg["model_dir"] = args.model_dir

    seed = int(cfg["seed"])
    set_seed(seed)

    out_dir = ensure_dir(ROOT / cfg["output_dir"])
    fig_dir = ensure_dir(out_dir / "figures")
    model_dir = ensure_dir(ROOT / cfg["model_dir"])
    save_json(cfg, out_dir / "used_config.json")

    training_rows: List[Dict] = []
    trained_agents: Dict[float, MultiCellDQNAgent] = {}

    for idx, alpha in enumerate(cfg["experiment"]["alphas"]):
        alpha = float(alpha)
        agent, rows = train_agent(alpha, cfg, model_dir, seed + idx * 1000)
        trained_agents[alpha] = agent
        training_rows.extend(rows)

    training_df = pd.DataFrame(training_rows)
    training_csv = out_dir / "training_history.csv"
    training_df.to_csv(training_csv, index=False)

    eval_df, robust_df = evaluate_all(trained_agents, cfg, seed=seed, skip_ga=args.no_ga)
    eval_csv = out_dir / "evaluation_results.csv"
    robust_csv = out_dir / "robustness_results.csv"
    eval_df.to_csv(eval_csv, index=False)
    robust_df.to_csv(robust_csv, index=False)

    summary = eval_df.groupby(["alpha", "policy"], as_index=False)[
        ["throughput_mbps", "sensing_score", "total_power_w", "avg_sinr_db", "avg_cqi", "avg_sensing_split", "reward"]
    ].mean()
    summary.to_csv(out_dir / "summary_by_policy.csv", index=False)

    plot_training(training_csv, fig_dir)
    plot_tradeoff(eval_csv, fig_dir)
    plot_robustness(robust_csv, fig_dir)

    print(f"Saved training CSV: {training_csv}")
    print(f"Saved evaluation CSV: {eval_csv}")
    print(f"Saved robustness CSV: {robust_csv}")
    print(f"Saved figures to: {fig_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
