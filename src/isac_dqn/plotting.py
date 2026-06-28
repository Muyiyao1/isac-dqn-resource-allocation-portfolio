from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")


def place_legend_below(ax, ncol: int = 4) -> None:
    legend = ax.get_legend()
    if legend is not None:
        legend.set_title(legend.get_title().get_text())
        sns.move_legend(ax, "upper center", bbox_to_anchor=(0.5, -0.18), ncol=ncol, frameon=False)


def plot_training(training_csv: str | Path, out_dir: str | Path) -> None:
    set_style()
    df = pd.read_csv(training_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    sns.lineplot(data=df, x="episode", y="episode_reward", hue="alpha", ax=ax)
    ax.set_title("DQN training reward")
    ax.set_ylabel("Episode reward")
    place_legend_below(ax, ncol=6)
    fig.tight_layout()
    fig.savefig(out_dir / "training_reward.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    sns.lineplot(data=df, x="episode", y="loss", hue="alpha", ax=ax)
    ax.set_title("DQN training loss")
    ax.set_ylabel("Huber loss")
    place_legend_below(ax, ncol=6)
    fig.tight_layout()
    fig.savefig(out_dir / "training_loss.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff(eval_csv: str | Path, out_dir: str | Path) -> None:
    set_style()
    df = pd.read_csv(eval_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("throughput_mbps", "Throughput (Mbps)"),
        ("sensing_score", "Sensing score"),
        ("total_power_w", "Total weighted power (W)"),
        ("avg_sinr_db", "Average SINR (dB)"),
        ("avg_cqi", "Average CQI"),
        ("avg_sensing_split", "Average sensing power split"),
        ("reward", "ISAC reward"),
    ]
    for metric, ylabel in metrics:
        fig, ax = plt.subplots(figsize=(9.8, 4.8))
        sns.barplot(data=df, x="alpha", y=metric, hue="policy", errorbar="sd", ax=ax)
        ax.set_title(f"Alpha trade-off: {ylabel}")
        ax.set_ylabel(ylabel)
        place_legend_below(ax, ncol=4)
        fig.tight_layout()
        fig.savefig(out_dir / f"tradeoff_{metric}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    dqn = df[df["policy"] == "dqn"].copy()
    if not dqn.empty:
        fig, ax1 = plt.subplots(figsize=(8.6, 4.8))
        summary = dqn.groupby("alpha", as_index=False)[["throughput_mbps", "sensing_score", "total_power_w", "avg_sensing_split"]].mean()
        ax1.plot(summary["alpha"], summary["throughput_mbps"], marker="o", label="Throughput")
        ax1.set_xlabel("alpha")
        ax1.set_ylabel("Throughput (Mbps)")
        ax2 = ax1.twinx()
        ax2.plot(summary["alpha"], summary["sensing_score"], marker="s", color="tab:orange", label="Sensing")
        ax2.set_ylabel("Sensing score")
        ax1.set_title("DQN communication-sensing trade-off")
        fig.tight_layout()
        fig.savefig(out_dir / "dqn_comm_sense_tradeoff.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.6, 4.8))
        ax.plot(summary["alpha"], summary["avg_sensing_split"], marker="o")
        ax.set_title("DQN learned sensing power split")
        ax.set_xlabel("alpha")
        ax.set_ylabel("Average sensing split")
        fig.tight_layout()
        fig.savefig(out_dir / "dqn_sensing_split_vs_alpha.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def plot_robustness(robust_csv: str | Path, out_dir: str | Path) -> None:
    set_style()
    df = pd.read_csv(robust_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric, ylabel in [
        ("throughput_mbps", "Throughput (Mbps)"),
        ("sensing_score", "Sensing score"),
        ("reward", "ISAC reward"),
    ]:
        fig, ax = plt.subplots(figsize=(9.4, 4.8))
        sns.lineplot(data=df, x="perturbation_db", y=metric, hue="alpha", marker="o", errorbar="sd", ax=ax)
        ax.set_title(f"Robustness under channel perturbation: {ylabel}")
        ax.set_ylabel(ylabel)
        place_legend_below(ax, ncol=6)
        fig.tight_layout()
        fig.savefig(out_dir / f"robustness_{metric}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)
