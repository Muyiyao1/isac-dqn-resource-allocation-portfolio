# ISAC DQN Resource Allocation

[中文说明](README.zh-CN.md)

This repository implements a Deep Q-Network (DQN) simulation for resource allocation in Integrated Sensing and Communication (ISAC) systems. The goal is to study how a learning-based power-allocation policy can balance communication throughput, sensing utility, and transmit-power cost under different preference settings.

## Project Background

ISAC systems reuse wireless infrastructure and radio resources for both data transmission and sensing. This creates a resource-allocation trade-off: allocating more power to communication can improve throughput, while allocating more resources to sensing can improve sensing utility. Static allocation rules are limited because channel conditions, interference, user positions, and sensing targets vary across episodes.

This project formulates the trade-off as a reinforcement learning problem and evaluates whether a DQN policy reacts coherently as the communication-sensing preference changes.

## System Model

The simulated environment uses a multi-cell downlink setting with communication users and one sensing target per cell.

Key components:

- discrete subband power actions;
- sensing power split levels;
- path loss, log-normal shadowing, Rayleigh fading, antenna gain, and inter-cell interference;
- SINR, CQI, throughput, sensing score, total power, and reward metrics;
- preference weight `alpha` for communication-sensing trade-off control.

The reward uses three terms:

```text
r_c = alpha * R_comm,c + (1 - alpha) * R_sense,c - lambda * R_power,c
```

where `alpha` controls the communication-sensing preference and `lambda` is the fixed power-penalty coefficient.

## Traditional Baselines

The DQN policy is compared against simple reference policies:

- `random`: randomly selects valid actions;
- `fixed_0.8w`: uses a fixed power level;
- `max_power`: selects maximum-power allocation;
- optional GA-style search helpers are available in `src/isac_dqn/baselines.py`.

These baselines are used as sanity checks for interpreting the learned policy behavior.

## Deep Learning Method

The DQN agent maps the environment state to per-cell discrete action scores.

Implementation details:

- PyTorch Q-network with configurable hidden layers;
- replay buffer for sampled updates;
- target network for more stable training;
- epsilon-greedy exploration;
- Smooth L1 loss and RMSprop optimizer;
- one output block per cell over the cell action space.

Core files:

| Path | Purpose |
| --- | --- |
| `src/isac_dqn/environment.py` | Environment state, action decoding, metrics, and reward. |
| `src/isac_dqn/dqn_agent.py` | Q-network, DQN update logic, and checkpoint saving. |
| `src/isac_dqn/radio.py` | Radio-channel and communication-metric helpers. |
| `run_experiment.py` | Training, evaluation, CSV export, and plotting pipeline. |

## Experimental Setup

The main configuration is stored in `configs/config.yaml`.

| Setting | Value |
| --- | --- |
| Cells | `5` |
| Users per cell | `5` |
| Subbands | `3` |
| Power levels | `0.4, 0.6, 0.8, 1.0, 1.2 W` |
| Sensing split levels | `0.1, 0.3, 0.5, 0.7` |
| Preference sweep | `alpha = 0.0, 0.1, ..., 1.0` |
| Training episodes | `800` per alpha |
| Evaluation episodes | `60` |
| Robustness perturbation | `0, 2, 4, 6 dB` |

## Result Figures

Included result files and figures are under `results/final/`.

As `alpha` increases, the policy shifts toward communication performance. In the included result set, learned average sensing split decreases from `0.610` to `0.182`, while throughput increases from `132.39 Mbps` to `146.32 Mbps`.

![DQN communication-sensing trade-off](results/final/figures/dqn_comm_sense_tradeoff.png)

![Average sensing split vs alpha](results/final/figures/dqn_sensing_split_vs_alpha.png)

Training and robustness views:

![Training reward](results/final/figures/training_reward.png)

![Robustness throughput](results/final/figures/robustness_throughput_mbps.png)

## Conclusion

The included results show that the DQN policy responds to the communication-sensing preference parameter in the expected direction. Higher communication preference increases throughput and reduces the learned sensing split. The sensing score remains stochastic because it depends on generated channel and target conditions.

The project should be read as a simulation-based ISAC resource-allocation study. The sensing score is a simplified proxy metric, not a full radar signal-processing chain.

## Improvements

Possible next steps:

- add stronger optimization baselines for more rigorous comparison;
- evaluate additional mobility and channel-variation scenarios;
- extend sensing metrics beyond the current proxy score;
- add automated tests for environment invariants and reward calculations;
- support configurable experiment sweeps from separate YAML files;
- publish regenerated model weights separately if they are needed.

## Project Structure

```text
.
|-- README.md
|-- README.zh-CN.md
|-- requirements.txt
|-- configs/
|   `-- config.yaml
|-- data/
|-- run_experiment.py
|-- run_quick_test.py
|-- quality_check.py
|-- src/
|   `-- isac_dqn/
|       |-- environment.py
|       |-- dqn_agent.py
|       |-- radio.py
|       |-- baselines.py
|       |-- plotting.py
|       |-- replay_buffer.py
|       `-- utils.py
|-- notebooks/
|-- docs/
`-- results/
    `-- final/
        |-- summary_by_policy.csv
        |-- evaluation_results.csv
        |-- robustness_results.csv
        |-- training_history.csv
        `-- figures/
```

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_quick_test.py
```

The quick test writes outputs to `results/quick/` and `models/quick/`.

Run the configured full experiment:

```powershell
python run_experiment.py
```

Run the full experiment without GA baselines:

```powershell
python run_experiment.py --no-ga --output-dir results/full_no_ga --model-dir models/full_no_ga
```

Check included result files and figures:

```powershell
python quality_check.py --results-dir results/final --skip-models
```
