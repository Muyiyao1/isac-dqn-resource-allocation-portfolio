# Code Architecture

## Main Flow

```text
configs/config.yaml
  -> run_experiment.py
  -> ISACPowerAllocationEnv
  -> MultiCellDQNAgent
  -> evaluation baselines
  -> CSV outputs
  -> result figures
```

## Key Files

| File | Responsibility |
| --- | --- |
| `run_experiment.py` | Coordinates training, evaluation, robustness checks, CSV writing, and figure generation. |
| `configs/config.yaml` | Stores environment, DQN, experiment, seed, and output settings. |
| `src/isac_dqn/environment.py` | Defines state construction, action decoding, metric computation, and reward. |
| `src/isac_dqn/dqn_agent.py` | Defines the Q-network, DQN update logic, epsilon schedule, and checkpoint saving. |
| `src/isac_dqn/replay_buffer.py` | Stores transitions for minibatch DQN updates. |
| `src/isac_dqn/radio.py` | Contains path loss, antenna gain, noise, SINR, CQI, and throughput helpers. |
| `src/isac_dqn/baselines.py` | Implements random, fixed-power, max-power, and optional GA-style baseline actions. |
| `src/isac_dqn/plotting.py` | Generates training, trade-off, and robustness plots from CSV outputs. |
| `quality_check.py` | Checks environment construction, CSV files, figures, and optional model files. |

## Data Outputs

| Directory | Content |
| --- | --- |
| `results/final/` | Included final CSV files and generated figures. |
| `results/quick/` | Smoke-test outputs generated locally. |
| `models/quick/` | Smoke-test model files generated locally. |
| `results/full/` | Full-run outputs generated locally. |
| `models/full/` | Full-run model files generated locally. |

Generated model directories are ignored by Git by default.
