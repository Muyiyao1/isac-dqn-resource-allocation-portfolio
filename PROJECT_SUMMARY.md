# Project Summary

## Problem

Integrated Sensing and Communication (ISAC) systems need to share limited radio resources between data transmission and sensing. Increasing transmit resources for communication can reduce the sensing budget, while sensing-oriented allocation can reduce throughput.

This project frames that trade-off as a reinforcement learning resource-allocation problem.

## Method

The implementation adapts a DQN-based downlink power-allocation framework into a simplified ISAC simulation.

Core modeling choices:

- Multi-cell downlink scenario.
- Discrete subband power actions.
- Path loss, log-normal shadowing, Rayleigh fading, antenna gain, inter-cell interference, SINR, CQI, and throughput.
- One sensing target per cell.
- Sensing-aware state features and a sensing power split.
- Reward with three terms: communication utility, sensing proxy utility, and power penalty.

The reward structure is:

```text
r_c = alpha * R_comm,c + (1 - alpha) * R_sense,c - lambda * R_power,c
```

where `alpha` controls the communication-sensing preference and `lambda` is a fixed empirical power-penalty coefficient.

## Evaluation

The experiments compare DQN against simple sanity-check baselines:

- random allocation,
- fixed-power allocation,
- maximum-power allocation.

The main sweep evaluates eleven preference settings:

```text
alpha in {0.0, 0.1, 0.2, ..., 1.0}
```

## Result Pattern

The result pattern is consistent with the intended trade-off:

- Higher `alpha` increases communication priority.
- Learned average sensing split decreases as communication priority increases.
- Throughput increases as the policy moves toward communication-oriented allocation.
- Sensing score follows an overall decreasing trend with stochastic variation.

Representative values from the included result set:

| Metric | alpha = 0.0 | alpha = 1.0 |
| --- | ---: | ---: |
| Average sensing split | 0.610 | 0.182 |
| Throughput | 132.39 Mbps | 146.32 Mbps |

## Engineering Value

The main value of the project is:

- defining a tractable simulation boundary,
- turning a domain trade-off into a reward function,
- validating whether the learned policy moves in the expected direction,
- using baselines and robustness checks to make results easier to interpret,
- keeping limitations explicit.

## Limitations

- The sensing score is a proxy metric, not a full radar signal processing model.
- The implementation focuses on one DQN-style framework rather than a broad algorithm comparison.
- Trained `.pt` model weights are not included, but the source code, configuration, final CSV outputs, and result figures are included.
