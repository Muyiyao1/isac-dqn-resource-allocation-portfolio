# Methodology And Formulas

This document maps the main mathematical ideas to the implementation.

## Reward

The per-cell reward combines communication utility, sensing utility, and a power penalty:

```text
r_c = alpha * R_comm,c + (1 - alpha) * R_sense,c - lambda * R_power,c
```

Implementation:

- `src/isac_dqn/environment.py`
- `ISACPowerAllocationEnv.evaluate_action`

Meaning:

- `alpha`: communication-sensing preference weight.
- `R_comm,c`: normalized communication utility.
- `R_sense,c`: normalized sensing proxy utility.
- `R_power,c`: normalized power usage.
- `lambda`: empirical power-penalty coefficient.

## Communication Metrics

The environment computes:

- path loss,
- shadowing,
- Rayleigh fading,
- antenna gain,
- interference,
- SINR,
- CQI,
- throughput.

Implementation:

- `src/isac_dqn/radio.py`
- `src/isac_dqn/environment.py`

## Sensing Proxy

The sensing score is a simplified proxy derived from sensing power, sensing target channel gain, sensing noise, and sensing demand. It is designed for resource-allocation experiments, not full radar detection.

Implementation:

- `ISACPowerAllocationEnv.evaluate_action`

## DQN Objective

The DQN maps the current state to per-cell discrete action scores. Training uses replay-buffer samples, a target network, and Smooth L1 loss.

Implementation:

- `src/isac_dqn/dqn_agent.py`
- `src/isac_dqn/replay_buffer.py`
