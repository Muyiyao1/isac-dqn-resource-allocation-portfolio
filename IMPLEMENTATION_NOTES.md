# Implementation Notes

## Modeling Boundary

The implementation is a simulation-based resource-allocation study. It models a simplified ISAC scenario rather than a complete physical-layer sensing system.

The communication pipeline is:

```text
power allocation
  -> antenna gain
  -> path loss + shadowing + Rayleigh fading
  -> SINR
  -> CQI
  -> throughput
```

The sensing extension is:

```text
sensing target features
  -> sensing power split rho
  -> sensing SNR / sensing score proxy
  -> reward = alpha R_comm + (1 - alpha) R_sense - lambda R_power
```

## Reward Design

`alpha` controls the communication-sensing preference:

- `alpha = 1.0`: communication-oriented reward.
- `alpha = 0.0`: sensing-oriented reward.
- intermediate values: mixed preference.

`lambda = 0.08` is a fixed empirical power-penalty coefficient used in this implementation.

## Action Design

Each cell selects a discrete power allocation and sensing split. The global action is represented as one selected cell action per cell, while the DQN output head is arranged as `num_cells * num_cell_actions`.

## Result Interpretation

Useful checks for this project:

- Throughput should generally improve as `alpha` increases.
- Average sensing split should generally decrease as `alpha` increases.
- Sensing score may fluctuate because channel generation and target conditions are stochastic.
- Power, SINR, CQI, throughput, sensing score, and reward should be read together rather than as isolated metrics.
