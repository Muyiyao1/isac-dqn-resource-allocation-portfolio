from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

from .environment import ISACPowerAllocationEnv


def fixed_action(env: ISACPowerAllocationEnv, power_w: float) -> np.ndarray:
    idx = env.nearest_action_index(power_w)
    return np.full(env.cfg.num_cells, idx, dtype=np.int64)


def max_power_action(env: ISACPowerAllocationEnv) -> np.ndarray:
    idx = env.max_power_action_index()
    return np.full(env.cfg.num_cells, idx, dtype=np.int64)


def genetic_search_action(
    env: ISACPowerAllocationEnv,
    population_size: int,
    generations: int,
    mutation_prob: float = 0.12,
    objective: str = "reward",
) -> np.ndarray:
    rng = env.rng
    population = rng.integers(0, env.num_cell_actions, size=(population_size, env.cfg.num_cells), dtype=np.int64)

    def fitness(individual: np.ndarray) -> float:
        metrics = env.evaluate_action(individual)
        if objective == "throughput":
            return float(metrics["throughput_mbps"])
        if objective == "sensing":
            return float(metrics["sensing_score"])
        return float(metrics["reward"])

    scores = np.asarray([fitness(ind) for ind in population], dtype=np.float64)
    elite_count = max(2, population_size // 8)

    for _ in range(generations):
        elite_idx = np.argsort(scores)[-elite_count:]
        elites = population[elite_idx].copy()
        new_pop = [e.copy() for e in elites]
        while len(new_pop) < population_size:
            p1 = tournament(population, scores, rng)
            p2 = tournament(population, scores, rng)
            cut = rng.integers(1, env.cfg.num_cells)
            child = np.concatenate([p1[:cut], p2[cut:]]).astype(np.int64)
            mask = rng.random(env.cfg.num_cells) < mutation_prob
            child[mask] = rng.integers(0, env.num_cell_actions, size=int(np.sum(mask)))
            new_pop.append(child)
        population = np.asarray(new_pop, dtype=np.int64)
        scores = np.asarray([fitness(ind) for ind in population], dtype=np.float64)

    return population[int(np.argmax(scores))].copy()


def tournament(population: np.ndarray, scores: np.ndarray, rng: np.random.Generator, k: int = 3) -> np.ndarray:
    idx = rng.choice(len(population), size=k, replace=False)
    return population[idx[int(np.argmax(scores[idx]))]].copy()


def evaluate_policy(
    env_factory: Callable[[], ISACPowerAllocationEnv],
    policy_name: str,
    policy_fn: Callable[[ISACPowerAllocationEnv, np.ndarray], np.ndarray],
    episodes: int,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for ep in range(episodes):
        env = env_factory()
        state = env.reset()
        action = policy_fn(env, state)
        metrics = env.evaluate_action(action)
        row = {k: float(v) for k, v in metrics.items() if np.isscalar(v)}
        row["episode"] = ep
        row["policy"] = policy_name
        rows.append(row)
    return rows
