from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple

import numpy as np


@dataclass
class TransitionBatch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 42) -> None:
        self.buffer: Deque[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]] = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, state: np.ndarray, action: np.ndarray, reward: np.ndarray | float, next_state: np.ndarray, done: bool) -> None:
        reward_arr = np.asarray(reward, dtype=np.float32)
        self.buffer.append((state.copy(), action.copy(), reward_arr.copy(), next_state.copy(), bool(done)))

    def sample(self, batch_size: int) -> TransitionBatch:
        idx = self.rng.choice(len(self.buffer), size=batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*(self.buffer[i] for i in idx))
        return TransitionBatch(
            states=np.asarray(states, dtype=np.float32),
            actions=np.asarray(actions, dtype=np.int64),
            rewards=np.stack(rewards).astype(np.float32),
            next_states=np.asarray(next_states, dtype=np.float32),
            dones=np.asarray(dones, dtype=np.float32),
        )
