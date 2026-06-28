from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch.optim import RMSprop

from .replay_buffer import ReplayBuffer
from .utils import linear_epsilon


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, output_dim: int, hidden_sizes: List[int]) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        last = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class DQNConfig:
    hidden_sizes: List[int]
    gamma: float
    learning_rate: float
    batch_size: int
    replay_capacity: int
    min_replay_size: int
    target_update_interval: int
    epsilon_start: float
    epsilon_end: float
    epsilon_decay_steps: int
    gradient_clip_norm: float


class MultiCellDQNAgent:
    def __init__(
        self,
        state_dim: int,
        num_cells: int,
        num_cell_actions: int,
        cfg: DQNConfig,
        seed: int = 42,
        device: Optional[str] = None,
    ) -> None:
        self.state_dim = state_dim
        self.num_cells = num_cells
        self.num_cell_actions = num_cell_actions
        self.output_dim = num_cells * num_cell_actions
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.q = QNetwork(state_dim, self.output_dim, cfg.hidden_sizes).to(self.device)
        self.target_q = QNetwork(state_dim, self.output_dim, cfg.hidden_sizes).to(self.device)
        self.target_q.load_state_dict(self.q.state_dict())
        self.target_q.eval()

        self.optim = RMSprop(self.q.parameters(), lr=cfg.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay = ReplayBuffer(cfg.replay_capacity, seed=seed)
        self.learn_steps = 0
        self.act_steps = 0

    def epsilon(self) -> float:
        return linear_epsilon(self.act_steps, self.cfg.epsilon_start, self.cfg.epsilon_end, self.cfg.epsilon_decay_steps)

    @torch.no_grad()
    def act(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        eps = self.epsilon() if explore else 0.0
        self.act_steps += int(explore)
        if explore and self.rng.random() < eps:
            return self.rng.integers(0, self.num_cell_actions, size=self.num_cells, dtype=np.int64)

        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        q_values = self.q(s).cpu().numpy()[0]
        actions = []
        for cell in range(self.num_cells):
            start = cell * self.num_cell_actions
            end = start + self.num_cell_actions
            actions.append(int(np.argmax(q_values[start:end])))
        return np.asarray(actions, dtype=np.int64)

    def observe(self, state: np.ndarray, action: np.ndarray, reward: np.ndarray | float, next_state: np.ndarray, done: bool) -> None:
        self.replay.add(state, action, reward, next_state, done)

    def update(self) -> Optional[Dict[str, float]]:
        if len(self.replay) < max(self.cfg.min_replay_size, self.cfg.batch_size):
            return None

        batch = self.replay.sample(self.cfg.batch_size)
        states = torch.as_tensor(batch.states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.device)
        if rewards.ndim == 1:
            rewards = rewards.unsqueeze(1).repeat(1, self.num_cells)
        next_states = torch.as_tensor(batch.next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=self.device)

        q_all = self.q(states).view(-1, self.num_cells, self.num_cell_actions)
        q_selected = q_all.gather(2, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            target_all = self.target_q(next_states).view(-1, self.num_cells, self.num_cell_actions)
            next_max = target_all.max(dim=2).values
            target = rewards + self.cfg.gamma * (1.0 - dones.unsqueeze(1)) * next_max

        loss = self.loss_fn(q_selected, target)
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.gradient_clip_norm)
        self.optim.step()

        self.learn_steps += 1
        if self.learn_steps % self.cfg.target_update_interval == 0:
            self.target_q.load_state_dict(self.q.state_dict())

        return {"loss": float(loss.item())}

    def save(self, path: str | Path, meta: Optional[dict] = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.q.state_dict(),
                "target_state_dict": self.target_q.state_dict(),
                "meta": meta or {},
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        payload = torch.load(path, map_location=self.device)
        self.q.load_state_dict(payload["state_dict"])
        self.target_q.load_state_dict(payload.get("target_state_dict", payload["state_dict"]))
