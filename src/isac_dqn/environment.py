from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .radio import (
    antenna_gain_linear,
    angle_deg,
    generate_power_combinations,
    noise_power_w,
    path_gain_linear,
    sinr_to_cqi,
    wrapped_angle_diff_deg,
)


@dataclass
class EnvConfig:
    num_cells: int
    users_per_cell: int
    num_subbands: int
    cell_radius_m: float
    min_user_radius_ratio: float
    subband_bandwidth_hz: float
    noise_density_dbm_hz: float
    receiver_noise_figure_db: float
    target_ber: float
    antenna_3db_deg: float
    antenna_max_attenuation_db: float
    bs_x_m: list
    bs_y_m: list
    power_levels_w: list
    rb_per_subband: list
    max_cell_power_w: float
    max_steps_per_episode: int
    include_shadowing: bool
    shadowing_std_db: float
    include_rayleigh: bool
    sensing_noise_factor: float
    throughput_scale_mbps: float
    sensing_scale: float
    power_penalty_lambda: float
    sensing_split_levels: list


class ISACPowerAllocationEnv:
    """Multi-cell downlink power allocation with sensing-aware state/reward.

    The communication chain follows Project 04:
    power -> directional antenna gain -> macro path loss -> SINR -> CQI -> throughput.

    The extension adds one sensing target per cell and optimizes a weighted
    communication/sensing reward controlled by alpha.
    """

    def __init__(
        self,
        cfg: EnvConfig,
        alpha: float = 0.5,
        seed: int = 42,
        channel_perturbation_std_db: float = 0.0,
    ) -> None:
        self.cfg = cfg
        self.alpha = float(alpha)
        self.rng = np.random.default_rng(seed)
        self.channel_perturbation_std_db = float(channel_perturbation_std_db)

        self.bs_xy = np.column_stack([np.asarray(cfg.bs_x_m, dtype=np.float64), np.asarray(cfg.bs_y_m, dtype=np.float64)])
        if self.bs_xy.shape[0] != cfg.num_cells:
            raise ValueError("num_cells must match bs_x_m/bs_y_m length.")

        self.power_combinations = generate_power_combinations(
            cfg.power_levels_w,
            cfg.num_subbands,
            cfg.rb_per_subband,
            cfg.max_cell_power_w,
        )
        self.sensing_split_levels = np.asarray(cfg.sensing_split_levels, dtype=np.float32)
        if np.any(self.sensing_split_levels <= 0.0) or np.any(self.sensing_split_levels >= 1.0):
            raise ValueError("sensing_split_levels must be in the open interval (0, 1).")
        self.action_power_index, self.action_split_index = self._build_action_table()
        self.num_cell_actions = self.action_power_index.shape[0]
        self.action_dim = self.num_cell_actions * cfg.num_cells
        self.state_dim = cfg.num_cells * cfg.users_per_cell * (cfg.num_subbands + 1 + 4)
        self.noise_w = noise_power_w(cfg.noise_density_dbm_hz, cfg.subband_bandwidth_hz, cfg.receiver_noise_figure_db)
        self.alpha_ber = -1.5 / np.log(5.0 * cfg.target_ber)
        self.max_total_power_w = float(cfg.num_cells * cfg.max_cell_power_w)

        self.step_count = 0
        self.user_xy: Optional[np.ndarray] = None
        self.serving_cell: Optional[np.ndarray] = None
        self.dist_user_bs: Optional[np.ndarray] = None
        self.angle_user_bs: Optional[np.ndarray] = None
        self.user_channel_gain: Optional[np.ndarray] = None
        self.target_xy: Optional[np.ndarray] = None
        self.target_distance: Optional[np.ndarray] = None
        self.target_angle_offset: Optional[np.ndarray] = None
        self.target_channel_gain: Optional[np.ndarray] = None
        self.sensing_demand: Optional[np.ndarray] = None
        self.current_action = np.zeros(cfg.num_cells, dtype=np.int64)
        self.last_metrics: Dict[str, float] = {}

    def reset(self) -> np.ndarray:
        self.step_count = 0
        self._generate_users()
        self._generate_sensing_targets()
        self._generate_channel_gains()

        min_power_action = int(np.argmin(np.sum(self.power_combinations, axis=1)))
        low_split_idx = int(np.argmin(self.sensing_split_levels))
        matches = np.where((self.action_power_index == min_power_action) & (self.action_split_index == low_split_idx))[0]
        self.current_action = np.full(self.cfg.num_cells, int(matches[0]), dtype=np.int64)
        self.last_metrics = self.compute_metrics_for_action(self.current_action)
        return self._build_state(self.last_metrics)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        self.step_count += 1
        action = np.asarray(action, dtype=np.int64)
        action = np.clip(action, 0, self.num_cell_actions - 1)
        self.current_action = action.copy()

        metrics = self.compute_metrics_for_action(action)
        reward = self.compute_reward(metrics)
        self.last_metrics = metrics
        done = self.step_count >= self.cfg.max_steps_per_episode
        next_state = self._build_state(metrics)
        info = dict(metrics)
        info["reward"] = float(reward)
        return next_state, float(reward), done, info

    def evaluate_action(self, action: np.ndarray) -> Dict[str, float]:
        metrics = self.compute_metrics_for_action(np.asarray(action, dtype=np.int64))
        metrics["reward"] = float(self.compute_reward(metrics))
        return metrics

    def compute_metrics_for_action(self, action: np.ndarray) -> Dict[str, float | np.ndarray]:
        power_matrix = self.action_to_power_matrix(action)
        self._active_sensing_split = self.action_to_split_vector(action)
        return self.compute_metrics(power_matrix)

    def action_to_power_matrix(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.int64)
        return self.power_combinations[self.action_power_index[action]]

    def action_to_split_vector(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.int64)
        return self.sensing_split_levels[self.action_split_index[action]].astype(np.float64)

    def nearest_action_index(self, target_power_w: float, sensing_split: float = 0.3) -> int:
        target = np.full(self.cfg.num_subbands, target_power_w, dtype=np.float64)
        power_dist = np.linalg.norm(self.power_combinations - target[None, :], axis=1)
        power_idx = int(np.argmin(power_dist))
        split_idx = int(np.argmin(np.abs(self.sensing_split_levels - sensing_split)))
        matches = np.where((self.action_power_index == power_idx) & (self.action_split_index == split_idx))[0]
        return int(matches[0])

    def max_power_action_index(self, sensing_split: float = 0.3) -> int:
        rb = np.asarray(self.cfg.rb_per_subband, dtype=np.float64)
        weighted_power = self.power_combinations @ rb
        power_idx = int(np.argmax(weighted_power))
        split_idx = int(np.argmin(np.abs(self.sensing_split_levels - sensing_split)))
        matches = np.where((self.action_power_index == power_idx) & (self.action_split_index == split_idx))[0]
        return int(matches[0])

    def random_action(self) -> np.ndarray:
        return self.rng.integers(0, self.num_cell_actions, size=self.cfg.num_cells, dtype=np.int64)

    def compute_metrics(self, power_matrix: np.ndarray) -> Dict[str, float | np.ndarray]:
        cfg = self.cfg
        assert self.user_channel_gain is not None
        assert self.serving_cell is not None
        assert self.target_channel_gain is not None
        assert self.sensing_demand is not None

        # In this ISAC extension, each selected action represents two decisions:
        # total subband power and a sensing power split. Communication and sensing
        # therefore compete for the same cell power budget.
        sensing_split = getattr(self, "_active_sensing_split", np.full(cfg.num_cells, 0.3, dtype=np.float64))
        comm_power_matrix = power_matrix * (1.0 - sensing_split[:, None])
        sensing_power_matrix = power_matrix * sensing_split[:, None]

        num_users = cfg.num_cells * cfg.users_per_cell
        sinr = np.zeros((num_users, cfg.num_subbands), dtype=np.float64)

        for user_idx in range(num_users):
            cell = int(self.serving_cell[user_idx])
            for f in range(cfg.num_subbands):
                signal = comm_power_matrix[cell, f] * self.user_channel_gain[user_idx, cell, f]
                interference = 0.0
                for other in range(cfg.num_cells):
                    if other != cell:
                        interference += comm_power_matrix[other, f] * self.user_channel_gain[user_idx, other, f]
                sinr[user_idx, f] = signal / (self.noise_w + interference + 1e-30)

        rate_per_user_subband = cfg.subband_bandwidth_hz * np.log2(1.0 + self.alpha_ber * sinr)
        selected_rate = np.zeros((cfg.num_cells, cfg.num_subbands), dtype=np.float64)
        selected_sinr = np.zeros((cfg.num_cells, cfg.num_subbands), dtype=np.float64)

        for cell in range(cfg.num_cells):
            start = cell * cfg.users_per_cell
            end = start + cfg.users_per_cell
            for f in range(cfg.num_subbands):
                local_rates = rate_per_user_subband[start:end, f]
                best_local = int(np.argmax(local_rates))
                selected_rate[cell, f] = local_rates[best_local]
                selected_sinr[cell, f] = sinr[start + best_local, f]

        throughput_mbps = float(np.sum(selected_rate) / 1e6)
        cell_throughput_mbps = np.sum(selected_rate, axis=1) / 1e6
        avg_sinr_db = float(np.mean(10.0 * np.log10(np.maximum(selected_sinr, 1e-18))))
        cqi = sinr_to_cqi(10.0 * np.log10(np.maximum(sinr, 1e-18)))
        avg_cqi = float(np.mean(cqi))

        target_snr = np.zeros(cfg.num_cells, dtype=np.float64)
        for cell in range(cfg.num_cells):
            sensing_signal = np.sum(sensing_power_matrix[cell, :] * self.target_channel_gain[cell, :])
            target_snr[cell] = sensing_signal / (self.noise_w * cfg.sensing_noise_factor + 1e-30)
        sensing_score_per_cell = self.sensing_demand * np.log2(1.0 + target_snr)
        sensing_score = float(np.mean(sensing_score_per_cell))

        rb = np.asarray(cfg.rb_per_subband, dtype=np.float64)
        total_power_w = float(np.sum(power_matrix * rb[None, :]))
        cell_power_w = np.sum(power_matrix * rb[None, :], axis=1)
        avg_sensing_split = float(np.mean(sensing_split))

        cell_comm_norm = np.tanh(cell_throughput_mbps / max(cfg.throughput_scale_mbps / cfg.num_cells, 1e-12))
        cell_sense_norm = np.tanh(sensing_score_per_cell / max(cfg.sensing_scale, 1e-12))
        cell_power_norm = cell_power_w / max(cfg.max_cell_power_w, 1e-12)
        per_cell_reward = (
            self.alpha * cell_comm_norm
            + (1.0 - self.alpha) * cell_sense_norm
            - cfg.power_penalty_lambda * cell_power_norm
        )

        return {
            "throughput_mbps": throughput_mbps,
            "sensing_score": sensing_score,
            "total_power_w": total_power_w,
            "avg_sinr_db": avg_sinr_db,
            "avg_cqi": avg_cqi,
            "comm_norm": float(np.tanh(throughput_mbps / cfg.throughput_scale_mbps)),
            "sense_norm": float(np.tanh(sensing_score / cfg.sensing_scale)),
            "power_norm": float(total_power_w / max(self.max_total_power_w, 1e-12)),
            "avg_sensing_split": avg_sensing_split,
            "per_cell_reward": per_cell_reward.astype(np.float32),
            "cell_throughput_mbps": cell_throughput_mbps.astype(np.float32),
            "cell_sensing_score": sensing_score_per_cell.astype(np.float32),
            "sinr_matrix": sinr,
            "cqi_matrix": cqi,
            "target_snr_db": 10.0 * np.log10(np.maximum(target_snr, 1e-18)),
        }

    def compute_reward(self, metrics: Dict[str, float]) -> float:
        if "per_cell_reward" in metrics:
            return float(np.mean(metrics["per_cell_reward"]))
        comm = float(metrics["comm_norm"])
        sense = float(metrics["sense_norm"])
        power = float(metrics["power_norm"])
        return self.alpha * comm + (1.0 - self.alpha) * sense - self.cfg.power_penalty_lambda * power

    def _build_state(self, metrics: Dict[str, float | np.ndarray]) -> np.ndarray:
        cfg = self.cfg
        assert self.dist_user_bs is not None
        assert self.serving_cell is not None
        assert self.target_distance is not None
        assert self.target_angle_offset is not None
        assert self.target_channel_gain is not None
        assert self.sensing_demand is not None

        cqi = np.asarray(metrics["cqi_matrix"], dtype=np.float64) / 15.0
        target_snr_db = np.asarray(metrics["target_snr_db"], dtype=np.float64)
        target_quality = np.tanh(np.maximum(target_snr_db, -20.0) / 20.0)
        target_quality = (target_quality + 1.0) / 2.0

        features = []
        for user_idx in range(cfg.num_cells * cfg.users_per_cell):
            cell = int(self.serving_cell[user_idx])
            user_features = []
            user_features.extend(cqi[user_idx, :].tolist())
            dist_to_serving = self.dist_user_bs[user_idx, cell]
            user_features.append(float(dist_to_serving / cfg.cell_radius_m >= 0.5))
            user_features.append(float(np.clip(self.target_distance[cell] / cfg.cell_radius_m, 0.0, 2.0) / 2.0))
            user_features.append(float(np.clip(target_quality[cell], 0.0, 1.0)))
            angle_gain = antenna_gain_linear(
                self.target_angle_offset[cell],
                cfg.antenna_3db_deg,
                cfg.antenna_max_attenuation_db,
            )
            user_features.append(float(np.clip(angle_gain, 0.0, 1.0)))
            user_features.append(float(np.clip(self.sensing_demand[cell], 0.0, 1.0)))
            features.extend(user_features)
        return np.asarray(features, dtype=np.float32)

    def _build_action_table(self) -> tuple[np.ndarray, np.ndarray]:
        power_idx = []
        split_idx = []
        for p in range(self.power_combinations.shape[0]):
            for s in range(len(self.sensing_split_levels)):
                power_idx.append(p)
                split_idx.append(s)
        return np.asarray(power_idx, dtype=np.int64), np.asarray(split_idx, dtype=np.int64)

    def _generate_users(self) -> None:
        cfg = self.cfg
        users = []
        serving = []
        r_min = cfg.min_user_radius_ratio * cfg.cell_radius_m
        for cell in range(cfg.num_cells):
            angles = self.rng.uniform(0.0, 2.0 * np.pi, size=cfg.users_per_cell)
            radii = np.sqrt(self.rng.uniform(r_min**2, cfg.cell_radius_m**2, size=cfg.users_per_cell))
            x = self.bs_xy[cell, 0] + radii * np.cos(angles)
            y = self.bs_xy[cell, 1] + radii * np.sin(angles)
            users.append(np.column_stack([x, y]))
            serving.extend([cell] * cfg.users_per_cell)
        self.user_xy = np.vstack(users)
        self.serving_cell = np.asarray(serving, dtype=np.int64)

        diff = self.user_xy[:, None, :] - self.bs_xy[None, :, :]
        self.dist_user_bs = np.linalg.norm(diff, axis=2)
        self.angle_user_bs = np.zeros((self.user_xy.shape[0], cfg.num_cells), dtype=np.float64)
        for bs in range(cfg.num_cells):
            self.angle_user_bs[:, bs] = angle_deg(self.bs_xy[bs, 0], self.bs_xy[bs, 1], self.user_xy[:, 0], self.user_xy[:, 1])

    def _generate_sensing_targets(self) -> None:
        cfg = self.cfg
        r_min = cfg.min_user_radius_ratio * cfg.cell_radius_m
        angles = self.rng.uniform(0.0, 2.0 * np.pi, size=cfg.num_cells)
        radii = np.sqrt(self.rng.uniform(r_min**2, cfg.cell_radius_m**2, size=cfg.num_cells))
        x = self.bs_xy[:, 0] + radii * np.cos(angles)
        y = self.bs_xy[:, 1] + radii * np.sin(angles)
        self.target_xy = np.column_stack([x, y])
        self.target_distance = radii
        self.sensing_demand = self.rng.uniform(0.5, 1.0, size=cfg.num_cells)

        beam_dirs = self._cell_beam_directions()
        target_angle = np.zeros(cfg.num_cells, dtype=np.float64)
        for cell in range(cfg.num_cells):
            target_angle[cell] = angle_deg(self.bs_xy[cell, 0], self.bs_xy[cell, 1], np.array([x[cell]]), np.array([y[cell]]))[0]
        nearest_beam = np.min(wrapped_angle_diff_deg(target_angle[:, None], beam_dirs), axis=1)
        self.target_angle_offset = nearest_beam

    def _generate_channel_gains(self) -> None:
        cfg = self.cfg
        assert self.dist_user_bs is not None
        assert self.angle_user_bs is not None
        assert self.target_distance is not None
        assert self.target_angle_offset is not None

        beam_dirs = self._cell_beam_directions()
        num_users = cfg.num_cells * cfg.users_per_cell
        gain = np.zeros((num_users, cfg.num_cells, cfg.num_subbands), dtype=np.float64)

        for bs in range(cfg.num_cells):
            angle_offsets = np.min(wrapped_angle_diff_deg(self.angle_user_bs[:, bs][:, None], beam_dirs[bs][None, :]), axis=1)
            ant_gain = antenna_gain_linear(angle_offsets, cfg.antenna_3db_deg, cfg.antenna_max_attenuation_db)
            path_gain = path_gain_linear(self.dist_user_bs[:, bs])
            for f in range(cfg.num_subbands):
                g = ant_gain * path_gain
                if cfg.include_shadowing:
                    shadow_db = self.rng.normal(0.0, cfg.shadowing_std_db, size=num_users)
                    if self.channel_perturbation_std_db > 0:
                        shadow_db += self.rng.normal(0.0, self.channel_perturbation_std_db, size=num_users)
                    g = g * 10.0 ** (shadow_db / 10.0)
                if cfg.include_rayleigh:
                    rayleigh_power = self.rng.exponential(scale=1.0, size=num_users)
                    g = g * rayleigh_power
                gain[:, bs, f] = np.maximum(g, 1e-30)
        self.user_channel_gain = gain

        target_gain = np.zeros((cfg.num_cells, cfg.num_subbands), dtype=np.float64)
        for cell in range(cfg.num_cells):
            ant_gain = antenna_gain_linear(self.target_angle_offset[cell], cfg.antenna_3db_deg, cfg.antenna_max_attenuation_db)
            base_gain = ant_gain * path_gain_linear(self.target_distance[cell])
            for f in range(cfg.num_subbands):
                g = base_gain
                if cfg.include_shadowing:
                    g *= 10.0 ** (self.rng.normal(0.0, cfg.shadowing_std_db) / 10.0)
                if cfg.include_rayleigh:
                    g *= self.rng.exponential(scale=1.0)
                target_gain[cell, f] = max(float(g), 1e-30)
        self.target_channel_gain = target_gain

    def _cell_beam_directions(self) -> np.ndarray:
        # Project 04 uses three 120-degree sectors. We align them to 60/180/300 deg.
        base = np.array([60.0, 180.0, 300.0], dtype=np.float64)
        return np.tile(base[None, :], (self.cfg.num_cells, 1))


def make_env_config(raw: dict) -> EnvConfig:
    return EnvConfig(**raw)
