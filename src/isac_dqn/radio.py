from __future__ import annotations

import itertools
from typing import Iterable, List

import numpy as np


def watt_to_dbm(power_w: np.ndarray | float) -> np.ndarray | float:
    power_w = np.maximum(power_w, 1e-18)
    return 10.0 * np.log10(power_w * 1000.0)


def dbm_to_watt(power_dbm: np.ndarray | float) -> np.ndarray | float:
    return 10.0 ** ((power_dbm - 30.0) / 10.0)


def noise_power_w(noise_density_dbm_hz: float, bandwidth_hz: float, noise_figure_db: float) -> float:
    noise_dbm = noise_density_dbm_hz + 10.0 * np.log10(bandwidth_hz) + noise_figure_db
    return float(dbm_to_watt(noise_dbm))


def macro_path_loss_db(distance_m: np.ndarray | float) -> np.ndarray | float:
    """Project-04 macro-cell path-loss model: PL=128.1+37.6log10(d_km)."""
    distance_km = np.maximum(distance_m, 1.0) / 1000.0
    return 128.1 + 37.6 * np.log10(distance_km)


def path_gain_linear(distance_m: np.ndarray | float) -> np.ndarray | float:
    return 10.0 ** (-macro_path_loss_db(distance_m) / 10.0)


def antenna_gain_linear(angle_offset_deg: np.ndarray | float, theta_3db: float, max_attenuation_db: float) -> np.ndarray | float:
    attenuation_db = -np.minimum(12.0 * (np.asarray(angle_offset_deg) / theta_3db) ** 2, max_attenuation_db)
    return 10.0 ** (attenuation_db / 10.0)


def angle_deg(x0: float, y0: float, x1: np.ndarray, y1: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan2(y1 - y0, x1 - x0)) % 360.0


def wrapped_angle_diff_deg(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray | float:
    return np.abs((np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0)


def sinr_to_cqi(sinr_db: np.ndarray | float) -> np.ndarray:
    """Approximate LTE-style CQI mapping used for compact state representation."""
    thresholds = np.array([-5, -2.5, 0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25, 27.5])
    return np.clip(np.digitize(sinr_db, thresholds) + 1, 1, 15)


def generate_power_combinations(
    power_levels_w: Iterable[float],
    num_subbands: int,
    rb_per_subband: Iterable[int],
    max_cell_power_w: float,
) -> np.ndarray:
    combos: List[List[float]] = []
    rb = np.asarray(list(rb_per_subband), dtype=np.float64)
    for combo in itertools.product(power_levels_w, repeat=num_subbands):
        arr = np.asarray(combo, dtype=np.float64)
        if float(np.sum(arr * rb)) <= max_cell_power_w + 1e-12:
            combos.append(arr.tolist())
    if not combos:
        raise ValueError("No feasible power combinations under max_cell_power_w.")
    return np.asarray(combos, dtype=np.float32)
