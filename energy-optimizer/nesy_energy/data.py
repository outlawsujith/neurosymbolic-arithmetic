"""Plant data: a synthetic generator for demos and a CSV loader for real history.

Feature order used everywhere: [sin(hour), cos(hour), weekend, temp_c, production].
"""
from __future__ import annotations

import csv

import numpy as np

from .plant import H

PRODUCTION = 4  # index of the production-rate feature


def features(hour, weekend, temp_c, production) -> np.ndarray:
    hour = np.asarray(hour, dtype=float)
    a = 2 * np.pi * hour / 24
    cols = [np.sin(a), np.cos(a), np.broadcast_to(np.asarray(weekend, dtype=float), hour.shape),
            np.asarray(temp_c, dtype=float), np.asarray(production, dtype=float)]
    return np.stack(cols, axis=1)


def _true_load(hour, weekend, temp_c, production):
    """Hidden ground truth of the synthetic plant's non-shiftable load (kW)."""
    staffed = ((hour >= 6) & (hour < 22)).astype(float) * (1 - 0.6 * weekend)
    return 110 + 90 * production + 3.0 * np.maximum(0, temp_c - 18) + 20 * staffed


def day_inputs(rng: np.random.Generator, weekend: bool = False):
    """Weather and production plan for one day."""
    hour = np.arange(H)
    temp = rng.uniform(12, 26) + 6 * np.sin(2 * np.pi * (hour - 9) / 24) + rng.normal(0, 0.8, H)
    shift = (hour >= 6) & (hour < 22)
    level = (0.25 if weekend else rng.uniform(0.7, 1.0))
    prod = np.where(shift, level, 0.25) + rng.normal(0, 0.04, H)
    return temp, np.clip(prod, 0, 1)


def synth_history(days: int = 60, seed: int = 0):
    """Hourly history: features X and measured base load y (kW, with sensor noise)."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    hour = np.arange(H)
    for d in range(days):
        weekend = float(d % 7 >= 5)
        temp, prod = day_inputs(rng, bool(weekend))
        X.append(features(hour, weekend, temp, prod))
        y.append(_true_load(hour, weekend, temp, prod) + rng.normal(0, 8, H))
    return np.concatenate(X), np.concatenate(y)


def load_history_csv(path: str):
    """Real plant history. Columns: hour, weekend, temp_c, production, load_kw."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{path} has no rows")
    col = lambda k: np.array([float(r[k]) for r in rows])
    return features(col("hour"), col("weekend"), col("temp_c"), col("production")), col("load_kw")


def pv_profile(kwp: float, cloud: float = 0.75) -> np.ndarray:
    """Simple clear-sky bell between 06:00 and 18:00, scaled by a cloud factor."""
    hour = np.arange(H)
    return kwp * cloud * np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None)
