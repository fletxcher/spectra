"""Assigns human-readable regime names to automatically-detected clusters,
ranked by mean load_fraction (the axis that actually separates them --
see evaluation notes on within-run feature collinearity)."""

from __future__ import annotations

import pandas as pd

# Ordered low -> high; a cluster count is mapped onto an evenly-spaced
# subset of this palette so names stay consistent across runs with
# different numbers of detected regimes.
LOAD_TIER_PALETTE = [
    "Minimum Load",
    "Low Load",
    "Below-Average Load",
    "Mid Load",
    "Above-Average Load",
    "High Load",
    "Near-Peak Load",
    "Peak Load",
]


def _tier_names(n: int) -> list[str]:
    if n == 1:
        return ["Steady Operating Point"]
    if n <= len(LOAD_TIER_PALETTE):
        if n == 2:
            return ["Off-Peak / Low Load", "Peak / High Load"]
        if n == 3:
            return ["Low Load", "Mid Load", "Peak / High Load"]
        step = (len(LOAD_TIER_PALETTE) - 1) / (n - 1)
        return [LOAD_TIER_PALETTE[round(i * step)] for i in range(n)]
    return [f"Load Band {i + 1}" for i in range(n)]


def name_clusters(points: pd.DataFrame, cluster_col: str = "kmeans_cluster") -> dict[int, str]:
    """Returns {cluster_label: name}, ranking clusters by mean load_fraction.
    Noise points (label -1, from HDBSCAN) are named separately."""
    real = points[points[cluster_col] != -1]
    means = real.groupby(cluster_col)["load_fraction"].mean().sort_values()
    names = _tier_names(len(means))
    mapping = {label: name for label, name in zip(means.index, names)}
    if (points[cluster_col] == -1).any():
        mapping[-1] = "Unclassified / Transitional"
    return mapping
