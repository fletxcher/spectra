"""Scoring: point-wise classification metrics, and window-level detection
(did the method find a usable >=60s steady window inside each true steady
region, and how much lag before it did)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.eval.labels import WINDOW_SECONDS, steady_window_label


@dataclass(frozen=True)
class PointMetrics:
    precision: float
    recall: float
    f1: float
    accuracy: float
    n_true_steady: int
    n_pred_steady: int
    n_total: int


def point_metrics(y_true: pd.Series, y_pred: pd.Series) -> PointMetrics:
    y_true = y_true.astype(bool)
    y_pred = y_pred.reindex(y_true.index).fillna(False).astype(bool)

    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else 0.0

    return PointMetrics(precision, recall, f1, accuracy, int(y_true.sum()), int(y_pred.sum()), len(y_true))


@dataclass(frozen=True)
class WindowMetrics:
    n_regions: int
    n_detected: int
    detection_rate: float
    median_lag_s: float | None
    mean_lag_s: float | None


def _contiguous_true_regions(mask: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    mask = mask.astype(bool)
    if not mask.any():
        return []
    group = (mask != mask.shift(fill_value=False)).cumsum()
    regions = []
    for _, g in mask[mask].groupby(group[mask]):
        regions.append((g.index[0], g.index[-1]))
    return regions


def window_metrics(
    raw_is_steady: pd.Series, y_pred: pd.Series, window_seconds: int = WINDOW_SECONDS
) -> WindowMetrics:
    """For each contiguous true-steady region (from the raw, per-sample
    ground truth) long enough to ever contain a valid window, check whether
    the method's predictions contain a qualifying >=window_seconds
    continuous steady stretch inside that region, and the lag between the
    earliest a window could legitimately be claimed (region start +
    window_seconds) and when the method actually found one."""
    regions = _contiguous_true_regions(raw_is_steady)
    y_pred = y_pred.reindex(raw_is_steady.index).fillna(False)
    pred_window = steady_window_label(y_pred, window_seconds)

    lags = []
    n_detected = 0
    n_eligible = 0
    for start, end in regions:
        duration = (end - start).total_seconds()
        if duration < window_seconds:
            continue
        n_eligible += 1
        earliest_possible = start + pd.Timedelta(seconds=window_seconds)
        window_in_region = pred_window.loc[start:end]
        hits = window_in_region[window_in_region]
        if len(hits) > 0:
            n_detected += 1
            lags.append((hits.index[0] - earliest_possible).total_seconds())

    detection_rate = n_detected / n_eligible if n_eligible > 0 else 0.0
    median_lag = float(np.median(lags)) if lags else None
    mean_lag = float(np.mean(lags)) if lags else None

    return WindowMetrics(n_eligible, n_detected, detection_rate, median_lag, mean_lag)
