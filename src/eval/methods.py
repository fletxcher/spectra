"""Adapters around indsl.detect functions: each one has a different API,
output polarity, and (for several) internally resamples to a different time
grid. This module normalizes all of that into one interface -- a boolean
Series aligned to the input index, True = steady -- so they can be scored
uniformly against the ground-truth steady-window label.

Parameter choices below are reasonable, dt-aware first-pass defaults (scaled
to the 60s steady-window target and the data's actual sampling interval),
not exhaustively tuned; see README for the evaluation writeup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

import indsl.detect as indsl_detect


def _align(result: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
    """Several indsl detectors internally resample to a coarser/different
    time grid (confirmed empirically for ssd_cpd, cpd_ed_pelt,
    unchanged_signal_detector, oscillation_detector). Re-align their output
    back onto our evaluation index by forward-filling the last known value
    (step-function semantics -- a detector's call holds until it changes)."""
    result = result.sort_index()
    combined_index = target_index.union(result.index)
    return result.reindex(combined_index).ffill().reindex(target_index).fillna(0)


def _far_from_events(flags: pd.Series, target_index: pd.DatetimeIndex, window_seconds: float) -> pd.Series:
    """True everywhere except within `window_seconds` after a flagged event
    (flags == 1) -- used to turn impulse-style change-point markers into a
    steady/transient call. Vectorized via searchsorted."""
    event_times = flags.index[flags == 1]
    if len(event_times) == 0:
        return pd.Series(True, index=target_index)
    event_ns = event_times.values.astype("datetime64[ns]").astype(np.int64)
    target_ns = target_index.values.astype("datetime64[ns]").astype(np.int64)
    pos = np.searchsorted(event_ns, target_ns, side="right") - 1
    has_prior = pos >= 0
    time_since = np.where(has_prior, target_ns - event_ns[np.clip(pos, 0, None)], np.inf)
    return pd.Series(time_since >= window_seconds * 1e9, index=target_index)


@dataclass(frozen=True)
class Method:
    name: str
    run: Callable[[pd.Series, float], pd.Series]  # (series, dt_seconds) -> bool Series, True=steady


def _ssd_cpd(series: pd.Series, dt_seconds: float) -> pd.Series:
    min_distance = max(int(round(60.0 / dt_seconds)), 2)
    out = indsl_detect.ssd_cpd(series, min_distance=min_distance, var_threshold=2.0, slope_threshold=-3.0)
    return _align(out, series.index) == 1


def _ssid(series: pd.Series, dt_seconds: float) -> pd.Series:
    out = indsl_detect.ssid(series, ratio_lim=2.5, alpha1=0.2, alpha2=0.1, alpha3=0.1)
    return _align(out, series.index) == 0  # docs: steady=0, transient=1


def _vma(series: pd.Series, dt_seconds: float) -> pd.Series:
    window_length = max(int(round(60.0 / dt_seconds)), 2)
    vma_out = _align(indsl_detect.vma(series, window_length=window_length), series.index)
    diff = vma_out.diff().abs()
    eps = 1e-4 * max(float(series.std()), 1e-9)
    return diff.fillna(0) < eps


def _unchanged(series: pd.Series, dt_seconds: float) -> pd.Series:
    out = indsl_detect.unchanged_signal_detector(series, duration=pd.Timedelta(seconds=60), min_nr_data_points=3)
    return _align(out, series.index) == 1


def _cpd_ed_pelt(series: pd.Series, dt_seconds: float) -> pd.Series:
    min_distance = max(int(round(60.0 / dt_seconds)), 2)
    cp = _align(indsl_detect.cpd_ed_pelt(series, min_distance=min_distance), series.index)
    return _far_from_events(cp, series.index, window_seconds=60.0)


def _cusum(series: pd.Series, dt_seconds: float) -> pd.Series:
    out = _align(indsl_detect.cusum(series, return_series_type="cusum_binary_result"), series.index)
    return out == 0


def _drift(series: pd.Series, dt_seconds: float) -> pd.Series:
    out = indsl_detect.drift(
        series, long_interval=pd.Timedelta(hours=2), short_interval=pd.Timedelta(minutes=15), std_threshold=3, detect="both"
    )
    return _align(out, series.index) == 0


def _oscillation(series: pd.Series, dt_seconds: float) -> pd.Series:
    out = indsl_detect.oscillation_detector(series, order=4, threshold=0.2)
    return _align(out, series.index) == 0


METHODS: list[Method] = [
    Method("ssd_cpd", _ssd_cpd),
    Method("ssid", _ssid),
    Method("vma", _vma),
    Method("unchanged_signal_detector", _unchanged),
    Method("cpd_ed_pelt", _cpd_ed_pelt),
    Method("cusum", _cusum),
    Method("drift", _drift),
    Method("oscillation_detector", _oscillation),
]
