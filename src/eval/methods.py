"""Adapters around indsl.detect functions: each one has a different API,
output polarity, and (for several) internally resamples to a different time
grid. This module normalizes all of that into one interface -- a boolean
Series aligned to the input index, True = steady -- so they can be scored
uniformly against the ground-truth steady-window label.

The `_tuned` variants of ssd_cpd and cusum replace indsl's library defaults
(which turned out to be badly mismatched to this data's units/timescale --
see the evaluation writeup) with parameters calibrated via grid search
against a held-out calibration slice (src/eval/calibrate.py), scored
against noise-floor estimates rather than guessed.
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


def _noise_std(series: pd.Series) -> float:
    return float(series.diff().std() / np.sqrt(2))


@dataclass(frozen=True)
class Method:
    name: str
    run: Callable[[pd.Series, float, str], pd.Series]  # (series, dt_seconds, channel) -> bool Series, True=steady


PELT_DT_S = 60.0
PELT_MIN_DISTANCE = 2  # in PELT_DT_S samples: no segment shorter than 2 minutes


def _minute_means(series: pd.Series) -> pd.Series:
    """indsl's ssd_cpd and cpd_ed_pelt force-resample any input to >= 60 s
    spacing by keeping only the samples that land on the 60 s grid, which
    throws away 3 of every 4 samples at 15 s without averaging any noise
    away. Averaging into 60 s blocks first makes that resample a no-op and
    keeps the noise reduction."""
    return series.resample(f"{PELT_DT_S:g}s").mean().dropna()


def run_ssd_cpd(series: pd.Series, var_threshold: float, slope_threshold: float) -> pd.Series:
    out = indsl_detect.ssd_cpd(
        _minute_means(series), min_distance=PELT_MIN_DISTANCE, var_threshold=var_threshold, slope_threshold=slope_threshold
    )
    return _align(out, series.index) == 1


def _ssd_cpd(series: pd.Series, dt_seconds: float, channel: str) -> pd.Series:
    return run_ssd_cpd(series, var_threshold=2.0, slope_threshold=-3.0)


# Calibrated via grid search (src/eval/calibrate.py) against a held-out
# calibration slice, distinct from the evaluation run set, keyed by the
# calibration slice length in days. ED-Pelt segments a shorter series more
# finely, so the best thresholds depend on the input length; the detector
# uses the set calibrated on the length closest to its input.
SSD_CPD_TUNED_PARAMS = {
    4.0: {
        "temperature": dict(var_threshold=50.0, slope_threshold=-4.0),
        "pressure": dict(var_threshold=5.0, slope_threshold=-4.0),
        "mass_flow": dict(var_threshold=50.0, slope_threshold=-6.0),
        "phi": dict(var_threshold=50.0, slope_threshold=-7.0),
    },
    1.0: {
        "temperature": dict(var_threshold=20.0, slope_threshold=-6.0),
        "pressure": dict(var_threshold=2.0, slope_threshold=-4.0),
        "mass_flow": dict(var_threshold=50.0, slope_threshold=-6.0),
        "phi": dict(var_threshold=20.0, slope_threshold=-7.0),
    },
}


def _ssd_cpd_tuned(series: pd.Series, dt_seconds: float, channel: str) -> pd.Series:
    span_days = (series.index[-1] - series.index[0]).total_seconds() / 86400.0
    horizon = min(SSD_CPD_TUNED_PARAMS, key=lambda d: abs(np.log(d) - np.log(max(span_days, 1e-3))))
    return run_ssd_cpd(series, **SSD_CPD_TUNED_PARAMS[horizon][channel])


def _ssid(series: pd.Series, dt_seconds: float, channel: str) -> pd.Series:
    out = indsl_detect.ssid(series, ratio_lim=2.5, alpha1=0.2, alpha2=0.1, alpha3=0.1)
    return _align(out, series.index) == 0  # docs: steady=0, transient=1


def _vma(series: pd.Series, dt_seconds: float, channel: str) -> pd.Series:
    window_length = max(int(round(60.0 / dt_seconds)), 2)
    vma_out = _align(indsl_detect.vma(series, window_length=window_length), series.index)
    diff = vma_out.diff().abs()
    eps = 1e-4 * max(float(series.std()), 1e-9)
    return diff.fillna(0) < eps


def _unchanged(series: pd.Series, dt_seconds: float, channel: str) -> pd.Series:
    out = indsl_detect.unchanged_signal_detector(series, duration=pd.Timedelta(seconds=60), min_nr_data_points=3)
    return _align(out, series.index) == 1


def _cpd_ed_pelt(series: pd.Series, dt_seconds: float, channel: str) -> pd.Series:
    cp = _align(indsl_detect.cpd_ed_pelt(_minute_means(series), min_distance=PELT_MIN_DISTANCE), series.index)
    return _far_from_events(cp, series.index, window_seconds=60.0)


def _cusum(series: pd.Series, dt_seconds: float, channel: str) -> pd.Series:
    out = _align(indsl_detect.cusum(series, return_series_type="cusum_binary_result"), series.index)
    return out == 0


# Calibrated via grid search (by MCC), applied relative to a per-series
# noise-floor estimate (replaces indsl's default drift formula, which goes
# deeply negative -- and thus fires constantly -- on absolute-scale signals
# like Kelvin or Pascal).
CUSUM_K_DRIFT = 2.0
CUSUM_K_THRESH = 12.0


def _cusum_tuned(series: pd.Series, dt_seconds: float, channel: str) -> pd.Series:
    noise_std = _noise_std(series)
    drift = CUSUM_K_DRIFT * noise_std
    threshold = CUSUM_K_THRESH * noise_std
    out = _align(
        indsl_detect.cusum(series, threshold=threshold, drift=drift, return_series_type="cusum_binary_result"),
        series.index,
    )
    return out == 0


def _drift(series: pd.Series, dt_seconds: float, channel: str) -> pd.Series:
    out = indsl_detect.drift(
        series, long_interval=pd.Timedelta(hours=2), short_interval=pd.Timedelta(minutes=15), std_threshold=3, detect="both"
    )
    return _align(out, series.index) == 0


def _oscillation(series: pd.Series, dt_seconds: float, channel: str) -> pd.Series:
    out = indsl_detect.oscillation_detector(series, order=4, threshold=0.2)
    return _align(out, series.index) == 0


def _always_steady(series: pd.Series, dt_seconds: float, channel: str) -> pd.Series:
    """Trivial baseline: anything that can't beat this isn't detecting."""
    return pd.Series(True, index=series.index)


METHODS: list[Method] = [
    Method("always_steady", _always_steady),
    Method("ssd_cpd", _ssd_cpd),
    Method("ssd_cpd_tuned", _ssd_cpd_tuned),
    Method("ssid", _ssid),
    Method("vma", _vma),
    Method("unchanged_signal_detector", _unchanged),
    Method("cpd_ed_pelt", _cpd_ed_pelt),
    Method("cusum", _cusum),
    Method("cusum_tuned", _cusum_tuned),
    Method("drift", _drift),
    Method("oscillation_detector", _oscillation),
]
