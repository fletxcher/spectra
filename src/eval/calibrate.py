"""Grid-search calibration for ssd_cpd and cusum against a noise-floor
estimate, run on a small held-out slice of calibration runs (distinct from
the 40-run evaluation set) to avoid tuning against the same data we score
on. Both indsl detectors ship with defaults that are badly mismatched to
our timescale/units (see findings in the eval writeup); this replaces those
defaults with values chosen from the data itself.
"""

from __future__ import annotations

import itertools
import warnings
from pathlib import Path

import indsl.detect as indsl_detect
import numpy as np
import pandas as pd

from src.eval.labels import steady_window_label, to_datetime_index
from src.eval.metrics import point_metrics

CALIBRATION_RUN_IDS = ["run_00001", "run_00051", "run_00101", "run_00151", "run_00176"]
CALIBRATION_DAYS = 4
CHANNELS = {"temperature": "temperature", "pressure": "pressure", "mass_flow": "mdot_air", "phi": "phi_cmd"}

CUSUM_K_DRIFT = [0.25, 0.5, 1.0, 2.0, 4.0]
CUSUM_K_THRESH = [3.0, 5.0, 8.0, 12.0, 20.0]

SSD_VAR_THRESHOLD = [5.0, 10.0, 20.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0]
SSD_SLOPE_THRESHOLD = [-6.0, -5.0, -4.0]


def _noise_std(series: pd.Series) -> float:
    return float(series.diff().std() / np.sqrt(2))


def _load_calibration_slice(run_id: str, channel_col: str) -> tuple[pd.Series, pd.Series]:
    df = pd.read_parquet(Path("datasets") / run_id / "data.parquet")
    df = df[df.time < CALIBRATION_DAYS * 86400.0]
    idx = to_datetime_index(df["time"])
    series = pd.Series(df[channel_col].to_numpy(), index=idx)
    raw_is_steady = pd.Series(df["is_steady"].to_numpy(), index=idx)
    y_true = steady_window_label(raw_is_steady)
    return series, y_true


def calibrate_cusum() -> dict:
    """Single global (k_drift, k_thresh) pair -- noise_std normalization
    should already make this roughly channel-agnostic."""
    results = []
    for k_drift, k_thresh in itertools.product(CUSUM_K_DRIFT, CUSUM_K_THRESH):
        f1s = []
        for run_id in CALIBRATION_RUN_IDS:
            for channel_col in CHANNELS.values():
                series, y_true = _load_calibration_slice(run_id, channel_col)
                noise_std = _noise_std(series)
                drift = k_drift * noise_std
                threshold = k_thresh * noise_std
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    out = indsl_detect.cusum(series, threshold=threshold, drift=drift, return_series_type="cusum_binary_result")
                y_pred = out == 0
                f1s.append(point_metrics(y_true, y_pred).f1)
        results.append({"k_drift": k_drift, "k_thresh": k_thresh, "mean_f1": float(np.mean(f1s))})
        print(results[-1], flush=True)

    best = max(results, key=lambda r: r["mean_f1"])
    return {"all_results": results, "best": best}


def calibrate_ssd_cpd() -> dict:
    """Per-channel (var_threshold, slope_threshold) grid, since the std/n
    segment-length bias and ED-Pelt's own segmentation behavior differ by
    channel noise/dynamics."""
    per_channel_best = {}
    all_results = {}
    for channel_label, channel_col in CHANNELS.items():
        results = []
        for var_threshold, slope_threshold in itertools.product(SSD_VAR_THRESHOLD, SSD_SLOPE_THRESHOLD):
            f1s = []
            for run_id in CALIBRATION_RUN_IDS:
                series, y_true = _load_calibration_slice(run_id, channel_col)
                min_distance = max(int(round(60.0 / 15.0)), 2)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    out = indsl_detect.ssd_cpd(
                        series, min_distance=min_distance, var_threshold=var_threshold, slope_threshold=slope_threshold
                    )
                y_pred = out.reindex(series.index).ffill().fillna(0) == 1
                f1s.append(point_metrics(y_true, y_pred).f1)
            results.append({"var_threshold": var_threshold, "slope_threshold": slope_threshold, "mean_f1": float(np.mean(f1s))})
            print(channel_label, results[-1], flush=True)

        best = max(results, key=lambda r: r["mean_f1"])
        per_channel_best[channel_label] = best
        all_results[channel_label] = results

    return {"all_results": all_results, "best": per_channel_best}
