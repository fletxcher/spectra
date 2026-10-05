"""Grid-search calibration for ssd_cpd and cusum against a noise-floor
estimate, run on a small held-out slice of calibration runs (distinct from
the 40-run evaluation set) to avoid tuning against the same data we score
on. Both indsl detectors ship with defaults that are badly mismatched to
our timescale/units (see findings in the eval writeup); this replaces those
defaults with values chosen from the data itself.

Calibration is per (method, channel) so each pair can run in its own
process:

    .venv/bin/python -m src.eval.calibrate ssd_cpd temperature
    .venv/bin/python -m src.eval.calibrate cusum pressure

Each writes calib_<method>_<channel>.json. cusum uses one global
(k_drift, k_thresh) pair, chosen by mean MCC across the four per-channel
result files (see `best_cusum`).
"""

from __future__ import annotations

import itertools
import json
import sys
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

CUSUM_K_DRIFT = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
CUSUM_K_THRESH = [0.5, 1.0, 2.0, 3.0, 5.0, 12.0, 20.0, 40.0, 80.0]

SSD_VAR_THRESHOLD = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0, 2500.0]
SSD_SLOPE_THRESHOLD = [-7.0, -6.0, -5.0, -4.0, -3.0]


def _noise_std(series: pd.Series) -> float:
    return float(series.diff().std() / np.sqrt(2))


def _load_calibration_slices(channel_col: str) -> list[tuple[pd.Series, pd.Series]]:
    slices = []
    for run_id in CALIBRATION_RUN_IDS:
        df = pd.read_parquet(Path("datasets") / run_id / "data.parquet", columns=["time", "is_steady", channel_col])
        df = df[df.time < CALIBRATION_DAYS * 86400.0]
        idx = to_datetime_index(df["time"])
        series = pd.Series(df[channel_col].to_numpy(), index=idx)
        y_true = steady_window_label(pd.Series(df["is_steady"].to_numpy(), index=idx))
        slices.append((series, y_true))
    return slices


def calibrate_cusum(channel: str) -> list[dict]:
    slices = _load_calibration_slices(CHANNELS[channel])
    results = []
    for k_drift, k_thresh in itertools.product(CUSUM_K_DRIFT, CUSUM_K_THRESH):
        scores = []
        for series, y_true in slices:
            noise_std = _noise_std(series)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = indsl_detect.cusum(
                    series, threshold=k_thresh * noise_std, drift=k_drift * noise_std, return_series_type="cusum_binary_result"
                )
            y_pred = out.reindex(series.index).ffill().fillna(1) == 0
            scores.append(point_metrics(y_true, y_pred))
        results.append({"k_drift": k_drift, "k_thresh": k_thresh, **_summarize(scores)})
        print(channel, results[-1], flush=True)
    return results


def calibrate_ssd_cpd(channel: str) -> list[dict]:
    """Per-channel (var_threshold, slope_threshold) grid, since the std/n
    segment-length bias and ED-Pelt's own segmentation behavior differ by
    channel noise/dynamics."""
    slices = _load_calibration_slices(CHANNELS[channel])
    min_distance = max(int(round(60.0 / 15.0)), 2)
    results = []
    for var_threshold, slope_threshold in itertools.product(SSD_VAR_THRESHOLD, SSD_SLOPE_THRESHOLD):
        scores = []
        for series, y_true in slices:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = indsl_detect.ssd_cpd(
                    series, min_distance=min_distance, var_threshold=var_threshold, slope_threshold=slope_threshold
                )
            y_pred = out.reindex(series.index).ffill().fillna(0) == 1
            scores.append(point_metrics(y_true, y_pred))
        results.append({"var_threshold": var_threshold, "slope_threshold": slope_threshold, **_summarize(scores)})
        print(channel, results[-1], flush=True)
    return results


def best_cusum(result_dir: str | Path = ".") -> dict:
    """Global (k_drift, k_thresh) with the best MCC averaged over channels."""
    by_params: dict[tuple[float, float], list[float]] = {}
    for channel in CHANNELS:
        for r in json.loads((Path(result_dir) / f"calib_cusum_{channel}.json").read_text()):
            by_params.setdefault((r["k_drift"], r["k_thresh"]), []).append(r["mean_mcc"])
    (k_drift, k_thresh), mccs = max(by_params.items(), key=lambda kv: np.mean(kv[1]))
    return {"k_drift": k_drift, "k_thresh": k_thresh, "mean_mcc": float(np.mean(mccs))}


def best_ssd_cpd(channel: str, result_dir: str | Path = ".") -> dict:
    results = json.loads((Path(result_dir) / f"calib_ssd_cpd_{channel}.json").read_text())
    return max(results, key=lambda r: r["mean_mcc"])


def _summarize(scores: list) -> dict:
    """Selection is on mean_mcc; F1 kept alongside for comparison with v1."""
    return {
        "mean_mcc": float(np.mean([s.mcc for s in scores])),
        "mean_f1": float(np.mean([s.f1 for s in scores])),
        "mean_pred_steady_rate": float(np.mean([s.n_pred_steady / s.n_total for s in scores])),
    }


if __name__ == "__main__":
    method, channel = sys.argv[1], sys.argv[2]
    fn = {"ssd_cpd": calibrate_ssd_cpd, "cusum": calibrate_cusum}[method]
    Path(f"calib_{method}_{channel}.json").write_text(json.dumps(fn(channel), indent=2))
    print("DONE", flush=True)
