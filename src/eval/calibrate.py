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

Each writes results/calibration/calib_<method>_<channel>.json. cusum uses one global
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
from src.eval.methods import run_ssd_cpd
from src.eval.metrics import point_metrics

CALIBRATION_RUN_IDS = ["run_00001", "run_00051", "run_00101", "run_00151", "run_00176"]
CALIBRATION_DAYS = 4
RESULTS_DIR = Path("results/calibration")
CHANNELS = {"temperature": "temperature", "pressure": "pressure", "mass_flow": "mdot_air", "phi": "phi_cmd"}

CUSUM_K_DRIFT = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
CUSUM_K_THRESH = [0.5, 1.0, 2.0, 3.0, 5.0, 12.0, 20.0, 40.0, 80.0]

SSD_VAR_THRESHOLD = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0, 2500.0]
SSD_SLOPE_THRESHOLD = [-7.0, -6.0, -5.0, -4.0, -3.0]


def _noise_std(series: pd.Series) -> float:
    return float(series.diff().std() / np.sqrt(2))


def _load_calibration_slices(channel_col: str, days: float = CALIBRATION_DAYS) -> list[tuple[pd.Series, pd.Series]]:
    slices = []
    for run_id in CALIBRATION_RUN_IDS:
        df = pd.read_parquet(Path("datasets") / run_id / "data.parquet", columns=["time", "is_steady", channel_col])
        df = df[df.time < days * 86400.0]
        idx = to_datetime_index(df["time"])
        series = pd.Series(df[channel_col].to_numpy(), index=idx)
        y_true = steady_window_label(pd.Series(df["is_steady"].to_numpy(), index=idx))
        slices.append((series, y_true))
    return slices


def calibrate_cusum(channel: str, days: float = CALIBRATION_DAYS) -> list[dict]:
    slices = _load_calibration_slices(CHANNELS[channel], days)
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


def calibrate_ssd_cpd(channel: str, days: float = CALIBRATION_DAYS) -> list[dict]:
    """Per-channel (var_threshold, slope_threshold) grid, since the std/n
    segment-length bias and ED-Pelt's own segmentation behavior differ by
    channel noise/dynamics."""
    slices = _load_calibration_slices(CHANNELS[channel], days)
    results = []
    for var_threshold, slope_threshold in itertools.product(SSD_VAR_THRESHOLD, SSD_SLOPE_THRESHOLD):
        scores = []
        for series, y_true in slices:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y_pred = run_ssd_cpd(series, var_threshold=var_threshold, slope_threshold=slope_threshold)
            scores.append(point_metrics(y_true, y_pred))
        results.append({"var_threshold": var_threshold, "slope_threshold": slope_threshold, **_summarize(scores)})
        print(channel, results[-1], flush=True)
    return results


def _result_path(method: str, channel: str, suffix: str = "", result_dir: str | Path = RESULTS_DIR) -> Path:
    return Path(result_dir) / f"calib_{method}_{channel}{suffix}.json"


def best_cusum(result_dir: str | Path = RESULTS_DIR, suffix: str = "") -> dict:
    """Global (k_drift, k_thresh) with the best MCC averaged over channels."""
    by_params: dict[tuple[float, float], list[float]] = {}
    for channel in CHANNELS:
        for r in json.loads(_result_path("cusum", channel, suffix, result_dir).read_text()):
            by_params.setdefault((r["k_drift"], r["k_thresh"]), []).append(r["mean_mcc"])
    (k_drift, k_thresh), mccs = max(by_params.items(), key=lambda kv: np.mean(kv[1]))
    return {"k_drift": k_drift, "k_thresh": k_thresh, "mean_mcc": float(np.mean(mccs))}


def best_ssd_cpd(channel: str, result_dir: str | Path = RESULTS_DIR, suffix: str = "") -> dict:
    results = json.loads(_result_path("ssd_cpd", channel, suffix, result_dir).read_text())
    return max(results, key=lambda r: r["mean_mcc"])


def _summarize(scores: list) -> dict:
    """Selection is on mean_mcc; F1 kept alongside for comparison with v1."""
    return {
        "mean_mcc": float(np.mean([s.mcc for s in scores])),
        "mean_f1": float(np.mean([s.f1 for s in scores])),
        "mean_pred_steady_rate": float(np.mean([s.n_pred_steady / s.n_total for s in scores])),
    }


if __name__ == "__main__":
    # optional third argument: number of days per calibration run (default CALIBRATION_DAYS);
    # a non-default value writes calib_<method>_<channel>_<N>d.json alongside the default results
    method, channel = sys.argv[1], sys.argv[2]
    days = float(sys.argv[3]) if len(sys.argv) > 3 else CALIBRATION_DAYS
    suffix = "" if days == CALIBRATION_DAYS else f"_{days:g}d"
    fn = {"ssd_cpd": calibrate_ssd_cpd, "cusum": calibrate_cusum}[method]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _result_path(method, channel, suffix).write_text(json.dumps(fn(channel, days), indent=2))
    print("DONE", flush=True)
