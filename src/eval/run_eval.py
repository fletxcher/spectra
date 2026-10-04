"""Runs every indsl.detect method against every channel of every simulated
run, scoring each against the 60-second steady-window ground truth, and
writes one row of results per (run, channel, method) combination."""

from __future__ import annotations

import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from src.eval.labels import WINDOW_SECONDS, steady_window_label, to_datetime_index
from src.eval.methods import METHODS
from src.eval.metrics import point_metrics, window_metrics

CHANNELS = {
    "temperature": "temperature",  # chamber temperature (TIT proxy), observed/noisy
    "pressure": "pressure",
    "mass_flow": "mdot_air",
    "phi": "phi_cmd",
}


def evaluate_run(run_dir: Path, window_seconds: int = WINDOW_SECONDS) -> list[dict]:
    df = pd.read_parquet(run_dir / "data.parquet")
    idx = to_datetime_index(df["time"])
    dt_seconds = float(df["time"].diff().median())

    raw_is_steady = pd.Series(df["is_steady"].to_numpy(), index=idx)
    y_true = steady_window_label(raw_is_steady, window_seconds)

    rows = []
    for channel_label, column in CHANNELS.items():
        series = pd.Series(df[column].to_numpy(), index=idx)
        for method in METHODS:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    y_pred = method.run(series, dt_seconds, channel_label)
            except Exception as e:  # noqa: BLE001 -- keep going on a single method/channel failure
                rows.append(
                    {
                        "run_id": run_dir.name,
                        "channel": channel_label,
                        "method": method.name,
                        "error": str(e),
                    }
                )
                continue

            pm = point_metrics(y_true, y_pred)
            wm = window_metrics(raw_is_steady, y_pred, window_seconds)
            rows.append(
                {
                    "run_id": run_dir.name,
                    "channel": channel_label,
                    "method": method.name,
                    "error": None,
                    "precision": pm.precision,
                    "recall": pm.recall,
                    "f1": pm.f1,
                    "accuracy": pm.accuracy,
                    "n_true_steady": pm.n_true_steady,
                    "n_pred_steady": pm.n_pred_steady,
                    "n_total": pm.n_total,
                    "n_regions": wm.n_regions,
                    "n_detected": wm.n_detected,
                    "detection_rate": wm.detection_rate,
                    "median_lag_s": wm.median_lag_s,
                    "mean_lag_s": wm.mean_lag_s,
                }
            )
    return rows


def evaluate_sweep(dataset_dir: str | Path, run_ids: list[str] | None = None, workers: int = 1) -> pd.DataFrame:
    dataset_dir = Path(dataset_dir)
    if run_ids is None:
        run_dirs = sorted(p for p in dataset_dir.glob("run_*") if p.is_dir())
    else:
        run_dirs = [dataset_dir / r for r in run_ids]

    all_rows = []
    if workers <= 1:
        for i, run_dir in enumerate(run_dirs):
            all_rows.extend(evaluate_run(run_dir))
            print(f"[{i + 1}/{len(run_dirs)}] evaluated {run_dir.name}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(evaluate_run, run_dir): run_dir for run_dir in run_dirs}
            for i, future in enumerate(as_completed(futures)):
                all_rows.extend(future.result())
                print(f"[{i + 1}/{len(run_dirs)}] evaluated {futures[future].name}", flush=True)

    return pd.DataFrame.from_records(all_rows)

