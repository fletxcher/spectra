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

# ED-Pelt's pruning only works when it finds many change points. Pressure is
# noise-dominated (the controller holds it near target), so it finds few and
# hits the O(n^2) worst case: ~2 min per call on 30 days of 60 s means (vs
# ~12 s on temperature), which would roughly double the evaluation time for a
# channel carrying almost no operating-point information. Skipped pairs are
# recorded with error="skipped" rather than silently dropped.
SKIP = {("pressure", "ssd_cpd"), ("pressure", "ssd_cpd_tuned"), ("pressure", "cpd_ed_pelt")}


def region_name(run_dir: Path) -> str:
    for line in (run_dir / "conditions.txt").read_text().splitlines():
        if line.startswith("region:"):
            return line.split(":", 1)[1].strip()
    raise ValueError(f"no region in {run_dir}/conditions.txt")


def evaluate_run(run_dir: Path, window_seconds: int = WINDOW_SECONDS, days: float | None = None) -> list[dict]:
    region = region_name(run_dir)
    df = pd.read_parquet(run_dir / "data.parquet")
    if days is not None:
        df = df[df["time"] < days * 86400.0].reset_index(drop=True)
    idx = to_datetime_index(df["time"])
    dt_seconds = float(df["time"].diff().median())

    raw_is_steady = pd.Series(df["is_steady"].to_numpy(), index=idx)
    y_true = steady_window_label(raw_is_steady, window_seconds)

    skip = SKIP if days is None or days > 1 else set()  # the O(n^2) case is cheap on a single day
    rows = []
    for channel_label, column in CHANNELS.items():
        series = pd.Series(df[column].to_numpy(), index=idx)
        for method in METHODS:
            if (channel_label, method.name) in skip:
                rows.append({"run_id": run_dir.name, "region": region, "channel": channel_label, "method": method.name, "error": "skipped"})
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    y_pred = method.run(series, dt_seconds, channel_label)
            except Exception as e:  # noqa: BLE001 -- keep going on a single method/channel failure
                rows.append(
                    {
                        "run_id": run_dir.name,
                        "region": region,
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
                    "region": region,
                    "channel": channel_label,
                    "method": method.name,
                    "error": None,
                    "precision": pm.precision,
                    "recall": pm.recall,
                    "f1": pm.f1,
                    "accuracy": pm.accuracy,
                    "mcc": pm.mcc,
                    "balanced_accuracy": pm.balanced_accuracy,
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


def evaluate_sweep(
    dataset_dir: str | Path, run_ids: list[str] | None = None, workers: int = 1, days: float | None = None
) -> pd.DataFrame:
    dataset_dir = Path(dataset_dir)
    if run_ids is None:
        run_dirs = sorted(p for p in dataset_dir.glob("run_*") if p.is_dir())
    else:
        run_dirs = [dataset_dir / r for r in run_ids]

    all_rows = []
    if workers <= 1:
        for i, run_dir in enumerate(run_dirs):
            all_rows.extend(evaluate_run(run_dir, days=days))
            print(f"[{i + 1}/{len(run_dirs)}] evaluated {run_dir.name}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(evaluate_run, run_dir, days=days): run_dir for run_dir in run_dirs}
            for i, future in enumerate(as_completed(futures)):
                all_rows.extend(future.result())
                print(f"[{i + 1}/{len(run_dirs)}] evaluated {futures[future].name}", flush=True)

    return pd.DataFrame.from_records(all_rows)


# every 5th run, disjoint from the calibration runs in src/eval/calibrate.py
EVAL_RUN_IDS = [f"run_{i:05d}" for i in range(0, 200, 5)]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Score every detector on every channel of the evaluation runs.")
    parser.add_argument("--days", type=float, default=None, help="only use the first N days of each run")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--runs", choices=["eval", "all"], default="eval",
        help="eval: the 40-run evaluation set; all: every run except the calibration runs",
    )
    parser.add_argument("--out", default=None, help="default: eval_results[_all][_<N>d].parquet")
    args = parser.parse_args()

    if args.runs == "all":
        from src.eval.calibrate import CALIBRATION_RUN_IDS

        run_ids = sorted(p.name for p in Path("datasets").glob("run_*") if p.is_dir() and p.name not in CALIBRATION_RUN_IDS)
    else:
        run_ids = EVAL_RUN_IDS
    suffix = ("_all" if args.runs == "all" else "") + ("" if args.days is None else f"_{args.days:g}d")
    out = args.out or f"eval_results{suffix}.parquet"
    results = evaluate_sweep("datasets", run_ids, workers=args.workers, days=args.days)
    results.to_parquet(out, index=False)
    print(f"wrote {len(results)} rows to {out}", flush=True)

