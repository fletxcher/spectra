"""Extracts steady points from a run: each contiguous stretch the
simulator's ground-truth `is_steady` label marks as settled (and which lasts
at least `min_window_s`) becomes one steady point, summarized by the mean of
its operating-point and emissions signals over that dwell. Stretches are
split on every transient, not grouped by dispatch segment, since a
disturbance can break one segment into several distinct steady windows
(e.g. before a fuel shift, on its plateau, and after it)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

MW_NO = 30.006
MW_NO2 = 46.005
MW_CO = 28.010

FEATURE_COLUMNS = ["temperature", "pressure", "mdot_air", "mdot_fuel", "phi", "load_fraction", "nox_ppm", "co_ppm"]
WINDOW_SECONDS_MIN = 60


def extract_steady_points(run_dir: Path, min_window_s: float = WINDOW_SECONDS_MIN) -> pd.DataFrame:
    df = pd.read_parquet(run_dir / "data.parquet")
    region_id = (df["is_steady"] != df["is_steady"].shift()).cumsum()
    steady = df[df["is_steady"]]
    if steady.empty:
        return pd.DataFrame(columns=["run_id", "segment_id", "start_time", "end_time", "duration_s", *FEATURE_COLUMNS])

    rows = []
    for _, g in steady.groupby(region_id[df["is_steady"]]):
        duration = float(g["time"].max() - g["time"].min())
        if duration < min_window_s or len(g) < 2:
            continue
        nox_ppm = float(((g["Y_NO"] / MW_NO + g["Y_NO2"] / MW_NO2) * g["mean_molecular_weight"]).mean() * 1e6)
        co_ppm = float((g["Y_CO"] / MW_CO * g["mean_molecular_weight"]).mean() * 1e6)
        rows.append(
            {
                "run_id": run_dir.name,
                "segment_id": int(g["segment_id"].iloc[0]),
                "start_time": float(g["time"].min()),
                "end_time": float(g["time"].max()),
                "duration_s": duration,
                "temperature": float(g["temperature_truth"].mean()),
                "pressure": float(g["pressure_truth"].mean()),
                "mdot_air": float(g["mdot_air_truth"].mean()),
                "mdot_fuel": float(g["mdot_fuel_truth"].mean()),
                "phi": float(g["phi_cmd"].mean()),
                "load_fraction": float(g["load_cmd"].mean()),
                "nox_ppm": nox_ppm,
                "co_ppm": co_ppm,
            }
        )
    return pd.DataFrame.from_records(rows)
