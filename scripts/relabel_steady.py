"""Recomputes the ground-truth `is_steady` column of an existing dataset in
place, without re-running the Cantera physics. Cases are regenerated
deterministically from the sweep seed; the label only depends on the
profile and the stored load_cmd/phi_cmd columns.

Run from the repo root (must match the sweep that produced the dataset):
    .venv/bin/python scripts/relabel_steady.py
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src.sweep.grid import sample_sweep

DATASETS_DIR = Path("datasets")
N_CASES, SEED = 200, 0


def relabel(case) -> tuple[str, float, float]:
    path = DATASETS_DIR / case.run_id / "data.parquet"
    df = pd.read_parquet(path)
    profile = case.profile

    # the stored commands must come from this exact profile, or relabeling is invalid
    for t, load in zip(df.time.iloc[::5000], df.load_cmd.iloc[::5000]):
        if not np.isclose(profile.setpoint_at(t).load_fraction, load, rtol=0, atol=1e-9):
            raise RuntimeError(f"{case.run_id}: stored load_cmd doesn't match regenerated profile at t={t}")

    old_rate = float(df.is_steady.mean())
    settled = np.array([profile.label_at(t)[1] for t in df.time])
    df["is_steady"] = profile.steady_labels(df.time.to_numpy(), df.load_cmd.to_numpy(), df.phi_cmd.to_numpy(), settled)

    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return case.run_id, old_rate, float(df.is_steady.mean())


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    cases = sample_sweep(n_cases=N_CASES, seed=SEED, fine_dt=15.0, coarse_dt=15.0, fine_window=0.0)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for run_id, old, new in pool.map(relabel, cases):
            print(f"{run_id}: steady {old:.3f} -> {new:.3f}", flush=True)
