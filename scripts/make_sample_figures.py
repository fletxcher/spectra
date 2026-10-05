"""Regenerates everything under samples/ that README.md embeds:

- samples/<region>/{conditions.txt,temperature.png,emissions.png}: copied
  from the first run of each region (runs cycle regions as i % 9).
- samples/<region>/detection_before_after.png: ssd_cpd defaults vs. tuned.
- samples/<region>/clusters_on_timeseries.png: detected groups on the trace.
- samples/<region>/clustering_example.png: groups in load/temperature space.

Region directories are always derived from each run's conditions.txt, never
hardcoded, so a figure can't land under the wrong region.

Run from the repo root:
    .venv/bin/python scripts/make_sample_figures.py [grouping run ids...]
"""

from __future__ import annotations

import shutil
import sys
import warnings
from pathlib import Path

import indsl.detect as indsl_detect
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.cluster.cluster import cluster_steady_points
from src.cluster.extract import extract_steady_points
from src.cluster.naming import name_clusters
from src.eval.labels import to_datetime_index
from src.eval.methods import SSD_CPD_TUNED_PARAMS
from src.sim.regions import REGIONS

DATASETS_DIR = Path("datasets")
SAMPLES_DIR = Path("samples")
DEFAULT_GROUPING_RUNS = ["run_00000"]


def region_of(run_id: str) -> tuple[str, str]:
    """(region key, display name) for a run, read from its conditions.txt."""
    for line in (DATASETS_DIR / run_id / "conditions.txt").read_text().splitlines():
        if line.startswith("region:"):
            name = line.split(":", 1)[1].strip()
            return next(k for k, r in REGIONS.items() if r.name == name), name
    raise ValueError(f"no region in {run_id}/conditions.txt")


def copy_region_samples() -> None:
    for i in range(len(REGIONS)):
        run_id = f"run_{i:05d}"
        key, _ = region_of(run_id)
        out = SAMPLES_DIR / key
        out.mkdir(parents=True, exist_ok=True)
        for name in ("conditions.txt", "temperature.png", "emissions.png"):
            shutil.copy(DATASETS_DIR / run_id / name, out / name)
        print(f"copied {run_id} -> {out}/")


def _cluster(run_id: str, cluster_col: str) -> pd.DataFrame:
    points = extract_steady_points(DATASETS_DIR / run_id)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = cluster_steady_points(points)
    points["kmeans_cluster"] = result.kmeans_labels
    points["hdbscan_cluster"] = result.hdbscan_labels
    points["group_name"] = points[cluster_col].map(name_clusters(points, cluster_col))
    return points


def make_detection_before_after(run_id: str = "run_00000", days: float = 3.0) -> None:
    """ssd_cpd library-default vs. tuned output, overlaid on the actual
    temperature trace. Everything is drawn in one coordinate system: the
    detector's 0/1 call sits flush at the bottom/top edge of the axes, and
    the ground-truth shading spans the same full height."""
    key, name = region_of(run_id)
    df = pd.read_parquet(DATASETS_DIR / run_id / "data.parquet")
    window = df[df.time < days * 86400].copy()
    idx = to_datetime_index(window["time"])
    series = pd.Series(window["temperature"].to_numpy(), index=idx)
    time_days = window.time.to_numpy() / 86400.0
    is_steady = window.is_steady.to_numpy()
    temp = window.temperature.to_numpy()

    def detect(**params) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = indsl_detect.ssd_cpd(series, min_distance=4, **params)
        return out.reindex(idx).ffill().fillna(0).to_numpy()

    pred_default = detect(var_threshold=2.0, slope_threshold=-3.0)
    pred_tuned = detect(**SSD_CPD_TUNED_PARAMS["temperature"])

    tmin, tmax = temp.min(), temp.max()
    pad = 0.05 * (tmax - tmin)
    y_low, y_high = tmin - pad, tmax + pad

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    panels = [
        (axes[0], pred_default, "ssd_cpd (library defaults)"),
        (axes[1], pred_tuned, "ssd_cpd_tuned (calibrated)"),
    ]
    for ax, pred, title in panels:
        ax.set_ylim(y_low, y_high)
        ax.fill_between(
            time_days, y_low, np.where(is_steady, y_high, y_low), step="post",
            color="gray", alpha=0.2, label="ground truth steady", zorder=0,
        )
        ax.plot(time_days, temp, color="tab:red", lw=0.6, label="chamber temperature (observed)", zorder=2)
        ax.plot(time_days, np.where(pred == 1, y_high, y_low), color="tab:blue", lw=1.4,
                drawstyle="steps-post", label="detector", zorder=3)
        ax.set_ylabel("temperature [K]")
        ax.set_title(f"{title}: {run_id} ({name})")
        ax.legend(loc="lower right", fontsize=8)

    axes[1].set_xlabel("time [days]")
    fig.tight_layout()
    out_path = SAMPLES_DIR / key / "detection_before_after.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"saved {out_path}")


def make_clusters_on_timeseries(run_id: str = "run_00000") -> None:
    key, name = region_of(run_id)
    df = pd.read_parquet(DATASETS_DIR / run_id / "data.parquet")
    points = _cluster(run_id, "kmeans_cluster")

    df["group_name"] = None
    for p in points.itertuples():
        in_window = (df.time >= p.start_time) & (df.time <= p.end_time)
        df.loc[in_window, "group_name"] = p.group_name

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(df.time / 86400.0, df.temperature, color="lightgray", lw=0.4, zorder=1, label="_nolegend_")
    colors = plt.cm.tab10.colors
    for i, (group, g) in enumerate(df[df.group_name.notna()].groupby("group_name")):
        ax.scatter(g.time / 86400.0, g.temperature, s=3, color=colors[i % 10], label=group, zorder=2)
    ax.set_xlabel("time [days]")
    ax.set_ylabel("temperature [K]")
    ax.set_title(f"{run_id} ({name}): steady points colored by detected group")
    ax.legend(markerscale=4, fontsize=9, loc="lower right")
    fig.tight_layout()
    out_path = SAMPLES_DIR / key / "clusters_on_timeseries.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"saved {out_path}")


def make_clustering_example(run_id: str, cluster_col: str = "kmeans_cluster") -> None:
    key, name = region_of(run_id)
    points = _cluster(run_id, cluster_col)
    n_groups = points.loc[points[cluster_col] != -1, cluster_col].nunique()
    method = "HDBSCAN" if cluster_col == "hdbscan_cluster" else "KMeans"

    fig, ax = plt.subplots(figsize=(11, 4.5))
    for group, g in points.groupby("group_name"):
        ax.scatter(g.load_fraction, g.temperature, s=22, label=group, alpha=0.85)
    ax.set_xlabel("load_fraction")
    ax.set_ylabel("temperature [K]")
    ax.set_title(f"{run_id}: {name} -- {n_groups} regimes ({method})")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    out_path = SAMPLES_DIR / key / "clustering_example.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    for stale in SAMPLES_DIR.glob("*/clustering_example.png"):
        stale.unlink()
    copy_region_samples()
    make_detection_before_after()
    make_clusters_on_timeseries()
    for arg in sys.argv[1:] or DEFAULT_GROUPING_RUNS:
        run_id, _, col = arg.partition(":")
        make_clustering_example(run_id, f"{col}_cluster" if col else "kmeans_cluster")
