"""Regenerates the extra figures embedded in README.md under samples/ that
aren't produced by the main simulation/eval pipelines: the detection
before/after comparison and the cluster-on-timeseries plot.

Run from the repo root: .venv/bin/python scripts/make_sample_figures.py
"""

from __future__ import annotations

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

DATASETS_DIR = Path("datasets")
SAMPLES_DIR = Path("samples")


def make_detection_before_after(run_id: str = "run_00000", days: float = 3.0) -> None:
    """ssd_cpd library-default vs. tuned output, overlaid on the actual
    temperature trace. Ground truth is a full-height background shading on
    its own twin axis; the detector's 0/1 call is drawn directly in the
    temperature axis's own coordinates, pinned to a fixed low band so it
    sits at the identical position in both panels."""
    df = pd.read_parquet(DATASETS_DIR / run_id / "data.parquet")
    window = df[df.time < days * 86400].copy()
    idx = to_datetime_index(window["time"])
    series = pd.Series(window["temperature"].to_numpy(), index=idx)
    time_days = window.time.to_numpy() / 86400.0
    is_steady = window.is_steady.to_numpy()
    temp = window.temperature.to_numpy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred_default = (
            indsl_detect.ssd_cpd(series, min_distance=4, var_threshold=2.0, slope_threshold=-3.0)
            .reindex(idx)
            .ffill()
            .fillna(0)
            .to_numpy()
        )
        pred_tuned = (
            indsl_detect.ssd_cpd(series, min_distance=4, var_threshold=5.0, slope_threshold=-5.0)
            .reindex(idx)
            .ffill()
            .fillna(0)
            .to_numpy()
        )

    tmin, tmax = temp.min(), temp.max()
    trange = tmax - tmin
    # baseline sits below the data (0 = not steady); the "1" level rises up
    # into the lower portion of the actual temperature trace, so the signal
    # visibly points up toward the series instead of staying in a separate
    # strip entirely beneath it. Both levels are fixed from the same tmin/
    # tmax for this window, so they land at the identical position in both
    # panels below.
    y_low = tmin - 0.08 * trange
    y_high = tmin + 0.15 * trange

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    panels = [
        (axes[0], pred_default, "ssd_cpd (library defaults)"),
        (axes[1], pred_tuned, "ssd_cpd_tuned (calibrated)"),
    ]
    for ax, pred, title in panels:
        ax_bg = ax.twinx()
        ax_bg.fill_between(time_days, 0, is_steady, step="post", color="gray", alpha=0.2, label="ground truth steady", zorder=0)
        ax_bg.set_ylim(0, 1)
        ax_bg.set_yticks([])

        ax.plot(time_days, temp, color="tab:red", lw=0.6, label="chamber temperature (observed)", zorder=2)
        detector_y = np.where(pred == 1, y_high, y_low)
        ax.plot(time_days, detector_y, color="tab:blue", lw=1.4, drawstyle="steps-post", label="detector", zorder=3)
        ax.set_ylim(y_low - 0.02 * trange, tmax + 0.05 * trange)
        ax.set_ylabel("temperature [K]")
        ax.set_title(title)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_bg.get_legend_handles_labels()
        ax.legend(lines2 + lines1, labels2 + labels1, loc="lower right", fontsize=8)

    axes[1].set_xlabel("time [days]")
    fig.tight_layout()
    out_path = SAMPLES_DIR / "united_states" / "detection_before_after.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"saved {out_path}")


def make_clusters_on_timeseries(run_id: str = "run_00000", region_label: str = "United States") -> None:
    df = pd.read_parquet(DATASETS_DIR / run_id / "data.parquet")
    points = extract_steady_points(DATASETS_DIR / run_id)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = cluster_steady_points(points)
    points["kmeans_cluster"] = result.kmeans_labels
    names = name_clusters(points, "kmeans_cluster")
    points["group_name"] = points.kmeans_cluster.map(names)

    seg_to_group = dict(zip(points.segment_id, points.group_name))
    df["group_name"] = df.segment_id.map(seg_to_group)

    days = df.time / 86400.0
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(days, df.temperature, color="lightgray", lw=0.4, zorder=1, label="_nolegend_")

    colors = plt.cm.tab10.colors
    for i, (name, g) in enumerate(df[df.group_name.notna()].groupby("group_name")):
        ax.scatter(g.time / 86400.0, g.temperature, s=3, color=colors[i % 10], label=name, zorder=2)

    ax.set_xlabel("time [days]")
    ax.set_ylabel("temperature [K]")
    ax.set_title(f"{run_id} ({region_label}): steady points colored by detected group")
    ax.legend(markerscale=4, fontsize=9, loc="lower right")
    fig.tight_layout()
    out_path = SAMPLES_DIR / "united_states" / "clusters_on_timeseries.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    make_detection_before_after()
    make_clusters_on_timeseries()
