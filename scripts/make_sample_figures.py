"""Regenerates everything under samples/ that README.md embeds:

- samples/<region>/{conditions.txt,temperature.png,emissions.png}: copied
  from the first run of each region (runs cycle regions as i % 9).
- samples/<region>/detection_before_after.{png,html}: ssd_cpd defaults vs.
  tuned (the .html is an interactive, zoomable Plotly version).
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
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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


def _detection_data(run_id: str, days: float) -> dict:
    """ssd_cpd library-default vs. tuned output on the temperature channel,
    plus the levels both plots draw them at: everything in one coordinate
    system, with the detector's 0/1 call flush at the bottom/top of the axes
    and the ground-truth shading spanning the same full height."""
    key, name = region_of(run_id)
    df = pd.read_parquet(DATASETS_DIR / run_id / "data.parquet")
    window = df[df.time < days * 86400].copy()
    idx = to_datetime_index(window["time"])
    series = pd.Series(window["temperature"].to_numpy(), index=idx)
    temp = window.temperature.to_numpy()

    def detect(**params) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = indsl_detect.ssd_cpd(series, min_distance=4, **params)
        return out.reindex(idx).ffill().fillna(0).to_numpy()

    tmin, tmax = temp.min(), temp.max()
    pad = 0.05 * (tmax - tmin)
    return {
        "key": key,
        "name": name,
        "run_id": run_id,
        "time_days": window.time.to_numpy() / 86400.0,
        "temp": temp,
        "is_steady": window.is_steady.to_numpy(),
        "panels": [
            ("ssd_cpd (library defaults)", detect(var_threshold=2.0, slope_threshold=-3.0)),
            ("ssd_cpd_tuned (calibrated)", detect(**SSD_CPD_TUNED_PARAMS["temperature"])),
        ],
        "y_low": tmin - pad,
        "y_high": tmax + pad,
    }


def make_detection_before_after(d: dict) -> None:
    y_low, y_high = d["y_low"], d["y_high"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for ax, (title, pred) in zip(axes, d["panels"]):
        ax.set_ylim(y_low, y_high)
        ax.fill_between(
            d["time_days"], y_low, np.where(d["is_steady"], y_high, y_low), step="post",
            color="gray", alpha=0.2, label="ground truth steady", zorder=0,
        )
        ax.plot(d["time_days"], d["temp"], color="tab:red", lw=0.6, label="chamber temperature (observed)", zorder=2)
        ax.plot(d["time_days"], np.where(pred == 1, y_high, y_low), color="tab:blue", lw=1.4,
                drawstyle="steps-post", label="detector", zorder=3)
        ax.set_ylabel("temperature [K]")
        ax.set_title(f"{title}: {d['run_id']} ({d['name']})")
        ax.legend(loc="lower right", fontsize=8)

    axes[1].set_xlabel("time [days]")
    fig.tight_layout()
    out_path = SAMPLES_DIR / d["key"] / "detection_before_after.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"saved {out_path}")


def make_detection_html(d: dict) -> None:
    """Interactive version of make_detection_before_after: same data, but
    zoomable, with both panels' x-axes linked so zooming one zooms both."""
    y_low, y_high = d["y_low"], d["y_high"]
    t = d["time_days"]
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=[f"{title}: {d['run_id']} ({d['name']})" for title, _ in d["panels"]],
    )
    for row, (_, pred) in enumerate(d["panels"], start=1):
        first = row == 1
        fig.add_trace(go.Scatter(
            x=t, y=np.where(d["is_steady"], y_high, y_low), line_shape="hv", fill="tozeroy",
            line=dict(width=0), fillcolor="rgba(128,128,128,0.2)", name="ground truth steady",
            legendgroup="gt", showlegend=first, hoverinfo="skip",
        ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=t, y=d["temp"], line=dict(color="#d62728", width=0.8), name="chamber temperature (observed)",
            legendgroup="temp", showlegend=first, hovertemplate="day %{x:.4f}<br>%{y:.1f} K<extra></extra>",
        ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=t, y=np.where(pred == 1, y_high, y_low), line_shape="hv", line=dict(color="#1f77b4", width=1.5),
            name="detector", legendgroup="det", showlegend=first, hoverinfo="skip",
        ), row=row, col=1)
        fig.update_yaxes(title_text="temperature [K]", range=[y_low, y_high], row=row, col=1)

    fig.update_xaxes(title_text="time [days]", row=2, col=1)
    fig.update_layout(height=750, template="simple_white", hovermode="x unified",
                      legend=dict(orientation="h", y=-0.12))
    out_path = SAMPLES_DIR / d["key"] / "detection_before_after.html"
    fig.write_html(out_path, include_plotlyjs="cdn")
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
    detection = _detection_data("run_00000", days=3.0)
    make_detection_before_after(detection)
    make_detection_html(detection)
    make_clusters_on_timeseries()
    for arg in sys.argv[1:] or DEFAULT_GROUPING_RUNS:
        run_id, _, col = arg.partition(":")
        make_clustering_example(run_id, f"{col}_cluster" if col else "kmeans_cluster")
