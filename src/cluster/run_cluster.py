"""Runs steady-point extraction + clustering independently for every run in
a dataset directory, writing one row of summary stats per run plus a
combined table of every extracted steady point with its cluster label."""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from src.cluster.cluster import cluster_steady_points
from src.cluster.extract import extract_steady_points


def cluster_run(run_dir: Path) -> tuple[dict, pd.DataFrame]:
    points = extract_steady_points(run_dir)
    if len(points) < 4:
        return {"run_id": run_dir.name, "n_points": len(points), "error": "too few steady points"}, points

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = cluster_steady_points(points)

    points = points.copy()
    points["hdbscan_cluster"] = result.hdbscan_labels
    points["kmeans_cluster"] = result.kmeans_labels

    summary = {
        "run_id": run_dir.name,
        "n_points": len(points),
        "error": None,
        "hdbscan_n_clusters": result.hdbscan_n_clusters,
        "hdbscan_n_noise": result.hdbscan_n_noise,
        "hdbscan_silhouette": result.hdbscan_silhouette,
        "kmeans_k": result.kmeans_k,
        "kmeans_silhouette": result.kmeans_silhouette,
    }
    return summary, points


def cluster_sweep(dataset_dir: str | Path, run_ids: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_dir = Path(dataset_dir)
    if run_ids is None:
        run_dirs = sorted(p for p in dataset_dir.glob("run_*") if p.is_dir())
    else:
        run_dirs = [dataset_dir / r for r in run_ids]

    summaries = []
    all_points = []
    for i, run_dir in enumerate(run_dirs):
        summary, points = cluster_run(run_dir)
        summaries.append(summary)
        if len(points) > 0:
            all_points.append(points)
        if (i + 1) % 20 == 0 or (i + 1) == len(run_dirs):
            print(f"[{i + 1}/{len(run_dirs)}] clustered {run_dir.name}", flush=True)

    return pd.DataFrame.from_records(summaries), pd.concat(all_points, ignore_index=True)


RESULTS_DIR = Path("results/clustering")

if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary, points = cluster_sweep("datasets")
    summary.to_parquet(RESULTS_DIR / "cluster_summary.parquet", index=False)
    points.to_parquet(RESULTS_DIR / "cluster_points.parquet", index=False)
    print(f"wrote {len(summary)} runs / {len(points)} steady points to {RESULTS_DIR}/", flush=True)
