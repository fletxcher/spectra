"""Clusters a run's steady points by similarity (operating point + emissions
features). Uses HDBSCAN (picks cluster count automatically, flags points
that don't fit any regime as noise) as the primary method, cross-checked
against KMeans with the cluster count selected by silhouette score."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.cluster.extract import FEATURE_COLUMNS


@dataclass(frozen=True)
class ClusterResult:
    hdbscan_labels: np.ndarray
    hdbscan_n_clusters: int
    hdbscan_n_noise: int
    hdbscan_silhouette: float | None
    kmeans_labels: np.ndarray | None
    kmeans_k: int | None
    kmeans_silhouette: float | None


def cluster_steady_points(points: pd.DataFrame, k_range: range = range(2, 9)) -> ClusterResult:
    X = points[FEATURE_COLUMNS].to_numpy()
    X_scaled = StandardScaler().fit_transform(X)

    min_cluster_size = max(3, len(points) // 20)
    hdb = HDBSCAN(min_cluster_size=min_cluster_size)
    hdb_labels = hdb.fit_predict(X_scaled)
    n_hdb_clusters = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
    n_noise = int((hdb_labels == -1).sum())
    hdb_silhouette = (
        float(silhouette_score(X_scaled, hdb_labels)) if n_hdb_clusters >= 2 and n_noise < len(points) else None
    )

    best_k, best_score, best_labels = None, -1.0, None
    for k in k_range:
        if k >= len(points):
            break
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels

    return ClusterResult(
        hdbscan_labels=hdb_labels,
        hdbscan_n_clusters=n_hdb_clusters,
        hdbscan_n_noise=n_noise,
        hdbscan_silhouette=hdb_silhouette,
        kmeans_labels=best_labels,
        kmeans_k=best_k,
        kmeans_silhouette=best_score if best_labels is not None else None,
    )
