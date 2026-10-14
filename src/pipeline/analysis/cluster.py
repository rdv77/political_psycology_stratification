from __future__ import annotations

import json
import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.utils.logging import get_logger


logger = get_logger("analysis.cluster")


def _choose_k(X: np.ndarray, k_range: Tuple[int, int]) -> int:
    k_best = None
    best_score = -1.0
    for k in range(k_range[0], k_range[1] + 1):
        km = KMeans(n_clusters=k, n_init="auto", random_state=42)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        if score > best_score:
            best_score, k_best = score, k
    return int(k_best or k_range[0])


def cluster_factors(config: dict) -> None:
    data_dir = config["io"]["data_dir"]
    scores_path = os.path.join(data_dir, "factors.parquet")
    out_path = os.path.join(data_dir, "clusters.parquet")

    df = pd.read_parquet(scores_path)
    X = df.values

    cl_cfg = config["analysis"]["cluster"]
    # Allow explicit n_clusters override from config
    n_clusters_cfg = int(cl_cfg.get("n_clusters", 0) or 0)
    if n_clusters_cfg > 0:
        k = n_clusters_cfg
    else:
        k_sel = str(cl_cfg.get("k_selection", "silhouette")).lower()
        if k_sel == "silhouette":
            k_range = cl_cfg["k_range"]
            k = _choose_k(X, (int(k_range[0]), int(k_range[1])))
        else:
            k = int(cl_cfg.get("k", 4))

    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = km.fit_predict(X)
    centers = km.cluster_centers_

    # Distances to centroid
    dists = ((X - centers[labels]) ** 2).sum(axis=1) ** 0.5
    out = df.copy()
    out["cluster_id"] = labels
    out["dist_to_center"] = dists
    out.to_parquet(out_path, index=True)
    logger.info("Saved clusters to %s | k=%d", out_path, k)


