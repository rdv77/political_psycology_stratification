from __future__ import annotations

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

from src.utils.logging import get_logger


logger = get_logger("analysis.tsne")


def tsne_plot(config: dict) -> None:
    data_dir = config["io"]["data_dir"]
    in_path = os.path.join(data_dir, "clusters.parquet")
    out_plot = os.path.join(config["io"]["plots_dir"], "tsne.png")
    os.makedirs(config["io"]["plots_dir"], exist_ok=True)

    df = pd.read_parquet(in_path)
    feature_cols = [c for c in df.columns if c.startswith("F")]  # factor columns
    X = df[feature_cols].values

    tsne_conf = config["analysis"]["tsne"]
    requested_perplexity = float(tsne_conf["perplexity"]) if isinstance(tsne_conf["perplexity"], (int, float, str)) else 30.0
    # Clamp perplexity: must be < n_samples, typically in [5, 50]
    max_allowed = max(1.0, min(50.0, X.shape[0] - 1.0 - 1e-3))
    perplexity = min(requested_perplexity, max_allowed)
    if perplexity != requested_perplexity:
        logger.info("Adjusted t-SNE perplexity from %.2f to %.2f (n_samples=%d)", requested_perplexity, perplexity, X.shape[0])

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=tsne_conf["learning_rate"],
        random_state=int(tsne_conf["random_state"]),
    )
    X2 = tsne.fit_transform(X)

    df["x"] = X2[:, 0]
    df["y"] = X2[:, 1]

    # Prepare injected parties list
    injected = set()
    try:
        injected = set((config.get("user_input", {}) or {}).get("parties", {}).keys())
    except Exception:
        injected = set()

    plt.figure(figsize=(10, 7))
    # Base scatter: all points colored by cluster
    for cid, group in df.groupby("cluster_id"):
        plt.scatter(group["x"], group["y"], s=28, alpha=0.8, label=f"Cluster {cid}")

    # Overlay injected parties with distinct marker/color
    inj_mask = df.index.isin(injected)
    if inj_mask.any():
        plt.scatter(
            df.loc[inj_mask, "x"],
            df.loc[inj_mask, "y"],
            s=90,
            marker="*",
            c="#d62728",
            edgecolors="black",
            linewidths=0.6,
            label="Injected parties",
            zorder=5,
        )
        # Add labels only for injected parties
        for name, x, y in zip(df.index[inj_mask], df.loc[inj_mask, "x"], df.loc[inj_mask, "y"]):
            try:
                plt.annotate(
                    str(name),
                    xy=(float(x), float(y)),
                    xytext=(6, 6),
                    textcoords="offset points",
                    fontsize=8,
                    color="#d62728",
                    ha="left",
                    va="bottom",
                    zorder=6,
                )
            except Exception:
                pass

    plt.legend()
    plt.title("t-SNE of factor scores (all parties)")
    plt.tight_layout()
    plt.savefig(out_plot, dpi=150)
    logger.info("Saved t-SNE plot to %s", out_plot)


