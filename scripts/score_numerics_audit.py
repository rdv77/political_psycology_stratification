"""Reproduce original score arithmetic and compare a pseudoinverse diagnostic.

Reuses saved loadings, preserving factor count, rotation, and trait semantics.
The pseudoinverse is a numerical sensitivity analysis, not an accepted new model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from factor_analyzer.utils import corr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from threadpoolctl import threadpool_limits

from methodology_audit import cluster_grid, cluster_stability


def run(data: Path, out: Path) -> dict:
    traits = json.loads((data / "traits.json").read_text(encoding="utf-8"))
    frame = pd.read_parquet(data / "scores.parquet").set_index("party")[traits]
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.fillna(frame.mean())
    if frame.isna().any().any() or (frame.std() <= 0).any():
        raise ValueError("This saved-loading comparison requires all original traits to be usable")
    x = frame.to_numpy()
    z = (x - x.mean(axis=0)) / x.std(axis=0)
    r = corr(x)  # Same correlation implementation used by original FactorAnalyzer.
    loadings = np.load(data / "factor_loadings.npy")
    saved = pd.read_parquet(data / "factors.parquet").loc[frame.index]
    labels = pd.read_parquet(data / "clusters.parquet").loc[frame.index, "cluster_id"]
    if loadings.shape != (len(traits), saved.shape[1]):
        raise ValueError("Saved loadings and scores have incompatible dimensions")
    weights_direct = np.linalg.solve(r, loadings)
    direct = z @ weights_direct
    stable_weights = np.linalg.pinv(r, rcond=1e-10) @ loadings
    stable = z @ stable_weights
    result = {
        "scope": "Same saved varimax loadings, same standardized inputs; numerical sensitivity only",
        "pseudo_inverse_rcond": 1e-10,
        "correlation_rank": int(np.linalg.matrix_rank(r)),
        "correlation_condition_number": float(np.linalg.cond(r)),
        "direct_weights_l2": np.linalg.norm(weights_direct, axis=0).tolist(),
        "pinv_weights_l2": np.linalg.norm(stable_weights, axis=0).tolist(),
        "saved_vs_reproduced_direct_rmse": np.sqrt(np.mean((saved.to_numpy() - direct) ** 2, axis=0)).tolist(),
        "saved_vs_pinv_rmse": np.sqrt(np.mean((saved.to_numpy() - stable) ** 2, axis=0)).tolist(),
        "saved_vs_pinv_correlations": [float(np.corrcoef(saved.iloc[:, i], stable[:, i])[0, 1])
                                       for i in range(saved.shape[1])],
        "saved_std": saved.std(ddof=0).to_dict(),
        "pinv_std": dict(zip(saved.columns, stable.std(axis=0).tolist())),
        "pinv_cluster_grid": cluster_grid(stable, 50),
    }
    best_k = max(result["pinv_cluster_grid"], key=lambda row: row["silhouette"])["k"]
    rng = np.random.default_rng(20261014)
    result["pinv_cluster_stability"] = []
    for k in sorted(set([best_k, int(labels.nunique())])):
        diagnostics = cluster_stability(stable, k, rng, 100, 50)
        predicted = KMeans(n_clusters=k, n_init=50, random_state=42).fit_predict(stable)
        diagnostics["ari_vs_saved_partition"] = float(adjusted_rand_score(labels, predicted))
        diagnostics["silhouette"] = float(silhouette_score(stable, predicted))
        result["pinv_cluster_stability"].append(diagnostics)
    out.mkdir(parents=True, exist_ok=True)
    (out / "score_numerics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pd.DataFrame(stable, columns=saved.columns, index=saved.index).to_csv(out / "diagnostic_pinv_scores.csv")
    print(json.dumps({"saved_vs_direct_rmse": result["saved_vs_reproduced_direct_rmse"],
                      "saved_vs_pinv_corr": result["saved_vs_pinv_correlations"],
                      "best_k": best_k,
                      "cluster_comparisons": result["pinv_cluster_stability"]}, ensure_ascii=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/audit"))
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
