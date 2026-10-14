"""Offline sensitivity audit; preserves the original pipeline and its artifacts.

PCA parallel analysis is a dimensionality diagnostic, NOT common-factor retention.
Cluster resampling is conditional on saved factor scores; it does not validate
LLM scoring, factor extraction, or the existence of population strata.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def leading_retained(observed: np.ndarray, cutoff: np.ndarray) -> int:
    """Count leading eigenvalues above the reference; stop at first failure."""
    failed = np.flatnonzero(observed <= cutoff)
    return int(failed[0]) if len(failed) else len(observed)


def congruence(reference: np.ndarray, candidate: np.ndarray) -> list[float]:
    """Match factors allowing permutations and arbitrary sign reversal."""
    a = reference / np.linalg.norm(reference, axis=0)
    b = candidate / np.linalg.norm(candidate, axis=0)
    similarities = np.abs(a.T @ b)
    rows, cols = linear_sum_assignment(-similarities)
    return similarities[rows, cols].tolist()


def summarize(values: list[float]) -> dict:
    return {"median": float(np.median(values)), "q05": float(np.quantile(values, .05)),
            "min": float(np.min(values)), "max": float(np.max(values))}


def aligned_phi(loadings: np.ndarray, structure: np.ndarray | None) -> np.ndarray:
    """Recover Phi in loading-column order from S = L Phi.

    factor-analyzer 0.5.1 reorders loadings/structure but not phi_ after MINRES.
    Using phi_ directly can therefore misalign 4-factor oblique reconstructions.
    """
    if structure is None:
        return np.eye(loadings.shape[1])
    return np.linalg.lstsq(loadings, structure, rcond=None)[0]


def cluster_grid(x: np.ndarray, restarts: int) -> list[dict]:
    output = []
    for k in range(2, min(15, len(x) - 1) + 1):
        km = KMeans(n_clusters=k, n_init=restarts, random_state=42).fit(x)
        counts = np.bincount(km.labels_)
        output.append({"k": k, "silhouette": float(silhouette_score(x, km.labels_)),
                       "inertia": float(km.inertia_), "min_size": int(counts.min()),
                       "max_size": int(counts.max())})
    return output


def cluster_stability(x: np.ndarray, k: int, rng: np.random.Generator,
                      repeats: int, restarts: int) -> dict:
    baseline = KMeans(n_clusters=k, n_init=restarts, random_state=42).fit(x)
    labels = baseline.labels_
    seed_ari, subset_ari = [], []
    jaccards = [[] for _ in range(k)]
    for seed in range(repeats):
        alternative = KMeans(n_clusters=k, n_init=1, random_state=seed).fit_predict(x)
        seed_ari.append(float(adjusted_rand_score(labels, alternative)))
        subset = np.sort(rng.choice(len(x), size=int(.8 * len(x)), replace=False))
        model = KMeans(n_clusters=k, n_init=restarts, random_state=seed).fit(x[subset])
        predicted = model.predict(x)
        # Compute ARI and best-match Jaccard on the shared, sampled objects.
        original, resampled = labels[subset], predicted[subset]
        subset_ari.append(float(adjusted_rand_score(original, resampled)))
        for cid in range(k):
            a = original == cid
            if not a.any():
                continue
            matches = []
            for other in range(k):
                b = resampled == other
                union = (a | b).sum()
                matches.append(float((a & b).sum() / union) if union else 0.)
            jaccards[cid].append(max(matches))
    return {"k": k, "single_start_seed_ari": summarize(seed_ari),
            "subsample_ari": summarize(subset_ari),
            "cluster_jaccard": {str(i): summarize(v) for i, v in enumerate(jaccards) if v},
            "baseline_sizes": np.bincount(labels).tolist(),
            "scope": "80% subsampling; fixed saved scores; shared objects only; not whole-pipeline stability"}


def run(data: Path, out: Path, permutations: int, repeats: int) -> dict:
    rng = np.random.default_rng(20261014)
    source = pd.read_parquet(data / "scores.parquet").set_index("party")
    declared_traits = json.loads((data / "traits.json").read_text(encoding="utf-8"))
    numeric = source[declared_traits].apply(pd.to_numeric, errors="coerce")
    missing = numeric.isna()
    usable = numeric.dropna(axis=1, how="all")
    usable = usable.loc[:, usable.std() > 0]
    clean = usable.fillna(usable.mean())
    x = clean.to_numpy()
    z = StandardScaler().fit_transform(x)
    correlation = np.corrcoef(z, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(correlation)[::-1]
    high_rows, high_cols = np.where(np.triu(np.abs(correlation) >= .95, k=1))
    pairs = sorted([{"a": clean.columns[i], "b": clean.columns[j],
                     "r": float(correlation[i, j])} for i, j in zip(high_rows, high_cols)],
                   key=lambda item: -abs(item["r"]))
    parties = json.loads((data / "parties_map.json").read_text(encoding="utf-8"))
    saved = pd.read_parquet(data / "factors.parquet").loc[clean.index]
    clusters = pd.read_parquet(data / "clusters.parquet").loc[clean.index]
    loadings = np.load(data / "factor_loadings.npy")
    if loadings.shape[0] != len(declared_traits):
        raise ValueError("Saved loading rows do not match declared trait count")
    meta = json.loads((data / "factors_meta.json").read_text(encoding="utf-8"))
    result = {
        "audit_date": "2026-10-14", "seed": 20261014,
        "parameters": {"permutations": permutations, "cluster_repeats": repeats,
                       "restarts": 50, "threads": 1},
        "versions": {name: importlib.metadata.version(name) for name in
                     ["numpy", "pandas", "scipy", "scikit-learn", "factor-analyzer", "pyarrow"]},
        "python": platform.python_version(),
        "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(data.iterdir()) if p.is_file()},
        "data": {"parties": len(clean), "declared_traits": len(declared_traits),
                 "usable_traits": clean.shape[1], "countries": len(set(parties.values())),
                 "missing_cells": int(missing.sum().sum()),
                 "missing_by_party": {str(k): int(v) for k, v in missing.sum(axis=1).items() if v},
                 "out_of_range": int(((numeric < 0) | (numeric > 10)).sum().sum()),
                 "centered_rank": int(np.linalg.matrix_rank(z)),
                 "correlation_rank": int(np.linalg.matrix_rank(correlation)),
                 "correlation_condition_number": float(np.linalg.cond(correlation)),
                 "duplicate_columns": int(clean.T.duplicated().sum()),
                 "pairs_abs_r_ge_095": len(pairs), "top_correlated_pairs": pairs[:30],
                 "classical_kmo_bartlett": "Not used: singular correlation matrix, p >= n; inverse/determinant diagnostics are unreliable."},
        "saved_factors": {"n_factors": saved.shape[1], "variance_shares": meta["explained_variance"],
                          "score_std": saved.std().to_dict(),
                          "score_variance_fraction": (saved.var() / saved.var().sum()).to_dict(),
                          "communalities_min": float((loadings ** 2).sum(axis=1).min()),
                          "communalities_max": float((loadings ** 2).sum(axis=1).max()),
                          "negative_pole_counts": {
                              col: {"negative_any": int((loadings[:, j] < 0).sum()),
                                    "negative_below_minus_03": int((loadings[:, j] <= -.3).sum())}
                              for j, col in enumerate(saved.columns)}},
    }
    print("Data diagnostics complete", flush=True)
    null_eigenvalues = []
    for _ in range(permutations):
        randomized = np.column_stack([rng.permutation(z[:, i]) for i in range(z.shape[1])])
        # Nonzero spectrum from smaller n x n Gram matrix, identical to Z'Z / n.
        eig = np.linalg.eigvalsh(randomized @ randomized.T / len(z))[::-1]
        null_eigenvalues.append(np.maximum(eig, 0))
    cutoff = np.quantile(null_eigenvalues, .95, axis=0)
    observed = eigenvalues[:len(cutoff)]
    result["pca_parallel_analysis"] = {
        "scope": "PCA/correlation spectrum diagnostic, not EFA common-factor selection",
        "permutations": permutations, "retained_leading_components": leading_retained(observed, cutoff),
        "observed_first_12": observed[:12].tolist(), "null_p95_first_12": cutoff[:12].tolist()}
    print("PCA parallel diagnostic complete", flush=True)
    candidates = []
    reference3 = None
    for n in (3, 4):
        for rotation in ("varimax", "oblimin"):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fa = FactorAnalyzer(n_factors=n, rotation=rotation, method="minres", use_smc=False)
                fa.fit(x)
            phi = aligned_phi(fa.loadings_, fa.structure_)
            reconstructed = fa.loadings_ @ phi @ fa.loadings_.T
            residual = (correlation - reconstructed)[np.triu_indices(len(correlation), k=1)]
            entry = {"n_factors": n, "rotation": rotation, "use_smc": False,
                     "phi": phi.tolist(), "off_diagonal_residual_rms": float(np.sqrt(np.mean(residual ** 2))),
                     "warnings": sorted(set(str(w.message) for w in caught))}
            if fa.phi_ is not None:
                entry["library_phi_raw"] = fa.phi_.tolist()
                entry["phi_alignment_note"] = "Recovered in loading-column order from structure = loadings @ phi"
            if rotation == "varimax":
                entry["variance_shares"] = fa.get_factor_variance()[1].tolist()
                if n == 3:
                    reference3 = fa.loadings_
                if n == 4:
                    entry["congruence_with_saved_loadings"] = congruence(loadings, fa.loadings_)
            candidates.append(entry)
            print(f"FA sensitivity {n} {rotation} complete", flush=True)
    result["factor_sensitivity"] = {
        "scope": "MINRES with constant starting uniquenesses (use_smc=False); avoids singular SMC inverse; diagnostic alternatives",
        "candidates": candidates}
    country_sensitivity = []
    country_series = pd.Series([parties[name] for name in clean.index], index=clean.index)
    for country in sorted(country_series.unique()):
        subset = clean.loc[country_series != country]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fa = FactorAnalyzer(n_factors=3, rotation="varimax", use_smc=False)
            fa.fit(subset.to_numpy())
        country_sensitivity.append({"excluded_country": country, "remaining_parties": len(subset),
                                    "matched_congruence": congruence(reference3, fa.loadings_),
                                    "warnings": sorted(set(str(w.message) for w in caught))})
    result["leave_country_out_factors"] = {
        "scope": "3-factor varimax use_smc=False, loadings matched by absolute Tucker congruence; same LLM and traits",
        "results": country_sensitivity}
    print("Leave-country-out factor sensitivity complete", flush=True)
    saved_x = saved.to_numpy()
    scaled_x = StandardScaler().fit_transform(saved_x)
    configurations = {"saved_4_raw": saved_x, "saved_4_standardized": scaled_x,
                      "saved_first3_raw": saved_x[:, :3]}
    grids = {name: cluster_grid(features, 50) for name, features in configurations.items()}
    result["cluster_grid"] = grids
    result["saved_clustering"] = {
        "k": int(clusters.cluster_id.nunique()),
        "silhouette": float(silhouette_score(saved_x, clusters.cluster_id)),
        "sizes": {str(k): int(v) for k, v in clusters.cluster_id.value_counts().sort_index().items()}}
    stability = {}
    baseline = clusters.cluster_id.to_numpy()
    for name, features in configurations.items():
        best_k = max(grids[name], key=lambda row: row["silhouette"])["k"]
        stability[name] = []
        for k in sorted(set([best_k, int(clusters.cluster_id.nunique())])):
            diagnostics = cluster_stability(features, k, rng, repeats, 50)
            fitted = KMeans(n_clusters=k, n_init=50, random_state=42).fit_predict(features)
            diagnostics["ari_vs_saved_partition"] = float(adjusted_rand_score(baseline, fitted))
            stability[name].append(diagnostics)
        print(f"Cluster sensitivity {name} complete", flush=True)
    result["cluster_stability"] = stability
    out.mkdir(parents=True, exist_ok=True)
    (out / "methodology_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pd.DataFrame([{"configuration": name, **row} for name, rows in grids.items() for row in rows]).to_csv(
        out / "cluster_grid.csv", index=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(range(1, 13), observed[:12], "o-", label="Observed PCA eigenvalues")
        ax[0].plot(range(1, 13), cutoff[:12], "o--", label="Permutation 95th percentile")
        ax[0].set(xlabel="Component", ylabel="Eigenvalue", title="Dimensionality diagnostic (PCA, not EFA)")
        ax[0].legend(fontsize=8)
        for name, grid in grids.items():
            ax[1].plot([r["k"] for r in grid], [r["silhouette"] for r in grid], "o-", label=name)
        ax[1].set(xlabel="Number of clusters", ylabel="Silhouette", title="Cluster sensitivity")
        ax[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "diagnostics.png", dpi=160)
        plt.close(fig)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/audit"))
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    if args.permutations < 20 or args.repeats < 10:
        parser.error("Use at least 20 permutations and 10 clustering repeats")
    with threadpool_limits(limits=1):
        run(args.data_dir, args.output_dir, args.permutations, args.repeats)
    print(f"Saved audit to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
