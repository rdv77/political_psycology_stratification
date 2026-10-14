"""Offline probabilistic common-factor benchmark; does not alter the main pipeline.

FA model in standardized coordinates: z = mu + L f + epsilon,
f ~ N(0, I), epsilon ~ N(0, diag(psi)). For complete observations:
E[f|z] = (z-mu) diag(psi)^-1 L (I + L' diag(psi)^-1 L)^-1.
Only the q x q posterior system is solved; the empirical p x p correlation
matrix is never inverted. PCA is a separate geometric baseline, not EFA.

Training gaps are mean-imputed before sklearn's Gaussian ML fit. Held-out
likelihood and subsequent posterior projection marginalize missing features.
The positive noise floor is imposed AFTER fit: this is explicit diagnostic
regularization, not a claim of constrained maximum-likelihood optimization.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def posterior_scores(z: np.ndarray, loadings: np.ndarray, noise: np.ndarray,
                     mean: np.ndarray | None = None) -> np.ndarray:
    """Gaussian posterior means, grouping rows by their observed features.

    NaN means unobserved, never a neutral scale response. A completely missing
    object maps to the prior mean (zero); consumers should flag it as unmeasured.
    """
    x = np.asarray(z, dtype=float)
    l = np.asarray(loadings, dtype=float)
    psi = np.asarray(noise, dtype=float)
    if x.ndim != 2 or l.ndim != 2 or x.shape[1] != l.shape[0]:
        raise ValueError("Data and loading feature dimensions must agree")
    if psi.shape != (x.shape[1],) or not np.all(np.isfinite(psi)) or np.any(psi <= 0):
        raise ValueError("Noise must be finite, strictly positive, and feature-aligned")
    if not np.all(np.isfinite(l)) or np.isinf(x).any():
        raise ValueError("Only finite parameters and finite/NaN observations are supported")
    center = np.zeros(x.shape[1]) if mean is None else np.asarray(mean, dtype=float)
    if center.shape != psi.shape or not np.all(np.isfinite(center)):
        raise ValueError("Mean must be finite and feature-aligned")
    masks, membership = np.unique(np.isfinite(x), axis=0, return_inverse=True)
    result = np.zeros((len(x), l.shape[1]))
    for group, observed in enumerate(masks):
        if not observed.any():
            continue
        indices = np.flatnonzero(membership == group)
        local_l = l[observed]
        weighted = local_l / psi[observed, None]
        system = np.eye(l.shape[1]) + local_l.T @ weighted
        projection = (x[np.ix_(indices, observed)] - center[observed]) @ weighted
        result[indices] = np.linalg.solve(system, projection.T).T
    return result


def observed_log_likelihood(z: np.ndarray, loadings: np.ndarray,
                            noise: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Log density of observed coordinates under L L' + diag(psi).

    Uses the determinant lemma and Woodbury formula; no p x p inverse.
    """
    x = np.asarray(z, dtype=float)
    # Validate the shared parameters and observation shape using posterior code.
    posterior_scores(x[:0], loadings, noise, mean)
    if np.isinf(x).any():
        raise ValueError("Infinite observations are invalid")
    masks, membership = np.unique(np.isfinite(x), axis=0, return_inverse=True)
    result = np.zeros(len(x))
    for group, observed in enumerate(masks):
        indices = np.flatnonzero(membership == group)
        local_l, local_psi = loadings[observed], noise[observed]
        weighted = local_l / local_psi[:, None]
        system = np.eye(loadings.shape[1]) + local_l.T @ weighted
        sign, logdet_small = np.linalg.slogdet(system)
        if sign <= 0:
            raise ArithmeticError("Posterior covariance system is not positive definite")
        centered = x[np.ix_(indices, observed)] - mean[observed]
        v = centered @ weighted
        quadratic = (centered ** 2 / local_psi).sum(axis=1)
        quadratic -= (v * np.linalg.solve(system, v.T).T).sum(axis=1)
        logdet = np.log(local_psi).sum() + logdet_small
        result[indices] = -.5 * (observed.sum() * np.log(2 * np.pi) + logdet + quadratic)
    return result


def varimax_rotation(loadings: np.ndarray, iterations: int = 200,
                     tolerance: float = 1e-8) -> np.ndarray:
    """Return an orthogonal varimax rotation; no axis-wise score rescaling."""
    p, q = loadings.shape
    rotation = np.eye(q)
    previous = 0.
    for _ in range(iterations):
        rotated = loadings @ rotation
        objective = loadings.T @ (rotated ** 3 - rotated @ np.diag(
            (rotated ** 2).sum(axis=0)) / p)
        left, singular, right = np.linalg.svd(objective, full_matrices=False)
        rotation = left @ right
        value = singular.sum()
        if previous > 0 and value <= previous * (1 + tolerance):
            break
        previous = value
    return rotation


def fit_model(raw: np.ndarray, q: int, noise_floor: float = 1e-6,
              seed: int = 20261014) -> tuple[dict, dict]:
    """Fit using only training data and return portable, fixed parameters."""
    if not np.isfinite(noise_floor) or noise_floor <= 0:
        raise ValueError("Noise floor must be finite and positive")
    raw = np.asarray(raw, dtype=float)
    if np.isinf(raw).any() or np.isnan(raw).all(axis=0).any():
        raise ValueError("Features need at least one finite training observation")
    if not 1 <= q < min(raw.shape):
        raise ValueError("Require 1 <= q < min(n, p)")
    imputation = np.nanmean(raw, axis=0)
    filled = np.where(np.isfinite(raw), raw, imputation)
    scaler = StandardScaler().fit(filled)
    standardized = scaler.transform(filled)
    model = FactorAnalysis(n_components=q, svd_method="lapack", max_iter=2000,
                           tol=1e-4, random_state=seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(standardized)
    loadings = model.components_.T.copy()
    original_noise = model.noise_variance_.copy()
    noise = np.maximum(original_noise, noise_floor)
    rotation = varimax_rotation(loadings)
    parameters = {
        "format_version": 1, "q": q,
        "training_imputation": imputation, "training_mean": scaler.mean_,
        "training_scale": scaler.scale_, "standardized_model_mean": model.mean_,
        "loadings": loadings, "noise_variance": noise, "rotation": rotation,
        "noise_floor": noise_floor,
        "projection": "posterior mean; missing input features marginalized",
    }
    diagnostics = {
        "iterations": int(model.n_iter_),
        "convergence_warning": any(issubclass(w.category, ConvergenceWarning) for w in caught),
        "minimum_noise_before_floor": float(original_noise.min()),
        "minimum_noise_after_floor": float(noise.min()),
        "noise_floor_active_features": int((original_noise < noise_floor).sum()),
        "posterior_system_condition": float(np.linalg.cond(
            np.eye(q) + loadings.T @ (loadings / noise[:, None]))),
        "training_missing_cells_mean_imputed": int(np.isnan(raw).sum()),
    }
    return parameters, diagnostics


def transform_new_objects(raw: np.ndarray, parameters: dict,
                          rotated: bool = True) -> np.ndarray:
    """Project with fixed training parameters and EXACT saved trait order.

    Do not fit a scaler on new objects. Missing is NaN; zero is an observed
    response. The saved parameters include trait_order to align input columns.
    """
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != len(parameters["training_mean"]):
        raise ValueError("New observations must use the saved trait order")
    z = (raw - parameters["training_mean"]) / parameters["training_scale"]
    scores = posterior_scores(z, parameters["loadings"], parameters["noise_variance"],
                              parameters["standardized_model_mean"])
    return scores @ parameters["rotation"] if rotated else scores


def raw_observed_log_likelihood(raw: np.ndarray, parameters: dict) -> np.ndarray:
    """Observed Gaussian density in original scale units, after noise flooring."""
    z = (raw - parameters["training_mean"]) / parameters["training_scale"]
    likelihood = observed_log_likelihood(z, parameters["loadings"],
                                         parameters["noise_variance"],
                                         parameters["standardized_model_mean"])
    jacobian = np.isfinite(raw) @ np.log(parameters["training_scale"])
    return likelihood - jacobian


def descriptive_range(values: list[float]) -> dict:
    a = np.asarray(values)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)),
            "min": float(a.min()), "max": float(a.max())}


def run(data: Path, output: Path, repeats: int = 5,
        seed: int = 20261014, noise_floor: float = 1e-6) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    traits = json.loads((data / "traits.json").read_text(encoding="utf-8"))
    frame = pd.read_parquet(data / "scores.parquet").set_index("party")
    if not frame.index.is_unique or len(set(traits)) != len(traits):
        raise ValueError("Object and trait identifiers must be unique")
    frame = frame[traits].apply(pd.to_numeric, errors="raise")
    retained = frame.notna().any(axis=0) & (frame.std() > 0)
    dropped_traits = frame.columns[~retained].tolist()
    frame = frame.loc[:, retained]
    raw, traits = frame.to_numpy(dtype=float), frame.columns.tolist()
    splits = list(ShuffleSplit(n_splits=repeats, test_size=.2,
                              random_state=seed).split(raw))
    summary_rows, split_rows, numerical_rows = [], [], []
    for q in range(2, 9):
        fitted, diag = fit_model(raw, q, noise_floor, seed)
        fitted["trait_order"] = traits
        fitted["training_object_order"] = frame.index.tolist()
        full_scores = transform_new_objects(raw, fitted, rotated=False)
        rotated_scores = full_scores @ fitted["rotation"]
        rotated_loadings = fitted["loadings"] @ fitted["rotation"]
        factors = [f"F{i + 1}" for i in range(q)]
        joblib.dump(fitted, output / f"fa_q{q}_parameters.joblib")
        pd.DataFrame(rotated_loadings, index=pd.Index(traits, name="trait"),
                     columns=factors).to_csv(output / f"fa_q{q}_loadings.csv", encoding="utf-8-sig")
        pd.DataFrame(rotated_scores, index=pd.Index(frame.index, name="party"),
                     columns=factors).to_csv(output / f"fa_q{q}_scores.csv", encoding="utf-8-sig")
        numerical_rows.append({"q": q, **diag})
        likelihoods, similarities, deviations = [], [], []
        reference_distances = pdist(full_scores)
        for split, (train, test) in enumerate(splits):
            candidate, local_diag = fit_model(raw[train], q, noise_floor, seed)
            value = float(raw_observed_log_likelihood(raw[test], candidate).mean())
            projected = transform_new_objects(raw, candidate, rotated=False)
            rotation, _ = orthogonal_procrustes(projected, full_scores)
            relative_rmse = float(np.linalg.norm(projected @ rotation - full_scores)
                                  / np.linalg.norm(full_scores))
            similarity = float(spearmanr(reference_distances, pdist(projected)).statistic)
            likelihoods.append(value)
            similarities.append(similarity)
            deviations.append(relative_rmse)
            split_rows.append({"q": q, "split": split, "n_train": len(train),
                               "n_test": len(test), "mean_observed_log_likelihood": value,
                               "all_object_distance_spearman": similarity,
                               "all_object_aligned_score_relative_rmse": relative_rmse,
                               **local_diag})
        summary_rows.append({"q": q,
                             "heldout_log_likelihood": descriptive_range(likelihoods),
                             "distance_spearman": descriptive_range(similarities),
                             "aligned_score_relative_rmse": descriptive_range(deviations)})
        print(f"q={q}: heldout mean log density={np.mean(likelihoods):.3f}, "
              f"distance similarity={np.mean(similarities):.3f}", flush=True)
    # PCA uses full SVD and is only a geometric descriptive baseline.
    imputation = np.nanmean(raw, axis=0)
    filled = np.where(np.isfinite(raw), raw, imputation)
    scaler = StandardScaler().fit(filled)
    z = scaler.transform(filled)
    pca = PCA(n_components=8, svd_solver="full").fit(z)
    pca_scores = pca.transform(z)
    pc_names = [f"PC{i + 1}" for i in range(8)]
    pd.DataFrame(pca_scores, index=pd.Index(frame.index, name="party"),
                 columns=pc_names).to_csv(output / "pca_scores.csv", encoding="utf-8-sig")
    pd.DataFrame(pca.components_.T, index=pd.Index(traits, name="trait"),
                 columns=pc_names).to_csv(output / "pca_axes.csv", encoding="utf-8-sig")
    joblib.dump({"format_version": 1, "method": "PCA full SVD", "trait_order": traits,
                 "training_imputation": imputation, "training_mean": scaler.mean_,
                 "training_scale": scaler.scale_, "components": pca.components_,
                 "standardized_model_mean": pca.mean_}, output / "pca_parameters.joblib")
    rows = []
    for row in summary_rows:
        flat = {"q": row["q"]}
        for measure in ("heldout_log_likelihood", "distance_spearman", "aligned_score_relative_rmse"):
            flat.update({f"{measure}_{stat}": value for stat, value in row[measure].items()})
        rows.append(flat)
    pd.DataFrame(rows).to_csv(output / "dimension_summary.csv", index=False)
    pd.DataFrame(split_rows).to_csv(output / "heldout_splits.csv", index=False)
    pd.DataFrame(numerical_rows).to_csv(output / "full_fit_diagnostics.csv", index=False)
    best = max(summary_rows, key=lambda item: item["heldout_log_likelihood"]["mean"])["q"]
    result = {
        "benchmark_date": "2026-10-14", "seed": seed,
        "versions": {name: importlib.metadata.version(name) for name in
                     ("numpy", "pandas", "scipy", "scikit-learn", "joblib")},
        "inputs_sha256": {name: hashlib.sha256((data / name).read_bytes()).hexdigest()
                          for name in ("scores.parquet", "traits.json")},
        "data": {"n_objects": len(raw), "p_traits": len(traits),
                 "missing_cells": int(np.isnan(raw).sum()), "dropped_traits": dropped_traits,
                 "centered_rank_after_mean_imputation": int(np.linalg.matrix_rank(z)),
                 "trait_order": traits, "object_order": frame.index.tolist()},
        "method": {"model": "Gaussian common-factor ML using sklearn FactorAnalysis",
                   "svd_method": "lapack", "noise_floor": noise_floor,
                   "floor_policy": "Applied after fitting; diagnostic regularization, not constrained ML",
                   "candidate_dimensions": list(range(2, 9)), "holdout_repeats": repeats,
                   "holdout_fraction": .2, "likelihood_units": "original feature units, observed coordinates only",
                   "standardization": "fitted on training objects separately within each split",
                   "rotation": "orthogonal varimax; posterior scores rotated by same matrix",
                   "covariance": "L L.T + diag(psi), with positive diagonal noise",
                   "posterior_formula": "(z - mu) diag(psi)^-1 L [I_q + L.T diag(psi)^-1 L]^-1",
                   "stability_scope": "80% training splits; all objects projected with each fitted model; orthogonal alignment; not independent replication"},
        "summary": summary_rows, "full_fit_diagnostics": numerical_rows,
        "exploratory_best_mean_heldout_q": best,
        "pca_baseline": {"method": "PCA full SVD, not common-factor analysis",
                         "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                         "cumulative_explained_variance_ratio": np.cumsum(pca.explained_variance_ratio_).tolist()},
        "projection_contract": {
            "artifact": "fa_q{q}_parameters.joblib: dictionary of numeric arrays and ordered feature IDs",
            "loader": "joblib.load(path), then transform_new_objects(raw, parameters)",
            "columns": "Raw new-object matrix must match saved trait_order exactly",
            "fixed_parameters": "Do not recompute training mean, scale, noise, loadings, or rotation",
            "missing": "NaN is unobserved and marginalized; numerical zero remains a measured response",
            "training_gaps": "Mean imputation only during sklearn fit; included in saved training_imputation",
            "out_of_domain": "Projection exists mathematically but requires empirical/semantic validation"},
        "limitations": [
            "Gaussian continuous likelihood is an approximation for bounded ordinal 0..10 or -5..5 ratings; it can assign probability beyond scale limits.",
            "The model assumes diagonal residual noise, Gaussian unit-variance factors, and conditional linear dependence; semantic and country dependence may violate these assumptions.",
            "Only 95 objects and overlapping random holdouts: mean/sd/min/max are descriptive ranges, not inferential confidence intervals; q selection is exploratory.",
            "Training mean imputation changes covariance slightly; likelihood for held-out missing coordinates uses the corresponding marginal distribution.",
            "LLM rating dependence, redundant descriptors, object sampling, shared country history, and substantive interpretation are not validated by this numerical benchmark.",
            "Posterior scores are shrinkage estimates, not latent factors observed without error; varimax naming/axis direction and factor-score uncertainty must remain explicit.",
            "Numerically valid new-object projection does not establish universal coverage or applicability to new kinds of objects.",
            "Choose q using interpretability, residual checks, repeatability, sample size, and external validation together; the maximum holdout score is not a final recommendation.",
        ],
    }
    (output / "benchmark.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("reports/latent_benchmark"))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--noise-floor", type=float, default=1e-6)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("At least two held-out splits are needed for descriptive spread")
    with threadpool_limits(limits=1):
        result = run(args.data, args.output, args.repeats, args.seed, args.noise_floor)
    print(f"Exploratory best mean held-out q={result['exploratory_best_mean_heldout_q']}; "
          "not a final dimension decision.", flush=True)
