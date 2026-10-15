"""Deterministic offline analysis for the semantic-space service MVP.

No ratings are generated here. Unknown observations remain NaN. Gaussian FA
uses the q-dimensional posterior system from the audited benchmark; empirical
correlation matrices are never inverted. Every selection rule is reported.
"""
from __future__ import annotations

import hashlib
import json
import math
import textwrap
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from scripts.latent_space_benchmark import (
    fit_model, raw_observed_log_likelihood, transform_new_objects,
)

SEED = 20261015
PARALLEL_PERMUTATIONS = 500
STABILITY_REPEATS = 30
LOADING_THRESHOLD = .35


class InsufficientMeasurementError(ValueError):
    """Valid measurements fail the fixed minimum-data policy, with a coverage audit."""

    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def _finite_number(value, lower, upper, label):
    if isinstance(value, bool):
        raise ValueError(f"{label}: требуется числовая оценка")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label}: требуется числовая оценка") from None
    if not np.isfinite(number) or not lower <= number <= upper:
        raise ValueError(f"{label}: допустимый диапазон от {lower} до {upper}")
    return number


def _prepare(study, ratings):
    """Apply the predeclared scale-first coverage policy to actual observations.

    Every iteration removes under-observed/constant scales using the current
    objects, then removes inadequately measured objects using the surviving
    scales. Repeat until the retained rectangle is stable. This order prevents
    sparse, subsequently discarded scales from excluding well-measured objects.
    """
    objects, scales = study.get("objects", []), study.get("scales", [])
    for label, values in (("объектов", objects), ("шкал", scales)):
        ids = [str(item.get("id", "")) for item in values]
        if any(not identity for identity in ids) or len(ids) != len(set(ids)):
            raise ValueError(f"Идентификаторы {label} должны быть непустыми и уникальными")
    object_map = {str(item["id"]): item for item in objects}
    scale_map = {str(item["id"]): item for item in scales}
    matrix = pd.DataFrame(np.nan, index=list(object_map), columns=list(scale_map))
    mass = matrix.copy()
    seen, observed_count, difference_only = set(), 0, 0
    for rating in ratings:
        oid, sid = str(rating.get("object_id", "")), str(rating.get("scale_id", ""))
        if oid not in object_map or sid not in scale_map:
            raise ValueError(f"Оценка ссылается на неизвестный объект или шкалу: {oid}/{sid}")
        key = (oid, sid)
        if key in seen:
            raise ValueError(f"Повторная оценка одной ячейки: {oid}/{sid}")
        seen.add(key)
        status = rating.get("status", "observed")
        if status == "unknown":
            continue
        if status != "observed":
            raise ValueError(f"Неизвестный статус оценки: {status}")
        if rating.get("positive") is not None and rating.get("negative") is not None:
            positive = _finite_number(rating["positive"], 0, 5, f"{oid}/{sid}: positive")
            negative = _finite_number(rating["negative"], 0, 5, f"{oid}/{sid}: negative")
            value = positive - negative
            if rating.get("value") is not None:
                imported = _finite_number(rating["value"], -5, 5, f"{oid}/{sid}: value")
                if not np.isclose(value, imported):
                    raise ValueError(f"Противоречивые value и positive-negative: {oid}/{sid}")
            mass.loc[oid, sid] = positive + negative
        elif rating.get("value") is not None:
            value = _finite_number(rating["value"], -5, 5, f"{oid}/{sid}: value")
            difference_only += 1
        else:
            raise ValueError(f"У наблюдаемой оценки нет двух полюсов или value: {oid}/{sid}")
        matrix.loc[oid, sid] = value
        observed_count += 1
    original = matrix.copy()
    excluded_objects, excluded_scales = [], []
    preprocessing_iterations = []
    # Fixed, monotone scale-first policy; never select an order for a nicer map.
    # Unknown remains unobserved throughout; it never becomes a neutral response.
    while True:
        before = matrix.shape
        scale_required = max(8, math.ceil(.6 * len(matrix)))
        scale_counts = matrix.notna().sum(axis=0)
        keep_cols = (scale_counts >= scale_required) & (matrix.std() > 1e-10)
        removed_scales = matrix.columns[~keep_cols].tolist()
        for sid in removed_scales:
            excluded_scales.append({"id": sid, "name": scale_map[sid].get("name", sid),
                                    "observed_objects": int(scale_counts[sid]), "required_objects": scale_required,
                                    "reason": "constant_or_insufficient_observations"})
        matrix = matrix.loc[:, keep_cols]
        row_required = max(4, math.ceil(.6 * matrix.shape[1]))
        row_counts = matrix.notna().sum(axis=1)
        keep_rows = row_counts >= row_required
        removed_objects = matrix.index[~keep_rows].tolist()
        for oid in removed_objects:
            count = int(row_counts[oid])
            originally_observed = int(original.loc[oid].notna().sum())
            excluded_objects.append({"id": oid, "name": object_map[oid].get("name", oid),
                                     "observed_scales": count, "required_scales": row_required,
                                     "original_observed_scales": originally_observed,
                                     "reason": "no_observations" if originally_observed == 0 else "insufficient_coverage"})
        matrix = matrix.loc[keep_rows]
        preprocessing_iterations.append({"iteration": len(preprocessing_iterations) + 1,
                                         "input_objects": before[0], "input_scales": before[1],
                                         "required_objects_per_scale": scale_required,
                                         "excluded_scale_ids": removed_scales,
                                         "required_scales_per_object": row_required,
                                         "excluded_object_ids": removed_objects,
                                         "retained_objects": len(matrix), "retained_scales": matrix.shape[1]})
        if matrix.shape == before or matrix.empty:
            break
    n, p = matrix.shape
    coverage = float(matrix.notna().to_numpy().mean()) if matrix.size else None
    input_cells = len(objects) * len(scales)
    diagnostics = {
        "observed_input_ratings": observed_count, "difference_only_ratings": difference_only,
        "input_objects": len(objects), "input_scales": len(scales),
        "total_input_cells": input_cells, "unknown_input_cells": input_cells - observed_count,
        "input_coverage": observed_count / input_cells if input_cells else None,
        "retained_objects": n, "retained_scales": p,
        "retained_object_ids": matrix.index.tolist(), "retained_scale_ids": matrix.columns.tolist(),
        "input_object_coverage": [{"id": oid, "name": item.get("name", oid),
                                   "observed_scales": int(original.loc[oid].notna().sum()),
                                   "total_scales": len(scales),
                                   "coverage": float(original.loc[oid].notna().mean()) if len(scales) else None,
                                   "retained": oid in matrix.index}
                                  for oid, item in object_map.items()],
        "input_scale_coverage": [{"id": sid, "name": item.get("name", sid),
                                  "observed_objects": int(original[sid].notna().sum()),
                                  "observed_retained_objects": int(original.loc[matrix.index, sid].notna().sum()),
                                  "total_objects": len(objects),
                                  "coverage": float(original[sid].notna().mean()) if len(objects) else None,
                                  "retained": sid in matrix.columns}
                                 for sid, item in scale_map.items()],
        "retained_coverage": coverage, "missing_retained_cells": int(matrix.isna().sum().sum()),
        "excluded_objects": excluded_objects, "excluded_scales": excluded_scales,
        "zero_differences": int((matrix == 0).sum().sum()),
        "zero_difference_with_mass": int(((original == 0) & (mass > 0)).sum().sum()),
        "mass_mean": float(np.nanmean(mass.to_numpy())) if mass.notna().any().any() else None,
        "measurement": "primary value = positive - negative; mass = positive + negative",
        "missing_policy": "unknown is NaN, never zero; iteratively filter scales before objects until stable; training gaps mean-imputed; posterior uses only observed coordinates",
        "minimum_coverage_policy": "each iteration: first scales observed on at least 60% and 8 current objects and nonconstant; then objects observed on at least 60% and 4 surviving scales; repeat until stable; retained overall coverage at least 75%",
        "preprocessing_order": "scales_then_objects_until_stable",
        "preprocessing_iterations": preprocessing_iterations,
        "preprocessing_report": "На каждой итерации сначала исключаются постоянные шкалы и шкалы, измеренные менее чем у 60% текущих объектов или менее чем у 8 объектов. Затем исключаются объекты с оценками менее чем по 60% оставшихся шкал или менее чем по 4 шкалам. Процедура повторяется до стабильности состава. В итоговой матрице требуется не менее 75% известных оценок. Неизвестные значения не заменяются нулём.",
    }
    if n < 12 or p < 6:
        raise InsufficientMeasurementError("Недостаточно реальных оценок: после исключения неизвестных объектов, "
                                           "редко измеренных и постоянных шкал нужны минимум 12 объектов и 6 шкал",
                                           {**diagnostics, "failure_code": "minimum_retained_dimensions"})
    if coverage < .75:
        raise InsufficientMeasurementError("Для анализа требуется не менее 75% известных оценок в используемой матрице",
                                           {**diagnostics, "failure_code": "minimum_retained_coverage"})
    return matrix, mass.loc[matrix.index, matrix.columns], object_map, scale_map, diagnostics


def _leading_count(observed, reference):
    failure = np.flatnonzero(observed <= reference)
    return int(failure[0]) if len(failure) else len(observed)


def _dimension_analysis(raw):
    filled = np.where(np.isfinite(raw), raw, np.nanmean(raw, axis=0))
    z = StandardScaler().fit_transform(filled)
    n, p = z.shape
    rank = int(np.linalg.matrix_rank(z))
    if rank < 2:
        raise ValueError("Матрица оценок содержит менее двух независимых измерений (ранг < 2). "
                         "Нужны различающиеся шкалы и реальные различия между объектами")
    eigenvalues = np.linalg.eigvalsh(z @ z.T / (n - 1))[::-1][:min(n - 1, p)]
    rng = np.random.default_rng(SEED)
    null = np.empty((PARALLEL_PERMUTATIONS, len(eigenvalues)))
    for repeat in range(PARALLEL_PERMUTATIONS):
        permuted = np.column_stack([rng.permutation(z[:, column]) for column in range(p)])
        null[repeat] = np.linalg.eigvalsh(permuted @ permuted.T / (n - 1))[::-1][:len(eigenvalues)]
    reference = np.quantile(null, .95, axis=0)
    retained = _leading_count(eigenvalues, reference)
    splits = list(ShuffleSplit(n_splits=3, test_size=.25, random_state=SEED).split(raw))
    qmax = min(8, p - 1, min(len(train) for train, _ in splits) - 1)
    covariance_moments = p * (p + 1) // 2
    accounting = []
    for q in range(2, qmax + 1):
        # Subtract orthogonal rotation indeterminacy; the diagonal noise adds p.
        # Nonnegative nominal df is necessary, not sufficient, for identification.
        parameter_count = p * q + p - q * (q - 1) // 2
        accounting.append({"q": q, "parameter_count": parameter_count,
                           "covariance_moments": covariance_moments,
                           "nominal_df": covariance_moments - parameter_count})
    admissible = [item for item in accounting if item["nominal_df"] >= 0]
    if not admissible:
        raise ValueError("Нет допустимых решений FA с q>=2: число параметров превышает число ковариационных моментов")
    covariance = z.T @ z / n
    off_diagonal = ~np.eye(p, dtype=bool)
    candidates, models = [], {}
    for dimension in admissible:
        q = dimension["q"]
        model, fitting = fit_model(raw, q, seed=SEED)
        models[q] = model
        rotated = model["loadings"] @ model["rotation"]
        strong = (np.abs(rotated) >= LOADING_THRESHOLD).sum(axis=0)
        residual = covariance - model["loadings"] @ model["loadings"].T
        residual -= np.diag(model["noise_variance"])
        values, warnings_seen = [], 0
        for train, test in splits:
            trained, local = fit_model(raw[train], q, seed=SEED)
            values.append(float(raw_observed_log_likelihood(raw[test], trained).mean()))
            warnings_seen += int(local["convergence_warning"])
        candidates.append({
            **dimension, "heldout_log_likelihood_mean": float(np.mean(values)),
            "heldout_log_likelihood_sd": float(np.std(values, ddof=1)),
            "heldout_log_likelihood_splits": values,
            "residual_rms_off_diagonal": float(np.sqrt(np.mean(residual[off_diagonal] ** 2))),
            "residual_max_abs_off_diagonal": float(np.max(np.abs(residual[off_diagonal]))),
            "residual_fraction_abs_above_010": float(np.mean(np.abs(residual[off_diagonal]) > .1)),
            "strong_traits_per_factor": strong.astype(int).tolist(),
            "all_factors_have_two_strong_traits": bool(np.all(strong >= 2)),
            "convergence_warnings": warnings_seen + int(fitting["convergence_warning"]),
            "common_variance_share": float((rotated ** 2).sum() / np.var(z, axis=0).sum()),
        })
    acceptable = [row for row in candidates if row["all_factors_have_two_strong_traits"]
                  and row["convergence_warnings"] == 0]
    pool = acceptable or candidates
    best = max(pool, key=lambda row: row["heldout_log_likelihood_mean"])
    # An explicit exploratory parsimony heuristic; overlapping split SD is NOT CI.
    tolerance = max(1., best["heldout_log_likelihood_sd"] / np.sqrt(len(splits)))
    shortlist = [row for row in pool if row["heldout_log_likelihood_mean"] >=
                 best["heldout_log_likelihood_mean"] - tolerance]
    chosen = min(row["q"] for row in shortlist)
    selection = {
        "recommended_q": chosen, "best_mean_heldout_q": best["q"],
        "parsimony_tolerance_log_likelihood": float(tolerance),
        "rule": "Only fit q with nominal_df=p(p+1)/2-[p*q+p-q(q-1)/2]>=0, a necessary parameter-count screen, not sufficient identification. Among converged candidates with at least two |loading|>=0.35 descriptors per factor, select the smallest q within max(1, SD_best/sqrt(3)) of the best three-split mean held-out likelihood; fall back to all candidates if none meet the loading criterion. This is an exploratory heuristic, not an inferential one-SE test.",
        "parameter_count_constraint": {"criterion": "nominal_df >= 0", "necessary_not_sufficient": True,
                                       "parameter_count_formula": "p*q+p-q*(q-1)/2",
                                       "covariance_moments": covariance_moments, "qmax_before_parameter_screen": qmax,
                                       "admissible_q": [item["q"] for item in admissible],
                                       "excluded_candidates": [item for item in accounting if item["nominal_df"] < 0]},
        "parallel_pca": {"retained_components": retained, "permutations": PARALLEL_PERMUTATIONS,
                         "quantile": .95, "observed_eigenvalues": eigenvalues.tolist(),
                         "reference_eigenvalues": reference.tolist(),
                         "scope": "PCA component permutation diagnostic, not common-factor retention; not a hard lower/upper bound on FA q"},
        "rank": rank,
        "selection_is_exploratory": True,
    }
    return chosen, candidates, models, selection


def _cluster_stability(scores, labels, k):
    rng = np.random.default_rng(SEED + k)
    agreements, jaccards = [], {group: [] for group in range(k)}
    for repeat in range(STABILITY_REPEATS):
        subset = np.sort(rng.choice(len(scores), size=max(k + 1, int(.8 * len(scores))), replace=False))
        sampled = KMeans(n_clusters=k, n_init=50, random_state=SEED + repeat).fit_predict(scores[subset])
        original = labels[subset]
        agreements.append(float(adjusted_rand_score(original, sampled)))
        for group in range(k):
            baseline = original == group
            if not baseline.any():
                continue
            jaccards[group].append(max(float(((sampled == other) & baseline).sum() /
                                              ((sampled == other) | baseline).sum())
                                        for other in range(k)))
    medians = [float(np.median(values)) if values else 0. for values in jaccards.values()]
    return {"mean_ari": float(np.mean(agreements)), "median_ari": float(np.median(agreements)),
            "min_ari": float(np.min(agreements)), "per_cluster_median_jaccard": medians,
            "min_cluster_median_jaccard": min(medians), "repeats": STABILITY_REPEATS,
            "scope": "80% objects, shared sampled objects, fixed factor scores; conditional cluster stability, not whole-pipeline validation"}


def _clustering(scores, requested_k):
    unique = len(np.unique(np.round(scores, decimals=12), axis=0))
    kmax = min(10, len(scores) // 4, unique)
    if kmax < 2:
        raise ValueError("У объектов нет хотя бы двух различающихся профилей для кластеризации")
    if requested_k is not None and (isinstance(requested_k, bool) or not isinstance(requested_k, (int, np.integer))
                                    or not 2 <= requested_k <= kmax):
        raise ValueError(f"Число кластеров должно быть целым от 2 до {kmax}")
    candidates, models = [], {}
    for k in range(2, kmax + 1):
        model = KMeans(n_clusters=k, n_init=50, random_state=SEED).fit(scores)
        models[k] = model
        sizes = np.bincount(model.labels_, minlength=k)
        stability = _cluster_stability(scores, model.labels_, k)
        candidates.append({"k": k, "silhouette": float(silhouette_score(scores, model.labels_)),
                           "inertia": float(model.inertia_), "sizes": sizes.astype(int).tolist(),
                           "min_size": int(sizes.min()), "max_size": int(sizes.max()),
                           "stability_mean_ari": stability["mean_ari"], "stability": stability})
    minimum_size = max(3, math.ceil(.05 * len(scores)))
    admissible = [row for row in candidates if row["min_size"] >= minimum_size and
                  row["stability"]["median_ari"] >= .65 and
                  row["stability"]["min_cluster_median_jaccard"] >= .60]
    pool = admissible or [row for row in candidates if row["min_size"] >= minimum_size] or candidates
    best_silhouette = max(row["silhouette"] for row in pool)
    chosen = min(row["k"] for row in pool if row["silhouette"] >= best_silhouette - .02)
    selection = {
        "recommended_k": chosen, "used_k": int(requested_k or chosen),
        "overridden": requested_k is not None, "stable_candidates_available": bool(admissible),
        "minimum_size_preference": minimum_size,
        "rule": "Prefer groups >=max(3,ceil(0.05*n)), median resampling ARI>=0.65 and every group's median Jaccard>=0.60; among admissible candidates choose smallest k within 0.02 of the best silhouette. If none qualify, use size-admissible candidates, then all candidates, and explicitly mark uncertainty. User override is allowed.",
        "interpretation": "Numerical choice is a reviewable proposal; final grouping requires semantic interpretation. KMeans also partitions continuous spaces.",
    }
    return models[int(requested_k or chosen)], candidates, selection, models


def _factor_descriptions(loadings, scales, denominator):
    factors = []
    for column in range(loadings.shape[1]):
        positive, negative = [], []
        for index in np.argsort(-np.abs(loadings[:, column])):
            value = float(loadings[index, column])
            if abs(value) < LOADING_THRESHOLD:
                continue
            scale = scales[index]
            trait = {"id": str(scale["id"]), "name": scale.get("name", str(scale["id"])),
                     "positive": scale.get("positive", ""), "negative": scale.get("negative", ""),
                     "loading": value,
                     "high_factor_property": scale.get("positive", "") if value > 0 else scale.get("negative", ""),
                     "low_factor_property": scale.get("negative", "") if value > 0 else scale.get("positive", "")}
            (positive if value > 0 else negative).append(trait)
        positive_names = ", ".join(item["positive"] or item["name"] for item in positive[:3])
        negative_names = ", ".join(item["positive"] or item["name"] for item in negative[:3])
        if positive and negative:
            label = f"{positive[0]['positive'] or positive[0]['name']} ↔ {negative[0]['positive'] or negative[0]['name']}"
        elif positive:
            label = ", ".join(item["positive"] or item["name"] for item in positive[:2])
        elif negative:
            label = "Обратная связь: " + ", ".join(item["positive"] or item["name"] for item in negative[:2])
        else:
            label = "Слабая семантическая определённость"
        sentences = []
        if positive:
            sentences.append(f"Положительные нагрузки: {positive_names}.")
        if negative:
            sentences.append(f"Отрицательные нагрузки: {negative_names}.")
        else:
            sentences.append("Сильных отрицательных нагрузок нет; второй полюс отдельно не именуется.")
        if not positive:
            sentences.append("Сильных положительных нагрузок нет.")
        high_properties = [item["positive"] or item["name"] for item in positive[:3]]
        high_properties += [item["negative"] for item in negative[:3] if item["negative"]]
        if high_properties:
            sentences.append("Высокая координата связана со свойствами: " + ", ".join(high_properties) + ".")
        if positive and negative:
            low_properties = [item["positive"] or item["name"] for item in negative[:3]]
            low_properties += [item["negative"] for item in positive[:3] if item["negative"]]
            sentences.append("Низкая координата связана со свойствами: " + ", ".join(low_properties) + ".")
        sentences.append("Знак оси условен. Названия описывают связи оценок; это не причинное объяснение.")
        factors.append({"id": f"F{column + 1}", "name": label, "label": label,
                        "positive_traits": positive, "negative_traits": negative,
                        "strong_loading_threshold": LOADING_THRESHOLD,
                        "variance_share": float((loadings[:, column] ** 2).sum() / denominator),
                        "interpretation": " ".join(sentences)})
    return factors


def _object_and_cluster_profiles(frame, scores, clustering, factors, object_map, scale_map, candidate):
    ids = frame.index.tolist()
    raw = frame.to_numpy()
    overall = np.nanmean(raw, axis=0)
    deviations = np.nanstd(raw, axis=0)
    labels, centers = clustering.labels_, clustering.cluster_centers_
    distances = clustering.transform(scores)
    pairwise = np.sqrt(np.maximum(((scores[:, None] - scores[None]) ** 2).sum(axis=2), 0))
    profiles, clusters = [], []
    for row, oid in enumerate(ids):
        order = np.argsort(distances[row])
        margin = float((distances[row, order[1]] - distances[row, order[0]]) /
                       max(distances[row, order[1]], 1e-12))
        neighbors = [int(index) for index in np.argsort(pairwise[row]) if index != row][:5]
        observed = int(frame.iloc[row].notna().sum())
        focus_axes = np.argsort(-np.abs(scores[row]))[:2]
        description = "; ".join(f"{factors[index]['id']}: {scores[row, index]:+.2f}" for index in focus_axes)
        profiles.append({"id": oid, "name": object_map[oid].get("name", oid),
                         "cluster_id": f"C{int(labels[row]) + 1}",
                         "factor_scores": {item["id"]: float(scores[row, index]) for index, item in enumerate(factors)},
                         "observed_scales": observed, "missing_scales": frame.shape[1] - observed,
                         "boundary_margin": margin, "mixed_or_boundary": margin < .15,
                         "nearest_objects": [{"id": ids[index], "name": object_map[ids[index]].get("name", ids[index]),
                                              "distance": float(pairwise[row, index])} for index in neighbors],
                         "profile": f"Наиболее выраженные координаты: {description}. " +
                                    ("Профиль находится у границы групп." if margin < .15 else
                                     "Ближайшая группа определяется расстоянием в полном факторном пространстве.")})
    for group in range(clustering.n_clusters):
        indices = np.flatnonzero(labels == group)
        mean = np.nanmean(raw[indices], axis=0)
        differences = (mean - overall) / np.maximum(deviations, 1e-12)
        finite = [int(index) for index in np.argsort(-np.nan_to_num(np.abs(differences), nan=-1.))
                  if np.isfinite(differences[index])][:6]
        distinguishing = [{"id": str(frame.columns[index]),
                           "name": scale_map[str(frame.columns[index])].get("name", str(frame.columns[index])),
                           "positive": scale_map[str(frame.columns[index])].get("positive", ""),
                           "negative": scale_map[str(frame.columns[index])].get("negative", ""),
                           "difference": float(differences[index]), "group_mean": float(mean[index]),
                           "overall_mean": float(overall[index]),
                           "direction": "positive" if differences[index] > 0 else "negative" if differences[index] < 0 else "neutral",
                           "observed_group_count": int(np.isfinite(raw[indices, index]).sum())} for index in finite]
        representatives = indices[np.argsort(distances[indices, group])[:5]]
        members = [{"id": ids[index], "name": object_map[ids[index]].get("name", ids[index])} for index in indices]
        phrases = [f"«{item['name']}» ({item['difference']:+.2f} σ)" for item in distinguishing[:3]]
        interpretation = "Отличия от среднего по изученным объектам: " + ", ".join(phrases) + ". "
        interpretation += "Общность описывается сочетанием оценённых свойств; совпадение электората или причин общности требует внешней проверки."
        clusters.append({"id": f"C{group + 1}", "name": f"Группа {group + 1}",
                         "count": len(indices), "members": members,
                         "representatives": [{"id": ids[index], "name": object_map[ids[index]].get("name", ids[index]),
                                              "distance": float(distances[index, group])} for index in representatives],
                         "factor_center": {factor["id"]: float(centers[group, index]) for index, factor in enumerate(factors)},
                         "distinguishing_scales": distinguishing, "interpretation": interpretation,
                         "boundary_objects": [profile["id"] for profile in profiles if profile["cluster_id"] == f"C{group + 1}"
                                              and profile["mixed_or_boundary"]],
                         "stability": {"median_jaccard": candidate["stability"]["per_cluster_median_jaccard"][group],
                                       "conditional_mean_ari": candidate["stability"]["mean_ari"]}})
    return profiles, clusters


def _semantic_candidate_profiles(frame, scores, models, factors, object_map, scale_map, candidates):
    """Attach a complete, compact description of each candidate partition.

    Uses already-fitted KMeans models and the identical full factor coordinates.
    This supplies evidence for substantive review of k, not automatic LLM assent.
    """
    for candidate in candidates:
        _, groups = _object_and_cluster_profiles(frame, scores, models[candidate["k"]], factors,
                                                 object_map, scale_map, candidate)
        semantic = []
        for group in groups:
            ordered = sorted(group["factor_center"].items(), key=lambda item: -abs(item[1]))[:3]
            labels = {factor["id"]: factor.get("label") or factor.get("name") for factor in factors}
            semantic.append({
                "id": group["id"], "size": group["count"],
                "member_ids": [member["id"] for member in group["members"]],
                "representatives": group["representatives"],
                "top_factors": [{"factor_id": identity, "label": labels[identity], "coordinate": coordinate,
                                 "direction": "positive" if coordinate > 1e-10 else "negative" if coordinate < -1e-10 else "neutral"}
                                for identity, coordinate in ordered],
                "distinguishing_scales": group["distinguishing_scales"],
            })
        candidate["semantic_profiles"] = semantic


def _tsne_initialization(matrix):
    centered = matrix - matrix.mean(axis=0)
    # sklearn PCA initialization divides by its leading coordinate's std.
    # Coincident vectors would produce NaN; reproducible random init is safe.
    return "random" if np.linalg.matrix_rank(centered) < 2 else "pca"


def _readable_labels(ax, coordinates, names, candidates, maximum=8):
    """Label representative points while rejecting overlapping annotation boxes."""
    ax.margins(x=.16, y=.16)
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    occupied = []
    middle = (np.min(coordinates, axis=0) + np.max(coordinates, axis=0)) / 2
    for index in candidates:
        right, top = coordinates[index] >= middle
        label = textwrap.fill(str(names[index]), width=25)
        annotation = ax.annotate(label, coordinates[index], xytext=(-6 if right else 6, -5 if top else 5),
                                 textcoords="offset points", fontsize=8,
                                 ha="right" if right else "left", va="top" if top else "bottom")
        extent = annotation.get_window_extent(renderer).expanded(1.05, 1.12)
        if any(extent.overlaps(other) for other in occupied):
            annotation.remove()
            continue
        occupied.append(extent)
        if len(occupied) >= maximum:
            break


def _save_charts(output, scores, loadings, factors, clusters, profiles, scales, study):
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    charts = []
    labels = np.array([int(item["cluster_id"][1:]) - 1 for item in profiles])
    names = [item["name"] for item in profiles]
    focus = study.get("focus_object_ids", []) or []
    if study.get("focus_object_id"):
        focus = [*focus, study["focus_object_id"]]
    focus = set(str(identity) for identity in focus)
    focus_text = str(study.get("focus") or "").strip().casefold()
    if focus_text:
        focus.update(item["id"] for item in profiles if focus_text in item["name"].casefold() or focus_text == item["id"].casefold())
    annotated = {index for index, item in enumerate(profiles) if item["id"] in focus}
    representatives = {item["representatives"][0]["id"] for item in clusters}
    annotated.update(index for index, item in enumerate(profiles) if item["id"] in representatives)
    annotated = sorted(annotated, key=lambda index: profiles[index]["id"] not in focus)[:12]

    def save(fig, filename, title, caption):
        fig.savefig(assets / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
        charts.append({"title": title, "file": f"assets/{filename}", "caption": caption})

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    shares = [100 * item["variance_share"] for item in factors]
    ax.bar([item["id"] for item in factors], shares, color="#3867a8")
    ax.set(ylabel="Доля наблюдаемой дисперсии, %", title="Дисперсия общих факторов")
    for index, share in enumerate(shares):
        ax.text(index, share + .3, f"{share:.1f}%", ha="center", fontsize=9)
    save(fig, "variance.png", "Дисперсия факторов",
         "Суммы квадратов вращённых нагрузок / общая дисперсия стандартизованных шкал. Описательные доли FA; это не доли дисперсии компонент PCA.")

    def objects_plot(coordinates, filename, title, xlabel, ylabel, caption):
        fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
        colors = plt.get_cmap("tab10")
        for group in range(len(clusters)):
            present = labels == group
            ax.scatter(coordinates[present, 0], coordinates[present, 1], s=44, alpha=.85,
                       color=colors(group % 10), label=f"C{group + 1} ({present.sum()})")
        _readable_labels(ax, coordinates, names, annotated, maximum=8)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        ax.axhline(0, color="#ccd0d6", linewidth=.6)
        ax.axvline(0, color="#ccd0d6", linewidth=.6)
        ax.legend(fontsize=8, loc="best")
        save(fig, filename, title, caption)

    objects_plot(scores[:, :2], "objects_f1_f2.png", "Объекты в факторных координатах", "F1", "F2",
                 "Точки — posterior scores объектов по F1/F2. Группы рассчитаны во всех выбранных факторах; двухмерная проекция может скрывать различия.")

    def traits_plot(coordinates, filename, title, xlabel, ylabel, caption):
        fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
        ax.scatter(coordinates[:, 0], coordinates[:, 1], s=32, alpha=.7, color="#267b6b")
        selected = np.argsort(-np.linalg.norm(loadings[:, :2], axis=1))
        _readable_labels(ax, coordinates, [scale.get("name", scale["id"]) for scale in scales], selected, maximum=8)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        ax.axhline(0, color="#ccd0d6", linewidth=.6)
        ax.axvline(0, color="#ccd0d6", linewidth=.6)
        save(fig, filename, title, caption)

    traits_plot(loadings[:, :2], "traits_f1_f2.png", "Семантические шкалы: нагрузки F1/F2", "Нагрузка на F1", "Нагрузка на F2",
                "Точки — шкалы, координаты — их факторные нагрузки. Эти точки отличаются от координат объектов; единицы и геометрия не взаимозаменяемы.")
    fig, ax = plt.subplots(figsize=(8, min(13, max(5, len(profiles) * .16))), constrained_layout=True)
    order = np.argsort(labels, kind="stable")
    image = ax.imshow(scores[order], aspect="auto", cmap="RdBu_r",
                      vmin=-max(1., float(np.max(np.abs(scores)))), vmax=max(1., float(np.max(np.abs(scores)))))
    step = max(1, math.ceil(len(order) / 25))
    ticks = list(range(0, len(order), step))
    ax.set_yticks(ticks, [names[order[index]][:30] for index in ticks], fontsize=8)
    ax.set_xticks(range(len(factors)), [factor["id"] for factor in factors])
    ax.set_title("Профили объектов по факторам")
    fig.colorbar(image, ax=ax, label="Posterior score")
    save(fig, "heatmap.png", "Профили объектов", "Объекты упорядочены по группам; цвет показывает оценённую факторную координату. Подписи строк прорежены для читаемости.")
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for cluster in clusters:
        ax.plot(range(len(factors)), list(cluster["factor_center"].values()), marker="o",
                label=f"{cluster['id']} ({cluster['count']})")
    ax.set_xticks(range(len(factors)), [factor["id"] for factor in factors])
    ax.set(ylabel="Средняя факторная координата", title="Средние профили групп")
    ax.axhline(0, linewidth=.8, color="#9ca3af")
    ax.legend(fontsize=8)
    save(fig, "cluster_profile.png", "Профили групп", "Центры групп в исходном выбранном факторном пространстве. Отдельная стандартизация факторных осей не применялась.")
    tsne_settings = {}
    for key, matrix in (("objects", scores), ("traits", loadings)):
        perplexity = float(min(30, max(2, (len(matrix) - 1) / 3)))
        initialization = _tsne_initialization(matrix)
        projected = TSNE(n_components=2, perplexity=perplexity, init=initialization, learning_rate="auto",
                         random_state=SEED, max_iter=1000, n_jobs=1).fit_transform(matrix)
        tsne_settings[key] = {"seed": SEED, "perplexity": perplexity, "input_dimensions": matrix.shape[1],
                              "n_points": len(matrix), "iterations": 1000, "initialization": initialization}
        if key == "objects":
            objects_plot(projected, "tsne_objects.png", "t-SNE объектов", "t-SNE 1", "t-SNE 2",
                         "t-SNE строится по всем выбранным факторным координатам. Его оси не являются факторами; глобальные расстояния и видимые пустоты нельзя трактовать как свойства пространства.")
        else:
            traits_plot(projected, "tsne_traits.png", "t-SNE семантических шкал", "t-SNE 1", "t-SNE 2",
                        "t-SNE строится по векторам нагрузок шкал на все выбранные факторы. Оси t-SNE не имеют самостоятельной семантической интерпретации.")
        identities = [item["id"] for item in profiles] if key == "objects" else [str(item["id"]) for item in scales]
        pd.DataFrame(projected, index=pd.Index(identities, name="id"), columns=["tsne_1", "tsne_2"]).to_csv(
            output / f"tsne_{key}.csv", encoding="utf-8-sig")
    return charts, tsne_settings


def analyze_study(study: dict, ratings: list[dict], output_dir: Path,
                  factor_count=None, cluster_count=None) -> dict:
    """Analyze actual approved-study ratings and save a reproducible report.

    Approvals are enforced by the caller. ``value`` imports are precomputed
    differences in [-5,5], not 0..10 party ratings silently reinterpreted here.
    Returns JSON-safe metadata; artifact paths are relative to output_dir.
    """
    output = Path(output_dir)
    frame, mass, object_map, scale_map, diagnostics = _prepare(study, ratings)
    raw = frame.to_numpy()
    n, p = raw.shape
    with threadpool_limits(limits=1):
        recommended_q, dimensions, models, selection = _dimension_analysis(raw)
        qmax = max(models)
        if factor_count is not None and (isinstance(factor_count, bool) or not isinstance(factor_count, (int, np.integer))
                                         or not 2 <= factor_count <= qmax):
            raise ValueError(f"Число факторов должно быть целым от 2 до {qmax}")
        q = int(factor_count or recommended_q)
        parameters = models[q]
        parameters["trait_order"] = frame.columns.tolist()
        parameters["training_object_order"] = frame.index.tolist()
        scores = transform_new_objects(raw, parameters)
        loadings = parameters["loadings"] @ parameters["rotation"]
        standardized = (np.where(np.isfinite(raw), raw, parameters["training_imputation"]) -
                        parameters["training_mean"]) / parameters["training_scale"]
        denominator = float(np.var(standardized, axis=0).sum())
        scales = [scale_map[identity] for identity in frame.columns]
        factors = _factor_descriptions(loadings, scales, denominator)
        clustering, cluster_candidates, cluster_selection, candidate_models = _clustering(scores, cluster_count)
        _semantic_candidate_profiles(frame, scores, candidate_models, factors, object_map, scale_map, cluster_candidates)
        selected_cluster = next(item for item in cluster_candidates if item["k"] == clustering.n_clusters)
        profiles, clusters = _object_and_cluster_profiles(frame, scores, clustering, factors,
                                                         object_map, scale_map, selected_cluster)
        output.mkdir(parents=True, exist_ok=True)
        charts, tsne_settings = _save_charts(output, scores, loadings, factors, clusters, profiles, scales, study)
    factor_ids = [item["id"] for item in factors]
    matrix_export = frame.rename_axis("object_id")
    matrix_export.to_csv(output / "matrix.csv", encoding="utf-8-sig")
    mass.rename_axis("object_id").to_csv(output / "mass.csv", encoding="utf-8-sig")
    score_frame = pd.DataFrame(scores, index=pd.Index(frame.index, name="object_id"), columns=factor_ids)
    score_frame.insert(0, "name", [object_map[identity].get("name", identity) for identity in frame.index])
    score_frame.to_csv(output / "scores.csv", encoding="utf-8-sig")
    loading_frame = pd.DataFrame(loadings, index=pd.Index(frame.columns, name="scale_id"), columns=factor_ids)
    loading_frame.insert(0, "name", [scale_map[identity].get("name", identity) for identity in frame.columns])
    loading_frame.to_csv(output / "loadings.csv", encoding="utf-8-sig")
    pd.DataFrame([{"object_id": profile["id"], "name": profile["name"], "cluster_id": profile["cluster_id"],
                   "boundary_margin": profile["boundary_margin"]} for profile in profiles]).to_csv(
        output / "cluster_labels.csv", index=False, encoding="utf-8-sig")
    parameters["cluster_centers"] = clustering.cluster_centers_
    parameters["cluster_ids"] = [item["id"] for item in clusters]
    joblib.dump(parameters, output / "parameters.joblib")
    selection.update({"used_q": q, "overridden": factor_count is not None})
    warnings = ["Исследование разведочное; численные группы могут возникать и в непрерывном пространстве.",
                "Гауссова линейная FA приблизительна для ограниченных порядковых оценок; независимость ошибок шкал может нарушаться.",
                "Сходимость и устойчивость алгоритма не подтверждают точность исходных оценок или полноту пространства.",
                "Кластеризация условна на выбранных факторных баллах; устойчивость всего сбора и анализа отдельно не проверена."]
    if n < max(50, 3 * p):
        warnings.append("Число объектов невелико относительно числа шкал; число факторов и интерпретации требуют проверки на новых объектах.")
    if diagnostics["missing_retained_cells"]:
        warnings.append("При обучении пропуски заполнены средними обучающей выборки. Это техническое допущение; при проекции неизвестные шкалы маргинализируются.")
    if not cluster_selection["stable_candidates_available"]:
        warnings.append("Ни один кандидат кластеризации не выполнил одновременно пороги размера и устойчивости; выбранное разбиение особенно предварительно.")
    diagnostics.update({"dimension_selection": selection, "cluster_selection": cluster_selection,
                        "seed": SEED, "tsne": tsne_settings,
                        "variance_denominator": denominator,
                        "variance_definition": "sum of squared orthogonally rotated FA loadings / total observed standardized variance; descriptive common-factor shares, not PCA variance",
                        "posterior_formula": "B=I+L.T diag(psi)^-1 L; E[f|z]=B^-1 L.T diag(psi)^-1 (z-mu); Cov(f|z)=B^-1",
                        "noise_floor": float(parameters["noise_floor"]),
                        "noise_floor_policy": "positive floor applied after ML fit; explicitly diagnostic regularization",
                        "minimum_noise": float(parameters["noise_variance"].min()),
                        "posterior_condition_number": float(np.linalg.cond(np.eye(q) + parameters["loadings"].T @
                            (parameters["loadings"] / parameters["noise_variance"][:, None]))),
                        "input_sha256": hashlib.sha256(json.dumps({"study": study, "ratings": ratings},
                                                   sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()})
    result = {"factor_count": q, "cluster_count": int(clustering.n_clusters), "n_objects": n, "n_scales": p,
              "variance_total": float(sum(factor["variance_share"] for factor in factors)),
              "factors": factors, "clusters": clusters, "cluster_candidates": cluster_candidates,
              "dimension_candidates": dimensions, "charts": charts, "object_profiles": profiles,
              "loading_table": [{"id": str(scale["id"]), "name": scale["name"],
                                 "loadings": {factor["id"]: float(loadings[index, column])
                                              for column, factor in enumerate(factors)}}
                                for index, scale in enumerate(scales)],
              "diagnostics": diagnostics, "warnings": warnings,
              "artifacts": {"matrix_csv": "matrix.csv", "mass_csv": "mass.csv", "scores_csv": "scores.csv",
                            "loadings_csv": "loadings.csv", "cluster_labels_csv": "cluster_labels.csv",
                            "parameters_joblib": "parameters.joblib", "metadata_json": "metadata.json",
                            "tsne_objects_csv": "tsne_objects.csv", "tsne_traits_csv": "tsne_traits.csv"}}
    (output / "metadata.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return result
