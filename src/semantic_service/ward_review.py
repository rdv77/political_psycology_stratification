"""Reviewable Ward partitions of the frozen, three-dimensional FA posterior.

No k is selected here. Subsampling, analytical imputations and axis rescaling
are separate sensitivity experiments, not independent expert measurements.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score, silhouette_samples, silhouette_score
from threadpoolctl import threadpool_limits

from scripts.latent_space_benchmark import fit_model, transform_new_objects
from src.semantic_service.analysis import SEED, _prepare
from src.semantic_service.factor_preview import _align_loading_columns
from src.semantic_service.imputation_review import impute_from_training


DISPLAY_SIGNS = np.array([-1., 1., 1.])
DISPLAY_AXES = {
    "X1": {"raw_axis": "F1", "multiplier": -1, "label": "Преемственность и поддержка курса"},
    "X2": {"raw_axis": "F2", "multiplier": 1, "label": "Обновление курса и раскрытие проблем"},
    "X3": {"raw_axis": "F3", "multiplier": 1, "label": "Доступность благ и профилактика относительно ценностной аргументации"},
}


def ward_partition(scores, k):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or not np.isfinite(scores).all() or scores.shape[1] != 3:
        raise ValueError("Finite n-by-3 posterior scores required")
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not 2 <= k < len(scores):
        raise ValueError("Require 2 <= k < n")
    if len(np.unique(np.round(scores, 12), axis=0)) < k:
        raise ValueError("Fewer distinct measured profiles than proposed groups")
    labels = cut_tree(linkage(scores, method="ward", metric="euclidean"), n_clusters=[k]).ravel()
    # Labels are local to k. Order by native F1 centroid, then F2/F3, then row.
    order = sorted(range(k), key=lambda g: (*scores[labels == g].mean(axis=0), int(np.flatnonzero(labels == g)[0])))
    mapping = {old: new for new, old in enumerate(order)}
    return np.array([mapping[int(group)] for group in labels], dtype=int)


def _align_partition(reference, candidate, k):
    table = np.zeros((k, k), dtype=int)
    np.add.at(table, (reference, candidate), 1)
    rows, columns = linear_sum_assignment(-table)
    mapping = {int(column): int(row) for row, column in zip(rows, columns)}
    return np.array([mapping[int(group)] for group in candidate]), table


def summarize_partition_variants(reference, variants, object_ids, k):
    """Conditional co-membership denominators include only jointly present rows."""
    reference = np.asarray(reference, dtype=int)
    n = len(reference)
    seen = np.zeros((n, n), dtype=int)
    same = np.zeros((n, n), dtype=int)
    distributions = np.zeros((n, k), dtype=int)
    object_seen = np.zeros(n, dtype=int)
    aris, records = [], []
    jaccards = [[] for _ in range(k)]
    for repeat, variant in enumerate(variants):
        indices = np.asarray(variant["indices"], dtype=int)
        labels = np.asarray(variant["labels"], dtype=int)
        if len(indices) != len(set(indices)) or len(indices) != len(labels):
            raise ValueError("Each variant needs unique shared-object indices")
        original = reference[indices]
        aligned, overlap = _align_partition(original, labels, k)
        ari = float(adjusted_rand_score(original, labels))
        aris.append(ari)
        seen[np.ix_(indices, indices)] += 1
        same[np.ix_(indices, indices)] += labels[:, None] == labels[None, :]
        object_seen[indices] += 1
        distributions[indices, aligned] += 1
        for group in range(k):
            before, after = original == group, aligned == group
            if before.any():
                jaccards[group].append(float(np.count_nonzero(before & after) / np.count_nonzero(before | after)))
        records.append({"repeat": repeat, "object_ids": [object_ids[i] for i in indices],
                        "aligned_cluster_ids": [f"C{g+1}" for g in aligned],
                        "adjusted_rand": ari, "overlap_table": overlap.tolist(), **variant.get("metadata", {})})
    frequency = np.divide(same, seen, out=np.zeros_like(same, dtype=float), where=seen > 0)
    objects = []
    for i, oid in enumerate(object_ids):
        count = int(object_seen[i])
        objects.append({"id": oid, "included_repeats": count, "reference_cluster": f"C{reference[i]+1}",
                        "same_cluster_fraction": float(distributions[i, reference[i]] / count) if count else None,
                        "cluster_fractions": {f"C{g+1}": float(distributions[i, g] / count) if count else None for g in range(k)}})
    return {"repeats": len(variants), "mean_adjusted_rand": float(np.mean(aris)) if aris else None,
            "median_adjusted_rand": float(np.median(aris)) if aris else None,
            "adjusted_rand_q05_q50_q95": np.quantile(aris, [.05, .5, .95]).tolist() if aris else None,
            "per_cluster_jaccard": [{"id": f"C{g+1}", "represented_repeats": len(values),
                                      "mean": float(np.mean(values)) if values else None,
                                      "median": float(np.median(values)) if values else None}
                                     for g, values in enumerate(jaccards)],
            "objects": objects, "pair_object_order": object_ids,
            "pair_jointly_present_count": seen.tolist(),
            "pair_co_membership_frequency": [[float(frequency[i, j]) if seen[i, j] else None for j in range(n)] for i in range(n)],
            "records": records,
            "alignment_note": "Cluster labels aligned by maximum overlap on shared objects. Pair co-membership is label invariant. Jaccard uses the globally matched group, conditional on original-group presence; no cluster is treated as stable merely because labels were renumbered."}


def _object_boundaries(scores, labels, object_ids, names):
    k = int(labels.max()) + 1
    centers = np.array([scores[labels == group].mean(axis=0) for group in range(k)])
    distances = cdist(scores, centers)
    silhouettes = silhouette_samples(scores, labels)
    result = []
    for i, oid in enumerate(object_ids):
        group = int(labels[i])
        others = [g for g in range(k) if g != group]
        other = min(others, key=lambda g: distances[i, g])
        own_d, other_d = float(distances[i, group]), float(distances[i, other])
        margin = (other_d - own_d) / max(own_d, other_d, 1e-12)
        result.append({"id": oid, "name": names[oid], "cluster_id": f"C{group+1}",
                       "silhouette": float(silhouettes[i]), "own_centroid_distance": own_d,
                       "closest_alternative_centroid": f"C{other+1}", "alternative_centroid_distance": other_d,
                       "centroid_margin": float(margin), "assigned_group_has_nearest_centroid": own_d <= other_d,
                       "note": "Centroid margin is descriptive only: Ward assignments come from the hierarchy and are never replaced by nearest-centroid labels."})
    return result, centers


def group_profiles(raw, scores, labels, object_ids, names, variable_ids, variable_names):
    """Observed-only property summaries: unknown is neither zero nor imputation."""
    groups = []
    overall_mean, overall_sd = np.nanmean(raw, axis=0), np.nanstd(raw, axis=0)
    for group in range(int(labels.max()) + 1):
        indices = np.flatnonzero(labels == group)
        center = scores[indices].mean(axis=0)
        distances = np.linalg.norm(scores[indices] - center, axis=1)
        properties = []
        for j, sid in enumerate(variable_ids):
            values = raw[indices, j]
            values = values[np.isfinite(values)]
            mean = float(values.mean()) if len(values) else None
            delta = float((mean - overall_mean[j]) / overall_sd[j]) if mean is not None and overall_sd[j] > 1e-12 else None
            properties.append({"id": sid, "name": variable_names[sid], "observed_n": len(values),
                               "group_n": len(indices), "missing_n": int(len(indices) - len(values)),
                               "observed_mean": mean, "observed_median": float(np.median(values)) if len(values) else None,
                               "overall_observed_mean": float(overall_mean[j]), "standardized_difference_from_overall": delta})
        required = min(len(indices), max(2, math.ceil(.5 * len(indices))))
        usable = [x for x in properties if x["observed_n"] >= required and x["standardized_difference_from_overall"] is not None]
        top = sorted(usable, key=lambda x: -abs(x["standardized_difference_from_overall"]))[:4]
        representative_indices = indices[np.argsort(distances)[:min(5, len(indices))]]
        groups.append({"id": f"C{group+1}", "count": len(indices),
                       "members": [{"id": object_ids[i], "name": names[object_ids[i]]} for i in indices],
                       "representatives": [{"id": object_ids[i], "name": names[object_ids[i]],
                                            "distance_to_centroid": float(np.linalg.norm(scores[i] - center))} for i in representative_indices],
                       "raw_factor_center": {f"F{j+1}": float(center[j]) for j in range(3)},
                       "display_factor_center": {f"X{j+1}": float(center[j] * DISPLAY_SIGNS[j]) for j in range(3)},
                       "observed_trait_profiles": properties, "distinguishing_traits": top,
                       "minimum_observed_n_for_distinguishing_summary": required,
                       "interpretation_status": "substantive_name_pending",
                       "interpretation_limit": "Group means describe recorded public properties within this selected sample. Lower expression does not imply an unmeasured opposite, a shared electorate or a common cause."})
    return groups


def _uncertain_pairs(summary, labels, object_ids, names):
    pairs = []
    for i in range(len(object_ids)):
        for j in range(i + 1, len(object_ids)):
            frequency = summary["pair_co_membership_frequency"][i][j]
            if frequency is None:
                continue
            pairs.append({"first_id": object_ids[i], "first_name": names[object_ids[i]],
                          "second_id": object_ids[j], "second_name": names[object_ids[j]],
                          "baseline_same_cluster": bool(labels[i] == labels[j]),
                          "co_membership_frequency": frequency,
                          "joint_repeats": summary["pair_jointly_present_count"][i][j]})
    return {"closest_to_half": sorted(pairs, key=lambda x: abs(x["co_membership_frequency"] - .5))[:15],
            "least_retained_within_group": sorted([x for x in pairs if x["baseline_same_cluster"]], key=lambda x: x["co_membership_frequency"])[:15],
            "most_merged_between_groups": sorted([x for x in pairs if not x["baseline_same_cluster"]], key=lambda x: -x["co_membership_frequency"])[:15]}


def _membership_sort_key(item):
    values = [item[key] for key in ("subsample_same_cluster_fraction", "imputation_same_cluster_fraction")]
    return min(value if value is not None else -1. for value in values)


def review_ward_q3(study, ratings, preview, output_dir: Path, *, k_values=(2, 3, 4, 5, 6),
                   subsample_repeats=100, imputation_draws=10, seed=SEED):
    matrix, _, object_map, scale_map, diagnostics = _prepare(study, ratings)
    ids, variables = matrix.index.tolist(), matrix.columns.tolist()
    raw = matrix.to_numpy(dtype=float)
    names = {oid: object_map[oid].get("name", oid) for oid in ids}
    variable_names = {sid: scale_map[sid].get("name", sid) for sid in variables}
    model = next(x for x in preview["models"] if x["q"] == 3)
    if preview["diagnostics"]["training_object_order"] != ids or preview["diagnostics"]["trait_order"] != variables:
        raise ValueError("Preview and current admitted matrix use different rows or variables")
    if len(set(k_values)) != len(k_values) or any(isinstance(k, bool) or not isinstance(k, int) or not 2 <= k < len(ids) for k in k_values):
        raise ValueError("Distinct integer cluster counts 2 <= k < n required")
    if subsample_repeats < 2 or imputation_draws < 2:
        raise ValueError("At least two sensitivity replicates required")
    with threadpool_limits(limits=1):
        parameters, fit_diagnostics = fit_model(raw, 3, seed=seed)
        loadings = parameters["loadings"] @ parameters["rotation"]
        saved = np.array([[x["loadings"][f"F{j+1}"] for j in range(3)] for x in model["loading_table"]])
        delta = float(np.max(np.abs(loadings - saved)))
        if delta > 1e-10:
            raise ValueError("FA does not reproduce the frozen q3 preview")
        scores = transform_new_objects(raw, parameters, rotated=True)
        parameters.update(trait_order=variables, training_object_order=ids)
        standard_deviation = scores.std(axis=0)
        if np.any(standard_deviation <= 1e-12):
            raise ValueError("A posterior score axis is constant")
        # Analytical copies only; canonical unknown statuses remain untouched.
        imputation_scores, imputation_records = [], []
        bounds = [(0., 5.) if scale_map[sid].get("kind") == "unipolar" else (-5., 5.) for sid in variables]
        for draw in range(imputation_draws):
            complete, imputer_fit = impute_from_training(raw, raw, method="bayesian_ridge", posterior=True,
                                                        bounds=bounds, seed=seed + 5000 + draw)
            local, local_fit = fit_model(complete, 3, seed=seed)
            local_loadings = local["loadings"] @ local["rotation"]
            aligned, congruence = _align_loading_columns(loadings, local_loadings)
            # Only the identical sign/permutation, never a fitted continuous rotation.
            signed = loadings.T @ local_loadings / np.maximum(
                np.linalg.norm(loadings, axis=0)[:, None] * np.linalg.norm(local_loadings, axis=0)[None, :], 1e-12)
            rows, columns = linear_sum_assignment(-np.abs(signed))
            transform = np.zeros((3, 3))
            for reference_axis, candidate_axis in zip(rows, columns):
                transform[candidate_axis, reference_axis] = 1. if signed[reference_axis, candidate_axis] >= 0 else -1.
            if not np.allclose(local_loadings @ transform, aligned, rtol=0, atol=1e-12):
                raise ValueError("Loading matching and score-column alignment disagree")
            local_scores = transform_new_objects(complete, local, rotated=True) @ transform
            imputation_scores.append(local_scores)
            imputation_records.append({"draw": draw, "imputation_seed": seed + 5000 + draw,
                                       "loading_congruence_to_reference": congruence.tolist(),
                                       "FA_convergence_warning": local_fit["convergence_warning"],
                                       "imputer_fit": imputer_fit})
        candidates = []
        for k in k_values:
            labels = ward_partition(scores, k)
            reflected = ward_partition(scores * DISPLAY_SIGNS, k)
            if adjusted_rand_score(labels, reflected) != 1.:
                raise ValueError("Display reflection unexpectedly changed the Ward partition")
            boundaries, centers = _object_boundaries(scores, labels, ids, names)
            groups = group_profiles(raw, scores, labels, ids, names, variables, variable_names)
            variants = []
            rng = np.random.default_rng(seed + 10000 + k)
            for repeat in range(subsample_repeats):
                indices = np.sort(rng.choice(len(ids), size=max(k + 1, int(.8 * len(ids))), replace=False))
                variants.append({"indices": indices, "labels": ward_partition(scores[indices], k)})
            resampling = summarize_partition_variants(labels, variants, ids, k)
            multiple = summarize_partition_variants(labels, [{"indices": np.arange(len(ids)), "labels": ward_partition(s, k),
                    "metadata": imputation_records[draw]} for draw, s in enumerate(imputation_scores)], ids, k)
            scaled_labels = ward_partition((scores - scores.mean(axis=0)) / standard_deviation, k)
            scaled_aligned, _ = _align_partition(labels, scaled_labels, k)
            sub_objects = {x["id"]: x for x in resampling["objects"]}
            mi_objects = {x["id"]: x for x in multiple["objects"]}
            uncertain = []
            for boundary in boundaries:
                oid = boundary["id"]
                boundary.update(subsample_same_cluster_fraction=sub_objects[oid]["same_cluster_fraction"],
                                imputation_same_cluster_fraction=mi_objects[oid]["same_cluster_fraction"],
                                scaled_cluster_id=f"C{scaled_aligned[ids.index(oid)]+1}")
                if (_membership_sort_key(boundary) < .8 or
                    boundary["centroid_margin"] < .1 or boundary["silhouette"] < 0):
                    uncertain.append(boundary)
            for group in groups:
                g = int(group["id"][1:]) - 1
                group["subsample_jaccard"] = resampling["per_cluster_jaccard"][g]
                group["imputation_jaccard"] = multiple["per_cluster_jaccard"][g]
            sizes = np.bincount(labels, minlength=k)
            candidates.append({"k": k, "sizes": sizes.tolist(), "min_size": int(sizes.min()),
                "silhouette": float(silhouette_score(scores, labels)),
                "within_cluster_sum_squares": float(np.sum((scores - centers[labels])**2)),
                "labels_by_object": {oid: f"C{labels[i]+1}" for i, oid in enumerate(ids)},
                "clusters": groups, "object_assignments": boundaries,
                "stability": {"object_subsampling": resampling, "multiple_imputation": multiple,
                    "empirical_axis_standardization": {"adjusted_rand": float(adjusted_rand_score(labels, scaled_labels)),
                        "matched_labels_by_object": {oid: f"C{scaled_aligned[i]+1}" for i, oid in enumerate(ids)},
                        "changed_objects": [oid for i, oid in enumerate(ids) if labels[i] != scaled_aligned[i]]}},
                "uncertain_participants": sorted(uncertain, key=_membership_sort_key),
                "pair_diagnostics": {"object_subsampling": _uncertain_pairs(resampling, labels, ids, names),
                                     "multiple_imputation": _uncertain_pairs(multiple, labels, ids, names)}})
    objects = []
    distances = cdist(scores, scores)
    for i, oid in enumerate(ids):
        near = [int(j) for j in np.argsort(distances[i]) if j != i][:5]
        objects.append({"id": oid, "name": names[oid], "raw_scores": {f"F{j+1}": float(scores[i, j]) for j in range(3)},
                        "display_scores": {f"X{j+1}": float(scores[i, j] * DISPLAY_SIGNS[j]) for j in range(3)},
                        "observed_properties": int(np.isfinite(raw[i]).sum()),
                        "property_measurements": [{"id": sid, "name": variable_names[sid],
                            "status": "observed" if np.isfinite(raw[i, j]) else "unknown",
                            "value": float(raw[i, j]) if np.isfinite(raw[i, j]) else None} for j, sid in enumerate(variables)],
                        "nearest_objects": [{"id": ids[j], "name": names[ids[j]], "distance": float(distances[i, j])} for j in near]})
    result = {"format_version": 1, "method": "Ward hierarchical clustering, Euclidean native posterior factor scores",
        "factor_count": 3, "selected_k": None, "selection_status": "awaiting_substantive_review",
        "n_objects": len(ids), "n_variables": len(variables), "seed": seed,
        "object_order": ids, "trait_order": variables, "matrix_observed": [[float(x) if np.isfinite(x) else None for x in row] for row in raw],
        "matrix_csv_sha256": hashlib.sha256(matrix.to_csv().encode()).hexdigest(), "diagnostics": diagnostics,
        "factor_model": {"loadings_max_abs_difference_from_preview": delta, "variance_total": model["variance_total"],
                         "factors": model["factors"], "fit_diagnostics": fit_diagnostics,
                         "parameters": {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in parameters.items()},
                         "loading_table": model["loading_table"],
                         "projection": "Exact existing transform_new_objects: posterior mean on observed coordinates; NaN features marginalized. Training fit uses its original mean completion."},
        "display_axes": DISPLAY_AXES,
        "score_scaling": {"primary": "No empirical z-rescaling; native rotated latent posterior units",
            "mean": scores.mean(axis=0).tolist(), "standard_deviation_ddof0": standard_deviation.tolist(),
            "minimum": scores.min(axis=0).tolist(), "maximum": scores.max(axis=0).tolist(),
            "covariance_ddof0": np.cov(scores, rowvar=False, ddof=0).tolist(),
            "distinct_profiles_rounded_12": len(np.unique(np.round(scores, 12), axis=0)),
            "sensitivity": "Per-axis empirical standardization is a separate partition comparison; it can upweight a narrow or weak axis. Display sign reflection is exactly distance preserving."},
        "objects": objects, "candidates": candidates,
        "stability_design": {"object_subsampling": {"repeats": subsample_repeats, "fraction": .8,
                "actual_rows_by_k": {str(k): max(k+1, int(.8 * len(ids))) for k in k_values}, "replacement": False,
                "seed_formula": "seed+10000+k", "factor_scores_fixed": True},
            "multiple_imputation": {"draws": imputation_draws, "method": "bounded IterativeImputer(BayesianRidge), sample_posterior=True",
                "seed_formula": "seed+5000+draw", "FA_q": 3,
                "scope": "Each analytical completion refits FA and its scaler. Scores use that completed analytical dataset. Matching loading columns only fixes display orientation and does not change Euclidean distances. Observed source cells are never changed; canonical unknown stays unknown."},
            "uncertain_participant_rule": "Flag for review if conditional same-cluster fraction <.8 in either experiment, silhouette<0 or descriptive centroid margin<.1. These are heuristic review thresholds, not significance or posterior probabilities of a true group."},
        "limitations": ["q3 remains an exploratory working hypothesis; previous loading/imputation and strict-source sensitivity findings still apply.",
            "Subsampling evaluates a fixed score geometry, not source accuracy or full factor-selection uncertainty. The ten imputation realizations are descriptive and do not cure MNAR or validate the corpus.",
            "Ward partitions continuous geometry even without natural social classes; silhouette is not the selection rule. Hierarchical ties can depend on the saved object order.",
            "Groups summarize public-property evidence, not private beliefs, causes, shared electorate, social strata or statistical significance.",
            "Only admitted objects receive coordinates; excluded unknown objects are not assigned zero or a guessed group. No sparse Russia block is imputed."]}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "cluster_candidates_q3"
    data_dir.mkdir(exist_ok=True)
    joblib.dump(parameters, data_dir / "factor_model_q3.joblib")
    (data_dir / "factor_parameters_q3.json").write_text(json.dumps(result["factor_model"]["parameters"], ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pd.DataFrame(loadings, index=variables, columns=["F1", "F2", "F3"]).rename_axis("trait_id").to_csv(data_dir / "loadings_q3.csv", encoding="utf-8-sig")
    matrix.rename_axis("object_id").to_csv(data_dir / "matrix_observed.csv", encoding="utf-8-sig")
    pd.DataFrame(scores, index=ids, columns=["F1", "F2", "F3"]).rename_axis("object_id").to_csv(data_dir / "scores_raw.csv", encoding="utf-8-sig")
    pd.DataFrame(scores * DISPLAY_SIGNS, index=ids, columns=["X1", "X2", "X3"]).rename_axis("object_id").to_csv(data_dir / "scores_display.csv", encoding="utf-8-sig")
    pd.DataFrame(linkage(scores, method="ward", metric="euclidean"), columns=["child_a", "child_b", "distance", "n_members"]).to_csv(data_dir / "ward_linkage.csv", index=False)
    for candidate in candidates:
        pd.DataFrame([{"object_id": oid, "name": names[oid], "cluster_id": candidate["labels_by_object"][oid]} for oid in ids]).to_csv(data_dir / f"labels_k{candidate['k']}.csv", index=False, encoding="utf-8-sig")
    result["artifacts"] = ["cluster_candidates_q3.json", "cluster_candidates_q3.md", "cluster_candidates_q3/factor_model_q3.joblib",
                           "cluster_candidates_q3/scores_raw.csv", "cluster_candidates_q3/scores_display.csv",
                           "cluster_candidates_q3/matrix_observed.csv", "cluster_candidates_q3/ward_linkage.csv"]
    result["artifacts"] += [f"cluster_candidates_q3/labels_k{k}.csv" for k in k_values]
    result["artifacts"] += ["cluster_candidates_q3/factor_parameters_q3.json", "cluster_candidates_q3/loadings_q3.csv"]
    (output_dir / "cluster_candidates_q3.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "cluster_candidates_q3.md").write_text(render_ward_review(result), encoding="utf-8")
    return result


def render_ward_review(result):
    lines = ["# Ward-кандидаты для трёхфакторного решения", "",
        f"Матрица {result['n_objects']} × {result['n_variables']}; используются все три posterior factor scores существующего ядра. Число групп **не выбрано**: нужна содержательная оценка составов, профилей и ограничений.", "",
        "Первичная геометрия — исходные единицы факторных баллов, без эмпирической стандартизации осей. Для карт X1=−F1, X2=F2, X3=F3; это отражение не меняет расстояния или разбиение. Стандартные отклонения F1/F2/F3: " + ", ".join(f"{x:.4f}" for x in result["score_scaling"]["standard_deviation_ddof0"]) + ".", "",
        "| k | Размеры | Silhouette | Subsample median ARI | Imputation median ARI | ARI при z-масштабировании осей |",
        "|---:|---|---:|---:|---:|---:|"]
    for c in result["candidates"]:
        s = c["stability"]
        lines.append(f"| {c['k']} | {c['sizes']} | {c['silhouette']:.3f} | {s['object_subsampling']['median_adjusted_rand']:.3f} | {s['multiple_imputation']['median_adjusted_rand']:.3f} | {s['empirical_axis_standardization']['adjusted_rand']:.3f} |")
    lines += ["", f"Subsampling: {result['stability_design']['object_subsampling']['repeats']} раз по 80% объектов без возвращения, исходные баллы фиксированы. Отдельно {result['stability_design']['multiple_imputation']['draws']} posterior-реализаций заполнения пропусков, с новой FA q3 в каждой. Это условные проверки, не независимая экспертиза и не доверительные интервалы. Оси исходной FA остаются содержательно и измерительно не вполне устойчивыми."]
    names = {x["id"]: x["name"] for x in result["objects"]}
    for c in result["candidates"]:
        lines += ["", f"## Кандидат k={c['k']}", ""]
        for group in c["clusters"]:
            center = "; ".join(f"{key}={value:+.2f}" for key, value in group["display_factor_center"].items())
            lines += [f"### {group['id']}: {group['count']} объектов", "", "Состав: " + "; ".join(x["name"] for x in group["members"]) + ".", "",
                "Центр: " + center + ".", "Представители возле центроида: " + "; ".join(x["name"] for x in group["representatives"][:3]) + ".", "",
                "| Измеренное свойство | Среднее группы | Среднее всей выборки | Известно в группе | Отличие, σ |",
                "|---|---:|---:|---:|---:|"]
            for trait in group["observed_trait_profiles"]:
                mean = f"{trait['observed_mean']:.2f}" if trait["observed_mean"] is not None else "unknown"
                delta = f"{trait['standardized_difference_from_overall']:+.2f}" if trait["standardized_difference_from_overall"] is not None else "—"
                lines.append(f"| {trait['name']} | {mean} | {trait['overall_observed_mean']:.2f} | {trait['observed_n']}/{trait['group_n']} | {delta} |")
            lines += ["", f"Медиана matched Jaccard: subsampling {group['subsample_jaccard']['median']:.3f}; imputation {group['imputation_jaccard']['median']:.3f}."]
        lines += ["", "Участники для проверки границ/устойчивости:", ""]
        for item in c["uncertain_participants"]:
            fractions = [f"{item[key]:.2f}" if item[key] is not None else "не участвовал" for key in ("subsample_same_cluster_fraction", "imputation_same_cluster_fraction")]
            lines.append(f"- {item['name']} ({item['cluster_id']}): сохранение группы subsample={fractions[0]}, imputation={fractions[1]}; silhouette={item['silhouette']:.2f}, centroid margin={item['centroid_margin']:+.2f}.")
        for method, pair_data in c["pair_diagnostics"].items():
            lines += ["", f"Пары с неоднозначным совместным членством ({method}, до пяти):"]
            for pair in pair_data["closest_to_half"][:5]:
                lines.append(f"- {pair['first_name']} / {pair['second_name']}: вместе {pair['co_membership_frequency']:.2f}, совместно наблюдались в {pair['joint_repeats']} повторах.")
    lines += ["", f"Centroid margin — диагностическое расстояние, не правило членства Ward. Отрицательный margin возможен и не вызывает переназначения. Названия групп пока не утверждены. Профили исходных свойств рассчитаны только по observed; неизвестное не подменено нулём или импутацией. Неравномерное покрытие и ограниченный корпус влияют на смысл средних. Все {result['n_objects']} координат, полные pair co-membership матрицы, частоты участников, seed, настройки и маска измерений доступны в JSON/CSV. Более низкая выраженность свойства не означает выраженность его антонима, а близость не доказывает общий электорат или причину общности."]
    return "\n".join(lines) + "\n"
