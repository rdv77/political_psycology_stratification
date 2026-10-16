"""Read frozen q3/Ward results; write a separate, conditional missing-value audit.

No baseline refit, API calls or canonical writes. Integer-grid counts are sensitivity
coverage of a stipulated grid, never probabilities of unobserved political views.
Only the separate100 posterior-imputation sensitivity refits its local FA models;
the first ten reproduce the frozen partitions and loading congruences.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.latent_space_benchmark import fit_model, transform_new_objects
from src.semantic_service.factor_preview import _align_loading_columns
from src.semantic_service.imputation_review import impute_from_training
from src.semantic_service.ward_review import DISPLAY_SIGNS, _align_partition, ward_partition

OBJECT_IDS = ("paul-kagame", "diane-rwigara")
ARRAY_PARAMETERS = ("training_imputation", "training_mean", "training_scale", "standardized_model_mean", "loadings", "noise_variance", "rotation")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score_diagnostics(row, parameters, trait_ids, names):
    """Conditional latent covariance and additive OBSERVED-cell contributions."""
    row = np.asarray(row, dtype=float)
    observed = np.isfinite(row)
    local_l = parameters["loadings"][observed]
    weighted = local_l / parameters["noise_variance"][observed, None]
    system = np.eye(parameters["q"]) + local_l.T @ weighted
    covariance = np.linalg.solve(system, np.eye(parameters["q"]))
    rotation = parameters["rotation"]
    rotated_covariance = rotation.T @ covariance @ rotation
    coefficient = np.linalg.solve(system, weighted.T).T @ rotation
    centered = ((row[observed] - parameters["training_mean"][observed]) /
                parameters["training_scale"][observed] - parameters["standardized_model_mean"][observed])
    contributions = centered[:, None] * coefficient
    score = transform_new_objects(row[None, :], parameters, rotated=True)[0]
    np.testing.assert_allclose(contributions.sum(axis=0), score, rtol=0, atol=1e-12)
    selected = np.flatnonzero(observed)
    return {"known_n": int(observed.sum()), "unknown_n": int((~observed).sum()),
        "raw_scores": {f"F{j+1}": float(score[j]) for j in range(3)},
        "display_scores": {f"X{j+1}": float(score[j] * DISPLAY_SIGNS[j]) for j in range(3)},
        "conditional_covariance_raw_rotated": rotated_covariance.tolist(),
        "conditional_latent_SD_raw_rotated": np.sqrt(np.diag(rotated_covariance)).tolist(),
        "observed_trait_contributions": [{"id": trait_ids[j], "name": names[trait_ids[j]], "observed_value": float(row[j]),
            "standardized_centered_value": float(centered[local]),
            "posterior_coefficients_raw_rotated": coefficient[local].tolist(),
            "additive_contribution_raw_rotated": contributions[local].tolist()}
            for local, j in enumerate(selected)],
        "contribution_scope": "Additive decomposition under a fixed Gaussian posterior. Coefficients depend on the row's observation mask. Not causal effects, newly observed values or independent evidence.",
        "uncertainty_scope": "Conditional latent covariance given the saved loading/noise/scaler parameters and actual observed cells. Does not incorporate parameter, ordinal-measurement, source or missing-mechanism uncertainty; not an inferential CI."}


def pair_stability(summary, first, second):
    """Recount the pair directly from saved records, not marginal retention."""
    together = joint = 0
    for record in summary["records"]:
        ids = record["object_ids"]
        if first not in ids or second not in ids:
            continue
        joint += 1
        labels = record["aligned_cluster_ids"]
        together += labels[ids.index(first)] == labels[ids.index(second)]
    order = summary["pair_object_order"]
    i, j = order.index(first), order.index(second)
    assert summary["pair_jointly_present_count"][i][j] == joint
    frequency = together / joint if joint else None
    assert summary["pair_co_membership_frequency"][i][j] == frequency
    return {"same_cluster_repeats": together, "jointly_present_repeats": joint,
            "total_repeats": len(summary["records"]), "conditional_co_membership_frequency": frequency}


def extended_imputation(raw, parameters, ward, baseline_labels, indices, *, draws=100):
    """Additional conditional sensitivity; first ten reproduce the saved runs."""
    if draws not in (0, 100):
        raise ValueError("Use0 for grid-only or the explicitly requested100 posterior draws")
    if draws==0:
        return {"status":"not_performed","draws":0}
    reference=parameters["loadings"]@parameters["rotation"]
    ids=ward["object_order"]
    saved=next(x for x in ward["candidates"] if x["k"]==4)["stability"]["multiple_imputation"]["records"]
    records=[]
    with threadpool_limits(limits=1):
        for draw in range(draws):
            complete,imputer_fit=impute_from_training(raw,raw,method="bayesian_ridge",posterior=True,bounds=[(0.,5.)]*raw.shape[1],seed=ward["seed"]+5000+draw)
            assert np.array_equal(complete[np.isfinite(raw)],raw[np.isfinite(raw)])
            local,fit=fit_model(complete,3,seed=ward["seed"])
            local_loadings=local["loadings"]@local["rotation"]
            aligned,congruence=_align_loading_columns(reference,local_loadings)
            signed=reference.T@local_loadings/np.maximum(np.linalg.norm(reference,axis=0)[:,None]*np.linalg.norm(local_loadings,axis=0)[None,:],1e-12)
            rows,columns=linear_sum_assignment(-np.abs(signed));transform=np.zeros((3,3))
            for j,k in zip(rows,columns):transform[k,j]=1. if signed[j,k]>=0 else -1.
            np.testing.assert_allclose(local_loadings@transform,aligned,rtol=0,atol=1e-12)
            local_scores=transform_new_objects(complete,local,rotated=True)@transform
            labels=ward_partition(local_scores,4);matched,overlap=_align_partition(baseline_labels,labels,4)
            labels_text=[f"C{x+1}" for x in matched]
            record={"draw":draw,"imputation_seed":ward["seed"]+5000+draw,"FA_seed":ward["seed"],
                "same_cluster":bool(labels[indices[0]]==labels[indices[1]]),
                "aligned_cluster_ids":labels_text,"object_ids":ids,"overlap_with_baseline":overlap.tolist(),
                "adjusted_rand_to_baseline":float(adjusted_rand_score(baseline_labels,labels)),
                "pair_distance":float(np.linalg.norm(local_scores[indices[0]]-local_scores[indices[1]])),
                "pair_raw_scores_aligned":local_scores[indices].tolist(),"loading_congruence":congruence.tolist(),
                "aligned_loadings":aligned.tolist(),"FA_fit_diagnostics":fit,"imputer_fit":imputer_fit}
            if draw<10:
                assert record["aligned_cluster_ids"]==saved[draw]["aligned_cluster_ids"]
                np.testing.assert_allclose(congruence,saved[draw]["loading_congruence_to_reference"],rtol=0,atol=1e-12)
                record["saved_first_ten_reproduced"]=True
            records.append(record)
            if (draw+1)%25==0:
                print(json.dumps({"posterior_imputation_completed":draw+1,"requested":draws,"pair_together":sum(x["same_cluster"] for x in records)}),flush=True)
    first_ten=sum(x["same_cluster"] for x in records[:10]);assert first_ten==6
    together=sum(x["same_cluster"] for x in records)
    return {"status":"complete","draws":draws,"same_cluster_draws":together,"different_cluster_draws":draws-together,
        "conditional_co_membership_frequency":together/draws,"first_ten_same_cluster_draws":first_ten,
        "first_ten_saved_partitions_and_loading_congruences_reproduced":True,
        "FA_convergence_warning_count":sum(x["FA_fit_diagnostics"]["convergence_warning"] for x in records),
        "settings":{"imputer":"IterativeImputer(BayesianRidge)","sample_posterior":True,"bounds":[0,5],"max_iter":20,"tol":.001,
            "imputation_seed_formula":"20261015+5000+draw (draw=0..99)","FA_seed":ward["seed"],"q":3,"Ward_k":4,
            "matrix_fixed_shape":[37,8],"observed_values_preserved":True,"FA_and_scaler_refit_per_completed_matrix":True},
        "loading_congruence_median_by_axis":np.median([x["loading_congruence"] for x in records],axis=0).tolist(),
        "quarter_pair_counts":[{"draw_from":start,"draw_to":start+24,"same_cluster_draws":sum(x["same_cluster"] for x in records[start:start+25]),"total":25} for start in range(0,draws,25)],
        "interpretation":"Additional100 conditional analytical completions, extending the saved10 without choosing k again. Frequencies are sensitivity under this imputation/FA model, not probabilities of political similarity or source truth. Observed cells and canonical files unchanged; source and model misspecification remain outside this experiment.",
        "records":records}


def review(output_dir, *, source_dir, posterior_draws=100):
    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    if output_dir == source_dir or output_dir == source_dir.parent:
        raise ValueError("Use a separate audit directory, never the original output directory")
    sources = {"ward_candidates": source_dir / "cluster_candidates_q3.json",
        "parameters": source_dir / "cluster_candidates_q3/factor_parameters_q3.json",
        "frozen_study": source_dir / "cluster_candidates_q3/source_study.json",
        "frozen_ratings": source_dir / "cluster_candidates_q3/source_ratings.json",
        "frozen_preview": source_dir / "cluster_candidates_q3/source_factor_preview.json",
        "model_core":ROOT/"scripts/latent_space_benchmark.py","imputation_core":ROOT/"src/semantic_service/imputation_review.py",
        "ward_core":ROOT/"src/semantic_service/ward_review.py"}
    hashes = {key: digest(path) for key, path in sources.items()}
    ward = load(sources["ward_candidates"])
    parameters = load(sources["parameters"])
    parameters = {key: np.asarray(value, dtype=float) if key in ARRAY_PARAMETERS else value for key, value in parameters.items()}
    ids, traits = ward["object_order"], ward["trait_order"]
    raw = np.asarray(ward["matrix_observed"], dtype=float)
    assert raw.shape == (37, 8) and parameters["q"] == 3
    assert parameters["trait_order"] == traits and parameters["training_object_order"] == ids
    indices = [ids.index(oid) for oid in OBJECT_IDS]
    names = {obj["id"]: obj["name"] for obj in ward["objects"]}
    trait_names = {item["id"]: item["name"] for item in ward["objects"][0]["property_measurements"]}
    scores = transform_new_objects(raw, parameters, rotated=True)
    expected = np.array([[obj["raw_scores"][f"F{j+1}"] for j in range(3)] for obj in ward["objects"]])
    np.testing.assert_array_equal(scores, expected)
    selected = next(candidate for candidate in ward["candidates"] if candidate["k"] == 4)
    original_labels = np.array([int(selected["labels_by_object"][oid][1:])-1 for oid in ids])
    assert adjusted_rand_score(original_labels, ward_partition(scores, 4)) == 1.
    distances = cdist(scores, scores)
    objects = []
    for index in indices:
        partner = indices[1] if index == indices[0] else indices[0]
        nearest = [int(j) for j in np.argsort(distances[index]) if j != index]
        obj = {"id": ids[index], "name": names[ids[index]],
            "measurements": [{"id": sid, "name": trait_names[sid], "status": "observed" if np.isfinite(raw[index,j]) else "unknown",
                "value": float(raw[index,j]) if np.isfinite(raw[index,j]) else None} for j,sid in enumerate(traits)],
            "baseline_k4_cluster": selected["labels_by_object"][ids[index]],
            "partner_neighbour_rank_of_36": nearest.index(partner)+1,
            "nearest_five": [{"id": ids[j], "name": names[ids[j]], "distance": float(distances[index,j])} for j in nearest[:5]],
            **score_diagnostics(raw[index], parameters, traits, trait_names)}
        objects.append(obj)
    common = np.isfinite(raw[indices[0]]) & np.isfinite(raw[indices[1]])
    shared = [{"id": sid, "name": trait_names[sid], "kagame": float(raw[indices[0],j]), "rwigara": float(raw[indices[1],j]),
        "equal": bool(raw[indices[0],j]==raw[indices[1],j]), "rotated_loadings": (parameters["loadings"]@parameters["rotation"])[j].tolist(),
        "training_observed_mean": float(parameters["training_imputation"][j])} for j,sid in enumerate(traits) if common[j]]
    missing = np.flatnonzero(~np.isfinite(raw[indices[1]]))
    assert len(missing) == 3
    alternatives = []
    for candidate in ward["candidates"]:
        labels = candidate["labels_by_object"]
        containing = next(group for group in candidate["clusters"] if group["id"]==labels[OBJECT_IDS[0]])
        alternatives.append({"k": candidate["k"], "cluster_ids": {oid: labels[oid] for oid in OBJECT_IDS},
            "same_cluster": labels[OBJECT_IDS[0]]==labels[OBJECT_IDS[1]], "containing_group_n": containing["count"],
            "containing_group_members": containing["members"],
            "object_subsampling_pair": pair_stability(candidate["stability"]["object_subsampling"], *OBJECT_IDS),
            "multiple_imputation_pair": pair_stability(candidate["stability"]["multiple_imputation"], *OBJECT_IDS)})

    def altered_partition(new_score, index):
        variant = scores.copy()
        variant[index] = new_score
        candidate = ward_partition(variant, 4)
        aligned, overlap = _align_partition(original_labels, candidate, 4)
        return {"kagame_cluster_aligned": f"C{aligned[indices[0]]+1}", "rwigara_cluster_aligned": f"C{aligned[indices[1]]+1}",
            "same_cluster": bool(candidate[indices[0]]==candidate[indices[1]]),
            "adjusted_rand_to_baseline": float(adjusted_rand_score(original_labels,candidate)),
            "rwigara_group_size": int((candidate==candidate[indices[1]]).sum()),
            "labels_by_object_aligned": {oid: f"C{aligned[i]+1}" for i,oid in enumerate(ids)},
            "overlap_with_baseline": overlap.tolist()}

    filled = np.where(np.isfinite(raw[indices[1]]),raw[indices[1]],parameters["training_imputation"])
    filled_diag = score_diagnostics(filled,parameters,traits,trait_names)
    filled_score = np.array(list(filled_diag["raw_scores"].values()))
    common_row = raw[indices[0]].copy(); common_row[~common] = np.nan
    common_diag = score_diagnostics(common_row,parameters,traits,trait_names)
    common_score = np.array(list(common_diag["raw_scores"].values()))
    remove_continuation = raw[indices[1]].copy()
    continuation_index = traits.index("programmatic-renewal__negative")
    remove_continuation[continuation_index] = np.nan
    without_diag = score_diagnostics(remove_continuation,parameters,traits,trait_names)
    assert without_diag["known_n"]==4
    scenarios = {"rwigara_mean_filled_under_fixed_parameters": {
        "filled_values": {traits[j]: float(filled[j]) for j in missing}, "projection": filled_diag,
        "distance_to_original_kagame": float(np.linalg.norm(filled_score-scores[indices[0]])),
        "ward_sensitivity": altered_partition(filled_score,indices[1]),
        "note": "Analytical substitution only; original three unknown values remain unknown. This is not the actual observed-coordinate posterior."},
        "kagame_on_same_five_observed_properties": {"projection": common_diag,
            "distance_to_original_rwigara": float(np.linalg.norm(common_score-scores[indices[1]])),
            "ward_sensitivity": altered_partition(common_score,indices[0]),
            "note": "Erase Kagame's other three known inputs only in an analytical copy; same full eight-variable FA parameters, no FA fit on five variables."},
        "rwigara_continuation_cell_removed": {"removed_property": traits[continuation_index], "previous_value": float(raw[indices[1],continuation_index]),
            "projection": without_diag, "remaining_known":4,"required_known_in_retained_eight":5,
            "row_admissible":False,"official_cluster_assigned":False,
            "note":"Illustrative fixed-model projection only. Removing this disputed construct leaves4/8 known, below unchanged60% row admission. No new official cluster or map placement may be inferred from this diagnostic."}}

    grid, one_at_a_time = [], []
    with threadpool_limits(limits=1):
        for values in product(range(6), repeat=3):
            row = raw[indices[1]].copy();row[missing] = values
            new_score = transform_new_objects(row[None,:],parameters,rotated=True)[0]
            partition = altered_partition(new_score,indices[1])
            grid.append({"hypothetical_values": dict(zip([traits[j] for j in missing],values)),
                "raw_scores":new_score.tolist(),"display_scores":(new_score*DISPLAY_SIGNS).tolist(),
                "distance_to_original_kagame":float(np.linalg.norm(new_score-scores[indices[0]])),
                **partition})
        for j in missing:
            for value in range(6):
                row = raw[indices[1]].copy();row[j]=value
                new_score=transform_new_objects(row[None,:],parameters,rotated=True)[0]
                one_at_a_time.append({"property_id":traits[j],"property_name":trait_names[traits[j]],"hypothetical_value":value,
                    "other_two_unknown_remain_unknown":True,"raw_scores":new_score.tolist(),
                    **altered_partition(new_score,indices[1])})
    grid_summary = {"grid_size":len(grid),"same_cluster_grid_count":sum(row["same_cluster"] for row in grid),
        "different_cluster_grid_count":sum(not row["same_cluster"] for row in grid),
        "rwigara_aligned_cluster_counts":dict(Counter(row["rwigara_cluster_aligned"] for row in grid)),
        "by_property_value":[{"property_id":traits[j],"property_name":trait_names[traits[j]],"value":value,
            "grid_size":36,"same_cluster_count":sum(row["same_cluster"] for row in grid if row["hypothetical_values"][traits[j]]==value),
            "rwigara_aligned_cluster_counts":dict(Counter(row["rwigara_cluster_aligned"] for row in grid if row["hypothetical_values"][traits[j]]==value))}
            for j in missing for value in range(6)],
        "interpretation":"Exhaustive6^3 integer intensity scenarios0..5, not random draws and not a uniform prior. Counts are scenario counts, not probabilities. Full FA/scaler and all other36 scores fixed; Ward is rerun on37 rows and aligned by maximum overlap. Source known values are fixed, canonical unknown remains unknown."}
    extended=extended_imputation(raw,parameters,ward,original_labels,indices,draws=posterior_draws)
    delta = scores[indices[0]]-scores[indices[1]]
    result={"format_version":1,"audit":"Kagame/Rwigara frozen q3 Ward k4","factor_count":3,"cluster_count":4,
        "n_objects":37,"n_variables":8,"trait_order":traits,"display_axes":ward["display_axes"],
        "objects":objects,"pair_distance":float(distances[indices[0],indices[1]]),
        "pair_squared_distance_axis_contributions":[{"raw_axis":f"F{j+1}","difference":float(delta[j]),"squared":float(delta[j]**2),"share_of_squared_distance":float(delta[j]**2/(delta@delta))} for j in range(3)],
        "common_observed_properties":shared,"equal_common_properties":sum(x["equal"] for x in shared),
        "common_raw_euclidean_distance":float(np.linalg.norm(raw[indices[0],common]-raw[indices[1],common])),
        "alternative_k":alternatives,"projection_scenarios":scenarios,"grid_summary":grid_summary,
        "grid_scenarios":grid,"one_at_a_time_scenarios":one_at_a_time,
        "extended_posterior_imputation":extended,
        "limitations":["Similarity of coded public-property intensities does not establish political alliance, electorate, private beliefs or comparable institutional power.",
            "The same score geometry inherits source availability, ordinal coding and model assumptions. It is not a causal account of why two politicians behave alike.",
            "Subsampling fixes scores. MI10 changes analytical completion and fitted FA; neither is independent source validation. Pair co-membership differs from individual retention and matched Jaccard.",
            "A known intensity1 means limited supported expression under the coding protocol, not explicit opposition to the current course. Construct validity of that code needs separate source audit.",
            "The original factor model was not refitted; only the explicitly separate100 imputation sensitivity realizations refit their analytical FA. No fresh source rating was performed. Conditional covariance excludes source and parameter uncertainty. Grid values are hypothetical and not evidence.",
            "After deleting Rwigara's continuation cell, admission fails; its illustrative projection must not be reported as an official retained object or cluster."],
        "reproduction":{"script":"scripts/audit_rwanda_kagame_rwigara.py","command":"python -B -X utf8 scripts/audit_rwanda_kagame_rwigara.py",
            "source_files":{key:{"path":path.relative_to(ROOT).as_posix(),"sha256":hashes[key]} for key,path in sources.items()},
            "script_sha256":digest(Path(__file__)),"baseline_refitted":False,"sensitivity_FA_fits":posterior_draws,"no_APIs":True,"canonical_writes":False}}
    assert all(digest(path)==hashes[key] for key,path in sources.items())
    output_dir.mkdir(parents=True,exist_ok=True)
    flattened=[]
    for row in grid:
        flattened.append({**row["hypothetical_values"],**{f"F{j+1}":row["raw_scores"][j] for j in range(3)},
            **{key:row[key] for key in ("kagame_cluster_aligned","rwigara_cluster_aligned","same_cluster","adjusted_rand_to_baseline","rwigara_group_size","distance_to_original_kagame")}})
    pd.DataFrame(flattened).to_csv(output_dir/"grid.csv",index=False,encoding="utf-8-sig")
    if extended["draws"]:
        pd.DataFrame([{key:row[key] for key in ("draw","imputation_seed","same_cluster","adjusted_rand_to_baseline","pair_distance")} for row in extended["records"]]).to_csv(output_dir/"posterior_imputation100.csv",index=False,encoding="utf-8-sig")
    (output_dir/"audit.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    (output_dir/"audit.md").write_text(render(result),encoding="utf-8")
    return result


def render(r):
    a,b=r["objects"]
    lines=["# Почему Кагаме и Рвигара сблизились в сохранённой модели", "",
        "Это аудит готовой FA q3 и Ward k4, без новых оценок или изменения канонических данных. Исходная модель не переобучается; отдельная описательная чувствительность дополнительно обучает FA для100 аналитических заполнений. Общая группа C4 содержит23 разнородных профиля; её наличие не означает политический союз.","",
        "| Измеренное свойство | Кагаме | Рвигара |","|---|---:|---:|"]
    for x,y in zip(a["measurements"],b["measurements"]):
        lines.append(f"| {x['name']} | {x['value'] if x['value'] is not None else 'unknown'} | {y['value'] if y['value'] is not None else 'unknown'} |")
    lines += ["",f"Общих измеренных свойств5, из них совпадают3; расстояние исходных известных интенсивностей на этих пяти —{r['common_raw_euclidean_distance']:.5f}. Неизвестные не считаются совпадением с нулём.","",
        f"Евклидово расстояние в3D={r['pair_distance']:.8f}. Рвигара —{a['partner_neighbour_rank_of_36']}-й ближайший сосед Кагаме из36, Кагаме —{b['partner_neighbour_rank_of_36']}-й сосед Рвигары. Ближайший обоим — Франсуа-Ксавье Калинда; пара не является взаимными первыми соседями.","",
        "| Человек | Raw F1/F2/F3 | Display X1/X2/X3 | Conditional latent SD F1/F2/F3 |","|---|---|---|---|"]
    for obj in r["objects"]:
        lines.append("| "+obj["name"]+" | "+" / ".join(f"{v:+.5f}" for v in obj["raw_scores"].values())+" | "+" / ".join(f"{v:+.5f}" for v in obj["display_scores"].values())+" | "+" / ".join(f"{v:.5f}" for v in obj["conditional_latent_SD_raw_rotated"])+" |")
    lines += ["", "## Из каких оценок складываются координаты", "",
        "Баллы — линейное условное среднее при фиксированной FA. Таблица даёт аддитивные численные вклады известных стандартизованных свойств, не причинные эффекты. Коэффициенты проекции зависят от маски наблюдений.","",
        "| Человек / свойство | Вклад F1 | Вклад F2 | Вклад F3 |","|---|---:|---:|---:|"]
    for obj in r["objects"]:
        for trait in obj["observed_trait_contributions"]:
            lines.append(f"| {obj['name']}: {trait['name']} | "+" | ".join(f"{v:+.5f}" for v in trait["additive_contribution_raw_rotated"])+" |")
    lines += ["", "Продолжение курса имеет нагрузку около−0,966 на raw F1. У обоих записана интенсивность1 при среднем2,182, поэтому эта переменная даёт основной вклад в сходный высокий raw F1 (низкий display X1). Оценка1 означает ограниченное зафиксированное проявление, а не доказанное отрицание режима. Качество именно этой смысловой атрибуции требует отдельного источникового аудита.","",
        "## Совместное членство и число групп", "",
        "| k | Группа Кагаме / Рвигары | Размер группы Кагаме | Вместе / совместно присутствовали, subsample | Вместе / MI10 |","|---:|---|---:|---|---|"]
    for row in r["alternative_k"]:
        sub,mi=row["object_subsampling_pair"],row["multiple_imputation_pair"]
        lines.append(f"| {row['k']} | {' / '.join(row['cluster_ids'].values())} | {row['containing_group_n']} | {sub['same_cluster_repeats']}/{sub['jointly_present_repeats']} | {mi['same_cluster_repeats']}/{mi['jointly_present_repeats']} |")
    lines += ["", "Это парные частоты, а не доля индивидуального сохранения метки и не Jaccard. Фиксированная геометрия сохраняет пару при всех срезах k2..6. При изменении допустимых аналитических заполнений и FA они вместе6 из10 раз приk4; устойчивость не безусловна.","",
        "## Что делают три пропуска", "",
        "Сохранённый posterior использует только действительно известные координаты. Подстановка обучающих средних — другая, отдельная проекция. Условная латентная ковариация C=(I+L_OᵀΨ_O^(-1)L_O)^(-1), после вращения RᵀCR. Она рассчитана при фиксированных параметрах и не включает ошибку источников/разметки/оценки модели; это не доверительный интервал."]
    extended=r["extended_posterior_imputation"]
    if extended["draws"]:
        lines += ["", "### Дополнительная проверка100 posterior-реализаций", "",
            f"Пара сохранилась в{extended['same_cluster_draws']} из100 реализаций ({extended['conditional_co_membership_frequency']:.3f}). Первые10 точно воспроизвели сохранённые разбиения, конгруэнтности и совместное членство6/10. Предупреждений сходимости FA: {extended['FA_convergence_warning_count']}. Медианы Tucker по осям: "+" / ".join(f"{x:.4f}" for x in extended["loading_congruence_median_by_axis"])+".",
            "Наблюдаемые клетки фиксированы, в каждой ограниченной регрессионной posterior-реализации заполняются51 пропуск всей37×8 матрицы, заново оцениваются scaler/FA q3 и Ward k4. Seed=20261015+5000+draw, draw0..99; max_iter20, BayesianRidge, диапазон0..5. Это расширение проверки малых10, не повторный подбор k. Частота не является вероятностью политической близости и не исправляет неслучайный механизм пропусков. Полные100 records, нагрузки, согласование, параметры и seed сохранены вJSON."]
    for key in ["rwigara_mean_filled_under_fixed_parameters","kagame_on_same_five_observed_properties"]:
        row=r["projection_scenarios"][key]
        lines += ["", key+": F="+" / ".join(f"{v:+.5f}" for v in row["projection"]["raw_scores"].values())+f"; расстояние до исходного партнёра={next(v for k,v in row.items() if k.startswith('distance_to_')):.5f}; пара вместе при повторном Ward: {row['ward_sensitivity']['same_cluster']}."]
    removed=r["projection_scenarios"]["rwigara_continuation_cell_removed"]
    lines += ["", "Удаление только известной интенсивности продолжения курса у Рвигары оставляет4/8 — ниже требуемых5/8. Официальная группа такому изменённому профилю не назначается. Исключительно иллюстративная проекция под замороженной моделью: F="+" / ".join(f"{v:+.5f}" for v in removed["projection"]["raw_scores"].values())+".","",
        "## Условная сетка трёх неизвестных интенсивностей", "",
        f"Проверены все216 сочетаний0..5 для трёх пропусков Рвигары, остальные её5 значений,36 объектов и FA/scaler фиксированы. Ward k4 пересчитан на каждой аналитической копии. Вместе с Кагаме —{r['grid_summary']['same_cluster_grid_count']} сценариев, раздельно —{r['grid_summary']['different_cluster_grid_count']}. Это не случайная выборка, не равномерное априорное распределение и не вероятность неизвестных взглядов. Никакая ячейка сетки не объявляется новой реальной оценкой.","",
        "| Неизвестное свойство | Условная интенсивность | Вместе, из36 сеточных сочетаний |","|---|---:|---:|"]
    for row in r["grid_summary"]["by_property_value"]:
        lines.append(f"| {row['property_name']} | {row['value']} | {row['same_cluster_count']} |")
    lines += ["", "## По одному условному свойству, остальные два остаются unknown", "",
        "| Свойство | Условная оценка | Кагаме / Рвигара (aligned k4) | Вместе |","|---|---:|---|---|"]
    for row in r["one_at_a_time_scenarios"]:
        lines.append(f"| {row['property_name']} | {row['hypothetical_value']} | {row['kagame_cluster_aligned']} / {row['rwigara_cluster_aligned']} | {row['same_cluster']} |")
    lines += ["", "Метки сеточных вариантов согласованы по максимальному перекрытию с baseline, но группы могли изменить состав. Решающее совместное членство не зависит от этих меток. Подробные координаты, все216 разбиений, SHA источников и команда повторения —audit.json; плоская таблица —grid.csv. Канонические оценки и прежние отчёты не менялись."]
    return "\n".join(lines)+"\n"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir",type=Path,default=ROOT/"studies/rwanda-politicians-2021-2026/output")
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--posterior-draws",type=int,choices=[0,100],default=100)
    args=parser.parse_args()
    result=review(args.output_dir or args.source_dir/"kagame_rwigara_audit",source_dir=args.source_dir,posterior_draws=args.posterior_draws)
    print(json.dumps({"pair_distance":result["pair_distance"],"grid_summary":result["grid_summary"],
        "posterior100":{k:v for k,v in result["extended_posterior_imputation"].items() if k!='records'},
        "removed_continuation_projection":result["projection_scenarios"]["rwigara_continuation_cell_removed"],
        "one_at_a_time":[{k:v for k,v in row.items() if k not in ["labels_by_object_aligned","overlap_with_baseline"]} for row in result["one_at_a_time_scenarios"]]},ensure_ascii=False))


if __name__=="__main__":
    main()
