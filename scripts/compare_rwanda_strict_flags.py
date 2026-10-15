"""Reproduce strict flag-exclusion FA from current or frozen evidence inputs.

No API calls or canonical writes. Use --frozen-inputs and a different output
directory to reproduce an existing receipt without overwriting that audit.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.latent_space_benchmark import fit_model  # noqa: E402
from src.semantic_service.analysis import SEED, _prepare  # noqa: E402
from src.semantic_service.factor_preview import _align_loading_columns, preview_factors  # noqa: E402


def _write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def compare_strict(study, ratings, main, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    main_matrix, _, object_map, scale_map, main_diag = _prepare(study, ratings)
    if (main["diagnostics"]["training_object_order"] != main_matrix.index.tolist() or
        main["diagnostics"]["trait_order"] != main_matrix.columns.tolist()):
        raise ValueError("Reference preview belongs to a different admitted matrix")
    altered_study, altered_ratings = copy.deepcopy(study), copy.deepcopy(ratings)
    removed = []
    for row in altered_ratings:
        for pole, cell in row["pole_scores"].items():
            if cell.get("status") == "observed" and (cell.get("source_audit") or {}).get("decision") == "flag":
                removed.append({"object_id": row["object_id"], "scale_id": row["scale_id"], "pole": pole,
                                "previous_value": cell["value"], "reason": cell["source_audit"].get("reason")})
                cell["sensitivity_exclusion"] = {"rule": "drop_current_source_audit_flags_only_in_copy",
                    "previous_value": cell["value"], "flag_is_not_proven_error": True}
                cell["status"], cell["value"] = "unknown", None
        poles = row["pole_scores"]
        count = sum(x["status"] == "observed" for x in poles.values())
        row.update(positive=poles["positive"].get("value"), negative=poles.get("negative", {}).get("value"),
                   difference=None, mass=None, status="observed" if count == len(poles) else "partial" if count else "unknown")
        if next(s for s in study["scales"] if s["id"] == row["scale_id"]).get("kind") == "unipolar":
            row["value"] = poles["positive"].get("value")
    strict_matrix, _, _, _, strict_diag = _prepare(altered_study, altered_ratings)
    _write(output / "input_study.json", altered_study)
    _write(output / "input_ratings_without_flags.json", altered_ratings)
    strict = preview_factors(altered_study, altered_ratings, output, make_charts=True, bootstrap_repeats=0)
    shared = [sid for sid in main_matrix.columns if sid in strict_matrix.columns]
    comparisons = []
    for sm in strict["models"]:
        q = sm["q"]
        mm = next((x for x in main["models"] if x["q"] == q), None)
        if mm is None:
            continue
        full_main = np.array([[row["loadings"][f"F{j+1}"] for j in range(q)] for row in mm["loading_table"]])
        reference_fit, _ = fit_model(main_matrix.to_numpy(dtype=float), q, seed=SEED)
        if not np.allclose(full_main, reference_fit["loadings"] @ reference_fit["rotation"], rtol=0, atol=1e-10):
            raise ValueError("Reference preview loadings do not reproduce the supplied measurements")
        main_rows = {x["id"]: x for x in mm["loading_table"]}
        strict_rows = {x["id"]: x for x in sm["loading_table"]}
        reference = np.array([[main_rows[sid]["loadings"][f"F{j+1}"] for j in range(q)] for sid in shared])
        candidate = np.array([[strict_rows[sid]["loadings"][f"F{j+1}"] for j in range(q)] for sid in shared])
        aligned, congruence = _align_loading_columns(reference, candidate)
        denominator = np.linalg.norm(reference, axis=0)[:, None] * np.linalg.norm(candidate, axis=0)[None, :]
        signed = reference.T @ candidate / np.maximum(denominator, 1e-12)
        rows, columns = linear_sum_assignment(-np.abs(signed))
        matches = []
        for j, k in zip(rows, columns):
            matches.append({"main_factor_id": f"F{j+1}", "strict_original_factor_id": f"F{k+1}",
                "strict_sign_multiplier": 1 if signed[j, k] >= 0 else -1,
                "signed_congruence_before_alignment": float(signed[j, k]),
                "absolute_congruence_after_alignment": float(congruence[j]),
                "main_loading_energy_retained_in_shared_variables": float(np.sum(reference[:, j]**2) / np.sum(full_main[:, j]**2)),
                "main_factor_name": mm["factors"][j]["name"], "strict_original_factor_name": sm["factors"][k]["name"],
                "main_full_factor_variance_share": mm["factors"][j]["variance_share"],
                "strict_matched_factor_variance_share": sm["factors"][k]["variance_share"],
                "main_full_factor_variance_share_denominator_8": mm["factors"][j]["variance_share"],
                "strict_matched_factor_variance_share_denominator_7": sm["factors"][k]["variance_share"]})
        profiles = [{"id": sid, "name": scale_map[sid]["name"],
            "main_loadings": {f"F{j+1}": float(reference[i, j]) for j in range(q)},
            "strict_aligned_loadings": {f"F{j+1}": float(aligned[i, j]) for j in range(q)}} for i, sid in enumerate(shared)]
        comparisons.append({"q": q, "main_total_variance_share": mm["variance_total"],
            "strict_total_variance_share": sm["variance_total"], "main_variance_denominator": mm["variance_denominator"],
            "strict_variance_denominator": sm["variance_denominator"], "main_dimension_fit": mm["dimension_fit"],
            "strict_dimension_fit": sm["dimension_fit"], "axis_alignment": matches, "shared_loading_profiles": profiles,
            "signed_pairwise_tucker_matrix": signed.tolist()})
        pd.DataFrame([x["loadings"] for x in sm["loading_table"]], index=[x["id"] for x in sm["loading_table"]]).rename_axis("trait").to_csv(output / f"loadings_q{q}.csv", encoding="utf-8-sig")
    describe = lambda ids: [{"id": sid, "name": scale_map[sid]["name"]} for sid in ids]
    objects = lambda ids: [{"id": oid, "name": object_map[oid].get("name", oid)} for oid in ids]
    matrix_info = lambda matrix, diag: {"n_objects": len(matrix), "n_variables": matrix.shape[1],
        "retained_coverage": diag["retained_coverage"], "object_ids": matrix.index.tolist(),
        "variable_ids": matrix.columns.tolist(), "matrix_csv_sha256": hashlib.sha256(matrix.to_csv().encode()).hexdigest()}
    result = {"format_version": 1, "mode": "strict_flag_exclusion_factor_sensitivity", "at": datetime.now(timezone.utc).isoformat(),
        "exclusion_rule": "All currently observed source_audit.decision=flag cells become unknown only in a copy; flags are not automatically false measurements. No threshold changes.",
        "removed_count": len(removed), "removed_properties": removed, "main": matrix_info(main_matrix, main_diag),
        "strict": {**matrix_info(strict_matrix, strict_diag), "dimension_selection": strict["dimension_selection"], "dimension_candidates": strict["dimension_candidates"]},
        "common_variable_ids": shared,
        "newly_excluded_objects": objects([oid for oid in main_matrix.index if oid not in strict_matrix.index]),
        "newly_included_objects": objects([oid for oid in strict_matrix.index if oid not in main_matrix.index]),
        "excluded_main_variables": describe([sid for sid in main_matrix.columns if sid not in strict_matrix.columns]),
        "comparisons": comparisons, "canonical_source_files_unchanged": True,
        "weak_factor_warning": {"all_candidates_fail_two_strong_traits_rule": all(not x["all_factors_have_two_strong_traits"] for x in strict["dimension_candidates"]),
            "strong_traits_per_factor_by_q": {str(x["q"]): x["strong_traits_per_factor"] for x in strict["dimension_candidates"]},
            "weak_indicator_axes": [{"q": model["q"], "factor_id": factor["id"],
                "strong_indicators": factor["positive_traits"] + factor["negative_traits"]}
                for model in strict["models"] for factor in model["factors"]
                if len(factor["positive_traits"]) + len(factor["negative_traits"]) < 2],
            "note": "A necessary mathematical admission is not measurement validation. If all candidates fail the two-strong-trait rule, the reported q uses the existing fallback pool; a single-indicator axis is weak as a latent construct."},
        "limitations": ["Different objects, observed cells and feature sets; sensitivity comparison, not a repeated test on an identical matrix.",
            "Tucker matching uses shared loading coordinates, optimal permutation and sign only; no arbitrary continuous rotation.",
            "The original main axis can lose its defining indicator during feature selection. Its retained loading-energy fraction is reported; low congruence is not a pure test of the same fully measured construct.",
            "Total variance denominators differ; percentages and raw held-out likelihoods across feature spaces are not direct fit-improvement tests.",
            "Nominal df is necessary, not sufficient identification. No API, fresh source validation or additional resampling.",
            "Russia indicators remain insufficiently measured; no claim of presence or absence of a Russia factor."]}
    return result


def render(result):
    main, strict = result["main"], result["strict"]
    lines = ["# Строгая FA без помеченных оценок", "",
        f"В копии исключены {result['removed_count']} текущих observed с source_audit.decision=flag. Флаг не означает доказанную ошибку; пороги, алгоритм и seed сохранены. Канонические данные не изменены.", "",
        f"Основная матрица {main['n_objects']}×{main['n_variables']} (покрытие {100*main['retained_coverage']:.2f}%), строгая {strict['n_objects']}×{strict['n_variables']} ({100*strict['retained_coverage']:.2f}%). Дополнительно исключены: {', '.join(x['name'] for x in result['newly_excluded_objects'])}.",
        "Удалённые признаки: " + ", ".join(x["name"] for x in result["excluded_main_variables"]) + ".", "",
        "| q | Основная доля, % | Строгая доля, % | Строгие df | Строгий held-out среднее (SD) | Сильных признаков на ось |",
        "|---:|---:|---:|---:|---:|---|"]
    for row in result["comparisons"]:
        fit = row["strict_dimension_fit"]
        lines.append(f"| {row['q']} | {100*row['main_total_variance_share']:.2f} | {100*row['strict_total_variance_share']:.2f} | {fit['nominal_df']} | {fit['heldout_log_likelihood_mean']:.3f} ({fit['heldout_log_likelihood_sd']:.3f}) | {fit['strong_traits_per_factor']} |")
    lines += ["", f"В основной доле знаменатель — все {main['n_variables']} стандартизованных переменных; в строгой — все {strict['n_variables']}. Изменились переменные, объекты и оценки: эти проценты и held-out плотности разных размерностей не являются тестом улучшения/ухудшения качества. SD перекрывающихся разделений — описательный разброс, не доверительный интервал.", ""]
    q4 = next((entry for entry in strict["dimension_selection"]["parameter_count_constraint"]["excluded_candidates"] if entry["q"] == 4), None)
    if q4:
        moments = strict["n_variables"] * (strict["n_variables"] + 1) // 2
        lines += [f"При p={strict['n_variables']} q4 не обучалась: {q4['parameter_count']} ковариационных параметров на {moments} моментов, номинальные df={q4['nominal_df']}. Неотрицательные df допустимых решений — необходимое, не достаточное условие идентификации.", ""]
    for weak in result["weak_factor_warning"]["weak_indicator_axes"]:
        descriptors = "; ".join(f"{trait['name']} ({trait['loading']:+.3f})" for trait in weak["strong_indicators"]) or "нет"
        lines += [f"Слабая измерительная опора q{weak['q']}/{weak['factor_id']}: сильных индикаторов менее двух — {descriptors}.", ""]
    if result["weak_factor_warning"]["all_candidates_fail_two_strong_traits_rule"]:
        lines += [f"Ни один кандидат не проходит правило двух сильных признаков на каждую ось; рекомендация q{strict['dimension_selection']['recommended_q']} получена через существующий fallback, а не через успешную проверку этого правила.", ""]
    lines += [
        f"Сопоставление по {len(result['common_variable_ids'])} общим признакам использует оптимальное соответствие осей и выравнивание знаков. Непрерывного вращения для улучшения совпадения нет. Номера/знаки локальны каждому решению.", "",
        "| q | Основная ось | Строгая исходная ось | Знак строгой при сравнении | Tucker | Сохранённая энергия нагрузок основной оси, % |",
        "|---:|---|---|---:|---:|---:|"]
    for row in result["comparisons"]:
        for match in row["axis_alignment"]:
            lines.append(f"| {row['q']} | {match['main_factor_id']} | {match['strict_original_factor_id']} | {match['strict_sign_multiplier']:+d} | {match['absolute_congruence_after_alignment']:.4f} | {100*match['main_loading_energy_retained_in_shared_variables']:.1f} |")
    lines += ["", "Удалённый признак мог быть основным якорем исходной оси; потеря её энергии показана в таблице. Поэтому низкое совпадение соответствующей оси нельзя представить как чистую проверку того же полностью измеренного конструкта. Перестройка части профилей ограничивает вывод об устойчивой полной структуре; допустимость FA и высокая общая доля дисперсии сами по себе не подтверждают латентные смыслы.", "",
        "Обычные смыслы и полные нагрузки строгих решений находятся в factor_preview.md/.json и loadings_q2.csv/loadings_q3.csv. Полные сопоставленные профили, исключения, receipts и команда воспроизведения — в COMPARISON.json. Все российские переменные исключены по недостаточному измерению; наличие или отсутствие российского фактора не установлено."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=ROOT / "studies/rwanda-politicians-2021-2026")
    parser.add_argument("--frozen-inputs", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = (args.output_dir or args.study_dir / "output/strict_without_flags").resolve()
    paths = ([args.frozen_inputs / name for name in ("source_study.json", "source_ratings.json", "source_main_preview.json")]
             if args.frozen_inputs else [args.study_dir / "study.json", args.study_dir / "ratings.json", args.study_dir / "output/factor_preview.json"])
    blobs = [path.read_bytes() for path in paths]
    output.mkdir(parents=True, exist_ok=True)
    frozen = {}
    for name, blob in zip(("source_study.json", "source_ratings.json", "source_main_preview.json"), blobs):
        target = output / name
        if target.exists() and target.read_bytes() != blob:
            raise ValueError("Frozen source inputs differ: use another output directory")
        target.write_bytes(blob)
        frozen[name] = hashlib.sha256(blob).hexdigest()
    study, ratings, reference = [json.loads(blob.decode("utf-8-sig")) for blob in blobs]
    result = compare_strict(study, ratings, reference, output)
    result["source_file_sha256"] = {str(path): hashlib.sha256(blob).hexdigest() for path, blob in zip(paths, blobs)}
    result["copy_input_sha256"] = {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in ("input_study.json", "input_ratings_without_flags.json")}
    result["reproduction"] = {"script": "scripts/compare_rwanda_strict_flags.py", "frozen_input_sha256": frozen,
        "command": 'python -B scripts/compare_rwanda_strict_flags.py --frozen-inputs "studies/rwanda-politicians-2021-2026/output/strict_without_flags" --output-dir ".service_runtime/reproduce_strict_flags"',
        "no_APIs": True, "no_new_methods_or_resampling": True}
    for path, blob in zip(paths, blobs):
        if path.read_bytes() != blob:
            raise ValueError("Input changed concurrently; rerun after the snapshot is stable")
    _write(output / "COMPARISON.json", result)
    (output / "COMPARISON.md").write_text(render(result), encoding="utf-8")
    print(json.dumps({"output": str(output), "n_objects": result["strict"]["n_objects"], "n_variables": result["strict"]["n_variables"],
                      "q": [row["q"] for row in result["comparisons"]], "source_files_unchanged": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
