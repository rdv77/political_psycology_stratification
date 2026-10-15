"""Offline comparison of all admissible FA dimensions, without clustering/API.

Russia-related flags are numeric candidates only. The presence of three strong
indicators does not verify evidence independence, construct validity or causal
meaning. Every threshold-crossing candidate remains visible even when the
existing numerical parsimony rule recommends fewer factors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.semantic_service.analysis import (
    InsufficientMeasurementError, LOADING_THRESHOLD, SEED,
    _analysis_scales, _dimension_analysis, _factor_descriptions, _measurement_model, _prepare, _trait_properties,
)
from scripts.latent_space_benchmark import fit_model


MINIMUM_SHARE = .02
MINIMUM_FAMILIES = 3


def _metadata_field(item, field):
    nested = item.get("metadata")
    return item.get(field) or (nested.get(field) if isinstance(nested, dict) else None)


def _theme_configuration(study):
    requirements = study.get("research_requirements") or {}
    if not isinstance(requirements, dict):
        raise ValueError("research_requirements must be an object")
    # These preregistered gates cannot silently be loosened by a provider.
    expected = {"loading_threshold": LOADING_THRESHOLD,
                "minimum_explained_share": MINIMUM_SHARE, "minimum_indicators": MINIMUM_FAMILIES}
    for key, value in expected.items():
        if key in requirements:
            given = requirements[key]
            if isinstance(given, bool) or not isinstance(given, (int, float)) or not np.isclose(given, value, rtol=0, atol=1e-12):
                raise ValueError(f"The factor-preview protocol fixes {key}={value}")
    required = requirements.get("required_scales", [])
    if not isinstance(required, list):
        raise ValueError("research_requirements.required_scales must be an array")
    requested, duplicates = {}, []
    for item in required:
        if isinstance(item, str):
            identity, definition = item, {"id": item}
        elif isinstance(item, dict) and isinstance(item.get("id"), str):
            identity, definition = item["id"], item
        else:
            raise ValueError("required_scales items must be IDs or scale objects with a string id")
        if not identity.strip():
            raise ValueError("A required scale ID must not be empty")
        if identity in requested:
            duplicates.append(identity)
        requested[identity] = definition
    original = {str(item["id"]): item for item in study.get("scales", [])}
    actual = {str(item["id"]): item for item in _analysis_scales(study)}
    for identity, item in original.items():
        if _metadata_field(item, "research_domain") == "russia_relations":
            requested.setdefault(identity, item)
    requested_original = requested
    requested = {}
    for identity, definition in requested_original.items():
        variables = [item for item in actual.values()
                     if str(item["id"]) == identity or str(item.get("parent_scale_id", "")) == identity]
        if variables:
            for variable in variables:
                requested[str(variable["id"])] = {**definition, **variable}
        else:
            requested[identity] = definition
    families, sources = {}, {}
    for identity, definition in requested.items():
        scale = actual.get(identity, {})
        family = _metadata_field(scale, "semantic_family_id") or _metadata_field(definition, "semantic_family_id")
        parent = scale.get("parent_scale_id")
        families[identity] = str(family or parent or identity)
        sources[identity] = ("declared_family" if family else "parent_scale_id_fallback_not_independence_verified" if parent
                             else "scale_id_fallback_not_independence_verified")
    instructions = requirements.get("instructions", [])
    if not isinstance(instructions, list) or any(not isinstance(item, str) for item in instructions):
        raise ValueError("research_requirements.instructions must be an array of strings")
    mismatches = []
    for identity, definition in requested_original.items():
        if identity in original:
            fields = [field for field in ["positive", "negative", "definition"]
                      if field in definition and definition[field] != original[identity].get(field)]
            if fields:
                mismatches.append({"id": identity, "different_fields": fields})
    return {"configured_ids": list(requested), "configured_parent_ids": list(requested_original),
            "parent_by_id": {identity: actual.get(identity, {}).get("parent_scale_id", identity) for identity in requested},
            "family_by_id": families,
            "family_assignment_source": sources, "duplicate_required_ids": sorted(set(duplicates)),
            "required_definition_mismatches": mismatches, "instructions": instructions,
            "loading_threshold": LOADING_THRESHOLD, "minimum_explained_share": MINIMUM_SHARE,
            "minimum_indicators": MINIMUM_FAMILIES, "requested_definitions": requested}


def _russia_factor_review(loadings, scales, denominator, configuration):
    """Audit signed loadings with all-scale variance denominator and family dedup."""
    loadings = np.asarray(loadings, dtype=float)
    if loadings.ndim != 2 or loadings.shape[0] != len(scales) or not np.isfinite(loadings).all():
        raise ValueError("Finite ordered scale-by-factor loadings required")
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("A positive all-scale variance denominator is required")
    themed = set(configuration["configured_ids"])
    families = configuration["family_by_id"]
    rows = []
    for column in range(loadings.shape[1]):
        vector = loadings[:, column]
        energy = float(np.sum(vector ** 2))
        share = energy / denominator
        strong_theme, strong_other, all_theme = [], [], []
        for index in np.argsort(-np.abs(vector)):
            scale = scales[index]
            identity, value = str(scale["id"]), float(vector[index])
            high, low = _trait_properties(scale, value)
            item = {"id": identity, "name": scale.get("name", identity), "loading": value,
                    "positive": scale.get("positive", ""), "negative": scale.get("negative", ""),
                    "high_factor_property": high, "low_factor_property": low,
                    "parent_scale_id": scale.get("parent_scale_id", identity), "pole": scale.get("pole"),
                    "kind": scale.get("kind", "bipolar")}
            if identity in themed:
                item["semantic_family_id"] = families[identity]
                all_theme.append(item)
            if abs(value) >= LOADING_THRESHOLD:
                (strong_theme if identity in themed else strong_other).append(item)
        supported_families = sorted({item["semantic_family_id"] for item in strong_theme})
        count = len(supported_families)
        passes_share = share >= MINIMUM_SHARE
        if not themed:
            status = "not_configured"
        elif passes_share and count >= MINIMUM_FAMILIES:
            status = "numeric_candidate"
        elif passes_share and count == 2:
            status = "two_indicator_candidate"
        elif passes_share and count == 1:
            status = "single_indicator_only"
        elif count >= 2:
            status = "below_reporting_threshold"
        else:
            status = "no_numeric_theme_support"
        theme_energy = float(sum(item["loading"] ** 2 for item in all_theme))
        signs = {1 if item["loading"] > 0 else -1 for item in strong_theme}
        rows.append({"factor_id": f"F{column+1}", "variance_share": float(share),
                     "russia_variance_contribution": theme_energy / denominator,
                     "russia_loading_energy_fraction": theme_energy / energy if energy > 0 else 0.,
                     "strong_russia_indicators": strong_theme,
                     "all_russia_loadings": all_theme,
                     "strong_non_russia_indicators": strong_other,
                     "strong_indicator_count": len(strong_theme),
                     "distinct_family_count": count, "supporting_families": supported_families,
                     "status": status, "meets_variance_threshold": passes_share,
                     "mandatory_report": bool(passes_share and count >= 2),
                     "polarity": "mixed_signs" if len(signs) > 1 else "same_sign" if signs else "not_defined",
                     "scope_hint": "mixed_with_other_strong_scales" if strong_other else "no_other_strong_scales",
                     "evidence_audit": "pending", "semantic_specificity_audit": "pending",
                     "factor_resampling": "not_performed", "cross_q_rotation_review": "pending",
                     "claim": "Numeric screening only; indicator independence and a distinct Russia-specific construct are not verified"})
    return rows


def _align_loading_columns(reference, candidate):
    """Match varimax axes by absolute Tucker congruence, then align signs."""
    reference, candidate = np.asarray(reference), np.asarray(candidate)
    if reference.shape != candidate.shape or reference.ndim != 2 or not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("Equal finite loading matrices required for alignment")
    denominator = np.linalg.norm(reference, axis=0)[:, None] * np.linalg.norm(candidate, axis=0)[None, :]
    congruence = np.clip(reference.T @ candidate / np.maximum(denominator, 1e-12), -1., 1.)
    rows, columns = linear_sum_assignment(-np.abs(congruence))
    permutation = np.empty(reference.shape[1], dtype=int)
    permutation[rows] = columns
    matched = congruence[np.arange(reference.shape[1]), permutation]
    aligned = candidate[:, permutation] * np.where(matched < 0, -1., 1.)
    return aligned, np.abs(matched)


def _bootstrap_russia_candidates(raw, scales, configuration, models, fitted, repeats):
    """Conditional row bootstrap, retaining the identical variables and fixed q."""
    if isinstance(repeats, bool) or not isinstance(repeats, int) or not 0 <= repeats <= 500:
        raise ValueError("bootstrap_repeats must be an integer from 0 to 500")
    variable_positions = {str(scale["id"]): i for i, scale in enumerate(scales)}
    for model in models:
        targets = [item for item in model["russia_candidates"] if item["mandatory_report"]]
        if not targets:
            continue
        if not repeats:
            for target in targets:
                target["bootstrap_stability"] = {"status": "not_performed", "requested_repeats": 0}
            continue
        q = model["q"]
        reference = fitted[q]["loadings"] @ fitted[q]["rotation"]
        rng = np.random.default_rng(SEED + 1000 + q)
        records = {item["factor_id"]: [] for item in targets}
        failed, warned = [], 0
        for repeat in range(repeats):
            sample = raw[rng.integers(0, len(raw), size=len(raw))]
            try:
                if np.isnan(sample).all(axis=0).any() or np.any(np.nanstd(sample, axis=0) <= 1e-10):
                    raise ValueError("A fixed variable is unobserved or constant in this resample")
                parameters, fitting = fit_model(sample, q, seed=SEED + repeat)
                rotated = parameters["loadings"] @ parameters["rotation"]
                aligned, congruence = _align_loading_columns(reference, rotated)
                standardized = (np.where(np.isfinite(sample), sample, parameters["training_imputation"]) -
                                parameters["training_mean"]) / parameters["training_scale"]
                denominator = float(np.var(standardized, axis=0).sum())
                screened = _russia_factor_review(aligned, scales, denominator, configuration)
                warned += int(fitting["convergence_warning"])
                for target in targets:
                    column = int(target["factor_id"][1:]) - 1
                    records[target["factor_id"]].append({"congruence": float(congruence[column]),
                        "variance_share": screened[column]["variance_share"], "status": screened[column]["status"],
                        "loadings": aligned[:, column].copy()})
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                failed.append({"repeat": repeat, "reason": str(error)})
        for target in targets:
            values = records[target["factor_id"]]
            successful = len(values)
            congruences = [item["congruence"] for item in values]
            shares = [item["variance_share"] for item in values]
            indicators = []
            for indicator in target["strong_russia_indicators"]:
                vector = np.array([item["loadings"][variable_positions[indicator["id"]]] for item in values])
                indicators.append({"id": indicator["id"], "parent_scale_id": indicator["parent_scale_id"],
                                   "same_sign_fraction": float(np.mean(np.sign(vector) == np.sign(indicator["loading"]))) if successful else None,
                                   "abs_loading_ge_035_fraction": float(np.mean(np.abs(vector) >= LOADING_THRESHOLD)) if successful else None})
            numeric = sum(item["status"] == "numeric_candidate" for item in values)
            conditional = sum(item["status"] in {"numeric_candidate", "two_indicator_candidate"} for item in values)
            target["factor_resampling"] = "conditional_row_bootstrap"
            target["bootstrap_stability"] = {
                "status": "completed" if successful else "no_successful_replicates", "requested_repeats": repeats,
                "successful_repeats": successful, "failed_repeats": len(failed), "convergence_warning_repeats": warned,
                "seed": SEED + 1000 + q, "median_congruence": float(np.median(congruences)) if successful else None,
                "p10_congruence": float(np.quantile(congruences, .1)) if successful else None,
                "fraction_congruence_ge_085": float(np.mean(np.array(congruences) >= .85)) if successful else None,
                "numeric_candidate_fraction_successful": numeric / successful if successful else None,
                "numeric_candidate_fraction_all_replicates": numeric / repeats,
                "at_least_two_families_fraction_successful": conditional / successful if successful else None,
                "variance_share_descriptive_percentiles": dict(zip(["p10", "median", "p90"],
                    np.quantile(shares, [.1, .5, .9]).tolist())) if successful else None,
                "initial_strong_indicators": indicators, "failures": failed,
                "design": "n rows sampled with replacement; fixed retained variables and q; full FA and varimax refit; Hungarian absolute Tucker congruence matching and sign alignment",
                "limitations": "Conditional descriptive sensitivity, not inferential confidence intervals, evidence validation, or whole-pipeline stability. Thirty replicates are a screening diagnostic."}


def _draw_loadings(path, model, scales, themed_ids):
    values = np.array([[row["loadings"][f"F{j+1}"] for j in range(model["q"])] for row in model["loading_table"]])
    height = max(5., .38 * len(scales) + 2.)
    fig, ax = plt.subplots(figsize=(max(10., 7 + .75 * model["q"]), height), constrained_layout=True)
    image = ax.imshow(values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    labels = [textwrap.fill(str(scale.get("name", scale["id"])), width=44) for scale in scales]
    ax.set(yticks=range(len(scales)), yticklabels=labels,
           xticks=range(model["q"]), xticklabels=[f"{factor['id']}\n{100*factor['variance_share']:.1f}%" for factor in model["factors"]],
           title=f"q={model['q']}: полные знаковые нагрузки после varimax", xlabel="Факторы и описательные доли общей дисперсии")
    ax.tick_params(axis="y", labelsize=9)
    for label, scale in zip(ax.get_yticklabels(), scales):
        if str(scale["id"]) in themed_ids:
            label.set_color("#0F766E"); label.set_fontweight("bold")
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if abs(value) > .63 else "black")
    fig.colorbar(image, ax=ax, shrink=.8, label="Нагрузка; зелёным выделены исследуемые российские шкалы")
    fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)


def _md_cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def _factor_meaning_rows(model):
    """Summarize both directions from approved properties, never antonyms."""
    rows = []
    for factor in model["factors"]:
        strong = sorted([*factor["positive_traits"], *factor["negative_traits"]],
                        key=lambda item: -abs(item["loading"]))
        poles = []
        for key in ["high_factor_property", "low_factor_property"]:
            values = []
            for item in strong:
                value = item.get(key, "")
                if value and value not in values:
                    values.append(value)
                if len(values) == 2:
                    break
            poles.append("; ".join(values) or ("Нет сильных нагрузок ≥0.35" if not strong else "Полюс не задан в словаре"))
        ordered = sorted(model["loading_table"], key=lambda item: -abs(item["loadings"][factor["id"]]))[:3]
        keys = "; ".join(f"{item['name']} {item['loadings'][factor['id']]:+.2f}" for item in ordered)
        identity = factor["id"] + (" *" if len(strong) < 2 else "")
        rows.append([identity, poles[0], poles[1], f"{100*factor['variance_share']:.2f}%", keys])
    return rows


def _render_coverage_audit(result):
    diagnostics = result["diagnostics"]
    percent = lambda value: f"{value:.1%}" if value is not None else "не определено для пустой матрицы"
    lines = ["# Короткий аудит покрытия: данных для факторного анализа недостаточно", "",
             "FA не запускался. " + result["measurement_failure"], "",
             f"Входная выборка: {diagnostics['input_objects']} объектов × {diagnostics['input_scales']} шкал. "
             f"Наблюдаемых аналитических значений: {diagnostics['observed_input_ratings']} из {diagnostics['total_input_cells']} "
             f"({percent(diagnostics['input_coverage'])}). Неизвестных ячеек: {diagnostics['unknown_input_cells']}.", "",
             f"После фиксированного отбора: {result['n_objects']} объектов × {result['n_scales']} шкал; "
             f"покрытие — {percent(diagnostics['retained_coverage'])}.", "",
             diagnostics["preprocessing_report"], "",
             ("Каждый известный полюс учитывался самостоятельно; неизвестный второй полюс его не скрывал. "
              "Меньшая выраженность свойства не означает выраженность его антонима. "
              if diagnostics.get("measurement_model") == "independent_poles" else
              "Если один полюс известен, а другой неизвестен, точная разность не установлена. ") +
             "Неизвестное не означает нейтральность или невыраженность свойства. Пороги ради получения модели не понижались.", "",
             "## Наблюдения по всем входным шкалам", "",
             "Числа относятся к полной входной выборке до отбора, а не только к оставшимся объектам.", "",
             "| ID | Шкала | Наблюдений | Всего объектов | Покрытие | Осталась |", "|---|---|---|---|---|---|"]
    for item in diagnostics["input_scale_coverage"]:
        row = [item["id"], item["name"], item["observed_objects"], item["total_objects"],
               percent(item["coverage"]), "да" if item["retained"] else "нет"]
        lines.append("| " + " | ".join(_md_cell(value) for value in row) + " |")
    lines += ["", "## Наблюдения по всем входным объектам", "",
              "| ID | Объект | Наблюдаемых шкал | Всего шкал | Покрытие | Остался |", "|---|---|---|---|---|---|"]
    for item in diagnostics["input_object_coverage"]:
        row = [item["id"], item["name"], item["observed_scales"], item["total_scales"],
               percent(item["coverage"]), "да" if item["retained"] else "нет"]
        lines.append("| " + " | ".join(_md_cell(value) for value in row) + " |")
    lines += ["", "## Тема России и пределы выводов", "",
              "Статус: `insufficient_measurement`. Ни наличие, ни отсутствие отдельного российского фактора "
              "по этой матрице установить нельзя. Нагрузки и доли дисперсии не оценивались; они не равны нулю.", "",
              "Дальнейший шаг — собрать дополнительные датированные свидетельства по недостающим свойствам "
              "и повторить проверку покрытия. Подробные причины исключения и все итерации отбора сохранены в JSON."]
    return "\n".join(lines).strip() + "\n"


def _insufficient_preview(study, ratings, configuration, error):
    diagnostics = error.diagnostics
    configured = configuration["configured_ids"]
    retained = set(diagnostics["retained_scale_ids"])
    scale_coverage = {item["id"]: item for item in diagnostics["input_scale_coverage"]}
    reasons = {item["id"]: item["reason"] for item in diagnostics["excluded_scales"]}
    configured_families = {configuration["family_by_id"][identity] for identity in configured}
    retained_families = {configuration["family_by_id"][identity] for identity in configured if identity in retained}
    coverage = []
    for identity in configured:
        item = scale_coverage.get(identity, {})
        coverage.append({"id": identity, "name": item.get("name", configuration["requested_definitions"][identity].get("name", identity)),
                         "semantic_family_id": configuration["family_by_id"][identity],
                         "retained": identity in retained,
                         "observed_count": item.get("observed_retained_objects", 0),
                         "observed_input_count": item.get("observed_objects", 0),
                         "exclusion_reason": None if identity in retained else reasons.get(identity, "not_in_approved_dictionary")})
    return {"format_version": 1, "mode": "factor_preview_only", "measurement_status": "insufficient_data",
            "measurement_failure": str(error), "n_objects": diagnostics["retained_objects"],
            "n_scales": diagnostics["retained_scales"], "recommended_q": None,
            "measurement_context": {"measurement_mode": study.get("measurement_mode", "unspecified"),
                                    "source_supplement": study.get("source_supplement"),
                                    "approvals": study.get("approvals", {}),
                                    "expert_validation": "not_independently_validated"},
            "dimension_selection": {"recommended_q": None, "selection_is_exploratory": True,
                                    "status": "not_started_insufficient_measurement"},
            "dimension_candidates": [], "models": [],
            "russia_review": {**configuration, "retained_ids": [identity for identity in configured if identity in retained],
                              "excluded_ids": [identity for identity in configured if identity not in retained],
                              "configured_family_count": len(configured_families), "retained_family_count": len(retained_families),
                              "measurement_scope": "global_measurement_insufficient",
                              "unmeasured_families": sorted(configured_families - retained_families),
                              "status": "insufficient_measurement", "evidence_audit": "pending", "scale_coverage": coverage,
                              "pairwise_observed_counts": {}, "candidates": [], "mandatory_report_refs": [], "warnings": []},
            "diagnostics": {**diagnostics, "seed": SEED, "factor_analysis_started": False,
                            "trait_order": diagnostics["retained_scale_ids"],
                            "training_object_order": diagnostics["retained_object_ids"],
                            "input_sha256": hashlib.sha256(json.dumps({"study": study, "ratings": ratings}, sort_keys=True,
                                                                      ensure_ascii=False).encode("utf-8")).hexdigest()},
            "limitations": ["Недостаточно реальных наблюдений для начала FA; факторы, нагрузки, доли дисперсии и кластеры не оценивались.",
                            "Недостаточное покрытие не доказывает отсутствия политической позиции или отдельного тематического измерения."],
            "charts": [], "artifacts": [{"name": "factor_preview_json", "path": "factor_preview.json", "type": "json"},
                                        {"name": "factor_preview_md", "path": "factor_preview.md", "type": "markdown"}]}


def _save_preview(result, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "factor_preview.md").write_text(_render_markdown(result), encoding="utf-8")
    (output / "factor_preview.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _render_markdown(result):
    if result.get("measurement_status") == "insufficient_data":
        return _render_coverage_audit(result)
    review = result["russia_review"]
    selection = result["dimension_selection"]
    lines = ["# Предварительное сравнение факторных решений", "",
             f"Реальная матрица: {result['n_objects']} объектов × {result['n_scales']} пригодных аналитических переменных. "
             f"Известно {result['diagnostics']['retained_coverage']:.1%} ячеек; неизвестные не считались нейтральными.", "",
             f"Численная рекомендация экономности: q={selection['recommended_q']}. Все допустимые q показаны ниже. "
             "Эта рекомендация не отменяет рассмотрение тематических кандидатов при большем q.", "",
             "Доли — суммы квадратов ортогонально вращённых нагрузок, делённые на общую дисперсию всех стандартизованных шкал. "
             "Это описательные доли FA, не PCA и не доли политических взглядов населения.", "",
             "| q | Общая доля | Доли F1…Fq | Held-out LL, среднее ± SD | Остаток ковариации RMS | Предупреждения сходимости |",
             "|---|---|---|---|---|---|"]
    for model in result["models"]:
        c = model["dimension_fit"]
        shares = "; ".join(f"{100*f['variance_share']:.2f}%" for f in model["factors"])
        lines.append(f"| {model['q']} | {100*model['variance_total']:.2f}% | {shares} | "
                     f"{c['heldout_log_likelihood_mean']:.3f} ± {c['heldout_log_likelihood_sd']:.3f} | "
                     f"{c['residual_rms_off_diagonal']:.4f} | {c['convergence_warnings']} |")
    if result["diagnostics"].get("measurement_model") == "independent_poles":
        lines += ["", "Полюса измерены как независимые свойства 0–5. Известный полюс не требует знания второго; "
                  "низкая оценка свойства не является оценкой его антонима. Семантические семьи России считаются "
                  "по исходным шкалам, а не по числу их полюсов."]
    excluded = selection.get("parameter_count_constraint", {}).get("excluded_candidates", [])
    if excluded:
        lines += ["", "Не обучались из-за отрицательного номинального df: " +
                  ", ".join(f"q={item['q']} (df={item['nominal_df']})" for item in excluded) +
                  ". Неотрицательный df — необходимая, но не достаточная проверка идентифицируемости."]
    lines += ["", "SD относится к трём разделениям, а не к доверительному интервалу. Полные три значения и диагностика сохранены в JSON.", "",
              "## Смыслы всех факторов по решениям", "",
              "Номера и знаки факторов локальны для каждого q. При изменении q вся система переоценивается; "
              "одинаковый FID не гарантирует одинакового смысла. Ниже приведены два наиболее сильных свойства каждого направления "
              "и три ведущие шкалы с нагрузками. Полюса взяты из утверждённого словаря с учётом знака нагрузки; "
              "это краткое численное описание, а не проверка причин или истинных убеждений. "
              "Звёздочкой отмечены оси с менее чем двумя сильными нагрузками: их содержательная опора особенно ограничена.", ""]
    for model in result["models"]:
        lines += [f"### q={model['q']}", "",
                  "| Фактор | При высоких значениях | При низких значениях | Доля | Ключевые шкалы и нагрузки |",
                  "|---|---|---|---|---|"]
        for row in _factor_meaning_rows(model):
            lines.append("| " + " | ".join(_md_cell(value) for value in row) + " |")
        lines += [""]
    lines += ["## Проверка темы России", "",
              f"Настроено шкал: {len(review['configured_ids'])}; осталось в матрице: {len(review['retained_ids'])}. "
              f"Статус численного просмотра: `{review['status']}`.", ""]
    if review["excluded_ids"]:
        lines += ["Не вошли в пригодную матрицу: " + ", ".join(f"`{identity}`" for identity in review["excluded_ids"]) + ". "
                  "Это ограничение измерения, а не отсутствие соответствующей позиции.", ""]
    if review["configured_ids"]:
        lines += ["| ID | Утверждённая шкала | Семантическая семья | Наблюдений в пригодной выборке |",
                  "|---|---|---|---|"]
        for item in review["scale_coverage"]:
            lines.append(f"| {_md_cell(item['id'])} | {_md_cell(item['name'])} | {_md_cell(item['semantic_family_id'])} | {item['observed_count']} |")
        lines += [""]
    required = [item for item in review["candidates"] if item["mandatory_report"]]
    if required:
        lines += ["### Кандидаты обязательного рассмотрения", "",
                  "Порог ≥2% и число индикаторов обеспечивают показ, но не доказывают независимость свидетельств или существование отдельного российского конструкта. "
                  "Три семьи дают статус `numeric_candidate`; две — `two_indicator_candidate`. Аудит источников и устойчивости ещё не выполнен.", "",
                  "| Ссылка q/F | Доля | Статус | Российские индикаторы и знаковые нагрузки | Другие сильные шкалы (первые четыре) |",
                  "|---|---|---|---|---|"]
        for item in required:
            positives = "; ".join(f"{value['id']} {value['loading']:+.2f}" for value in item["strong_russia_indicators"])
            others = "; ".join(f"{value['name']} {value['loading']:+.2f}" for value in item["strong_non_russia_indicators"][:4]) or "Нет при пороге 0.35"
            lines.append(f"| q{item['q']}/{item['factor_id']} | {100*item['variance_share']:.2f}% | {item['status']} | {_md_cell(positives)} | {_md_cell(others)} |")
        lines += ["", "Все сильные общие шкалы, полные знаковые нагрузки и правильные свойства обоих полюсов приведены в JSON. "
                  "Смешанные знаки не позволяют автоматически назвать ось единой поддержкой России.", ""]
        bootstrapped = [item for item in required if item.get("bootstrap_stability", {}).get("requested_repeats", 0)]
        if bootstrapped:
            lines += ["### Условная устойчивость тематических кандидатов", "",
                      "Bootstrap объектов с возвращением при фиксированных переменных и q; FA и вращение обучены повторно. "
                      "Оси сопоставлены по нагрузкам с учётом перестановки и знака. Это описательная чувствительность, "
                      "не доверительный интервал и не проверка качества источников.", "",
                      "| q/F | Успешных повторов | Медианная congruence | Снова ≥2% и ≥3 семьи, доля всех повторов |",
                      "|---|---|---|---|"]
            for item in bootstrapped:
                b = item["bootstrap_stability"]
                congruence = f"{b['median_congruence']:.3f}" if b["median_congruence"] is not None else "не оценено"
                lines.append(f"| q{item['q']}/{item['factor_id']} | {b['successful_repeats']}/{b['requested_repeats']} | "
                             f"{congruence} | {b['numeric_candidate_fraction_all_replicates']:.1%} |")
            lines += [""]
    elif review["configured_ids"] and review["status"] == "insufficient_measurement":
        lines += ["Измерений российской темы недостаточно даже для численного вывода об отдельном измерении: "
                  f"в пригодной матрице осталось {review.get('retained_family_count', 0)} семантических семей. "
                  "Ни наличие, ни отсутствие российского фактора не установлено. Успешная общая FA не устраняет этот пробел.", ""]
    elif review["configured_ids"]:
        lines += ["Кандидатов, одновременно прошедших порог ≥2% и имевших хотя бы две разные сильные семьи российских индикаторов, не найдено. "
                  "Это результат просмотра только реально измеренной части темы, а не доказательство отсутствия российского измерения.", ""]
    else:
        lines += ["Российская тема не была настроена через required_scales или research_domain; тематические выводы не делаются.", ""]
    for warning in review["warnings"]:
        lines += [f"- {warning}"]
    lines += ["", "## Ограничения", ""]
    lines.extend(f"- {warning}" for warning in result["limitations"])
    if result["charts"]:
        lines += ["", "## Полные карты нагрузок отдельными файлами", ""]
        for chart in result["charts"]:
            lines += [f"- [{chart['title']}]({chart['file']})"]
    text = "\n".join(lines).strip() + "\n"
    for item in required:
        if f"q{item['q']}/{item['factor_id']}" not in text:
            raise AssertionError("A mandatory Russia-related candidate was omitted from preview")
    return text


def preview_factors(study: dict, ratings: list[dict], output_dir: Path, *, make_charts: bool = True,
                    bootstrap_repeats: int = 30) -> dict:
    """Return/export all q2…qmax fits and pending-evidence thematic candidates.

    Uses the stable core already fitted by _dimension_analysis, not an extra
    round of FA. No cluster, LLM, study-state mutation or automatic final q choice.
    """
    configuration = _theme_configuration(study)
    if isinstance(bootstrap_repeats, bool) or not isinstance(bootstrap_repeats, int) or not 0 <= bootstrap_repeats <= 500:
        raise ValueError("bootstrap_repeats must be an integer from 0 to 500")
    try:
        frame, _, _, scale_map, diagnostics = _prepare(study, ratings)
    except InsufficientMeasurementError as error:
        result = _insufficient_preview(study, ratings, configuration, error)
        _save_preview(result, output_dir)
        return result
    raw = frame.to_numpy()
    scales = [scale_map[identity] for identity in frame.columns]
    with threadpool_limits(limits=1):
        _, dimensions, fitted, selection = _dimension_analysis(raw)
    models, registry = [], []
    for dimension in dimensions:
        q = dimension["q"]
        parameters = fitted[q]
        standardized = (np.where(np.isfinite(raw), raw, parameters["training_imputation"]) -
                        parameters["training_mean"]) / parameters["training_scale"]
        denominator = float(np.var(standardized, axis=0).sum())
        rotated = parameters["loadings"] @ parameters["rotation"]
        factors = _factor_descriptions(rotated, scales, denominator)
        russia = _russia_factor_review(rotated, scales, denominator, configuration)
        for item in russia:
            item["q"] = q
            item["model_convergence_warnings"] = dimension["convergence_warnings"]
            if item["strong_indicator_count"]:
                registry.append(item)
        models.append({"q": q, "variance_total": float(sum(f["variance_share"] for f in factors)),
                       "variance_denominator": denominator, "factors": factors,
                       "dimension_fit": dimension, "russia_candidates": russia,
                       "loading_table": [{"id": str(scale["id"]), "name": scale.get("name", str(scale["id"])),
                                          "positive": scale.get("positive", ""), "negative": scale.get("negative", ""),
                                          "parent_scale_id": scale.get("parent_scale_id", str(scale["id"])),
                                          "kind": scale.get("kind", "bipolar"), "pole": scale.get("pole"),
                                          "loadings": {f["id"]: float(rotated[i, j]) for j, f in enumerate(factors)}}
                                         for i, scale in enumerate(scales)]})
    with threadpool_limits(limits=1):
        _bootstrap_russia_candidates(raw, scales, configuration, models, fitted, bootstrap_repeats)
    configured, retained = configuration["configured_ids"], set(frame.columns)
    retained_theme = [identity for identity in configured if identity in retained]
    measured_coverage = {item["id"]: item for item in diagnostics["input_scale_coverage"]}
    excluded_reasons = {item["id"]: item["reason"] for item in diagnostics["excluded_scales"]}
    coverage = [{"id": identity, "name": scale_map.get(identity, configuration["requested_definitions"][identity]).get("name", identity),
                 "semantic_family_id": configuration["family_by_id"][identity],
                 "retained": identity in retained,
                 "observed_count": measured_coverage.get(identity, {}).get("observed_retained_objects", 0),
                 "observed_input_count": measured_coverage.get(identity, {}).get("observed_objects", 0),
                 "exclusion_reason": None if identity in retained else excluded_reasons.get(identity, "not_in_approved_dictionary")}
                for identity in configured]
    warnings = []
    if any(value.endswith("not_independence_verified") for value in configuration["family_assignment_source"].values()):
        warnings.append("Для некоторых шкал semantic_family_id не задан: разные ID временно считаются разными семьями, их содержательная независимость не проверена.")
    if configuration["required_definition_mismatches"]:
        warnings.append("Определения части required_scales отличаются от утверждённого словаря; использованы утверждённые шкалы, требуется содержательная сверка.")
    if any(item["mandatory_report"] and item["model_convergence_warnings"] for item in registry):
        warnings.append("Часть тематических кандидатов получена в q с предупреждениями сходимости полного или отложенного fit; их численная надёжность требует проверки.")
    mandatory = [item for item in registry if item["mandatory_report"]]
    configured_families = {configuration["family_by_id"][identity] for identity in configured}
    retained_families = {configuration["family_by_id"][identity] for identity in retained_theme}
    measurement_scope = ("not_configured" if not configured else "insufficient_measurement" if len(retained_families) < 2
                         else "partial_thematic_coverage" if len(retained_theme) < len(configured)
                         else "all_declared_variables_retained")
    if configured and len(retained_families) < 2:
        warnings.append("Российская тема измерена недостаточно: осталось менее двух семантических семей. "
                        "Успешная общая FA не позволяет утверждать ни наличие, ни отсутствие российского фактора.")
    elif configured and len(retained_theme) < len(configured):
        warnings.append(f"Российская тема представлена частично: сохранены {len(retained_families)} из {len(configured_families)} семей "
                        f"и {len(retained_theme)} из {len(configured)} переменных. Выводы условны только на этот измеренный фрагмент; "
                        "неизмеренные свойства и возможные другие факторы не проверены.")
    status = ("not_configured" if not configured else "numeric_candidates_pending_audit" if any(item["status"] == "numeric_candidate" for item in mandatory)
              else "conditional_two_indicator_candidates" if mandatory else "insufficient_measurement" if len(retained_families) < 2
              else "no_numeric_candidate")
    pair_counts = {left: {right: int((frame[left].notna() & frame[right].notna()).sum()) for right in retained_theme}
                   for left in retained_theme}
    result = {"format_version": 1, "mode": "factor_preview_only", "n_objects": len(frame), "n_scales": frame.shape[1],
              "measurement_context": {"measurement_mode": study.get("measurement_mode", "unspecified"),
                                      "measurement_model": _measurement_model(study),
                                      "source_supplement": study.get("source_supplement"),
                                      "approvals": study.get("approvals", {}),
                                      "expert_validation": "not_independently_validated"},
              "dimension_selection": selection, "dimension_candidates": dimensions, "models": models,
              "russia_review": {**configuration, "retained_ids": retained_theme,
                                "excluded_ids": [identity for identity in configured if identity not in retained],
                                "configured_family_count": len(configured_families), "retained_family_count": len(retained_families),
                                "measurement_scope": measurement_scope,
                                "unmeasured_families": sorted(configured_families - retained_families),
                                "status": status, "evidence_audit": "pending", "scale_coverage": coverage,
                                "pairwise_observed_counts": pair_counts, "candidates": registry,
                                "mandatory_report_refs": [{"q": item["q"], "factor_id": item["factor_id"], "status": item["status"]} for item in mandatory],
                                "warnings": warnings},
              "diagnostics": {**diagnostics, "seed": SEED, "trait_order": frame.columns.tolist(),
                              "training_object_order": frame.index.tolist(),
                              "input_sha256": hashlib.sha256(json.dumps({"study": study, "ratings": ratings}, sort_keys=True,
                                                                        ensure_ascii=False).encode("utf-8")).hexdigest(),
                              "variance_definition": "sum squared orthogonally rotated FA loadings / total standardized variance after training mean-imputation; all retained scales, not only Russia block"},
              "limitations": ["Разведочная численная модель публичных образов; не независимая оценка убеждений или причин политических позиций.",
                              "FA гауссова и линейная, оценки ограниченные и порядковые; обучающее заполнение средним не устраняет смещения пропусков.",
                              "Три сильные семантические семьи дают только numeric_candidate: независимость событий, атрибуция и специфичность темы требуют аудита.",
                              "Bootstrap фиксирует состав переменных и q; он не проверяет весь сбор данных, независимость общих событий и все возможные вращения.",
                              "Номера и знаки факторов локальны для q; перенос смысла между решениями требует сопоставления полных нагрузок.",
                              "Кандидат ≥2% с двумя семьями показан условно; отсутствие кандидата при недостаточном измерении не означает отсутствия соответствующих позиций."],
              "charts": [], "artifacts": [{"name": "factor_preview_json", "path": "factor_preview.json", "type": "json"},
                                         {"name": "factor_preview_md", "path": "factor_preview.md", "type": "markdown"}]}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if make_charts:
        (output / "assets").mkdir(exist_ok=True)
        for model in models:
            file = f"assets/factor_preview_q{model['q']}.png"
            _draw_loadings(output / file, model, scales, set(configured))
            result["charts"].append({"q": model["q"], "file": file, "title": f"Полные нагрузки q={model['q']}"})
            result["artifacts"].append({"name": f"factor_preview_q{model['q']}", "path": file, "type": "image/png"})
    _save_preview(result, output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_json", type=Path)
    parser.add_argument("ratings_json", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--no-charts", action="store_true")
    arguments = parser.parse_args()
    study = json.loads(arguments.study_json.read_text(encoding="utf-8-sig"))
    ratings = json.loads(arguments.ratings_json.read_text(encoding="utf-8-sig"))
    result = preview_factors(study, ratings, arguments.output_dir, make_charts=not arguments.no_charts)
    print(json.dumps({"n_objects": result["n_objects"], "n_scales": result["n_scales"],
                      "recommended_q": result["dimension_selection"]["recommended_q"],
                      "russia_status": result["russia_review"]["status"],
                      "mandatory_report_refs": result["russia_review"]["mandatory_report_refs"],
                      "artifacts": result["artifacts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
