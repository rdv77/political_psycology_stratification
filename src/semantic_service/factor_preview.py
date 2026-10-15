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
from threadpoolctl import threadpool_limits

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.semantic_service.analysis import (
    LOADING_THRESHOLD, SEED, _dimension_analysis, _factor_descriptions, _prepare,
)


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
    actual = {str(item["id"]): item for item in study.get("scales", [])}
    for identity, item in actual.items():
        if _metadata_field(item, "research_domain") == "russia_relations":
            requested.setdefault(identity, item)
    families, sources = {}, {}
    for identity, definition in requested.items():
        scale = actual.get(identity, {})
        family = _metadata_field(scale, "semantic_family_id") or _metadata_field(definition, "semantic_family_id")
        families[identity] = str(family or identity)
        sources[identity] = "declared_family" if family else "scale_id_fallback_not_independence_verified"
    instructions = requirements.get("instructions", [])
    if not isinstance(instructions, list) or any(not isinstance(item, str) for item in instructions):
        raise ValueError("research_requirements.instructions must be an array of strings")
    mismatches = []
    for identity, definition in requested.items():
        if identity in actual:
            fields = [field for field in ["positive", "negative", "definition"]
                      if field in definition and definition[field] != actual[identity].get(field)]
            if fields:
                mismatches.append({"id": identity, "different_fields": fields})
    return {"configured_ids": list(requested), "family_by_id": families,
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
            item = {"id": identity, "name": scale.get("name", identity), "loading": value,
                    "positive": scale.get("positive", ""), "negative": scale.get("negative", ""),
                    "high_factor_property": scale.get("positive", "") if value > 0 else scale.get("negative", ""),
                    "low_factor_property": scale.get("negative", "") if value > 0 else scale.get("positive", "")}
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


def _render_markdown(result):
    review = result["russia_review"]
    selection = result["dimension_selection"]
    lines = ["# Предварительное сравнение факторных решений", "",
             f"Реальная матрица: {result['n_objects']} объектов × {result['n_scales']} пригодных шкал. "
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
    elif review["configured_ids"]:
        lines += ["Кандидатов, одновременно прошедших порог ≥2% и имевших хотя бы две разные сильные семьи российских индикаторов, не найдено. "
                  "Это результат численного просмотра этой матрицы. При недостаточном покрытии вывод о наличии отдельного измерения невозможен.", ""]
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


def preview_factors(study: dict, ratings: list[dict], output_dir: Path, *, make_charts: bool = True) -> dict:
    """Return/export all q2…qmax fits and pending-evidence thematic candidates.

    Uses the stable core already fitted by _dimension_analysis, not an extra
    round of FA. No cluster, LLM, study-state mutation or automatic final q choice.
    """
    configuration = _theme_configuration(study)
    frame, _, _, scale_map, diagnostics = _prepare(study, ratings)
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
                                          "loadings": {f["id"]: float(rotated[i, j]) for j, f in enumerate(factors)}}
                                         for i, scale in enumerate(scales)]})
    configured, retained = configuration["configured_ids"], set(frame.columns)
    retained_theme = [identity for identity in configured if identity in retained]
    observed_keys = {(str(rating["object_id"]), str(rating["scale_id"])) for rating in ratings
                     if rating.get("status", "observed") == "observed"}
    excluded_reasons = {item["id"]: item["reason"] for item in diagnostics["excluded_scales"]}
    coverage = [{"id": identity, "name": scale_map.get(identity, configuration["requested_definitions"][identity]).get("name", identity),
                 "semantic_family_id": configuration["family_by_id"][identity],
                 "retained": identity in retained,
                 "observed_count": sum(sid == identity and oid in frame.index for oid, sid in observed_keys),
                 "observed_input_count": sum(sid == identity for _, sid in observed_keys),
                 "exclusion_reason": None if identity in retained else excluded_reasons.get(identity, "not_in_approved_dictionary")}
                for identity in configured]
    warnings = []
    if any(value == "scale_id_fallback_not_independence_verified" for value in configuration["family_assignment_source"].values()):
        warnings.append("Для некоторых шкал semantic_family_id не задан: разные ID временно считаются разными семьями, их содержательная независимость не проверена.")
    if configuration["required_definition_mismatches"]:
        warnings.append("Определения части required_scales отличаются от утверждённого словаря; использованы утверждённые шкалы, требуется содержательная сверка.")
    if any(item["mandatory_report"] and item["model_convergence_warnings"] for item in registry):
        warnings.append("Часть тематических кандидатов получена в q с предупреждениями сходимости полного или отложенного fit; их численная надёжность требует проверки.")
    mandatory = [item for item in registry if item["mandatory_report"]]
    status = ("not_configured" if not configured else "numeric_candidates_pending_audit" if any(item["status"] == "numeric_candidate" for item in mandatory)
              else "conditional_two_indicator_candidates" if mandatory else "insufficient_measurement" if len({configuration['family_by_id'][i] for i in retained_theme}) < 2
              else "no_numeric_candidate")
    pair_counts = {left: {right: int((frame[left].notna() & frame[right].notna()).sum()) for right in retained_theme}
                   for left in retained_theme}
    result = {"format_version": 1, "mode": "factor_preview_only", "n_objects": len(frame), "n_scales": frame.shape[1],
              "measurement_context": {"measurement_mode": study.get("measurement_mode", "unspecified"),
                                      "source_supplement": study.get("source_supplement"),
                                      "approvals": study.get("approvals", {}),
                                      "expert_validation": "not_independently_validated"},
              "dimension_selection": selection, "dimension_candidates": dimensions, "models": models,
              "russia_review": {**configuration, "retained_ids": retained_theme,
                                "excluded_ids": [identity for identity in configured if identity not in retained],
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
                              "Подвыборочная устойчивость FA, альтернативные старты вращения и зависимость от общих событий ещё не проверены.",
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
    markdown = _render_markdown(result)
    (output / "factor_preview.md").write_text(markdown, encoding="utf-8")
    (output / "factor_preview.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
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
