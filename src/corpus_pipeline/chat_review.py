"""Validate source-grounded chat adjudication and publish safe derived results.

No model client, database, queue or canonical study is opened here. Literal
evidence stays in the private review; public output contains paraphrases only.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import date
import re

from .scoring_preparation import digest


VERSION = "rwanda-chat-semantic-review-v1"
VERDICTS = {"retain", "narrow", "hold", "exclude"}
ATTRIBUTION = {"supported", "corrected", "unresolved", "not_own", "homonym"}
DATE_STATUS = {"supported_exact", "supported_interval", "publication_only", "unresolved", "outside_window", "corrected_exact"}
READ_SCOPE = {"full_sources", "targeted_passages", "partial_long_source"}
EVENT_IDENTITY = {"single_document_event", "same_event_supported", "unresolved", "distinct_events_in_literal_group"}
_PRIVATE = re.compile(r"https?://|\b(?:fact-|doc-|literal-|job-)[a-zA-Z0-9]", re.I)


def safe_text(value):
    if not isinstance(value, str) or not value.strip() or _PRIVATE.search(value):
        raise ValueError("Unsafe or empty public explanation")
    return value


def _iso(value):
    if value is None:
        return None
    if not isinstance(value, str) or len(value) != 10:
        raise ValueError("Date precision must be explicit")
    date.fromisoformat(value)
    return value


def apply_adjudications(review_files, amendments):
    """Apply explicit, source-auditable calibration without changing submissions."""
    updated = deepcopy(review_files)
    by_id = {row["case_id"]: row for file in updated for row in file["cases"]}
    public_changes = []
    for amendment in amendments:
        row = by_id[amendment["case_id"]]
        action, sid = amendment["action"], amendment.get("scale_id")
        before, after = None, None
        if action in {"remove_rating", "replace_rating"}:
            rating = next(r for r in row["ratings"] if r["scale_id"] == sid)
            if rating["value"] != amendment["expected_value"]:
                raise ValueError("Calibration target changed since review")
            before = rating["value"]
            row["ratings"].remove(rating)
        elif action == "add_rating":
            if any(r["scale_id"] == sid for r in row["ratings"]):
                raise ValueError("Calibration must not silently replace a rating")
        elif action == "replace_field":
            if amendment["field"] not in {"event_identity", "event_date", "scope_summary_ru", "public_note_ru"}:
                raise ValueError("Calibration field is not allowed")
            row[amendment["field"]] = deepcopy(amendment["value"])
        else:
            raise ValueError("Unknown calibration action")
        if action in {"replace_rating", "add_rating"}:
            if amendment["rating"]["scale_id"] != sid:
                raise ValueError("Calibration property mismatch")
            row["ratings"].append(deepcopy(amendment["rating"]))
            after = amendment["rating"]["value"]
        reason = safe_text(amendment["reason_ru"])
        row["corrections"].append({"field": "rating:" + sid if sid else amendment["field"],
                                   "before": before, "after": after, "reason_ru": reason})
        public_changes.append({"case_id": row["case_id"], "action": action, "scale_id": sid,
                               "before_value": before, "after_value": after, "reason_ru": reason})
    return updated, public_changes


def compile_review(cases, dictionary, review_files):
    """Validate completeness, evidence and unknowns; never aggregate episodes."""
    if dictionary.get("stage_status") != "approved" or dictionary.get("approved") is not True:
        raise ValueError("Use the approved measurement dictionary")
    scales = dictionary["scales"]
    scale_ids = [s["id"] for s in scales]
    if len(set(scale_ids)) != len(scale_ids):
        raise ValueError("Duplicate property ID")
    originals = {c["case_id"]: c for c in cases}
    if len(originals) != len(cases):
        raise ValueError("Duplicate source case")
    reviews = {}
    for file in review_files:
        for row in file["cases"]:
            cid = row.get("case_id")
            if cid in reviews or cid not in originals:
                raise ValueError("Duplicate or foreign adjudication")
            reviews[cid] = row
    if reviews.keys() != originals.keys():
        raise ValueError("Every selected case and control needs adjudication")

    public_cases, cells = [], []
    private_issues = []
    for cid, source in originals.items():
        row = reviews[cid]
        if row.get("all_38_reviewed") is not True:
            raise ValueError("Every approved property must be considered")
        if row.get("verdict") not in VERDICTS or row["attribution"].get("status") not in ATTRIBUTION:
            raise ValueError("Invalid adjudication status")
        if row.get("source_read_scope") not in READ_SCOPE:
            raise ValueError("Declare actual source-reading scope")
        timing = row["event_date"]
        if timing.get("status") not in DATE_STATUS:
            raise ValueError("Invalid date status")
        exact, start, end = (_iso(timing.get(k)) for k in ("date", "date_from", "date_to"))
        if timing["status"] in {"supported_exact", "corrected_exact"} and exact is None:
            raise ValueError("An exact date requires an exact value")
        if timing["status"] == "supported_interval" and (not start or not end or exact is not None):
            raise ValueError("An interval must not become an exact day")
        if timing["status"] in {"publication_only", "unresolved"} and any((exact, start, end)):
            raise ValueError("Publication-only or unresolved time stays unknown")
        if bool(start) != bool(end) or start and start > end:
            raise ValueError("Invalid date interval")
        related = row["event_identity"].get("related_cases", [])
        if (row["event_identity"].get("status") not in EVENT_IDENTITY or not isinstance(related, list)
                or any(x not in originals or x == cid for x in related)):
            raise ValueError("Invalid cross-case episode reference")
        documents = {d["id"]: d for d in source["sources"]}
        date_sources = list(documents.values())
        if row["event_identity"]["status"] == "same_event_supported":
            for linked in related:
                if originals[linked]["actor_id"] == source["actor_id"]:
                    date_sources.extend(originals[linked]["sources"])
        date_quote = timing.get("evidence_quote")
        if date_quote and not any(date_quote in d["text"] for d in date_sources):
            raise ValueError("Date evidence is not a literal source span")
        if (timing["status"] in {"supported_exact", "corrected_exact", "supported_interval"}
                or timing["status"] == "outside_window" and any((exact, start, end))) and not date_quote:
            raise ValueError("Source-grounded event time requires a literal supporting span")
        supported = {}
        for rating in row["ratings"]:
            sid, value = rating.get("scale_id"), rating.get("value")
            if sid not in scale_ids or sid in supported:
                raise ValueError("Duplicate or unapproved property")
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 5:
                raise ValueError("Episode intensity must be an integer 0 to 5")
            if (row["verdict"] == "exclude" or row["attribution"]["status"] in {"not_own", "homonym", "unresolved"}
                    or timing["status"] == "outside_window"):
                raise ValueError("Ineligible personal attribution/time cannot produce scores")
            document = documents.get(rating.get("source_document_id"))
            quote = rating.get("support_quote")
            if not document or not isinstance(quote, str) or not quote.strip() or quote not in document["text"]:
                raise ValueError("Rating evidence must be an exact original source span")
            if value == 0 and rating.get("explicit_zero_basis") is not True:
                raise ValueError("A zero requires explicitly reviewed relevant non-expression")
            supported[sid] = {"scale_id": sid, "value": value, "reason_ru": safe_text(rating["reason_ru"]),
                              "status": "provisional_fragment_rating"}
        corrections = [{"field": safe_text(c["field"]), "reason_ru": safe_text(c["reason_ru"])} for c in row["corrections"]]
        public = {"case_id": cid, "sample_kind": source["sample_kind"],
                  "actor_id": source["actor_id"], "actor_name": safe_text(source["actor_profile"].get("name_original") or source["actor_profile"]["name"]),
                  "verdict": row["verdict"], "summary_ru": safe_text(row["scope_summary_ru"]),
                  "attribution": {k: safe_text(row["attribution"][k]) for k in ("status", "basis", "explanation_ru")},
                  "capacity_review_ru": safe_text(row["capacity_review_ru"]),
                  "event_date": {"status": timing["status"], "date": exact, "date_from": start, "date_to": end,
                                 "explanation_ru": safe_text(timing["explanation_ru"])},
                  "event_identity": {"status": row["event_identity"]["status"], "related_cases": related,
                                     "explanation_ru": safe_text(row["event_identity"]["explanation_ru"])},
                  "ratings": list(supported.values()), "unknown_reason_ru": safe_text(row["unknown_reason_ru"]),
                  "corrections": corrections, "confidence": row["confidence"],
                  "source_read_scope": row["source_read_scope"], "public_note_ru": safe_text(row["public_note_ru"]),
                  "usable_for_person_aggregation": False}
        public_cases.append(public)
        for sid in scale_ids:
            cells.append({"case_id": cid, "actor_id": source["actor_id"], "scale_id": sid,
                          "considered": True, "value": supported[sid]["value"] if sid in supported else None,
                          "status": "provisional_fragment_rating" if sid in supported else "unknown",
                          "included_in_canonical_study": False})
        private_issues.append({"case_id": cid, "review_content_sha256": digest(row), "source_case_sha256": digest(source)})

    selected = [c for c in public_cases if c["sample_kind"] == "selected_unit"]
    controls = [c for c in public_cases if c["sample_kind"] == "excluded_control"]
    counter = lambda values: dict(sorted(Counter(values).items()))
    public = {"version": VERSION, "method": "assistant_semantic_review_in_current_chat_no_local_paid_api",
              "dictionary_content_sha256": digest(dictionary), "measurement_dictionary_approved": True,
              "counts": {"selected_cases_reviewed": len(selected), "control_cases_reviewed": len(controls),
                         "source_documents_in_dossiers": sum(len(c["sources"]) for c in cases),
                         "unique_source_documents": len({d["id"] for c in cases for d in c["sources"]}),
                         "properties_considered_per_case": len(scales), "cells_considered": len(cells),
                         "selected_cases_with_ratings": sum(bool(c["ratings"]) for c in selected),
                         "controls_with_provisional_ratings": sum(bool(c["ratings"]) for c in controls),
                         "actors_with_provisional_ratings": len({c["actor_id"] for c in public_cases if c["ratings"]}),
                         "properties_with_provisional_ratings": len({r["scale_id"] for c in public_cases for r in c["ratings"]}),
                         "held_cases_with_provisional_ratings": sum(bool(c["ratings"]) and c["verdict"] == "hold" for c in public_cases),
                         "provisional_ratings": sum(len(c["ratings"]) for c in public_cases),
                         "unknown_cells": sum(c["value"] is None for c in cells)},
              "selected_verdicts": counter(c["verdict"] for c in selected),
              "control_verdicts": counter(c["verdict"] for c in controls),
              "attribution_statuses": counter(c["attribution"]["status"] for c in public_cases),
              "date_statuses": counter(c["event_date"]["status"] for c in public_cases),
              "source_read_scopes": counter(c["source_read_scope"] for c in public_cases),
              "rating_distribution": counter(str(r["value"]) for c in public_cases for r in c["ratings"]),
              "scales": [{"id": s["id"], "name": s["name"]} for s in scales],
              "cases": public_cases, "cells": cells,
              "api_calls": 0, "new_api_cost_usd": 0, "paid_execution_authorized": False,
              "canonical_study_modified": False, "person_level_aggregation_performed": False,
              "factor_analysis_performed": False, "raw_quotes_published": False,
              "limitations": ["Targeted risk sample, not an estimate of population extraction accuracy.",
                              "Assistant review is not an independent human expert study or external fact verification.",
                              "Literal source matching checks provenance, not semantic truth.",
                              "Ratings concern reviewed source fragments, not stable five-year personal profiles.",
                              "All other considered properties remain unknown; no zeros from silence.",
                              "Cross-case episode links are recorded, but no independence-weighted aggregation is performed.",
                              "Excluded controls stay separate from the selected sample and canonical ratings."]}
    return {"public": public, "private_validation": private_issues}


def render(public):
    c = public["counts"]
    labels = {"retain": "Сохранить содержание", "narrow": "Сузить интерпретацию", "hold": "Удержать до уточнения", "exclude": "Исключить из измерения"}
    lines = ["# Руанда: проверка 60 фрагментов и 12 контрольных случаев в чате", "",
             "Проверка выполнена ассистентом по сохранённым источникам, без платных запросов локального конвейера. "
             "Новые расходы API — **$0**. Это не независимая человеческая экспертиза и не внешняя проверка достоверности публикаций.", "",
             "## Основные результаты", "",
             f"Рассмотрено **{c['selected_cases_reviewed']} отобранных случаев** и **{c['control_cases_reviewed']} контролей**; "
             f"для каждого содержательно проверены все **{c['properties_considered_per_case']} утверждённых признаков**.", "",
             f"Предварительные баллы обоснованы в **{c['selected_cases_with_ratings']} отобранных случаях**; "
             f"всего **{c['provisional_ratings']} оценок**, неизвестных ячеек — **{c['unknown_cells']}**. "
             "Это выраженность в конкретном источнике, а не итоговые рейтинги политиков.", "",
             f"Источники включают **{c['unique_source_documents']} уникальные публикации**. "
             f"Полные тексты по случаям прочитаны в {public['source_read_scopes'].get('full_sources', 0)} случаях; "
             f"целевые контексты длинных источников — в {public['source_read_scopes'].get('targeted_passages', 0)} случаях. "
             "Буквальные повторы не считаются отдельной проверкой смысла.", "",
             "| Решение | Отобранные случаи | Контрольные случаи |", "|---|---:|---:|"]
    for code, label in labels.items():
        lines.append(f"| {label} | {public['selected_verdicts'].get(code, 0)} | {public['control_verdicts'].get(code, 0)} |")
    lines += ["", "Выборка целенаправленно насыщена редкими и рискованными случаями. Эти числа нельзя выдавать за долю ошибок во всём корпусе. "
              "Каждый исходный фрагмент и прежняя оценка сохранены; поправки находятся в отдельной версии.", "",
              "## Какие признаки удалось оценить", "", "| Признак | Случаи: предварительная выраженность |", "|---|---|"]
    for scale in public["scales"]:
        values = [f"{case['case_id']} ({case['actor_name']}): {r['value']}" for case in public["cases"]
                  for r in case["ratings"] if r["scale_id"] == scale["id"]]
        lines.append(f"| {scale['name']} | {'; '.join(values) if values else 'Недостаточно основания в этой выборке'} |")
    lines += ["", "Неизвестное не равно нулю. Частота и число перепечаток не увеличивают балл. "
              "Высокая выраженность одного эпизода не устанавливает устойчивую позицию за весь период.", "",
              "## Калибровка спорных решений", ""]
    for change in public.get("calibration_changes", []):
        if change["action"] == "replace_field":
            continue
        before = change["before_value"] if change["before_value"] is not None else "unknown"
        after = change["after_value"] if change["after_value"] is not None else "unknown"
        lines.append(f"- {change['case_id']}, {change['scale_id']}: {before} → {after}. {change['reason_ru']}")
    lines += ["", "Исходные ассистентские решения и причины последующих поправок сохранены отдельно. "
              "Обсуждение в одной команде не является слепым межэкспертным тестом.", "", "## Все отобранные случаи", ""]
    for case in public["cases"]:
        if case["sample_kind"] == "excluded_control" and case["case_id"] == "C01":
            lines += ["## Контрольные случаи", ""]
        lines += [f"### {case['case_id']}. {case['actor_name']}", "",
                  f"**{labels[case['verdict']]}**. {case['summary_ru']}", "",
                  "Атрибуция: " + case["attribution"]["explanation_ru"], "",
                  "Роль: " + case["capacity_review_ru"], ""]
        t = case["event_date"]
        when = t["date"] or (f"{t['date_from']} — {t['date_to']}" if t["date_from"] else "не установлена")
        lines += [f"Дата события: **{when}**. {t['explanation_ru']}", ""]
        for r in case["ratings"]:
            name = next(s["name"] for s in public["scales"] if s["id"] == r["scale_id"])
            lines += [f"- **{name}: {r['value']}/5** — {r['reason_ru']}"]
        if not case["ratings"]:
            lines += ["Числовые оценки не обоснованы; неизвестное сохраняется пропуском."]
        lines += ["", "Связь событий: " + case["event_identity"]["explanation_ru"], ""]
        if case["event_identity"]["related_cases"]:
            lines += ["Связанные случаи: " + ", ".join(case["event_identity"]["related_cases"]) + ".", ""]
    lines += ["## Границы результата и следующий шаг", "",
              f"Публичная таблица содержит {len(public['cases'])} × {len(public['scales'])} рассмотренных ячеек с неизвестными значениями. "
              "Полные цитаты, внутренние ID и исходные тексты хранятся локально. Подтверждающие строки каждой числовой оценки проверены "
              "на буквальное присутствие в соответствующем исходнике; содержательную правильность это само по себе не доказывает.", "",
              "Повторные материалы и несколько тематических фрагментов одного события не агрегированы в независимые эпизоды. "
              "Итоговые профили 42 политиков, факторные решения и кластеры этим пилотом не пересчитаны. "
              "Перед массовым этапом нужны доработка отбора говорящего, хронологии и связей событий, затем проверка покрытия новой матрицы.", "",
              "[Утверждённый словарь](APPROVAL.md) · [Решение о платных запросах](execution_decision.json) · [Состав пилота](OFFLINE_PILOT_2026-10-17.md)", ""]
    return "\n".join(lines)
