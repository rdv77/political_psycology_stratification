"""Whitelist-only quality aggregates from private review journals; no API.

The reviewer is an assistant LLM, the samples are nonrandom, and source
entailment is distinct from independent verification that a source is true.
This command writes review files only; it does not publish or modify SQLite.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sqlite3


JUDGMENTS = set("supported_recorded_channel source_supported_mischannelled source_supported_historical_outside_window uncertain_title_translation uncertain_primary_attribution uncertain_source_role_detail unsupported_actor_attribution uncertain_event_period unsupported_period_and_uncertain_identity source_supported_basis_inexact source_supported_trait_inference_uncertain source_partly_supported_and_mischannelled source_supported_event_date_misattributed".split())
SUPPORT = set("supported_as_source_assertion partially_supported_title_translation_uncertain supported_as_secondary_post_assertion partially_supported_role_detail_inconsistent not_supported_for_approved_target_identity supported_as_source_assertion_period_unresolved historical_source_assertion_not_supported_for_current_target partially_supported_claim_extension".split())
FAILURES = set("classification_language_indonesian_as_sw event_date_belongs_to_referenced_event external_appointment_not_own_action historical_1994_statement_outside_window historical_2017_action_outside_window historical_2019_action_outside_window homonym_alias_wrong_approved_actor indirect_news_report_marked_direct_statement possible_recycled_historical_event_period_unresolved primary_speech_attribution_not_verified_from_post reported_speech_typed_as_physical_action same_name_identity_unresolved source_role_detail_inconsistent source_supported_but_not_personal_action speculative_trait_candidates_from_nonresponse statement_title_not_substantive_direct_quote trait_candidate_not_entailed_multipartism trait_candidate_not_entailed_post_disaster trait_candidate_overgeneralized_pro_government trait_candidate_responsiveness_not_entailed statement_title_not_substantive_direct_quote trait_candidates_from_event_heading_or_role truncated_clause_extended_to_framework uncertain_un_rank_title_translation unverified_post_promoted_to_own_action".split())
CLASS_JUDGMENTS = set("supported conservative_hold_of_useful_material uncertain_low_substantive_relevance unsupported_actor_identity supported_for_supplied_passages uncertain unsupported supported_irrelevance false_negative_candidate unsupported_irrelevance confirmed_irrelevant potentially_relevant uncertain_relevance supported_negative no_false_negative_detected uncertain_false_negative".split())
NEGATIVE_JUSTIFIED = set("supported_review_sparse_source supported_review_low_relevance supported_review_unreadable_ocr supported_irrelevant supported_conservative_no_facts".split())
NEGATIVE_POTENTIAL = set("useful_material_held_by_identity_guard useful_context_held_by_mixed_gate missed_roster_actor_due_alias_order usable_source_lost_to_model_attribution_format usable_official_action_omitted usable_source_lost_to_short_name_guard".split())
NEGATIVE_UNCERTAIN = {"uncertain_language_ocr_overfilter"}
CLASS_JUDGMENTS |= NEGATIVE_JUSTIFIED | NEGATIVE_POTENTIAL | NEGATIVE_UNCERTAIN
CLASS_JUDGMENTS |= set("supported_contextual_identity unestablished plausible_ocr_identity name_only_not_substantive_actor unestablished_beyond_name historical_film_reference_only no_roster_actor_found different_nonroster_person supported_bizimana_not_provided_in_compact_roster named_image_reference supported_louise_reference supported_actor_in_full_source named_actor_in_truncated_title named_actor_without_substantive_statement supported_actor_attendance supported_biruta_personal_visit named_actor_in_ambiguous_comment unknown supported_mixed supported_other_indonesian likely_english_short_ocr supported_with_limited_translation_confidence".split())
NEGATIVE_FAILURES = set("alias_accent_kagame_guard_hold invisible_format_character_in_name_guard_hold mixed_allowed_languages_context_hold reverse_name_order_roster_recall_gap short_english_ocr_labelled_other invented_ellipsis_in_attribution_rejected short_name_local_guard_recall_loss supported_official_visit_omitted_by_model".split())
READING_MODES = {"full_supplied_source_passages", "targeted_head_tail_and_actor_windows_plus_lexical_census"}
LANGUAGES = {"en", "fr", "rw", "sw", "mixed", "other", "unknown"}
KINDS = {"action", "public_statement", "attributed_image", "institutional_context"}
RECEIPT_ISSUES = set("invalid_receipt duplicate_response_id conflicting_duplicate_response_id receipt_without_snapshot_job ambiguous_request_hash receipt_usage_unknown receipt_model_unpriced nonstandard_service_tier actual_service_tier_unreported pre_wrapper_without_receipt job_without_receipt multiple_provider_responses_for_job response_id_mismatch_or_ambiguous job_usage_mismatch job_cost_mismatch".split())
SEMANTIC_ANNOTATION_FIELDS = {"kind", "attribution_basis", "temporal_status", "measurement_status", "review_status", "event_date", "date_evidence"}


def _code(value, allowed):
    if value is None or value == "":
        return "not_recorded"
    return value if isinstance(value, str) and value in allowed else "unrecognized"


def _counts(values, allowed):
    return dict(sorted(Counter(_code(value, allowed) for value in values).items()))


def _findings(journal):
    rows = journal.get("findings", []) if isinstance(journal, dict) else []
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Review findings must be dictionaries")
    return rows


def _ids(rows, field, *, fact=False):
    ids = []
    for row in rows:
        value = row.get(field) or (row.get("original_fact") or {}).get(field)
        if isinstance(value, str) and value:
            ids.append(value)
        elif fact:
            raise ValueError("Every fact finding requires a fact_id")
    if fact and len(ids) != len(set(ids)):
        raise ValueError("Duplicate fact ID inside one review journal")
    return set(ids)


def _fact_summary(rows, agency=False):
    source_facts = [row.get("original_fact") or {} for row in rows]
    result = {"findings": len(rows), "unique_facts": len(_ids(rows, "fact_id", fact=True)),
        "distinct_documents": len(_ids(rows, "document_id")), "distinct_actors": len(_ids(rows, "actor_id")),
        "review_language_counts": _counts((row.get("language") for row in rows), LANGUAGES),
        "original_kind_counts": _counts((row.get("original_kind") or f.get("kind") for row, f in zip(rows, source_facts)), KINDS)}
    if agency:
        result["decision_counts"] = _counts((row.get("decision") for row in rows), {"keep", "context_only", "flag"})
    else:
        result["judgment_counts"] = _counts((row.get("judgment") for row in rows), JUDGMENTS)
        result["source_support_counts"] = _counts((row.get("source_support") for row in rows), SUPPORT)
        result["failure_code_counts"] = _counts((code for row in rows for code in row.get("failure_codes", [])), FAILURES)
    return result


def _classification_summary(rows, negative=False):
    judgments = [row.get("judgment") for row in rows]
    def dimension(name):
        return _counts(((value.get(name) if isinstance(value, dict) else value if name == "status" else None)
                        for value in judgments), CLASS_JUDGMENTS)
    result = {"findings": len(rows), "distinct_documents": len(_ids(rows, "document_id")),
        "distinct_returned_actors": len({actor for row in rows for actor in (row.get("original_decision") or {}).get("actor_ids", []) if isinstance(actor, str)}),
        "original_status_counts": _counts(((row.get("original_decision") or {}).get("status") for row in rows), {"relevant", "irrelevant", "review"}),
        "original_model_language_counts": _counts(((row.get("original_decision") or {}).get("language") for row in rows), LANGUAGES),
        "status_judgment_counts": dimension("status"), "identity_judgment_counts": dimension("identity"),
        "language_judgment_counts": dimension("language"),
        "category_counts": _counts((row.get("category") for row in rows), {"relevant_no_facts", "review", "irrelevant"}),
        "reading_mode_counts": _counts((row.get("reading_mode") for row in rows), READING_MODES),
        "failure_code_counts": _counts((code for row in rows for code in row.get("failure_codes", [])), FAILURES | NEGATIVE_FAILURES)}
    if negative:
        statuses = [(value.get("status") if isinstance(value, dict) else value) for value in judgments]
        result["targeted_summary_groups"] = {"supported_conservative": sum(v in NEGATIVE_JUSTIFIED for v in statuses),
            "potential_useful_material_loss_or_barrier": sum(v in NEGATIVE_POTENTIAL for v in statuses),
            "uncertain": sum(v in NEGATIVE_UNCERTAIN for v in statuses)}
    return result


def _semantic(annotation):
    return {key: value for key, value in annotation.items() if key in SEMANTIC_ANNOTATION_FIELDS}


def _safe_receipts(value):
    if value is None:
        return None
    value = value.get("public", value)
    fields = ("receipt_lines", "unique_receipt_records", "duplicate_response_records", "unpriced_receipt_units",
              "receipt_priced_cost_usd_known", "receipt_recomputed_cost_usd", "fully_priced_matched_job_cost_usd",
              "fully_priced_matched_receipt_cost_usd", "fully_priced_matched_cost_difference_usd",
              "max_abs_priced_job_cost_difference_usd", "expected_pre_wrapper_calls")
    result = {key: (value.get(key) if type(value.get(key)) in (int, float) and math.isfinite(value[key]) else None) for key in fields}
    result["issue_counts"] = {}
    totals = value.get("totals", {})
    for source, target in (("job_recorded_cost_usd", "job_recorded_cost_usd_total"),
                           ("outstanding_reserved_usd", "outstanding_reserved_usd")):
        amount = totals.get(source)
        result[target] = amount if type(amount) in (int, float) and math.isfinite(amount) else None
    for key, count in value.get("issue_counts", {}).items():
        if type(count) is int and count >= 0:
            safe = _code(key, RECEIPT_ISSUES)
            result["issue_counts"][safe] = result["issue_counts"].get(safe, 0) + count
    return result


def aggregate_quality(seed, agency, classification, negative=None, *, applied_reviews=None, receipt_reconciliation=None):
    """Public counts only. Unknown categories collapse to unrecognized.

    applied_reviews=None means not checked, whereas [] means a checked empty
    annotation table. No free text, identifiers or paths are copied to output.
    """
    first, second = _findings(seed), _findings(agency)
    classes = _findings(classification)
    negatives = _findings(negative) if negative is not None else None
    first_ids, second_ids = _ids(first, "fact_id", fact=True), _ids(second, "fact_id", fact=True)
    union = first_ids | second_ids
    proposal = {}
    for row in first:
        annotation = row.get("proposed_annotation")
        if isinstance(annotation, dict) and annotation:
            proposal.setdefault(row["fact_id"], []).append(_semantic(annotation))
    for row in agency.get("review_facts_decisions", []):
        if isinstance(row, dict) and row.get("fact_id") in second_ids and isinstance(row.get("annotation"), dict):
            proposal.setdefault(row["fact_id"], []).append(_semantic(row["annotation"]))
    applied = None
    if applied_reviews is not None:
        all_ids, matching, effective = set(), set(), {}
        for row in applied_reviews:
            identifier = row.get("fact_id")
            if not isinstance(identifier, str):
                continue
            all_ids.add(identifier)
            annotation = row.get("annotation", row.get("data", {}))
            if isinstance(annotation, str):
                annotation = json.loads(annotation)
            if isinstance(annotation, dict):
                effective.setdefault(identifier, {}).update(annotation)
        for identifier, candidates in proposal.items():
            annotation = effective.get(identifier, {})
            for candidate in candidates:
                if identifier in all_ids and _semantic(annotation) == candidate:
                    matching.add(identifier)
        applied = {"annotation_rows": len(applied_reviews), "distinct_annotated_facts": len(all_ids),
            "annotated_facts_in_review_union": len(all_ids & union), "annotated_facts_outside_review_union": len(all_ids - union),
            "proposed_with_any_annotation_facts": len(set(proposal) & all_ids),
            "proposed_without_annotation_facts": len(set(proposal) - all_ids),
            "proposal_semantically_matched_facts": len(matching),
            "proposed_with_different_annotation_facts": len((set(proposal) & all_ids) - matching),
            "proposed_not_matched_facts": len(set(proposal) - matching)}
    return {"version": "half-quality-public-v1", "api_calls": 0, "database_modified": False,
        "reviewer_type": "assistant_LLM_offline_not_independent_human_expert", "selection_type": "nonrandom_purposive_risk_and_actor_diversity",
        "independent_human_coders": 0, "external_source_truth_verified": False, "population_accuracy_estimate": None,
        "fact_review_union": {"seed_fact_findings": len(first), "agency_fact_findings": len(second),
            "unique_facts": len(union), "overlap_facts": len(first_ids & second_ids),
            "distinct_documents": len(_ids(first + second, "document_id")), "distinct_actors": len(_ids(first + second, "actor_id")),
            "expected_unique_facts": 100, "complete_expected_review": len(first) == 80 and len(second) == 20 and len(union) == 100},
        "seed_fact_review": _fact_summary(first), "agency_fact_review": _fact_summary(second, True),
        "classification_review": _classification_summary(classes),
        "negative_document_review": _classification_summary(negatives, negative=True) if negatives is not None else None,
        "annotations": {"proposed_distinct_facts": len(proposal),
            "proposed_note_only_facts": sum(all(not candidate for candidate in candidates) for candidates in proposal.values()),
            "actual_applied": applied},
        "receipt_reconciliation": _safe_receipts(receipt_reconciliation),
        "limits": {"categories_kept_separate": True, "targeted_error_fraction_is_not_population_error_rate": True,
            "missing_review_language_does_not_mean_unknown_source_language": True,
            "source_assertion_is_not_externally_verified_truth": True,
            "proposal_is_not_an_applied_annotation": True,
            "receipt_usage_reconciliation_is_not_provider_invoice": True}}


LABELS = {
    "supported_recorded_channel": "Источник поддерживает запись в указанном канале",
    "source_supported_mischannelled": "Содержание есть в источнике, канал измерения выбран неверно",
    "source_supported_historical_outside_window": "Исторический эпизод вне исследуемого периода",
    "source_supported_event_date_misattributed": "Дата отнесена к другому событию",
    "source_supported_basis_inexact": "Неточная маркировка основания атрибуции",
    "source_supported_trait_inference_uncertain": "Необоснованное расширение до признака",
    "source_partly_supported_and_mischannelled": "Частичное подтверждение и ошибка канала",
    "unsupported_actor_attribution": "Не подтверждена атрибуция выбранному участнику",
    "unsupported_period_and_uncertain_identity": "Проблема периода и неясная личность",
    "uncertain_title_translation": "Неясность перевода должности",
    "uncertain_primary_attribution": "Неясность первичной атрибуции",
    "uncertain_source_role_detail": "Противоречивая деталь роли в источнике",
    "uncertain_event_period": "Период события не установлен",
    "keep": "Сохранить измерительный канал", "context_only": "Отделить как контекст", "flag": "Дополнительно проверить",
    "supported": "Поддержано", "unsupported_actor_identity": "Не подтверждена личность",
    "supported_for_supplied_passages": "Поддержано для переданных фрагментов",
    "conservative_hold_of_useful_material": "Полезный материал консервативно оставлен на проверку",
    "uncertain_low_substantive_relevance": "Неясная содержательная релевантность",
    "not_recorded": "Не записано в журнале", "unrecognized": "Нераспознанная категория",
    "supported_review_sparse_source": "Проверка оправдана скудностью источника",
    "supported_review_low_relevance": "Проверка оправдана низкой релевантностью",
    "supported_review_unreadable_ocr": "Проверка оправдана нечитаемым OCR",
    "supported_irrelevant": "Нерелевантность поддержана источником",
    "supported_conservative_no_facts": "Консервативное отсутствие свидетельств поддержано",
    "useful_material_held_by_identity_guard": "Полезный материал остановлен проверкой имени",
    "useful_context_held_by_mixed_gate": "Полезный контекст остановлен языковым барьером",
    "missed_roster_actor_due_alias_order": "Участник пропущен из-за порядка имени",
    "usable_source_lost_to_model_attribution_format": "Полезный источник потерян из-за формы атрибуции",
    "usable_official_action_omitted": "Пропущено подтверждённое официальное действие",
    "usable_source_lost_to_short_name_guard": "Полезный источник остановлен короткой формой имени",
    "uncertain_language_ocr_overfilter": "Возможный избыточный отсев короткого OCR по языку",
    "full_supplied_source_passages": "Полностью прочитаны предоставленные исходные фрагменты",
    "targeted_head_tail_and_actor_windows_plus_lexical_census": "Целевые окна и лексическая проверка всего текста",
}


def _table(lines, title, values):
    lines.extend(["", "### " + title, "", "| Категория | Число |", "|---|---:|"])
    for key, count in values.items():
        lines.append(f"| {LABELS.get(key, key)} | {count} |")
    if not values:
        lines.append("| Нет записанных категорий | 0 |")


def render_quality(value):
    u = value["fact_review_union"]
    lines = ["# Проверка качества обработки половины корпуса Руанды", "",
        f"Прочитано **{u['unique_facts']} уникальных свидетельств**: основной журнал — {u['seed_fact_findings']}, целевой аудит агентности — {u['agency_fact_findings']}. Пересечений: {u['overlap_facts']}. Охвачено документов: {u['distinct_documents']}; участников: {u['distinct_actors']}.", "",
        "Проверял ассистент LLM по сохранённым исходным фрагментам. Выборки целевые, неслучайные: ошибки и спорные случаи специально искались. Это не независимое человеческое кодирование и не оценка общей точности всего корпуса. Соответствие записи источнику не доказывает истинность самого источника."]
    if not u["complete_expected_review"]:
        lines += ["", "**Проверка ещё не соответствует плану 80 + 20 = 100 уникальных свидетельств.** Частичные или пересекающиеся журналы не заменяются ожидаемыми числами."]
    for key, title, category in (("seed_fact_review", "Основная проверка свидетельств", "judgment_counts"),
                                  ("agency_fact_review", "Целевая проверка собственных действий и периода", "decision_counts")):
        block = value[key]
        lines += ["", "## " + title, "", f"Записей: {block['findings']}; документов: {block['distinct_documents']}; участников: {block['distinct_actors']}."]
        _table(lines, "Решения в собственной схеме журнала", block[category])
        _table(lines, "Языки, записанные рецензентом", block["review_language_counts"])
    lines += ["", "Отсутствие языковой отметки в журнале означает отсутствие этой отметки, а не отсутствие знания языка в исходном материале. Категории двух проверок не сливаются в искусственную общую долю «верно/неверно». Исторические годы внутри современного высказывания сами по себе не являются выходом за период."]
    _table(lines, "Коды проблем основной проверки", value["seed_fact_review"]["failure_code_counts"])
    for key, title in (("classification_review", "Проверка смыслового отбора"), ("negative_document_review", "Дополнительная проверка отрицательных документов")):
        block = value[key]
        lines += ["", "## " + title, ""]
        if block is None:
            lines.append("Отдельный журнал пока не передан; результат не заменяется нулём ошибок.")
            continue
        lines.append(f"Записей: {block['findings']}; уникальных документов: {block['distinct_documents']}; возвращённых моделью участников: {block['distinct_returned_actors']}.")
        if key == "negative_document_review":
            groups = block["targeted_summary_groups"]
            lines.append(f"В этой целевой проверке: {groups['supported_conservative']} поддержанных консервативных решений, {groups['potential_useful_material_loss_or_barrier']} потенциальных случаев потери полезного материала или ограничения барьером, {groups['uncertain']} неопределённых. Это не оценка доли пропусков во всём корпусе.")
        _table(lines, "Релевантность", block["status_judgment_counts"])
        _table(lines, "Идентификация участника", block["identity_judgment_counts"])
        _table(lines, "Определение языка", block["language_judgment_counts"])
        if key == "negative_document_review":
            _table(lines, "Объём чтения", block["reading_mode_counts"])
            lines.append("Целевые окна и лексическая проверка длинного текста не означают полный перевод всего источника.")
            _table(lines, "Коды барьеров и потерь", block["failure_code_counts"])
    a = value["annotations"]
    lines += ["", "## Предложения и фактически применённые аннотации", "", f"Предложения аннотаций относятся к {a['proposed_distinct_facts']} уникальным свидетельствам. Из них {a['proposed_note_only_facts']} содержат только замечание рецензента, без автоматического изменения семантических полей."]
    actual = a["actual_applied"]
    if actual is None:
        lines.append("Финальная таблица применённых аннотаций не передана; предложения нельзя объявлять выполненными.")
    else:
        lines += [f"В таблице корпуса записано {actual['annotation_rows']} аннотаций для {actual['distinct_annotated_facts']} уникальных свидетельств. Из них в рассмотренном объединении: {actual['annotated_facts_in_review_union']}; за его пределами: {actual['annotated_facts_outside_review_union']}.",
            f"Аннотация присутствует для {actual['proposed_with_any_annotation_facts']} предложенных свидетельств; не найдена для {actual['proposed_without_annotation_facts']}. Семантические поля предложения совпали с итоговым наложением аннотаций для {actual['proposal_semantically_matched_facts']}; для {actual['proposed_with_different_annotation_facts']} аннотация есть, но отличается от предложения. Такое отличие не означает неприменение: ручное решение могло быть уточнено. Несколько аннотаций одного свидетельства не являются несколькими проверенными фактами."]
    lines += ["", "## Отдельная сверка расходов", ""]
    receipt = value["receipt_reconciliation"]
    if receipt is None:
        lines.append("Финальная сверка квитанций здесь пока не приложена. Проверка качества текста не подтверждает финансовую сверку.")
    else:
        lines += [f"Строк квитанций: {receipt['receipt_lines']}; дубликатов response: {receipt['duplicate_response_records']}; единиц с неизвестной стоимостью: {receipt['unpriced_receipt_units']}.",
            f"Известный пересчитанный подытог: {receipt['receipt_priced_cost_usd_known']} USD. Разница по полностью оценённым сопоставленным заданиям: {receipt['fully_priced_matched_cost_difference_usd']} USD.",
            f"Записанные расходы заданий: {receipt['job_recorded_cost_usd_total']} USD; удержанный резерв: {receipt['outstanding_reserved_usd']} USD.",
            "Неизвестная оплата и удержанные резервы не превращаются в нулевые расходы. Это локальная сверка usage по тарифам, а не счёт провайдера."]
    lines += ["", "## Ограничения", "", "Целевые находки показывают конкретные риски атрибуции, агентности, датировки и чрезмерных смысловых обобщений. Они не дают случайной выборки ошибок, доверительного интервала точности или оценки всей совокупности. Повторы одного эпизода и независимость источников требуют отдельного контроля. Новых API-запросов при сборке этого документа нет.", "",
              "Агрегаты воспроизводятся скриптом `scripts/summarize_half_quality.py` из локальных журналов; исходные тексты, частные индексы и ключи в этот документ не включены.", ""]
    return "\n".join(lines)


def _read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("seed-review", "agency-review", "classification-review"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--negative-review", type=Path)
    parser.add_argument("--receipt-reconciliation", type=Path)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--corpus-id")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    applied = None
    if args.database:
        if not args.corpus_id:
            parser.error("--database requires --corpus-id")
        db = sqlite3.connect(args.database.resolve().as_uri() + "?mode=ro", uri=True, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            applied = [dict(row) for row in db.execute("SELECT fact_id,data FROM fact_reviews WHERE corpus_id=? ORDER BY created_at,id", (args.corpus_id,))]
        finally:
            db.close()
    result = aggregate_quality(_read(args.seed_review), _read(args.agency_review), _read(args.classification_review),
        _read(args.negative_review) if args.negative_review else None, applied_reviews=applied,
        receipt_reconciliation=_read(args.receipt_reconciliation) if args.receipt_reconciliation else None)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "quality_public_for_review.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.output_dir / "CORPUS_HALF_QUALITY_2026-10-17.md").write_text(render_quality(result), encoding="utf-8")
    print(json.dumps({"api_calls": 0, "unique_facts": result["fact_review_union"]["unique_facts"], "complete_expected_review": result["fact_review_union"]["complete_expected_review"]}))


if __name__ == "__main__":
    main()
