"""Publish only offline aggregates from a private half-corpus processing snapshot.

No API, queue, database or source post is modified. Actor names come from the
approved snapshot roster. Quotes, source URLs, authors, document/fact IDs and
payloads never enter the public output. Candidate labels remain unapproved.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import unicodedata
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.corpus_pipeline.models import _eligible_fact

CORPUS_ID = "corpus-b2d1452fae0acf9c20f5"
_STAGES = {"relevance", "extract", "describe", "score"}
_STATUSES = {"pending", "running", "done", "review", "error", "uncertain", "obsolete"}
_CONTACT = re.compile(r"https?://|www\.|[^\s@]+@[^\s@]+|(?<!\w)@[\w]+|(?:\+?\d[\s().-]*){7,}", re.I)


def _normal(value):
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def _label(value):
    """Keep short descriptor labels, never source excerpts/contact strings."""
    if (not isinstance(value, str) or not value.strip() or len(value) > 160
            or len(value.split()) > 20 or "\n" in value or "\r" in value or _CONTACT.search(value)
            or any(unicodedata.category(c).startswith("C") for c in value)):
        return None
    return " ".join(unicodedata.normalize("NFC", value).split())


def _number(value, *, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("Snapshot accounting requires finite nonnegative numbers")
    if integer and type(value) is not int:
        raise ValueError("Token counts must be integers")
    return value


def _model(job):
    value = (job.get("payload") or {}).get("model", "unknown")
    return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9._:-]{1,80}", value) else "unknown"


def _account(jobs):
    groups = {}
    total = {"jobs": 0, "input_tokens": 0, "output_tokens": 0, "estimated_usd": 0., "outstanding_reserved_usd": 0.}
    for job in jobs:
        stage = job.get("stage") if job.get("stage") in _STAGES else "unknown"
        status = job.get("status") if job.get("status") in _STATUSES else "unknown"
        key = stage, _model(job), status
        row = groups.setdefault(key, {"stage": stage, "model": key[1], "status": status,
                                     "jobs": 0, "input_tokens": 0, "output_tokens": 0,
                                     "estimated_usd": 0., "outstanding_reserved_usd": 0.})
        values = {"jobs": 1, "input_tokens": _number(job.get("input_tokens", 0), integer=True),
                  "output_tokens": _number(job.get("output_tokens", 0), integer=True),
                  "estimated_usd": _number(job.get("cost_usd", 0)),
                  "outstanding_reserved_usd": _number(job.get("reserved_usd", 0)) if status in {"running", "uncertain"} else 0.}
        for field, amount in values.items():
            total[field] += amount
            row[field] += amount
    for row in [total, *groups.values()]:
        for field in ("estimated_usd", "outstanding_reserved_usd"):
            row[field] = round(row[field], 8)
    return {"total": total, "by_stage_model_status": [groups[key] for key in sorted(groups)]}


def _document_coverage(jobs, stage):
    documents = defaultdict(list)
    for job in jobs:
        document = (job.get("context") or {}).get("document", {})
        if job.get("stage") == stage and document.get("id"):
            documents[document["id"]].append(job)
    counts = Counter()
    for group in documents.values():
        statuses = {j["status"] for j in group}
        for status in statuses:
            counts[status + "_documents"] += 1
        if statuses == {"done"}:
            counts["all_prepared_jobs_done_documents"] += 1
        if statuses <= {"done", "review"}:
            counts["all_prepared_jobs_terminal_done_or_review_documents"] += 1
        if stage == "relevance":
            outcomes = {(job.get("result") or {}).get("status") for job in group if job["status"] == "done"}
            if "relevant" in outcomes:
                counts["relevant_documents"] += 1
            elif outcomes == {"irrelevant"} and statuses == {"done"}:
                counts["irrelevant_documents"] += 1
    return {"documents_with_prepared_jobs": len(documents), **dict(sorted(counts.items()))}


def _host(url):
    try:
        return urlsplit(url).hostname if isinstance(url, str) else None
    except ValueError:
        return None


def _scale_groups(scales, facts, eligible):
    """Connected literal declared aliases only; definitions are never inferred."""
    entries, parents, labels_seen, dropped, unsupported = [], {}, {}, 0, 0

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for scale in scales:
        if not any(eid in eligible for eid in scale.get("evidence_ids", [])):
            # Later source reviews can withdraw the only usable evidence of
            # a once structurally valid proposal. Keep its receipt privately.
            unsupported += 1
            continue
        labels = [label for label in (_label(v) for v in [scale.get("name"), *scale.get("aliases", [])]) if label]
        if not labels:
            dropped += 1
            continue
        i = len(entries)
        entries.append((scale, labels))
        parents[i] = i
        for label in labels:
            key = _normal(label)
            if key in labels_seen:
                parents[find(i)] = find(labels_seen[key])
            else:
                labels_seen[key] = i
    groups = defaultdict(list)
    for i in range(len(entries)):
        groups[find(i)].append(entries[i])
    result = []
    for members in groups.values():
        forms = defaultdict(set)
        evidence = set()
        for scale, labels in members:
            for label in labels:
                forms[_normal(label)].add(label)
            evidence.update(eid for eid in scale.get("evidence_ids", []) if eid in facts)
        labels = sorted((sorted(values, key=lambda s: (len(s), s))[0] for values in forms.values()), key=_normal)
        support = [facts[eid] for eid in evidence]
        result.append({"group_id": "labels-" + hashlib.sha256(json.dumps(sorted(forms), ensure_ascii=False).encode()).hexdigest()[:12],
                       "name": labels[0], "labels": labels, "draft_count": len(members),
                       "supporting_facts": len(evidence), "eligible_public_position_facts": len(evidence & eligible),
                       "supporting_actors": len({f["actor_id"] for f in support}),
                       "supporting_documents": len({f["document_id"] for f in support}),
                       "definitions_differ": len({_normal(str(scale.get("definition", ""))) for scale, _ in members}) > 1,
                       "draft": True, "approved": False})
    return sorted(result, key=lambda r: (-r["supporting_actors"], -r["supporting_facts"], _normal(r["name"]))), dropped, unsupported


def _quality(audit, corpus_id, source_sha256, selected_count, facts):
    if audit is None:
        return {"available": False, "status": "not_yet_checked", "structural_checks": None,
                "semantic_review": {"status": "not_yet_checked", "accuracy_estimate": None}}
    private_keys = {"public_aggregate", "semantic_review_packet", "risk_review_packet",
                    "negative_review_packet", "structural_issues_private"}
    if (not isinstance(audit, dict) or private_keys & set(audit)
            or not isinstance(audit.get("audit_policy_version"), str)
            or not re.fullmatch(r"[A-Za-z0-9._-]{1,100}", audit["audit_policy_version"])
            or audit.get("corpus_id") != corpus_id or audit.get("source_sha256") != source_sha256):
        raise ValueError("Use only the matching public aggregate_for_review.json, never audit_private.json")

    def count(mapping, key):
        return _number(mapping.get(key, 0), integer=True)

    def counts(mapping):
        if not isinstance(mapping, dict):
            raise ValueError("Audit counters must be objects")
        if any(not isinstance(key, str) or not re.fullmatch(r"[a-z0-9_]{1,100}", key) for key in mapping):
            raise ValueError("Audit issue codes must be public counter labels")
        return {key: _number(value, integer=True) for key, value in sorted(mapping.items())}

    structural, semantic, scope = audit.get("structural_census"), audit.get("semantic_review"), audit.get("scope")
    if any(not isinstance(value, dict) for value in (structural, semantic, scope)):
        raise ValueError("Audit aggregate lacks required structural/semantic/scope counters")
    checked = count(structural, "active_accepted_facts_checked")
    actual_hash = hashlib.sha256(json.dumps(sorted(facts.values(), key=lambda fact: fact["id"]),
                                ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    supplied_hash = audit.get("facts_snapshot_sha256", (audit.get("provenance") or {}).get("facts_snapshot_sha256"))
    if supplied_hash is not None and (not isinstance(supplied_hash, str) or not re.fullmatch(r"[a-f0-9]{64}", supplied_hash)):
        raise ValueError("Invalid public audit fact-snapshot hash")
    return {"available": True, "status": "structural_audit_only", "audit_policy_version": audit["audit_policy_version"],
        "audit_scope": {"selected_documents": count(scope, "selected_document_frame"),
            "unfinished_active_jobs": count(scope, "unfinished_active_jobs"), "snapshot_is_partial": scope.get("snapshot_is_partial") is True},
        "snapshot_alignment": {"fact_count_matches": checked == len(facts),
            "selected_frame_count_matches": count(scope, "selected_document_frame") == selected_count,
            "exact_fact_snapshot_matches": supplied_hash == actual_hash if supplied_hash is not None else None},
        "structural_checks": {key: count(structural, key) for key in (
            "active_accepted_facts_checked", "inactive_facts_ignored", "facts_with_recheck_issues",
            "done_extract_jobs_missing_qc", "source_context_jobs_checked", "jobs_with_qc")},
        "structural_issue_counts": counts(structural.get("issue_counts", {})),
        "source_context_issue_counts": counts(structural.get("source_context_issue_counts", {})),
        "normalization": {"totals": counts(structural.get("original_normalization_totals", {})),
            "issue_counts": counts(structural.get("original_normalization_issue_counts", {}))},
        "semantic_review": {"status": "pending_source_review", "accuracy_estimate": None,
            "facts_selected_for_review": count(semantic, "facts_selected"),
            "risk_packet_facts": count(semantic, "risk_packet_facts"),
            "negative_packet_documents": count(semantic, "negative_packet_documents")}}


def summarize(snapshot, audit=None):
    manifest, corpus = snapshot["manifest"], snapshot["corpus"]
    config = corpus["config"]
    objects = config["objects"]
    actor_ids = [obj["id"] for obj in objects]
    if len(actor_ids) != len(set(actor_ids)):
        raise ValueError("Approved roster IDs must be unique")
    jobs_by_id = {}
    for job in snapshot["jobs"]:
        if job["id"] in jobs_by_id and job != jobs_by_id[job["id"]]:
            raise ValueError("Conflicting repeated job receipt")
        jobs_by_id[job["id"]] = job
    jobs = list(jobs_by_id.values())
    active = [job for job in jobs if job.get("active_recipe") is True]
    active_extract = {job["id"]: job for job in active if job["stage"] == "extract" and job["status"] == "done"}
    accounting, active_accounting = _account(jobs), _account(active)
    expected = manifest.get("selected_count", 12205)
    if type(expected) is not int or expected < 1:
        raise ValueError("Expected selected-document count must be a positive integer")
    coverage = {stage: _document_coverage(active, stage) for stage in ("relevance", "extract")}
    coverage["selected_documents"] = expected
    prepared = coverage["relevance"]["documents_with_prepared_jobs"]
    coverage["relevance"]["documents_not_prepared"] = max(0, expected - prepared)
    coverage["relevance"]["done_fraction_of_selected"] = coverage["relevance"].get("all_prepared_jobs_done_documents", 0) / expected
    coverage["extract"]["expected_relevant_documents"] = coverage["relevance"].get("relevant_documents", 0)
    coverage["extract"]["documents_not_prepared"] = max(0, coverage["extract"]["expected_relevant_documents"] - coverage["extract"]["documents_with_prepared_jobs"])
    coverage["complete_classification"] = prepared == expected and coverage["relevance"].get("all_prepared_jobs_terminal_done_or_review_documents", 0) == expected
    coverage["complete_extraction_of_relevant"] = (coverage["complete_classification"]
        and coverage["extract"]["documents_not_prepared"] == 0
        and coverage["extract"].get("all_prepared_jobs_done_documents", 0) == coverage["extract"]["expected_relevant_documents"])

    facts, skipped, unsafe, suppressed_context_traits = {}, Counter(), 0, 0
    for fact in snapshot["facts"]:
        if fact.get("actor_id") not in actor_ids:
            skipped["unapproved_actor"] += 1
        elif fact.get("extraction_job_id") not in active_extract:
            skipped["inactive_or_incomplete_extraction"] += 1
        elif fact.get("id") in facts:
            if fact != facts[fact["id"]]:
                raise ValueError("Conflicting repeated fact")
        else:
            facts[fact["id"]] = fact
    position_config = dict(config, aspect="public_positions")
    eligible = {identifier for identifier, fact in facts.items() if _eligible_fact(fact, position_config)}
    cards, global_labels = [], {}
    for obj in objects:
        actor_facts = [fact for fact in facts.values() if fact["actor_id"] == obj["id"]]
        own = [fact for fact in actor_facts if fact["id"] in eligible]
        traits, kinds, channels, languages = {}, Counter(), Counter(), Counter()
        for fact in actor_facts:
            kinds[fact.get("kind", "unknown")] += 1
            job = active_extract[fact["extraction_job_id"]]
            context = job.get("context") or {}
            channel = context.get("document", {}).get("source_type", "unknown")
            channels[channel if channel in {"article", "post", "repost", "comment"} else "unknown"] += 1
            language = context.get("relevance", {}).get("language", "unknown")
            languages[language if language in {"rw", "en", "fr", "sw", "mixed", "other"} else "unknown"] += 1
            local = {}
            trait_values = fact.get("trait_candidates", [])
            if (fact.get("kind") == "institutional_context" or fact.get("measurement_status") == "context_only"
                    or fact.get("review_status") in {"rejected", "excluded", "unreviewed"}):
                # Keep the evidence census, but do not advertise office labels,
                # historical background or rejected identities as properties.
                suppressed_context_traits += len(trait_values)
                trait_values = []
            for value in trait_values:
                label = _label(value)
                if label is None:
                    unsafe += 1
                else:
                    local.setdefault(_normal(label), label)
            for key, label in local.items():
                row = traits.setdefault(key, {"label": label, "fact_count": 0, "eligible_public_position_fact_count": 0,
                                              "documents": set(), "eligible_documents": set()})
                row["fact_count"] += 1
                row["documents"].add(fact["document_id"])
                if fact["id"] in eligible:
                    row["eligible_public_position_fact_count"] += 1
                    row["eligible_documents"].add(fact["document_id"])
                aggregate = global_labels.setdefault(key, {"label": label, "fact_count": 0,
                    "eligible_public_position_fact_count": 0, "actors": set(), "eligible_actors": set()})
                aggregate["fact_count"] += 1
                aggregate["actors"].add(obj["id"])
                if fact["id"] in eligible:
                    aggregate["eligible_public_position_fact_count"] += 1
                    aggregate["eligible_actors"].add(obj["id"])
        trait_rows = []
        for trait in traits.values():
            trait_rows.append({**trait, "documents": len(trait["documents"]), "eligible_documents": len(trait["eligible_documents"])})
        trait_rows.sort(key=lambda r: (-r["eligible_public_position_fact_count"], -r["fact_count"], _normal(r["label"])))
        cards.append({"actor_id": obj["id"], "name": _label(obj["name"]) or obj["id"],
            "facts": len(actor_facts), "eligible_public_positions": len(own), "evidence_kinds": dict(sorted(kinds.items())),
            "source_coverage": {"fact_documents": len({f["document_id"] for f in actor_facts}),
                "eligible_public_position_documents": len({f["document_id"] for f in own}),
                "source_hosts": len({host for f in actor_facts if (host := _host(f.get("source_url")))}),
                "eligible_source_hosts": len({host for f in own if (host := _host(f.get("source_url")))}),
                "explicit_event_dates": sum(bool(f.get("event_date")) for f in actor_facts),
                "unknown_event_dates": sum(not f.get("event_date") for f in actor_facts),
                "facts_by_source_type": dict(sorted(channels.items())), "facts_by_detected_document_language": dict(sorted(languages.items()))},
            "trait_candidates": trait_rows})
    candidate_rows = [{**row, "actors": len(row["actors"]), "eligible_actors": len(row["eligible_actors"])} for row in global_labels.values()]
    candidate_rows.sort(key=lambda r: (-r["eligible_actors"], -r["eligible_public_position_fact_count"], _normal(r["label"])))
    scale_groups, dropped_scales, unsupported_scales = _scale_groups(corpus.get("draft_scales", []), facts, eligible)
    total = accounting["total"]
    cap = _number(manifest.get("budget_cap_usd", 70.))
    start = _number(manifest.get("start_spent_usd", 0.))
    warnings = []
    if len(objects) != 42:
        warnings.append("roster_count_differs_from_42")
    if prepared > expected:
        warnings.append("prepared_document_count_exceeds_selected_count")
    if start > total["estimated_usd"] + 1e-7:
        warnings.append("baseline_cost_exceeds_snapshot_receipts")
    return {"version": "half-processing-public-aggregates-v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "corpus_id": corpus["id"], "source_sha256": manifest.get("source_sha256"),
        "accounting": {"all_receipts": accounting, "active_recipes": active_accounting,
            "budget_cap_usd": cap, "start_spent_usd": start,
            "estimated_since_start_usd": round(total["estimated_usd"] - start, 8),
            "uncommitted_remaining_usd": round(max(0., cap - total["estimated_usd"] - total["outstanding_reserved_usd"]), 8),
            "unresolved_requests": sum(job.get("status") in {"running", "uncertain"} for job in jobs)},
        "coverage": coverage, "facts": {"active": len(facts), "eligible_public_positions": len(eligible),
            "skipped_snapshot_rows": dict(sorted(skipped.items())), "actor_count": len(objects),
            "actors_with_facts": sum(bool(card["facts"]) for card in cards),
            "actors_with_eligible_public_positions": sum(bool(card["eligible_public_positions"]) for card in cards)},
        "trait_candidates": candidate_rows, "draft_dictionary": scale_groups, "actor_cards": cards,
        "public_safety": {"source_texts_published": False, "urls_published": False, "quotes_published": False,
            "authors_published": False, "unsafe_candidate_labels_omitted": unsafe,
            "context_or_rejected_trait_occurrences_omitted": suppressed_context_traits,
            "drafts_without_eligible_evidence_omitted": unsupported_scales,
            "drafts_without_safe_labels_omitted": dropped_scales},
        "quality_review": _quality(audit, corpus["id"], manifest.get("source_sha256", corpus.get("source_sha256")), expected, facts),
        "warnings": warnings, "limitations": [
            "Это машинный черновик: найденные характеристики и группы синонимов не утверждены пользователем.",
            "Число фрагментов, публикаций и источников не является интенсивностью политического свойства.",
            "Повторы одного события и независимость источников требуют отдельной проверки; домены не равны независимым источникам.",
            "Отсутствие свидетельств и статусы review/error/uncertain не означают нулевое свойство политика.",
            "Объединены только буквальные совпадения названий и заявленных aliases; смысловая эквивалентность не установлена.",
            "Покрытие done относится к подготовленным заданиям; полный отбор и извлечение показаны отдельно.",
            "Расход рассчитан по API usage и зафиксированным тарифам; окончательную оплату сверяют с биллингом.",
            "В машинных таблицах названия сохранены на языке модели; русский редакционный список приложен отдельно, утверждение словаря ещё требуется."]}


def _cell(value):
    value = str(value).replace("\n", " ").replace("\r", " ").replace("<", "&lt;").replace(">", "&gt;")
    return re.sub(r"([\\`*_\[\]|])", r"\\\1", value)


def render(result):
    accounting, coverage = result["accounting"], result["coverage"]
    total, rel, ext = accounting["all_receipts"]["total"], coverage["relevance"], coverage["extract"]
    lines = ["# Половина корпуса Руанды: обработка и найденные характеристики", "",
        "[Основные результаты](CORPUS_HALF_FINDINGS_2026-10-17.md) · [Характеристики на русском](CORPUS_HALF_PROPERTIES_RU_2026-10-17.md) · [Качество](CORPUS_HALF_QUALITY_2026-10-17.md)", "",
        f"Выбрано **{coverage['selected_documents']:,} публикаций**. Проверка релевантности, включая ответы review, завершена для **{rel.get('all_prepared_jobs_terminal_done_or_review_documents', 0):,} документов**; извлечение done — для **{ext.get('all_prepared_jobs_done_documents', 0):,} документов**.", "",
        f"Учтённый расход: **${total['estimated_usd']:.4f}** из ${accounting['budget_cap_usd']:.2f}. Неразрешённых запросов: {accounting['unresolved_requests']}; зарезервировано **${total['outstanding_reserved_usd']:.4f}**. Свободный остаток после резерваций: **${accounting['uncommitted_remaining_usd']:.4f}**.", "",
        "Это сводка извлечённых свидетельств и черновых характеристик, а не итоговые оценки политиков. Полные публикации, цитаты, URL, авторы и ответы модели остаются локально.", "",
        "## Покрытие обработки", "", "| Показатель | Документов |", "|---|---:|",
        f"| Выбранная половина | {coverage['selected_documents']} |",
        f"| Есть подготовленные задания отбора | {rel['documents_with_prepared_jobs']} |",
        f"| Отбор пока не подготовлен | {rel['documents_not_prepared']} |",
        f"| Отбор done по всем подготовленным частям | {rel.get('all_prepared_jobs_done_documents', 0)} |",
        f"| Отбор содержит review | {rel.get('review_documents', 0)} |",
        f"| Отбор содержит error / uncertain / running / pending | {rel.get('error_documents', 0)} / {rel.get('uncertain_documents', 0)} / {rel.get('running_documents', 0)} / {rel.get('pending_documents', 0)} |",
        f"| Признаны содержательно относящимися к участникам | {rel.get('relevant_documents', 0)} |",
        f"| Подготовлено извлечение | {ext['documents_with_prepared_jobs']} |",
        f"| Извлечение done по всем подготовленным частям | {ext.get('all_prepared_jobs_done_documents', 0)} |",
        f"| Извлечение содержит review / error / uncertain / running / pending | {ext.get('review_documents', 0)} / {ext.get('error_documents', 0)} / {ext.get('uncertain_documents', 0)} / {ext.get('running_documents', 0)} / {ext.get('pending_documents', 0)} |", "",
        "Статусы разных частей одного документа могут пересекаться. Review обозначает необходимость проверки, error — техническую ошибку, uncertain — неразрешённый исход запроса. Они не характеризуют политическую позицию.", "",
        "## Расходы по этапам и моделям", "", "Включены все квитанции корпуса, в том числе неактивные версии и неуспешные ответы. Резервации показаны отдельно от учтённой стоимости.", "",
        "| Этап | Модель | Статус | Запросов | Входных токенов | Выходных токенов | Учтено, $ | Резерв, $ |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in accounting["all_receipts"]["by_stage_model_status"]:
        lines.append(f"| {row['stage']} | {row['model']} | {row['status']} | {row['jobs']} | {row['input_tokens']} | {row['output_tokens']} | {row['estimated_usd']:.6f} | {row['outstanding_reserved_usd']:.6f} |")
    quality = result["quality_review"]
    lines += ["", "## Проверка качества", ""]
    if not quality["available"]:
        lines += ["Отдельная проверка качества ещё не выполнена или её публичная сводка не подключена. Содержательная точность не установлена."]
    else:
        structural, normalization, semantic = quality["structural_checks"], quality["normalization"], quality["semantic_review"]
        lines += ["Подключена автоматическая структурная проверка: наличие цитат в источнике, координаты, связи с контекстом и исходные решения нормализации. Эти проверки не доказывают правильность смысла, атрибуции или датировки события.", "",
            "| Проверка | Количество |", "|---|---:|",
            f"| Принятых фрагментов проверено структурно | {structural['active_accepted_facts_checked']} |",
            f"| Фрагментов с замечаниями при повторной проверке | {structural['facts_with_recheck_issues']} |",
            f"| Проверено контекстов запросов | {structural['source_context_jobs_checked']} |",
            f"| Завершённых извлечений с сохранённым контролем нормализации | {structural['jobs_with_qc']} |",
            f"| Завершённых извлечений без такой квитанции | {structural['done_extract_jobs_missing_qc']} |"]
        for field, title in (("input_facts", "Предложено фрагментов до нормализации"), ("skipped_facts", "Отсечено нормализацией"),
                             ("downgraded_facts", "Изменён тип/допуск нормализацией"), ("cleared_dates", "Сняты неподтверждённые даты")):
            lines.append(f"| {title} | {normalization['totals'].get(field, 0)} |")
        if quality["snapshot_alignment"]["exact_fact_snapshot_matches"] is not True:
            lines += ["", "Точное совпадение аудиторского и текущего наборов свидетельств не подтверждено; эти счётчики не следует распространять на более поздние записи."]
        for title, codes in (("Замечания структурной проверки", quality["structural_issue_counts"]),
                             ("Замечания контекстов", quality["source_context_issue_counts"]),
                             ("Исходные причины нормализации", normalization["issue_counts"])):
            if codes:
                lines += ["", title + ": " + "; ".join(f"{_cell(code)} — {count}" for code, count in codes.items()) + "."]
        lines += ["", f"Для дополнительного чтения автоматически подготовлено {semantic['facts_selected_for_review']} фрагментов основной выборки, {semantic['risk_packet_facts']} фрагментов повышенного риска и {semantic['negative_packet_documents']} документов для проверки пропусков. Эти наборы могут пересекаться; создание пакета не означает его содержательной проверки. Итоги выполненного чтения источников приводятся в [отдельной сводке качества](CORPUS_HALF_QUALITY_2026-10-17.md). Оценка точности всего корпуса по целевой выборке не рассчитывается."]
    facts = result["facts"]
    lines += ["", "## Что найдено", "",
        f"Активных фрагментов: **{facts['active']}**; допустимых свидетельств собственных публичных заявлений/действий: **{facts['eligible_public_positions']}**. Они есть у **{facts['actors_with_eligible_public_positions']} из {facts['actor_count']}** участников.", "",
        "Ниже частоты коротких названий из trait_candidates. Объединены регистр, Unicode и пробелы; отрицания сохранены. Частоты относятся к фрагментам, не независимым событиям и не выраженности установки. Названия не утверждены.", "",
        "| Кандидат характеристики | Допустимых фрагментов позиции | Участников с такими опорами | Всего фрагментов, включая образ |", "|---|---:|---:|---:|"]
    for row in result["trait_candidates"][:60]:
        lines.append(f"| {_cell(row['label'])} | {row['eligible_public_position_fact_count']} | {row['eligible_actors']} | {row['fact_count']} |")
    if len(result["trait_candidates"]) > 60:
        lines += ["", f"Показаны первые 60 из {len(result['trait_candidates'])} буквальных названий; полный частотный список находится в JSON."]
    lines += ["", "## Черновой словарь этапа describe", "",
        "Группы ниже связаны только буквальными совпадениями названий/заявленных aliases. Это не новые факторы и не автоматическое утверждение семантической эквивалентности. Различающиеся определения требуют проверки.", "",
        "| Название и варианты | Черновиков | Опорных фрагментов | Допустимых опор позиции | Участников | Различаются определения |", "|---|---:|---:|---:|---:|---|"]
    for row in result["draft_dictionary"]:
        lines.append(f"| {_cell('; '.join(row['labels']))} | {row['draft_count']} | {row['supporting_facts']} | {row['eligible_public_position_facts']} | {row['supporting_actors']} | {'да' if row['definitions_differ'] else 'нет'} |")
    if not result["draft_dictionary"]:
        lines += ["", "Пока нет завершённых предложений describe; это не отсутствие характеристик у участников."]
    lines += ["", "## Все участники", "", "Порядок сохранён из утверждённого справочника. В каждой карточке сначала указаны все извлечённые фрагменты, затем допустимые опоры публичной позиции. Образ, приписанный другими, не подменяет собственную позицию."]
    for card in result["actor_cards"]:
        sources = card["source_coverage"]
        lines += ["", f"### {_cell(card['name'])}", "",
            f"Фрагментов: **{card['facts']}**, допустимых опор позиции: **{card['eligible_public_positions']}**. Публикаций с опорами позиции: {sources['eligible_public_position_documents']}; доменов: {sources['eligible_source_hosts']}. Фрагментов без явной даты события: {sources['unknown_event_dates']}."]
        candidates = [row for row in card["trait_candidates"] if row["eligible_public_position_fact_count"]]
        if candidates:
            lines += ["", "Кандидаты по допустимым опорам позиции: " + "; ".join(f"{_cell(row['label'])} — {row['eligible_public_position_fact_count']} фр." for row in candidates[:8]) + "."]
        elif card["eligible_public_positions"]:
            lines += ["", "Допустимые опоры есть, но короткие названия характеристик ещё не сформированы."]
        else:
            lines += ["", "Допустимых опор собственной публичной позиции пока нет. Это неизвестность в данном корпусе, а не нулевая выраженность свойств."]
    lines += ["", "## Границы результата", "", *["- " + note for note in result["limitations"]], ""]
    if result["warnings"]:
        lines += ["Диагностические предупреждения: " + ", ".join(result["warnings"]) + ".", ""]
    return "\n".join(lines)


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--audit", type=Path, help="Only public aggregate_for_review.json; never audit_private.json")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args(argv)
    path = args.snapshot or args.root / ".corpus_runtime" / "pilots" / CORPUS_ID / "half-70-v1" / "results_private.json"
    raw = path.read_bytes()
    if args.audit and args.audit.name == "audit_private.json":
        parser.error("--audit accepts only public aggregate_for_review.json, never audit_private.json")
    audit_raw = args.audit.read_bytes() if args.audit else None
    result = summarize(json.loads(raw.decode("utf-8-sig")), json.loads(audit_raw.decode("utf-8-sig")) if audit_raw else None)
    result["private_snapshot_sha256"] = hashlib.sha256(raw).hexdigest()
    if audit_raw is not None:
        result["public_audit_aggregate_sha256"] = hashlib.sha256(audit_raw).hexdigest()
    if args.publish:
        _write(args.root / "reports/rwanda_half_processing_2026_10_17.json", json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        _write(args.root / "research/rwanda_politicians_2021_2026/CORPUS_HALF_PROCESSING_2026-10-17.md", render(result))
    print(json.dumps({"published": args.publish, "selected_documents": result["coverage"]["selected_documents"],
                      "facts": result["facts"], "accounting": result["accounting"]["all_receipts"]["total"]}, ensure_ascii=False, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
