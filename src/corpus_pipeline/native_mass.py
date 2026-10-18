"""Local journal for native semantic reading; no model client or keyword scores.

The private TSV is entered after reading source passages in the conversation.
All measurements remain episode judgements until event/date adjudication and
profile aggregation. A queue entry alone is never counted as reviewed.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import json

from .scoring_preparation import digest


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def parse_annotations(paths, queue, dictionary):
    scales = {s["id"] for s in dictionary["scales"]}
    index = {int(r["case_id"][1:]): r for r in queue}
    result = {}
    for path in sorted(paths):
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            number, numbers, verdict, reason = line.split("|", 3)
            number = int(number)
            if number not in index or number in result:
                raise ValueError("Unknown or repeated native annotation")
            if verdict not in {"retain", "hold", "exclude", "identity_uncertain"} or not reason.strip():
                raise ValueError("A native verdict and explanation are required")
            ratings = []
            for value in numbers.split(",") if numbers else []:
                sid, value = value.split("=")
                if sid not in scales or not value.isdigit() or not 0 <= int(value) <= 5:
                    raise ValueError("Invalid approved property intensity")
                if any(r["scale_id"] == sid for r in ratings):
                    raise ValueError("Repeated property")
                ratings.append({"scale_id": sid, "value": int(value), "reason_ru": reason})
            if verdict != "retain" and ratings:
                raise ValueError("Held or excluded passages cannot supply ratings")
            result[number] = {**index[number], "verdict": verdict,
                              "ratings": ratings, "reason_ru": reason,
                              "all_38_considered": True,
                              "source_read_scope": "quote_and_attribution_passages"}
    return result


def checkpoint_changes(before, after):
    """Missingness changes are substantive even when zero intensity is involved."""
    old = {(c["actor_id"], c["scale_id"]): c["value"] for c in before}
    new = {(c["actor_id"], c["scale_id"]): c["value"] for c in after}
    if old.keys() != new.keys():
        raise ValueError("Compare the same complete roster and dictionary")
    changes = []
    for key, value in new.items():
        prior = old[key]
        if prior == value:
            continue
        status_changed = (prior is None) != (value is None)
        delta = abs(value - prior) if prior is not None and value is not None else None
        changes.append({"actor_id": key[0], "scale_id": key[1],
                        "before": prior, "after": value,
                        "absolute_change": delta,
                        "substantive": status_changed or delta >= .5})
    return {"changed_cells": changes,
            "substantive_changes": sum(c["substantive"] for c in changes),
            "new_known": sum(c["before"] is None for c in changes),
            "lost_known": sum(c["after"] is None for c in changes)}


def stopping_decision(checkpoints, *, rare_actor_queue_exhausted):
    recent = checkpoints[-2:]
    return bool(rare_actor_queue_exhausted and len(recent) == 2
                and all(c["reviewed_cases_in_block"] == 500
                        and c["substantive_changes"] == 0
                        and c.get("profiles_aggregated") is True for c in recent))


def audit_journal(root):
    root = Path(root)
    private = root / ".corpus_runtime/native_mass_2026_10_18"
    dossiers = read_json(root / ".corpus_runtime/evidence_dossiers_2026_10_18/dossiers_private.json")
    queue_file = read_json(private / "queue_private.json")
    dictionary = read_json(root / "research/rwanda_politicians_2021_2026/source_scales_v2/approved_scales.json")
    if queue_file["source_sha256"] != dossiers["source_facts_sha256"]:
        raise ValueError("Frozen evidence queue mismatch")
    queue = queue_file["rows"]
    if len({r["fact_id"] for r in queue}) != len(queue):
        raise ValueError("Queue repeats source fragments")
    annotations = parse_annotations(private.glob("review_*.tsv"), queue, dictionary)
    public_cases, private_cases = [], []
    for number, row in sorted(annotations.items()):
        fact = dossiers["private_fact_index"][row["fact_id"]]["fact"]
        if not dossiers["private_fact_index"][row["fact_id"]]["retrieval_eligible"]:
            raise ValueError("Native queue contains a non-candidate")
        safe = {k: v for k, v in row.items() if k != "fact_id"}
        safe["actor_id"] = fact["actor_id"]
        safe["all_38_considered"] = len(dictionary["scales"]) == 38
        # Quote validation is literal; no claim that the whole document was read.
        if not fact.get("quote"):
            raise ValueError("Missing original source passage")
        private_cases.append({**row, "fact_content_sha256": digest(fact),
                              "support_quote": fact["quote"],
                              "attribution_passage": fact["attribution_evidence"],
                              "source_document_id": fact["document_id"]})
        public_cases.append(safe)
    initial = sum(r["phase"] == "shortlist" for r in annotations.values())
    summary = {"version": "native-mass-semantic-journal-v1",
               "dictionary_content_sha256": digest(dictionary),
               "queue_content_sha256": digest(queue_file),
               "native_journal_content_sha256": digest(private_cases),
               "queued_fragments": len(queue), "initial_shortlist": 751,
               "reviewed_shortlist": initial,
               "reviewed_remainder": len(annotations) - initial,
               "reviewed_fragments": len(annotations),
               "remaining_fragments": len(queue) - len(annotations),
               "provisional_fragment_property_ratings": sum(len(r["ratings"]) for r in public_cases),
               "verdicts": dict(Counter(r["verdict"] for r in public_cases)),
               "cases": public_cases, "api_calls": 0, "new_api_cost_usd": 0,
               "raw_quotes_published": False,
               "profile_aggregation_completed": False,
               "factor_analysis_performed": False,
               "mass_scoring_completed": False,
               "early_stopping_demonstrated": False,
               "limitations_ru": ["Числа относятся к фрагментам, а не к итоговым личностным профилям.",
                                  "Прочитаны цитаты и контекст атрибуции, не полные статьи.",
                                  "Объединение эпизодов и восстановление времени требуют отдельного проверяемого решения."]}
    (private / "validated_native_journal_private.json").write_text(
        json.dumps(private_cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def review_bundle(summary, dossiers, dictionary, *, last_case=None, timing_overrides=()):
    """Adapt native passage judgements to the existing conservative aggregator.

    Previously source-adjudicated dates are reused. New dates are not silently
    promoted from publication metadata. No lexical rule creates intensities.
    """
    queue_by_case = {r["case_id"]: r for r in dossiers["queue_rows"]}
    overrides = {r["case_id"]: r for r in timing_overrides}
    if len(overrides) != len(timing_overrides):
        raise ValueError("Repeated native time decision")
    cases = []
    for row in summary["cases"]:
        if last_case is not None and int(row["case_id"][1:]) > last_case:
            continue
        fact = dossiers["private_fact_index"][queue_by_case[row["case_id"]]["fact_id"]]["fact"]
        exact = fact.get("event_date") if fact.get("correction_origin") == "exact_source_adjudication" else None
        timing = {"status": "supported_exact" if exact else "unresolved", "date": exact,
                  "date_from": None, "date_to": None,
                  "explanation_ru": "Повторно использована дата ранее проверенного источника." if exact else
                  "Время нового эпизода не восстановлено из одной даты публикации."}
        if row["case_id"] in overrides:
            from datetime import date
            override = overrides[row["case_id"]]
            a, b = override["date_from"], override["date_to"]
            if date.fromisoformat(a) > date.fromisoformat(b) or not override["reason_ru"]:
                raise ValueError("Invalid native time decision")
            timing = {"status": "supported_exact" if a == b else "supported_interval",
                      "date": a if a == b else None, "date_from": a if a != b else None,
                      "date_to": b if a != b else None, "explanation_ru": override["reason_ru"]}
        cases.append({"case_id": row["case_id"], "actor_id": row["actor_id"],
                      "sample_kind": "selected_unit", "verdict": "retain" if row["verdict"] == "retain" else "hold",
                      "summary_ru": row["reason_ru"],
                      "attribution": {"status": "supported" if row["verdict"] == "retain" else "unresolved",
                                      "basis": fact["attribution_basis"]},
                      "capacity_review_ru": "Роль относится к источнику; международный мандат не переносится на правительство Руанды.",
                      "event_date": timing,
                      "event_identity": {"status": "single_document_event", "related_cases": [],
                                         "explanation_ru": "Междокументная независимость не подразумевается."},
                      "ratings": row["ratings"]})
    return {"measurement_dictionary_approved": True,
            "dictionary_content_sha256": digest(dictionary),
            "raw_quotes_published": False, "cases": cases}
