"""Offline audit census and actor-balanced review packets; never calls an API.

All output stays private by default. Only ``public_aggregate`` is suitable for
publication: it is an explicit allowlist without document IDs, URLs or source
prose. Structural validity is not semantic accuracy. Review packets are pending
purposive checks, not a random precision/recall estimate or automatic decisions.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import closing
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.corpus_pipeline.evidence import normalize_extraction
from src.corpus_pipeline.models import SCHEMAS, validate_result
from src.corpus_pipeline.service import chunks
from src.corpus_pipeline.source_spans import document_spans

AUDIT_POLICY_VERSION = "offline-fact-and-negative-review-v1"
_LANGUAGES = {"en", "fr", "rw", "sw", "mixed", "other", "unknown"}
_SOURCE_TYPES = {"article", "post", "comment", "repost", "unknown"}
_KINDS = {"public_statement", "action", "attributed_image", "institutional_context"}
_JOB_STATUSES = {"pending", "running", "done", "review", "error", "uncertain", "obsolete"}
_FACT_KEYS = tuple(SCHEMAS["extract"]["properties"]["facts"]["items"]["properties"])
_CHECKLIST = ["actor_identity", "speaker_and_attribution", "quote_entails_claim",
              "own_position_vs_image_or_role", "date_of_this_event", "planned_vs_completed",
              "same_event_or_reprint", "intensity_not_frequency"]


def _hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _month(document):
    value = str(document.get("published_at") or "")
    return value[:7] if re.match(r"^\d{4}-\d{2}-", value) else "unknown"


def _label(value, allowed):
    return value if isinstance(value, str) and value in allowed else "unknown"


def _signature(job):
    source = job["context"]["document"]
    try:
        return tuple((span["start"], span["end"]) for span in document_spans(source))
    except ValueError:
        return None


def _source_parts(document, fact, context_document=None):
    """Separate, bounded original passages; never manufacture a joined quote."""
    text = document.get("text", "")
    ranges = []
    try:
        candidates = document_spans(context_document) if context_document is not None else []
    except ValueError:
        candidates = []
    source_span = next((span for span in candidates if type(fact.get("quote_start")) is int
                        and span["start"] <= fact["quote_start"] < span["end"]), None)
    for key in ("quote", "attribution_evidence"):
        value = fact.get(key)
        if isinstance(value, str) and value:
            local = source_span["text"].find(value) if source_span else -1
            start = source_span["start"] + local if local >= 0 else text.find(value)
            if key == "quote" and type(fact.get("quote_start")) is int:
                start = fact["quote_start"]
            if start >= 0:
                ranges.append((max(0, start - 220), min(len(text), start + min(len(value), 1600) + 220)))
    merged = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1] and end - merged[-1][0] <= 2400:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return [{"start": start, "end": end, "text": text[start:end]} for start, end in merged]


def _family_map(documents, similar):
    parent = {identifier: identifier for identifier in documents}
    def find(identifier):
        while parent[identifier] != identifier:
            parent[identifier] = parent[parent[identifier]]
            identifier = parent[identifier]
        return identifier
    def union(left, right):
        if left not in parent or right not in parent:
            return False
        a, b = sorted((find(left), find(right)))
        parent[b] = a
        return True
    invalid = 0
    for document in documents.values():
        if document.get("duplicate_of") and not union(document["id"], document["duplicate_of"]):
            invalid += 1
    for pair in similar:
        if not union(pair.get("left_id"), pair.get("right_id")):
            invalid += 1
    return {identifier: find(identifier) for identifier in parent}, invalid


def build_audit(snapshot, documents, *, selection=None, similar=None, seed="half-review-v1",
                facts_per_actor=5, max_per_document=2, negatives_per_stratum=2, risk_limit=80):
    """Pure audit over the frozen old/new pilot snapshot shape and original docs.

    snapshot contains corpus, jobs (explicit active_recipe booleans), and merged
    active facts. Original docs include id/text and source metadata. selection,
    if supplied, is the frozen sampling manifest with documents[id, selected].
    No original receipt, source, annotation, score or queue is modified.
    """
    for value in (facts_per_actor, max_per_document, negatives_per_stratum, risk_limit):
        if type(value) is not int or value < 1:
            raise ValueError("Review limits must be positive integers")
    corpus = snapshot["corpus"]
    config = corpus["config"]
    objects = {actor["id"]: actor for actor in config["objects"]}
    if len(objects) != len(config["objects"]):
        raise ValueError("Duplicate approved actor IDs")
    by_doc = {document["id"]: document for document in documents}
    if len(by_doc) != len(documents):
        raise ValueError("Duplicate source document IDs")
    if snapshot.get("manifest"):
        for field, expected in (("corpus_id", corpus["id"]), ("source_sha256", corpus["source_sha256"])):
            if snapshot["manifest"].get(field, expected) != expected:
                raise ValueError("Snapshot manifest and corpus/source mismatch")
    jobs = snapshot["jobs"]
    if any(type(job.get("active_recipe")) is not bool for job in jobs):
        raise ValueError("Each frozen job must declare active_recipe")
    if len({job["id"] for job in jobs}) != len(jobs):
        raise ValueError("Duplicate job IDs")
    active = [job for job in jobs if job["active_recipe"]]
    extracts = {job["id"]: job for job in active if job["stage"] == "extract" and job["status"] == "done"}
    by_doc_jobs = defaultdict(lambda: defaultdict(list))
    for job in active:
        if job["stage"] in {"relevance", "extract"}:
            by_doc_jobs[job["context"]["document"]["id"]][job["stage"]].append(job)
    if selection is not None:
        if selection.get("corpus_id", corpus["id"]) != corpus["id"] or selection.get("source_sha256", corpus["source_sha256"]) != corpus["source_sha256"]:
            raise ValueError("Selection and frozen source/corpus mismatch")
        if len({row["id"] for row in selection["documents"]}) != len(selection["documents"]):
            raise ValueError("Duplicate IDs in frozen selection")
        selected_ids = {row["id"] for row in selection["documents"] if row["selected"]}
        frame_basis = "frozen_selected_documents"
    else:
        selected_ids = set(by_doc_jobs)
        frame_basis = "active_job_documents_only"
    if not selected_ids <= by_doc.keys():
        raise ValueError("Original sources missing for selected documents")

    families, invalid_pairs = _family_map(by_doc, similar or [])
    raw_facts = snapshot.get("facts", [])
    if len({fact["id"] for fact in raw_facts}) != len(raw_facts):
        raise ValueError("Duplicate fact IDs")
    facts = [fact for fact in raw_facts if fact.get("extraction_job_id") in extracts]
    issues, codes, fact_issues = [], Counter(), defaultdict(list)
    source_codes = defaultdict(list)
    source_checked = [job for job in active if job["stage"] in {"relevance", "extract"}]
    for job in source_checked:
        source = job["context"]["document"]
        original = by_doc.get(source["id"])
        try:
            spans = document_spans(source)
        except ValueError:
            source_codes[job["id"]].append("invalid_source_spans")
            continue
        if original is None:
            source_codes[job["id"]].append("original_source_missing")
        elif any(original.get("text", "")[span["start"]:span["end"]] != span["text"] for span in spans):
            source_codes[job["id"]].append("source_span_original_mismatch")

    def issue(fact, code):
        if code not in fact_issues[fact["id"]]:
            fact_issues[fact["id"]].append(code)
            codes[code] += 1
            issues.append({"fact_id": fact["id"], "document_id": fact.get("document_id"), "code": code})

    for fact in facts:
        job = extracts[fact["extraction_job_id"]]
        context = job["context"]
        source = context["document"]
        original = by_doc.get(fact.get("document_id"))
        if fact.get("actor_id") not in objects:
            issue(fact, "actor_outside_roster")
        if fact.get("document_id") != source["id"]:
            issue(fact, "fact_job_document_mismatch")
        if fact.get("document_id") not in selected_ids:
            issue(fact, "fact_outside_selected_frame")
        for code in source_codes[job["id"]]:
            issue(fact, code)
        try:
            normalized = normalize_extraction({"facts": [{key: fact.get(key) for key in _FACT_KEYS}]}, context, validate_result)
            for row in normalized["quality_control"]["issues"]:
                issue(fact, "recheck_" + row["code"])
            if not normalized["facts"]:
                issue(fact, "recheck_rejected")
        except (KeyError, TypeError, ValueError):
            issue(fact, "recheck_invalid_context_or_fact")
        start, end = fact.get("quote_start"), fact.get("quote_end")
        if original is None:
            issue(fact, "original_source_missing")
        elif (type(start) is not int or type(end) is not int or start < 0 or end < start
              or original.get("text", "")[start:end] != fact.get("quote")):
            issue(fact, "quote_global_coordinates_mismatch")
        if fact.get("source_url") != source.get("url", ""):
            issue(fact, "fact_job_url_mismatch")

    qc_totals, qc_issues = Counter(), Counter()
    qc_jobs = missing_qc = 0
    for job in extracts.values():
        qc = (job.get("result") or {}).get("quality_control")
        if not isinstance(qc, dict):
            missing_qc += 1
            continue
        qc_jobs += 1
        for field in ("input_facts", "skipped_facts", "downgraded_facts", "cleared_dates"):
            value = qc.get(field, 0)
            if type(value) is not int or value < 0:
                raise ValueError("Invalid frozen QC counter")
            qc_totals[field] += value
        for item in qc.get("issues", []):
            code = item.get("code")
            qc_issues[code if isinstance(code, str) and re.fullmatch(r"[a-z0-9_]+", code) else "unknown"] += 1

    doc_meta, funnel, negatives = {}, Counter(), defaultdict(list)
    context_counts = Counter()
    context_flags = {key: Counter({"known_true": 0, "known_false": 0, "unknown": 0})
                     for key in ("expanded_documents", "truncated_documents", "no_mention_head_fallback_documents")}
    source_characters = selected_characters = 0
    document_fact_count = Counter(fact.get("document_id") for fact in facts)
    bounded = config.get("text_selection", {}).get("mode", "full") != "full"
    for identifier in sorted(selected_ids):
        document = by_doc[identifier]
        rel = by_doc_jobs[identifier]["relevance"]
        ext = by_doc_jobs[identifier]["extract"]
        expected = 1 if bounded else len(list(chunks(document.get("text", ""))))
        rel_signatures = [_signature(job) for job in rel]
        expected_signatures = {((start, end),) for start, end, _ in chunks(document.get("text", ""))}
        rel_complete = (len(rel) == expected and len(set(rel_signatures)) == expected
                        and all(job["status"] in {"done", "review"} and not source_codes[job["id"]] for job in rel)
                        and (bounded or set(rel_signatures) == expected_signatures))
        relevant = [job for job in rel if job["status"] == "done" and (job.get("result") or {}).get("status") == "relevant"]
        is_review = any(job["status"] == "review" or (job.get("result") or {}).get("status") == "review" for job in rel)
        status = ("relevant" if relevant else "review" if is_review else "irrelevant" if rel_complete
                  and rel and all((job.get("result") or {}).get("status") == "irrelevant" for job in rel) else "incomplete")
        ext_complete = (bool(relevant) and rel_complete and len(ext) == len(relevant)
                        and {_signature(job) for job in ext} == {_signature(job) for job in relevant}
                        and all(job["status"] == "done" and not source_codes[job["id"]] for job in ext))
        actual_languages = {_label((job.get("result") or {}).get("language"), _LANGUAGES) for job in rel if job["status"] in {"done", "review"}}
        language = ("mixed" if "mixed" in actual_languages or len(actual_languages - {"unknown", "other"}) > 1
                    else next(iter(sorted(actual_languages - {"unknown", "other"})),
                              "other" if "other" in actual_languages else "unknown"))
        source_type = _label(next((job["context"]["document"].get("source_type") for job in rel if job.get("context")), None), _SOURCE_TYPES)
        source_characters += len(document.get("text", ""))
        ranges = []
        metadata = []
        for job in rel:
            source = job["context"]["document"]
            metadata.append(source.get("text_selection", {}))
            try:
                ranges.extend((span["start"], span["end"]) for span in document_spans(source))
            except ValueError:
                pass
        merged_ranges = []
        for start, end in sorted(ranges):
            if merged_ranges and start <= merged_ranges[-1][1]:
                merged_ranges[-1] = (merged_ranges[-1][0], max(end, merged_ranges[-1][1]))
            else:
                merged_ranges.append((start, end))
        selected_characters += sum(end - start for start, end in merged_ranges)
        context_counts["documents_with_relevance_context"] += bool(rel)
        for name, key in (("expanded_documents", "expanded"), ("truncated_documents", "truncated"),
                          ("no_mention_head_fallback_documents", "no_mentions_fallback")):
            state = ("known_true" if any(row.get(key) is True for row in metadata)
                     else "known_false" if metadata and all(row.get(key) is False for row in metadata)
                     else "unknown")
            context_flags[name][state] += 1
        doc_meta[identifier] = {"status": status, "language": language, "source_type": source_type,
                                "month": _month(document), "relevance_complete": rel_complete,
                                "extraction_complete": ext_complete,
                                "language_basis": "classification_of_supplied_passages",
                                "actor_ids": sorted({actor_id for job in rel for actor_id in (job.get("result") or {}).get("actor_ids", [])
                                                     if actor_id in objects})}
        funnel[status] += 1
        category = ("relevant_no_facts" if status == "relevant" and ext_complete and not document_fact_count[identifier]
                    else status if status in {"review", "irrelevant"} and rel_complete else None)
        if category:
            negatives[(category, language, source_type)].append(identifier)

    actor_facts = defaultdict(list)
    for fact in facts:
        if fact.get("actor_id") in objects and fact.get("document_id") in doc_meta:
            actor_facts[fact["actor_id"]].append(fact)
    taken_docs, chosen, actor_rows = Counter(), [], []
    for actor_id in sorted(objects, key=lambda identifier: (len(actor_facts[identifier]), identifier)):
        candidates = list(actor_facts[actor_id])
        seen = {dimension: set() for dimension in ("family", "kind", "language", "source_type", "month")}
        selected_facts = []
        while candidates and len(selected_facts) < facts_per_actor:
            available = [fact for fact in candidates if taken_docs[fact["document_id"]] < max_per_document]
            if not available:
                break
            def priority(fact):
                meta = doc_meta[fact["document_id"]]
                labels = dict(meta, family=families[fact["document_id"]], kind=fact.get("kind"))
                score = sum((4 if dimension == "family" else 1) for dimension in seen if labels[dimension] not in seen[dimension])
                return -score, _hash([seed, fact["id"]])
            fact = min(available, key=priority)
            candidates.remove(fact)
            selected_facts.append(fact)
            taken_docs[fact["document_id"]] += 1
            meta = doc_meta[fact["document_id"]]
            labels = dict(meta, family=families[fact["document_id"]], kind=fact.get("kind"))
            for dimension in seen:
                seen[dimension].add(labels[dimension])
        chosen.extend(selected_facts)
        actor_rows.append({"actor_id": actor_id, "name": objects[actor_id]["name"],
                           "available_facts": len(actor_facts[actor_id]), "selected_for_review": len(selected_facts),
                           "available_documents": len({fact["document_id"] for fact in actor_facts[actor_id]}),
                           "candidate_editorial_families": len({families[fact["document_id"]] for fact in actor_facts[actor_id]})})

    def packet(fact, reason):
        document = by_doc.get(fact.get("document_id"), {})
        return {"fact_id": fact["id"], "document_id": fact.get("document_id"),
                "actor_id": fact.get("actor_id"), "name": objects.get(fact.get("actor_id"), {}).get("name"),
                "kind": fact.get("kind"), "attribution_basis": fact.get("attribution_basis"),
                "quote": fact.get("quote"), "claim": fact.get("claim"),
                "attribution_evidence": fact.get("attribution_evidence"), "date_evidence": fact.get("date_evidence"),
                "event_date": fact.get("event_date"), "publication_date": document.get("published_at"),
                "source_url": fact.get("source_url"),
                "source_parts": _source_parts(document, fact, extracts[fact["extraction_job_id"]]["context"]["document"]),
                "source_metadata": doc_meta.get(fact.get("document_id")),
                "structural_issues": sorted(fact_issues[fact["id"]]), "selection_reason": reason,
                "review_status": "pending", "checklist": {key: None for key in _CHECKLIST}}

    risk_candidates = []
    for fact in facts:
        meta = doc_meta.get(fact.get("document_id"), {})
        risk = list(fact_issues[fact["id"]])
        if fact.get("event_date"):
            risk.append("asserted_event_date")
        if fact.get("kind") == "action" or fact.get("temporal_status") == "planned":
            risk.append("action_or_plan")
        if meta.get("source_type") in {"comment", "repost", "unknown"}:
            risk.append("non_article_attribution")
        if meta.get("language") in {"rw", "sw", "mixed", "unknown"}:
            risk.append("language_coverage_check")
        if risk:
            risk_candidates.append((fact, sorted(set(risk))))
    risk_candidates.sort(key=lambda item: (-bool(fact_issues[item[0]["id"]]), -len(item[1]), _hash([seed, item[0]["id"]])))
    risk_rows, risk_by_actor, risk_by_doc = [], Counter(), Counter()
    for fact, reasons in risk_candidates:
        if len(risk_rows) >= risk_limit:
            break
        if risk_by_actor[fact.get("actor_id")] >= facts_per_actor or risk_by_doc[fact.get("document_id")] >= max_per_document:
            continue
        risk_rows.append(packet(fact, reasons))
        risk_by_actor[fact.get("actor_id")] += 1
        risk_by_doc[fact.get("document_id")] += 1
    negative_packet, negative_actor_counts = [], Counter()
    for stratum in sorted(negatives):
        remaining = list(negatives[stratum])
        for _ in range(min(len(remaining), negatives_per_stratum)):
            identifier = min(remaining, key=lambda identifier: (
                min((negative_actor_counts[actor_id] for actor_id in doc_meta[identifier]["actor_ids"]), default=0),
                _hash([seed, identifier])))
            remaining.remove(identifier)
            negative_packet.append({"document_id": identifier, "stratum": list(stratum),
                                    "review_status": "pending", "source_url": by_doc[identifier].get("url"),
                                    "actor_ids": doc_meta[identifier]["actor_ids"],
                                    "source_text_private": by_doc[identifier].get("text", ""),
                                    "questions": {"relevant_actor_missed": None, "usable_claim_missed": None,
                                                  "selected_context_insufficient": None, "language_error": None}})
            negative_actor_counts.update(doc_meta[identifier]["actor_ids"])
    actor_rows.sort(key=lambda row: row["actor_id"])
    statuses = {stage: dict(sorted(Counter(_label(job["status"], _JOB_STATUSES) for job in active if job["stage"] == stage).items()))
                for stage in ("relevance", "extract", "describe", "score")}
    pending = sum(job["status"] in {"pending", "running", "error", "uncertain"} for job in active)
    repeated_quotes = Counter(_hash([fact.get("actor_id"), fact.get("kind"), fact.get("attribution_basis"),
                                    fact.get("quote")]) for fact in facts)
    facts_snapshot_sha = _hash(sorted(facts, key=lambda fact: fact["id"]))
    public = {
        "audit_policy_version": AUDIT_POLICY_VERSION, "corpus_id": corpus["id"],
        "source_sha256": corpus["source_sha256"], "api_calls": 0,
        "provenance": {"facts_snapshot_sha256": facts_snapshot_sha},
        "scope": {"selected_document_frame": len(selected_ids), "frame_basis": frame_basis,
                  "active_stage_statuses": statuses, "unfinished_active_jobs": pending,
                  "snapshot_is_partial": bool(pending or any(not meta["relevance_complete"] or meta["status"] == "relevant"
                                              and not meta["extraction_complete"] for meta in doc_meta.values()))},
        "structural_census": {
            "active_accepted_facts_checked": len(facts), "inactive_facts_ignored": len(raw_facts) - len(facts),
            "facts_with_recheck_issues": len({item["fact_id"] for item in issues}),
            "issue_counts": dict(sorted(codes.items())), "done_extract_jobs_missing_qc": missing_qc,
            "source_context_jobs_checked": len(source_checked),
            "source_context_issue_counts": dict(sorted(Counter(code for values in source_codes.values() for code in values).items())),
            "jobs_with_qc": qc_jobs, "original_normalization_totals": dict(sorted(qc_totals.items())),
            "original_normalization_issue_counts": dict(sorted(qc_issues.items())),
            "note": "Exact quotation/locality/coordinates are necessary checks, not verified attribution, truth or semantic accuracy."},
        "accepted_facts": {"by_kind": dict(sorted(Counter(_label(fact.get("kind"), _KINDS) for fact in facts).items())),
                           "event_date_known": sum(fact.get("event_date") is not None for fact in facts),
                           "event_date_unknown": sum(fact.get("event_date") is None for fact in facts),
                           "note": "Unknown event date is legitimate; publication date is not substituted."},
        "whole_document_funnel": dict(sorted(funnel.items())),
        "context_coverage": {**dict(context_counts),
                             **{name: counts["known_true"] if not counts["unknown"] else None
                                for name, counts in context_flags.items()},
                             "metadata_availability": {name: dict(counts) for name, counts in context_flags.items()},
                             "full_source_characters_in_selected_frame": source_characters,
                             "union_characters_in_supplied_relevance_contexts": selected_characters,
                             "note": "Missing selection metadata is unknown, not zero. Character coverage is not semantic recall; source passage coordinates remain verifiable."},
        "semantic_review": {"status": "pending_human_or_assistant_source_review", "facts_selected": len(chosen),
                            "per_actor_target": facts_per_actor, "global_fact_cap_per_document": max_per_document,
                            "actors": actor_rows, "zero_fact_actor_ids": sorted(actor_id for actor_id in objects if not actor_facts[actor_id]),
                            "risk_packet_facts": len(risk_rows), "negative_packet_documents": len(negative_packet),
                            "negative_strata": [{"category": key[0], "language": key[1], "source_type": key[2],
                                                 "available": len(values), "selected": min(len(values), negatives_per_stratum)}
                                                for key, values in sorted(negatives.items())],
                            "accuracy_estimate": None,
                            "note": "Purposive actor-balanced and risk/negative packets do not estimate population precision or recall."},
        "duplicate_context": {"candidate_links_supplied": len(similar or []), "invalid_links": invalid_pairs,
                              "documents_with_facts": len({fact.get("document_id") for fact in facts}),
                              "candidate_editorial_families_with_facts": len({families.get(fact.get("document_id")) for fact in facts
                                                                             if fact.get("document_id") in families}),
                              "repeated_literal_quote_groups": sum(count > 1 for count in repeated_quotes.values()),
                              "extra_records_in_repeated_literal_quote_groups": sum(count - 1 for count in repeated_quotes.values() if count > 1),
                              "note": "Candidate editorial families guide diversity only; they are not verified independent events or an instruction to delete."},
        "limitations": ["Only supplied frozen active receipts and original sources are audited; no new search or API call.",
                        "A quotation can be exact while actor attribution, translated meaning, claim or date semantics are wrong.",
                        "Review/unknown and zero facts do not mean political neutrality or no public position.",
                        "Source/type/language metadata are noisy; no language group is automatically excluded.",
                        "The half sample oversamples non-primary actors; frequencies are not population support, and intensity is not article/event count."]}
    return {"public_aggregate": public,
            "semantic_review_packet": [packet(fact, "actor_balanced_diversity") for fact in chosen],
            "risk_review_packet": risk_rows, "negative_review_packet": negative_packet,
            "structural_issues_private": issues,
            "provenance": {"facts_snapshot_sha256": facts_snapshot_sha,
                           "selection_sha256": _hash(selection) if selection is not None else None,
                           "audit_seed": seed, "model_receipts_modified": False, "api_calls": 0}}


def _file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    with args.snapshot.open(encoding="utf-8-sig") as stream:
        snapshot = json.load(stream)
    # Large request bodies must never be copied to the output packets.
    for job in snapshot["jobs"]:
        job.pop("payload", None)
    selection = json.loads(args.selection.read_text(encoding="utf-8-sig")) if args.selection else None
    corpus = snapshot["corpus"]
    database = args.root.resolve() / ".corpus_runtime/corpus.sqlite3"
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        db.execute("BEGIN")
        stored = db.execute("SELECT source_sha256 FROM corpora WHERE id=?", (corpus["id"],)).fetchone()
        if stored is None or stored[0] != corpus["source_sha256"]:
            raise ValueError("Original database source and frozen snapshot differ")
        documents = [dict(json.loads(row["data"]), id=row["id"], duplicate_of=row["duplicate_of"])
                     for row in db.execute("SELECT id,data,duplicate_of FROM documents WHERE corpus_id=?", (corpus["id"],))]
        similar = [json.loads(row[0]) for row in db.execute("SELECT data FROM similar WHERE corpus_id=?", (corpus["id"],))]
    result = build_audit(snapshot, documents, selection=selection, similar=similar)
    result["provenance"].update(snapshot_file_sha256=_file_hash(args.snapshot),
                                audit_source_sha256=_file_hash(Path(__file__)))
    output = args.output_dir or args.snapshot.parent / "offline_audit"
    _write(output / "audit_private.json", result)
    _write(output / "aggregate_for_review.json", result["public_aggregate"])
    print(json.dumps({"private_directory": str(output), "facts_checked": result["public_aggregate"]["structural_census"]["active_accepted_facts_checked"],
                      "structural_issue_facts": result["public_aggregate"]["structural_census"]["facts_with_recheck_issues"],
                      "semantic_review_pending": True, "published": False, "api_calls": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
