"""Offline full-v2 census summary; publish aggregates only with --publish.

No credentials, task preparation or API requests. The SQLite connection is
read-only. Current recipes are the frozen snapshot flags, never recalculated.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.corpus_pipeline.service import CorpusPipeline, chunks
from scripts.summarize_corpus_pilot import _ledger

STAGES = ("relevance", "extract", "describe", "score")


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _json_hash(value):
    return _sha(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8"))


def _span(job):
    document = job.get("context", {}).get("document", {})
    return document.get("chunk_start"), document.get("chunk_end")


def _span_complete(grouped, expected, statuses):
    return bool(expected) and set(grouped) == expected and all(
        len(grouped[s]) == 1 and grouped[s][0]["status"] in statuses for s in expected)


def _qc(jobs, stage):
    fields = ("input_facts", "skipped_facts", "downgraded_facts", "cleared_dates") if stage == "extract" else ("input_scales", "skipped_scales")
    result = Counter({field: 0 for field in fields})
    codes = Counter()
    present = missing = 0
    for job in jobs:
        if job["stage"] != stage or job["status"] != "done":
            continue
        qc = (job.get("result") or {}).get("quality_control")
        if not isinstance(qc, dict):
            missing += 1
            continue
        present += 1
        for field in fields:
            value = qc.get(field, 0)
            if type(value) is not int or value < 0:
                raise ValueError("Invalid normalization QA counter")
            result[field] += value
        for item in qc.get("issues", []):
            codes[str(item.get("code", "unknown"))] += 1
    return {"jobs_with_qc": present, "done_jobs_missing_qc": missing,
            "totals": dict(result), "issue_counts": dict(sorted(codes.items()))}


def build_summary(manifest, results, docs, similar):
    """Pure whole-document aggregation over ALL records, including exact copies.

    docs must include id/text/status/duplicate_of; source metadata may be present
    but is never copied into the public allowlist. Actor IDs/names are the public
    approved roster; document/account identifiers and source prose stay private.
    """
    corpus = results["corpus"]
    if (manifest.get("version") != "full-v2" or manifest["corpus_id"] != corpus["id"]
            or manifest["source_sha256"] != corpus["source_sha256"]
            or results["manifest"]["corpus_id"] != manifest["corpus_id"]
            or results["manifest"]["source_sha256"] != manifest["source_sha256"]):
        raise ValueError("Full-pilot corpus/source snapshot mismatch")
    config, jobs = corpus["config"], results["jobs"]
    if any(type(j.get("active_recipe")) is not bool for j in jobs):
        raise ValueError("Every snapshot job must declare active_recipe")
    by_id = {d["id"]: d for d in docs}
    raw = corpus["profile"]["rows_total"]
    if len(by_id) != len(docs) or len(docs) != raw or raw <= 0:
        raise ValueError("All unique record IDs, including exact copies/empty rows, are required")
    if len({j["id"] for j in jobs}) != len(jobs):
        raise ValueError("Duplicate job receipt IDs")
    active = [j for j in jobs if j["active_recipe"]]
    by_doc = defaultdict(lambda: defaultdict(list))
    flags = Counter()
    for job in active:
        if job["stage"] in {"relevance", "extract"}:
            identifier = job["context"]["document"]["id"]
            if identifier not in by_id:
                raise ValueError("Active job refers to an unknown document")
            by_doc[identifier][job["stage"]].append(job)

    parent = {identifier: identifier for identifier in by_id}
    def find(identifier):
        while parent[identifier] != identifier:
            parent[identifier] = parent[parent[identifier]]
            identifier = parent[identifier]
        return identifier
    def union(a, b):
        if a not in parent or b not in parent:
            raise ValueError("Duplicate candidate refers to an unknown document")
        parent[find(b)] = find(a)
    for d in docs:
        if d.get("duplicate_of"):
            union(d["id"], d["duplicate_of"])
    for pair in similar:
        union(pair["left_id"], pair["right_id"])

    funnel = Counter()
    exclusions = Counter()
    languages, dominant_languages = Counter(), Counter()
    diagnostics = []
    eligible = relevance_complete_docs = extraction_complete_docs = 0
    expected_extract_docs = 0
    for d in docs:
        identifier = d["id"]
        rel, ext = by_doc[identifier]["relevance"], by_doc[identifier]["extract"]
        expected = {(start, end) for start, end, _ in chunks(d.get("text", ""))}
        rel_groups, ext_groups = defaultdict(list), defaultdict(list)
        for j in rel:
            rel_groups[_span(j)].append(j)
        for j in ext:
            ext_groups[_span(j)].append(j)
        rel_complete = _span_complete(rel_groups, expected, {"done", "review"}) and all(
            j["status"] == "review" or isinstance(j.get("result"), dict)
            and j["result"].get("status") in {"relevant", "irrelevant", "review"} for j in rel)
        relevant_spans = {_span(j) for j in rel if j["status"] == "done"
                          and isinstance(j.get("result"), dict) and j["result"].get("status") == "relevant"}
        ext_complete = bool(relevant_spans) and rel_complete and _span_complete(ext_groups, relevant_spans, {"done"}) and all(
            isinstance(j.get("result"), dict) and isinstance(j["result"].get("facts"), list) for j in ext)
        if set(rel_groups) - expected or set(ext_groups) - relevant_spans:
            flags["unexpected_chunk_span"] += 1
        if any(len(group) != 1 for group in [*rel_groups.values(), *ext_groups.values()]):
            flags["duplicate_active_chunk_jobs"] += 1

        status = d.get("status", "unique")
        exclusion = None
        if status == "exact_duplicate":
            status = "exact_duplicate"
        elif status == "empty" or not d.get("text", "").strip():
            status = "empty"
        else:
            exclusion = CorpusPipeline._rule_exclusion(d, config)
            if exclusion:
                status = "rule_excluded"
                exclusions[exclusion] += 1
            else:
                eligible += 1
                relevance_complete_docs += rel_complete
                expected_extract_docs += bool(relevant_spans)
                extraction_complete_docs += ext_complete
                if relevant_spans:
                    status = "relevant"
                elif any(j["status"] == "review" or (j.get("result") or {}).get("status") == "review" for j in rel):
                    status = "review"
                elif rel_complete and all(j["status"] == "done" and j["result"].get("status") == "irrelevant" for j in rel):
                    status = "irrelevant"
                else:
                    status = "incomplete"
        funnel[status] += 1
        if status in {"exact_duplicate", "empty", "rule_excluded"} and (rel or ext):
            flags["active_jobs_outside_processing_frame"] += 1
        classified = [(max(0, (_span(j)[1] or 0) - (_span(j)[0] or 0)),
                       (j.get("result") or {}).get("language", "unknown"), j["id"])
                      for j in rel if j["status"] in {"done", "review"} and isinstance(j.get("result"), dict)]
        dominant = max(classified, default=(0, "unprocessed", ""), key=lambda x: (x[0], x[2]))[1]
        observed_langs = sorted({x[1] for x in classified})
        label = "mixed" if "mixed" in observed_langs or len(set(observed_langs) - {"unknown", "other"}) > 1 else dominant
        languages[label] += 1
        dominant_languages[dominant] += 1
        diagnostics.append({"document_id": identifier, "status": status, "rule_exclusion": exclusion,
                            "expected_relevance_chunks": len(expected), "relevance_complete": rel_complete,
                            "expected_extraction_chunks": len(relevant_spans), "extraction_complete": ext_complete,
                            "actual_language": label, "dominant_language": dominant, "chunk_languages": observed_langs})

    if manifest.get("population"):
        population = manifest["population"]
        actual = {"raw": raw, "unique": sum(d.get("status", "unique") == "unique" for d in docs),
                  "eligible_unique": eligible, "rule_exclusions": sum(exclusions.values())}
        if any(population.get(key) != value for key, value in actual.items()):
            raise ValueError("Full-pilot population frame differs from its frozen manifest")

    active_extract_jobs = {j["id"]: j for j in active if j["stage"] == "extract" and j["status"] == "done"}
    active_extract_ids = set(active_extract_jobs)
    facts = [f for f in results.get("facts", []) if f.get("extraction_job_id") in active_extract_ids]
    flags["inactive_facts_ignored"] += len(results.get("facts", [])) - len(facts)
    if len({f["id"] for f in facts}) != len(facts):
        raise ValueError("Duplicate persisted fact IDs")
    actor_rows = []
    approved = {actor["id"] for actor in config["objects"]}
    if any(f["actor_id"] not in approved or f["document_id"] not in by_id for f in facts):
        raise ValueError("Accepted fact is outside the roster/documents")
    if any(active_extract_jobs[f["extraction_job_id"]]["context"]["document"]["id"] != f["document_id"] for f in facts):
        raise ValueError("Persisted fact document differs from its extraction job")
    for actor in config["objects"]:
        selected = [f for f in facts if f["actor_id"] == actor["id"]]
        channels = Counter()
        for f in selected:
            own = ((f["kind"] == "public_statement" and f.get("attribution_basis") in {"direct_statement", "reported_statement"})
                   or (f["kind"] == "action" and f.get("attribution_basis") == "reported_action"))
            channels["own" if own else "image" if f["kind"] == "attributed_image" else "context"] += 1
        actor_rows.append({"actor_id": actor["id"], "name": actor["name"], "facts": len(selected),
                           "own_facts": channels["own"], "image_facts": channels["image"], "context_facts": channels["context"],
                           "unique_documents": len({f["document_id"] for f in selected}),
                           "candidate_editorial_families": len({find(f["document_id"]) for f in selected})})
    all_ledger = _ledger(jobs, config["pricing"])
    starting_ids = set(manifest.get("starting_job_ids", []))
    fresh_ledger = _ledger([j for j in jobs if j["id"] not in starting_ids], config["pricing"])
    starting_spend = manifest["start_spent_usd"]
    ledger = {"all_historical_and_new": all_ledger, "new_job_receipts": fresh_ledger,
              "frozen_start_spent_usd": starting_spend,
              "new_recorded_cost_delta_usd": all_ledger["recorded_cost_usd"] - starting_spend,
              "budget_cap_usd": manifest["budget_cap_usd"],
              "remaining_cap_after_reservations_usd": manifest["budget_cap_usd"] - all_ledger["recorded_cost_usd"] - all_ledger["outstanding_reserved_usd"]}
    blocked = {"pending", "running", "uncertain", "error"}
    rel_complete = relevance_complete_docs == eligible and not any(j["stage"] == "relevance" and j["status"] in blocked for j in active)
    ext_complete = rel_complete and expected_extract_docs == extraction_complete_docs and not any(j["stage"] == "extract" and j["status"] in blocked | {"review"} for j in active)
    forecast_stages = {}
    for stage in ("relevance", "extract", "describe"):
        group = [j for j in active if j["stage"] == stage]
        book = _ledger(group, config["pricing"])
        complete = rel_complete if stage == "relevance" else ext_complete if stage == "extract" else (
            bool(group) and ext_complete and all(j["status"] == "done" for j in group))
        if stage == "extract" and not rel_complete:
            complete = False
        equivalents = [row["uncached_equivalent_usd"] for row in book["by_model_stage_recipe"]]
        uncached = sum(equivalents) if all(value is not None for value in equivalents) else None
        forecast_stages[stage] = {"complete": complete, "active_jobs": len(group),
            "active_recorded_cost_usd": book["recorded_cost_usd"], "active_uncached_equivalent_usd": uncached,
            "full_32000_observed_cache_usd": book["recorded_cost_usd"] * 32000 / raw if complete else None,
            "full_32000_uncached_usd": uncached * 32000 / raw if complete and uncached is not None else None,
            "partial_incurred_scaled_to_32000_usd": book["recorded_cost_usd"] * 32000 / raw if not complete else None,
            "status_counts": dict(Counter(j["status"] for j in group))}
    measured = {"population": {"raw_records": raw, "unique_documents": sum(d.get("status", "unique") == "unique" for d in docs),
                                "exact_duplicate_extra_records": funnel["exact_duplicate"],
                                "empty_records": funnel["empty"], "eligible_unique_documents": eligible,
                                "rule_excluded_unique_documents": sum(exclusions.values()), "rule_exclusions": dict(exclusions)},
                "whole_document_funnel": {key: funnel[key] for key in ("exact_duplicate", "empty", "rule_excluded", "relevant", "review", "irrelevant", "incomplete")},
                "whole_document_completion": {"relevance_complete_documents": relevance_complete_docs,
                                               "extraction_expected_documents": expected_extract_docs,
                                               "extraction_complete_documents": extraction_complete_docs},
                "active_stage_statuses": {s: dict(Counter(j["status"] for j in active if j["stage"] == s)) for s in STAGES},
                "languages": {"document_labels": dict(languages), "dominant_by_longest_classified_chunk": dict(dominant_languages),
                              "note": "Mixed labels reflect multiple classified chunk languages; unprocessed is not an inferred language."},
                "quality_control": {"extract": _qc(active, "extract"), "describe": _qc(active, "describe")},
                "accepted_facts": {"total": len(facts), "by_kind": dict(Counter(f["kind"] for f in facts)),
                                   "by_attribution_basis": dict(Counter(f.get("attribution_basis", "legacy_unspecified") for f in facts)),
                                   "event_date_known": sum(f.get("event_date") is not None for f in facts),
                                   "event_date_unknown": sum(f.get("event_date") is None for f in facts)},
                "local_fact_reviews": {"annotated_facts": sum(bool(f.get("review_id")) for f in facts),
                                       "planned_context_facts": sum(f.get("temporal_status") == "planned" for f in facts),
                                       "raw_model_receipts_modified": False},
                "draft_scale_review": {"accepted_proposals": len(corpus.get("draft_scales", [])),
                                       "locally_reviewed_proposals": sum(bool(s.get("rubric_reviewer")) for s in corpus.get("draft_scales", [])),
                                       "scales_approved": bool(config.get("scales_approved")),
                                       "intensity_unit": "single_episode_expression; frequency_and_coverage_separate"},
                "actor_coverage": actor_rows, "zero_coverage": [{"actor_id": a["actor_id"], "name": a["name"]} for a in actor_rows if not a["facts"]],
                "zero_own_position_coverage": [{"actor_id": a["actor_id"], "name": a["name"]} for a in actor_rows if not a["own_facts"]],
                "roster_coverage": {"actors": len(actor_rows), "actors_with_any_facts": sum(a["facts"] > 0 for a in actor_rows),
                                    "actors_with_own_position_facts": sum(a["own_facts"] > 0 for a in actor_rows)},
                "editorial_families": {"near_candidate_pairs": len(similar), "candidate_family_count_all_records": len({find(key) for key in by_id}),
                                      "note": "Union of exact and near-duplicate candidate links, not verified independent episodes or evidence of true allegations."},
                "ledger": ledger, "flags": dict(sorted((k, v) for k, v in flags.items() if v))}
    public = {"schema_version": "corpus-full-pilot-summary-v1", "corpus_id": corpus["id"], "measured": measured,
              "forecast": {"target_raw_documents": 32000, "pilot_raw_document_denominator": raw,
                           "stage_costs": forecast_stages,
                           "basis": "Active-recipe equivalent cost × 32000 / all pilot records; no random-stratum extrapolation or confidence interval.",
                           "limitations": ["Three-day topic-heavy census is not representative of a future corpus.",
                                           "Observed cache is a transfer scenario, not a promised future discount.",
                                           "Partial incurred scaling is not a completed-stage price estimate.",
                                           "No cost estimate for whole-research scoring, human review, retries or revisions."],
                           "full_research_scores_estimated": False}}
    return {"public_aggregate": public, "document_diagnostics": diagnostics}


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def summarize(root, corpus_id, *, publish=False):
    root = Path(root).resolve()
    if not re.fullmatch(r"corpus-[A-Za-z0-9_-]+", corpus_id):
        raise ValueError("Invalid corpus ID")
    directory = root / ".corpus_runtime/pilots" / corpus_id / "full-v2"
    manifest_bytes = (directory / "manifest.json").read_bytes()
    results_bytes = (directory / "results_private.json").read_bytes()
    manifest, results = [json.loads(x.decode("utf-8-sig")) for x in (manifest_bytes, results_bytes)]
    path = root / ".corpus_runtime/corpus.sqlite3"
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        db.execute("BEGIN")
        stored = db.execute("SELECT source_sha256 FROM corpora WHERE id=?", (corpus_id,)).fetchone()
        if stored is None or stored[0] != manifest["source_sha256"]:
            raise ValueError("Database and frozen source hashes differ")
        docs = []
        for row in db.execute("SELECT * FROM documents WHERE corpus_id=? ORDER BY source_row", (corpus_id,)):
            docs.append(dict(json.loads(row["data"]), id=row["id"], status=row["status"], duplicate_of=row["duplicate_of"]))
        similar = [json.loads(r[0]) for r in db.execute("SELECT data FROM similar WHERE corpus_id=? ORDER BY left_id,right_id", (corpus_id,))]
    summary = build_summary(manifest, results, docs, similar)
    provenance = {"manifest_sha256": _sha(manifest_bytes), "results_private_sha256": _sha(results_bytes),
                  "source_csv_sha256": manifest["source_sha256"], "document_snapshot_sha256": _json_hash(docs),
                  "similar_candidate_snapshot_sha256": _json_hash(similar), "summarizer_sha256": _sha(Path(__file__).read_bytes()),
                  "jobs_basis": "Frozen snapshot active_recipe flags; no API or live recipe recalculation."}
    summary["provenance"] = provenance
    summary["public_aggregate"].update(provenance=provenance, generated_at=datetime.now(timezone.utc).isoformat())
    private = directory / "summary_private.json"
    _write(private, summary)
    public = root / "reports/corpus_full_pilot_2026_10_17.json"
    if publish:
        _write(public, summary["public_aggregate"])
    return {"private_summary": str(private), "public_summary": str(public) if publish else None,
            "published": bool(publish), "measured": summary["public_aggregate"]["measured"]["whole_document_funnel"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(summarize(args.root, args.corpus_id, publish=args.publish), ensure_ascii=False, allow_nan=False))
    except (ValueError, KeyError, FileNotFoundError, sqlite3.Error) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
