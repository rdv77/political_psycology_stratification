"""Offline document-level pilot accounting; publish aggregates only with --publish.

Reads the private immutable result snapshot and actual imported documents through
SQLite read-only mode. Never loads credentials, prepares work, or calls an API.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.corpus_pipeline.models import estimate_cost
from src.corpus_pipeline.pilot_estimates import estimate_costs
from src.corpus_pipeline.service import CorpusPipeline, chunks
from scripts.run_rwanda_corpus_pilot import stratum


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    temporary.replace(path)


def _nonnegative(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Invalid numeric ledger field: " + name)
    return value


def _has_usage(job):
    return bool(job.get("input_tokens", 0) or job.get("output_tokens", 0) or job.get("cost_usd", 0))


def _uncached(job, pricing):
    if not _has_usage(job):
        return None
    try:
        return estimate_cost(job["payload"]["model"], job["input_tokens"], job["output_tokens"], pricing=pricing)
    except (KeyError, TypeError, ValueError):
        return None


def _span(job):
    doc = job.get("context", {}).get("document", {})
    return doc.get("chunk_start"), doc.get("chunk_end")


def _job_sum(jobs, key, pricing):
    if not jobs:
        return None
    if key == "cost_usd":
        return sum(_nonnegative(j.get(key, 0), key) for j in jobs)
    values = [_uncached(j, pricing) for j in jobs]
    return sum(values) if all(v is not None for v in values) else None


def _ledger(jobs, pricing):
    grouped = defaultdict(list)
    for job in jobs:
        grouped[(job.get("payload", {}).get("model", "unknown"), job["stage"],
                 "active" if job["active_recipe"] else "inactive")].append(job)
    rows = []
    for (model, stage, state), group in sorted(grouped.items()):
        paid = [j for j in group if _has_usage(j)]
        uncached = [_uncached(j, pricing) for j in paid]
        rows.append({"model": model, "stage": stage, "recipe_state": state,
                     "jobs": len(group), "jobs_with_reported_usage_or_cost": len(paid),
                     "status_counts": dict(sorted(Counter(j["status"] for j in group).items())),
                     "input_tokens": sum(j.get("input_tokens", 0) for j in group),
                     "output_tokens": sum(j.get("output_tokens", 0) for j in group),
                     "recorded_cost_usd": sum(_nonnegative(j.get("cost_usd", 0), "cost_usd") for j in group),
                     "uncached_equivalent_usd": sum(uncached) if all(v is not None for v in uncached) else None,
                     "outstanding_reserved_usd": sum(_nonnegative(j.get("reserved_usd", 0), "reserved_usd")
                                                     for j in group if j["status"] in {"running", "uncertain"})})
    return {"by_model_stage_recipe": rows,
            "input_tokens": sum(r["input_tokens"] for r in rows),
            "output_tokens": sum(r["output_tokens"] for r in rows),
            "total_tokens": sum(r["input_tokens"] + r["output_tokens"] for r in rows),
            "recorded_cost_usd": sum(r["recorded_cost_usd"] for r in rows),
            "active_recorded_cost_usd": sum(r["recorded_cost_usd"] for r in rows if r["recipe_state"] == "active"),
            "inactive_recorded_cost_usd": sum(r["recorded_cost_usd"] for r in rows if r["recipe_state"] == "inactive"),
            "outstanding_reserved_usd": sum(r["outstanding_reserved_usd"] for r in rows),
            "note": "All snapshot jobs, including inactive recipes, refusals and failed validations. Reservations are possible charges, not confirmed expenditure. Uncached equivalents use snapshot-config rates."}


def build_summary(manifest, results, documents, *, target_count=32000):
    """Pure aggregation; documents are complete unique imported records."""
    corpus = results["corpus"]
    if corpus["id"] != manifest["corpus_id"] or results["manifest"]["random_document_ids"] != manifest["random_document_ids"]:
        raise ValueError("Manifest/result snapshot mismatch")
    if corpus["source_sha256"] != manifest["source_sha256"]:
        raise ValueError("Source fingerprint mismatch")
    config, jobs = corpus["config"], results["jobs"]
    if any(type(j.get("active_recipe")) is not bool for j in jobs):
        raise ValueError("Every result-snapshot job must declare active_recipe")
    eligible = [d for d in documents if CorpusPipeline._rule_exclusion(d, config) is None]
    docs = {d["id"]: d for d in eligible}
    if len(docs) != len(eligible) or len(eligible) != manifest["eligible_documents"]:
        raise ValueError("Eligible unique sampling frame changed; rebuild the pilot manifest")
    population = [{"id": d["id"], "stratum": stratum(d)} for d in eligible]
    random_ids = set(manifest["random_document_ids"])
    extraction_random = set(manifest.get("extraction_random_ids", []))
    selected_ids = random_ids | set(manifest.get("challenge_document_ids", [])) | set(manifest.get("extraction_document_ids", []))
    if len(random_ids) != len(manifest["random_document_ids"]) or not selected_ids <= docs.keys() or not extraction_random <= random_ids:
        raise ValueError("Invalid or ineligible manifest document selection")
    sampled_counts = Counter(stratum(docs[identifier]) for identifier in random_ids)
    frame_counts = Counter(d["stratum"] for d in population)
    for name, sizes in manifest["strata"].items():
        if sizes["population"] != frame_counts[name] or sizes["selected"] != sampled_counts[name]:
            raise ValueError("Manifest stratum allocation differs from actual sampling frame")

    active = [j for j in jobs if j["active_recipe"]]
    by_document = defaultdict(lambda: defaultdict(list))
    aggregation_flags = []
    for job in active:
        if job["stage"] not in {"relevance", "extract"}:
            continue
        identifier = job["context"]["document"]["id"]
        if identifier not in docs:
            aggregation_flags.append({"code": "active_job_outside_eligible_frame", "stage": job["stage"]})
            continue
        selected_ids.add(identifier)
        by_document[identifier][job["stage"]].append(job)
    active_extract_ids = {j["id"] for j in active if j["stage"] == "extract" and j["status"] == "done"}
    fact_counts = Counter()
    active_fact_total = 0
    for fact in results.get("facts", []):
        if fact.get("extraction_job_id") not in active_extract_ids:
            aggregation_flags.append({"code": "nonactive_fact_ignored"})
            continue
        fact_counts[fact["document_id"]] += 1
        active_fact_total += 1

    observations, doc_diagnostics = [], []
    for identifier in sorted(selected_ids):
        doc = docs[identifier]
        expected = {(start, end) for start, end, text in chunks(doc["text"])}
        rel = by_document[identifier]["relevance"]
        ext = by_document[identifier]["extract"]
        rel_by_span = defaultdict(list)
        ext_by_span = defaultdict(list)
        for job in rel:
            rel_by_span[_span(job)].append(job)
        for job in ext:
            ext_by_span[_span(job)].append(job)
        rel_complete = bool(expected) and set(rel_by_span) == expected and all(len(rel_by_span[s]) == 1 and rel_by_span[s][0]["status"] in {"done", "review"} for s in expected)
        relevant_spans = {_span(j) for j in rel if j["status"] == "done" and isinstance(j.get("result"), dict) and j["result"].get("status") == "relevant"}
        relevant = bool(relevant_spans & expected)
        review = (any(j["status"] == "review" or isinstance(j.get("result"), dict) and j["result"].get("status") == "review" for j in rel)
                  or any(j["status"] == "done" and (not isinstance(j.get("result"), dict) or j["result"].get("status") not in {"relevant", "irrelevant", "review"}) for j in rel)) and not relevant
        ext_complete = bool(relevant_spans) and rel_complete and set(ext_by_span) == relevant_spans and all(
            len(ext_by_span[s]) == 1 and ext_by_span[s][0]["status"] == "done"
            and isinstance(ext_by_span[s][0].get("result"), dict)
            and isinstance(ext_by_span[s][0]["result"].get("facts"), list) for s in relevant_spans)
        if set(rel_by_span) - expected or set(ext_by_span) - relevant_spans:
            aggregation_flags.append({"code": "unexpected_chunk_span", "document_id": identifier})
        if any(len(group) > 1 for group in (*rel_by_span.values(), *ext_by_span.values())):
            aggregation_flags.append({"code": "duplicate_active_chunk_jobs", "document_id": identifier})
        observations.append({"id": identifier, "stratum": stratum(doc), "random_sample": identifier in random_ids,
                             "relevance_complete": rel_complete, "relevant": relevant, "review": bool(review),
                             "relevance_cost_usd": _job_sum(rel, "cost_usd", config["pricing"]),
                             "relevance_uncached_cost_usd": _job_sum(rel, "uncached", config["pricing"]),
                             "extraction_complete": ext_complete, "extraction_random": identifier in extraction_random,
                             "extraction_cost_usd": _job_sum(ext, "cost_usd", config["pricing"]),
                             "extraction_uncached_cost_usd": _job_sum(ext, "uncached", config["pricing"]),
                             "facts": fact_counts[identifier] if ext else None})
        doc_diagnostics.append({"id": identifier, "expected_relevance_chunks": len(expected),
                                "active_relevance_jobs": len(rel), "expected_extraction_chunks": len(relevant_spans),
                                "active_extraction_jobs": len(ext), "active_facts": fact_counts[identifier]})

    describe_paid = [j for j in active if j["stage"] == "describe" and _has_usage(j)]
    descriptor_cost = sum(j.get("cost_usd", 0) for j in describe_paid)
    descriptor_inputs = sum(len(j.get("context", {}).get("facts", [])) for j in describe_paid)
    raw_count = corpus["profile"]["rows_total"]
    estimates = estimate_costs(population, observations, target_count=target_count, raw_count=raw_count,
                               descriptor_cost_usd=descriptor_cost, descriptor_input_facts=descriptor_inputs)
    ledger = _ledger(jobs, config["pricing"])
    cap = manifest.get("budget_cap_usd")
    totals = {"raw_documents": raw_count, "unique_documents": len(documents), "eligible_unique_documents": len(eligible),
              "rule_exclusions": len(documents) - len(eligible), "random_documents": len(random_ids),
              "targeted_documents": len(selected_ids - random_ids), "active_facts": active_fact_total,
              "active_job_statuses": dict(sorted(Counter(j["status"] for j in active).items())),
              "inactive_job_statuses": dict(sorted(Counter(j["status"] for j in jobs if not j["active_recipe"]).items()))}
    # Deliberate allowlist: never put manifest selection, contexts, payloads,
    # document/author IDs, source URLs, quotes or text into this aggregate.
    public = {"schema_version": "corpus-pilot-summary-v1", "corpus_id": corpus["id"],
              "generated_at": datetime.now(timezone.utc).isoformat(), "counts": totals,
              "sampling": {"seed": manifest["seed"], "strata": manifest["strata"],
                           "selection_basis": "stratified_random_core; targeted_challenges_excluded_from_forecast"},
              "windows": {"canonical_research": manifest.get("canonical_research_window"), "technical_pilot": manifest.get("technical_pilot_window")},
              "active_models": config["routes"], "pricing_per_million_tokens": config["pricing"],
              "all_paid_ledger": ledger, "estimates": estimates,
              "descriptor_measurement": {"active_paid_calls": len(describe_paid), "cost_usd": descriptor_cost,
                                         "input_fact_occurrences": descriptor_inputs, "denominator": "sum of facts in each paid input, not unique facts"},
              "budget": {"cap_usd": cap, "recorded_plus_outstanding_reserved_usd": ledger["recorded_cost_usd"] + ledger["outstanding_reserved_usd"],
                         "uncommitted_cap_usd": cap - ledger["recorded_cost_usd"] - ledger["outstanding_reserved_usd"] if cap is not None else None,
                         "note": "Conservative accounting, not confirmation of provider billing or authorization for further calls."},
              "aggregation_flags": [dict(code=code, count=count) for code, count in sorted(Counter(f["code"] for f in aggregation_flags).items())],
              "limitations": ["Snapshot cost forecast; unfinished current-recipe documents are not treated as irrelevant.",
                              "Relevance ambiguity scenarios are not confidence intervals or classification accuracy.",
                              "No guaranteed future cached-token rate. No scoring or whole-research forecast.",
                              "Extraction review/errors are not completed valid fact yields. Descriptor scaling assumes linear fact packaging."]}
    return {"public_aggregate": public, "population": population, "observations": observations,
            "document_diagnostics": doc_diagnostics, "aggregation_flags": aggregation_flags}


def summarize(root, corpus_id, *, publish=False, target_count=32000):
    root = Path(root).resolve()
    if not re.fullmatch(r"corpus-[A-Za-z0-9_-]+", corpus_id):
        raise ValueError("Invalid corpus ID")
    directory = root / ".corpus_runtime/pilots" / corpus_id
    manifest_path, results_path = directory / "manifest.json", directory / "results_private.json"
    # Hash the exact bytes parsed, even if another worker later refreshes a
    # snapshot file while this offline aggregation is running.
    manifest_bytes, results_bytes = manifest_path.read_bytes(), results_path.read_bytes()
    manifest, results = json.loads(manifest_bytes.decode("utf-8-sig")), json.loads(results_bytes.decode("utf-8-sig"))
    db_path = root / ".corpus_runtime/corpus.sqlite3"
    with closing(sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        stored = db.execute("SELECT source_sha256 FROM corpora WHERE id=?", (corpus_id,)).fetchone()
        if stored is None or stored[0] != manifest["source_sha256"]:
            raise ValueError("Imported corpus source does not match the private snapshot")
        documents = [json.loads(r[0]) for r in db.execute("SELECT data FROM documents WHERE corpus_id=? AND status='unique' ORDER BY source_row", (corpus_id,))]
    result = build_summary(manifest, results, documents, target_count=target_count)
    provenance = {"manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(), "results_private_sha256": hashlib.sha256(results_bytes).hexdigest(),
                  "source_csv_sha256": manifest["source_sha256"], "summarizer_sha256": sha(Path(__file__)),
                  "estimator_sha256": sha(ROOT / "src/corpus_pipeline/pilot_estimates.py"),
                  "service_sha256": sha(ROOT / "src/corpus_pipeline/service.py"),
                  "sampling_script_sha256": sha(ROOT / "scripts/run_rwanda_corpus_pilot.py"),
                  "document_population_sha256": hashlib.sha256(json.dumps(documents, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest(),
                  "jobs_basis": "saved results_private active_recipe flags, not a live reclassification"}
    result["provenance"] = provenance
    result["public_aggregate"]["provenance"] = provenance
    private_path = directory / "summary_private.json"
    write(private_path, result)
    public_path = root / "reports/corpus_pilot_2026_10_17.json"
    if publish:
        write(public_path, result["public_aggregate"])
    return {"private_summary": str(private_path), "published": bool(publish),
            "public_summary": str(public_path) if publish else None,
            "counts": result["public_aggregate"]["counts"], "all_paid_ledger": result["public_aggregate"]["all_paid_ledger"]}


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--target-count", type=int, default=32000)
    parser.add_argument("--publish", action="store_true", help="Write the aggregate-only public report")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(summarize(args.root, args.corpus_id, publish=args.publish, target_count=args.target_count), ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, FileNotFoundError, sqlite3.Error) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
