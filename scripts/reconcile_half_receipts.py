"""Offline reconciliation of private API receipts with a frozen job snapshot.

Never sends requests, retries a job, reads a key or changes the corpus database.
Both output files remain private until the caller explicitly reviews/publishes
the aggregate. Provider usage is not an invoice or account balance.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.corpus_pipeline.models import estimate_cost
from src.corpus_pipeline.service import digest

_STAGES = {"relevance", "extract", "describe", "score"}


def _stage(job):
    return job.get("stage") if job.get("stage") in _STAGES else "unknown"


def _attempted(job):
    return bool(job.get("started_at") or job.get("response_id") or job.get("input_tokens")
                or job.get("output_tokens") or job.get("cost_usd")
                or job.get("status") in {"running", "uncertain", "done", "review", "error"})


def _tokens(usage):
    if not isinstance(usage, dict):
        return None
    inputs, outputs = usage.get("input_tokens"), usage.get("output_tokens")
    details = usage.get("input_tokens_details") or {}
    if not isinstance(details, dict):
        return None
    cached = details.get("cached_tokens", 0)
    if any(type(value) is not int or value < 0 for value in (inputs, outputs, cached)) or cached > inputs:
        return None
    return inputs, outputs, cached


def reconcile(snapshot, receipts, *, manifest=None, pre_wrapper_calls=12):
    """Return public aggregates and private issues without mutating inputs.

    Repeated nonempty provider response IDs are counted only once. Unknown
    usage, unmatched receipts and conflicting duplicate IDs are surfaced; they
    are not evidence of a free request. Missing receipts for the first N started
    jobs are labelled expected legacy coverage, with their recorded cost intact.
    """
    if type(pre_wrapper_calls) is not int or pre_wrapper_calls < 0:
        raise ValueError("pre_wrapper_calls must be a nonnegative integer")
    manifest = manifest if manifest is not None else snapshot.get("manifest")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("pricing"), dict):
        raise ValueError("Frozen manifest.pricing is required")
    pricing = manifest["pricing"]
    jobs = snapshot["jobs"]
    jobs_by_hash, jobs_by_id = defaultdict(list), {}
    issues, issue_counts = [], Counter()

    def issue(kind, *, job=None, line=None, receipt=None):
        item = {"kind": kind}
        if job is not None:
            item["job_id"] = job["id"]
        if line is not None:
            item["receipt_line"] = line
        if isinstance(receipt, dict):
            for key in ("response_id", "request_sha256"):
                if isinstance(receipt.get(key), str):
                    item[key] = receipt[key]
        issues.append(item)
        issue_counts[kind] += 1

    attempted = []
    for job in jobs:
        if job["id"] in jobs_by_id:
            raise ValueError("Duplicate snapshot job ID")
        jobs_by_id[job["id"]] = job
        if isinstance(job.get("payload"), dict):
            jobs_by_hash[digest(job["payload"])].append(job)
        if _attempted(job):
            attempted.append(job)
    attempted.sort(key=lambda job: (job.get("started_at") or job.get("created_at") or "", job["id"]))
    legacy_ids = {job["id"] for job in attempted[:pre_wrapper_calls]}
    matched, seen_responses, normalized = defaultdict(list), {}, []
    unknown_pricing_units, duplicates = 0, 0
    for line, receipt in enumerate(receipts, 1):
        if not isinstance(receipt, dict) or not isinstance(receipt.get("request_sha256"), str):
            issue("invalid_receipt", line=line)
            unknown_pricing_units += 1
            continue
        response_id = receipt.get("response_id")
        if isinstance(response_id, str) and response_id:
            if response_id in seen_responses:
                duplicates += 1
                prior = seen_responses[response_id]
                same = all(prior.get(key) == receipt.get(key) for key in
                           ("request_sha256", "model", "usage", "actual_service_tier"))
                issue("duplicate_response_id" if same else "conflicting_duplicate_response_id", line=line, receipt=receipt)
                if not same:
                    unknown_pricing_units += 1
                continue
            seen_responses[response_id] = receipt
        candidates = jobs_by_hash.get(receipt["request_sha256"], [])
        if len(candidates) > 1 and response_id:
            exact = [job for job in candidates if job.get("response_id") == response_id]
            if len(exact) == 1:
                candidates = exact
        job = candidates[0] if len(candidates) == 1 else None
        if not candidates:
            issue("receipt_without_snapshot_job", line=line, receipt=receipt)
        elif job is None:
            issue("ambiguous_request_hash", line=line, receipt=receipt)
        tokens = _tokens(receipt.get("usage"))
        cost = None
        if tokens is None:
            issue("receipt_usage_unknown", job=job, line=line, receipt=receipt)
        else:
            model = receipt.get("model")
            if not model and job is not None:
                model = (job.get("payload") or {}).get("model")
            try:
                cost = estimate_cost(model, tokens[0], tokens[1], cached_input_tokens=tokens[2], pricing=pricing)
            except (TypeError, ValueError, KeyError):
                issue("receipt_model_unpriced", job=job, line=line, receipt=receipt)
        actual_tier = receipt.get("actual_service_tier")
        if actual_tier not in {None, "default"}:
            issue("nonstandard_service_tier", job=job, line=line, receipt=receipt)
            cost = None
        elif actual_tier is None:
            issue("actual_service_tier_unreported", job=job, line=line, receipt=receipt)
        if cost is None:
            unknown_pricing_units += 1
        row = {"receipt_line": line, "job": job, "receipt": receipt, "tokens": tokens, "cost": cost}
        normalized.append(row)
        if job is not None:
            matched[job["id"]].append(row)

    stage_totals = defaultdict(lambda: {"jobs": 0, "attempted_jobs": 0, "job_recorded_cost_usd": 0.,
        "job_input_tokens": 0, "job_output_tokens": 0, "outstanding_reserved_usd": 0.,
        "matched_jobs": 0, "matched_unique_receipts": 0, "receipt_priced_cost_usd_known": 0.,
        "receipt_unpriced_count": 0, "pre_wrapper_without_receipt_jobs": 0,
        "unexpected_without_receipt_jobs": 0, "cost_mismatch_jobs": 0})
    states, activity, differences = Counter(), Counter(), []
    complete_matched_job_cost, complete_matched_receipt_cost = 0., 0.
    for job in jobs:
        stage = stage_totals[_stage(job)]
        stage["jobs"] += 1
        stage["attempted_jobs"] += _attempted(job)
        stage["job_recorded_cost_usd"] += job.get("cost_usd", 0.)
        stage["job_input_tokens"] += job.get("input_tokens", 0)
        stage["job_output_tokens"] += job.get("output_tokens", 0)
        states[job.get("status", "unknown")] += 1
        activity["active" if job.get("active_recipe") else "inactive_or_unspecified"] += 1
        if job.get("status") in {"running", "uncertain"}:
            stage["outstanding_reserved_usd"] += job.get("reserved_usd", 0.)
        rows = matched.get(job["id"], [])
        if not rows:
            if _attempted(job):
                legacy = job["id"] in legacy_ids
                stage["pre_wrapper_without_receipt_jobs" if legacy else "unexpected_without_receipt_jobs"] += 1
                issue("pre_wrapper_without_receipt" if legacy else "job_without_receipt", job=job)
            continue
        stage["matched_jobs"] += 1
        stage["matched_unique_receipts"] += len(rows)
        stage["receipt_priced_cost_usd_known"] += sum(row["cost"] for row in rows if row["cost"] is not None)
        stage["receipt_unpriced_count"] += sum(row["cost"] is None for row in rows)
        if len(rows) > 1:
            issue("multiple_provider_responses_for_job", job=job)
        final_rows = [row for row in rows if row["receipt"].get("response_id") == job.get("response_id")]
        if len(final_rows) != 1:
            issue("response_id_mismatch_or_ambiguous", job=job)
        comparable = final_rows if len(final_rows) == 1 else rows if len(rows) == 1 else []
        if comparable:
            row = comparable[0]
            if row["tokens"] is not None and row["tokens"][:2] != (job.get("input_tokens", 0), job.get("output_tokens", 0)):
                issue("job_usage_mismatch", job=job, line=row["receipt_line"], receipt=row["receipt"])
        if all(row["cost"] is not None for row in rows):
            receipt_total = sum(row["cost"] for row in rows)
            job_cost = job.get("cost_usd", 0.)
            complete_matched_job_cost += job_cost
            complete_matched_receipt_cost += receipt_total
            delta = receipt_total - job_cost
            differences.append(delta)
            if abs(delta) > 1e-8:
                stage["cost_mismatch_jobs"] += 1
                issue("job_cost_mismatch", job=job)

    totals = {key: sum(value[key] for value in stage_totals.values()) for key in next(iter(stage_totals.values()), {})}
    known_receipt_cost = sum(row["cost"] for row in normalized if row["cost"] is not None)
    phase = (snapshot.get("phase_trace") or {}).get("stage")
    public = {"version": "half-receipt-reconciliation-v1", "api_calls": 0, "database_modified": False,
        "snapshot_phase": phase if phase in _STAGES else "unspecified", "statuses": dict(states),
        "recipe_activity": dict(activity), "expected_pre_wrapper_calls": pre_wrapper_calls,
        "receipt_lines": len(receipts), "unique_receipt_records": len(normalized), "duplicate_response_records": duplicates,
        "unpriced_receipt_units": unknown_pricing_units,
        "receipt_priced_cost_usd_known": known_receipt_cost,
        "receipt_recomputed_cost_usd": known_receipt_cost if not unknown_pricing_units else None,
        "fully_priced_matched_job_cost_usd": complete_matched_job_cost,
        "fully_priced_matched_receipt_cost_usd": complete_matched_receipt_cost,
        "fully_priced_matched_cost_difference_usd": complete_matched_receipt_cost - complete_matched_job_cost,
        "max_abs_priced_job_cost_difference_usd": max(map(abs, differences), default=None),
        "by_stage": dict(stage_totals), "totals": totals, "issue_counts": dict(sorted(issue_counts.items())),
        "method": {"request_match": "service.digest(original job payload); response ID disambiguates repeated hashes",
            "pricing": "Frozen manifest.pricing with returned-model snapshot aliases and actual cached input usage",
            "duplicates": "Same nonempty response ID counted once; conflicting duplicates invalidate a complete receipt total",
            "missing_usage": "Unknown/unpriced calls stay unpriced, not zero; known subtotal is explicitly partial",
            "legacy": "First N started jobs may predate the receipt wrapper; their recorded job cost remains in totals",
            "tiers": "Default-tier rates; a nondefault actual tier is unpriced; missing actual tier is flagged",
            "timing": "Use a final snapshot after workers join; unmatched live receipts may be newer than the snapshot",
            "scope": "Local usage-based reconciliation only; no provider invoice, balance lookup, retry or further API call"}}
    return {"public": public, "private": {"issues": issues}}


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--pre-wrapper-calls", type=int, default=12)
    args = parser.parse_args(argv)
    snapshot_raw, receipt_raw = args.snapshot.read_bytes(), args.receipts.read_bytes()
    snapshot = json.loads(snapshot_raw.decode("utf-8-sig"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig")) if args.manifest else None
    receipts = []
    for line in receipt_raw.decode("utf-8-sig").splitlines():
        try:
            receipts.append(json.loads(line))
        except (ValueError, TypeError):
            receipts.append(None)  # Preserve the line count, never emit raw text.
    result = reconcile(snapshot, receipts, manifest=manifest, pre_wrapper_calls=args.pre_wrapper_calls)
    result["private"]["input_sha256"] = {"snapshot": hashlib.sha256(snapshot_raw).hexdigest(),
        "receipts": hashlib.sha256(receipt_raw).hexdigest()}
    directory = args.output_dir or args.snapshot.parent
    _write(directory / "reconciliation_private.json", result)
    _write(directory / "reconciliation_public_for_review.json", result["public"])
    print(json.dumps({"api_calls": 0, "receipt_lines": len(receipts), "issue_counts": result["public"]["issue_counts"]}))


if __name__ == "__main__":
    main()
