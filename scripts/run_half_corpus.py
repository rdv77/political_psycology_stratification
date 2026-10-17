"""Execute the frozen half-sample under one durable, non-rebased API cap.

Source material, credentials, queues and receipts remain local. Explicit
--execute is required to send anything. Resume cannot increase the frozen cap.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_full_corpus import write
from src.corpus_pipeline.models import make_task, estimate_cost, payload_input_bound
from src.corpus_pipeline.service import CorpusPipeline, compact_objects, digest, now, packed, MAX_OUTPUT


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def emit(value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def directory(pipeline, identifier):
    return pipeline.runtime / "pilots" / identifier / "half-70-v1"


def _hold_basis(job, manifest):
    if (job["status"] != "uncertain" or job.get("response_id")
            or job.get("input_tokens", 0) != 0 or job.get("output_tokens", 0) != 0 or job.get("cost_usd", 0) != 0):
        raise ValueError("Only an unchanged uncertain request without response/usage can be held")
    payload = json.loads(job["payload"])
    reserve = job["reserved_usd"]
    expected = estimate_cost(payload["model"], payload_input_bound(payload), payload["max_output_tokens"], pricing=manifest["pricing"])
    if not math.isfinite(reserve) or reserve <= 0 or abs(reserve - expected) > 1e-10:
        raise ValueError("Held request must retain its full original positive reservation")
    return {"job_id": job["id"], "payload_sha256": digest(payload), "status": "uncertain", "reserved_usd": reserve,
            "started_at": job["started_at"], "finished_at": job["finished_at"]}


def validated_uncertain_holds(pipeline, manifest):
    path = directory(pipeline, manifest["corpus_id"]) / "uncertain_holds.json"
    if not path.exists():
        return {"job_ids": set(), "reserved_usd": 0.}
    journal = read(path)
    if journal.get("version") != "held-http503-v1" or journal.get("corpus_id") != manifest["corpus_id"]:
        raise ValueError("Uncertain-hold journal belongs to another policy/corpus")
    holds = journal.get("holds")
    if not isinstance(holds, list):
        raise ValueError("Invalid uncertain-hold journal")
    raw_lines = (path.parent / "api_receipts_private.jsonl").read_bytes().splitlines(keepends=True)
    ids, total = set(), 0.
    with pipeline.connection() as db:
        for hold in holds:
            if not isinstance(hold, dict) or hold.get("job_id") in ids:
                raise ValueError("Duplicate or invalid uncertain hold")
            row = db.execute("SELECT * FROM jobs WHERE corpus_id=? AND id=?", (manifest["corpus_id"], hold.get("job_id"))).fetchone()
            if row is None:
                raise ValueError("Held request is absent")
            basis = _hold_basis(dict(row), manifest)
            if any(hold.get(key) != value for key, value in basis.items()):
                raise ValueError("Held uncertain request changed after acknowledgement")
            matching_jobs = db.execute("SELECT id,payload FROM jobs WHERE corpus_id=?", (manifest["corpus_id"],)).fetchall()
            if sum(digest(json.loads(job["payload"])) == basis["payload_sha256"] for job in matching_jobs) != 1:
                raise ValueError("Uncertain request payload is not unique")
            index = hold.get("receipt_line")
            if type(index) is not int or not 1 <= index <= len(raw_lines):
                raise ValueError("Pinned uncertain receipt is absent")
            if hashlib.sha256(raw_lines[index - 1]).hexdigest() != hold.get("receipt_sha256"):
                raise ValueError("Pinned uncertain receipt changed")
            matches = []
            for line_number, raw in enumerate(raw_lines, 1):
                receipt = json.loads(raw)
                if receipt.get("request_sha256") == basis["payload_sha256"]:
                    matches.append((line_number, receipt))
            if len(matches) != 1 or matches[0][0] != index:
                raise ValueError("Uncertain request requires exactly one pinned provider receipt")
            receipt = matches[0][1]
            if receipt.get("http_status") != 503 or receipt.get("usage") is not None or receipt.get("response_id"):
                raise ValueError("Held provider receipt must be HTTP503 without usage/response ID")
            ids.add(basis["job_id"])
            total += basis["reserved_usd"]
    return {"job_ids": ids, "reserved_usd": total}


def acknowledge_uncertain_job(pipeline, manifest, job_id):
    """Acknowledge a worst-case reserve; never changes/retries the original job."""
    check(pipeline, manifest)
    existing = validated_uncertain_holds(pipeline, manifest)
    if job_id in existing["job_ids"]:
        return existing
    path = directory(pipeline, manifest["corpus_id"]) / "uncertain_holds.json"
    raw_lines = (path.parent / "api_receipts_private.jsonl").read_bytes().splitlines(keepends=True)
    with pipeline.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        if db.execute("SELECT 1 FROM jobs WHERE corpus_id=? AND status='running'", (manifest["corpus_id"],)).fetchone():
            raise ValueError("Join all running requests before acknowledging an uncertain hold")
        row = db.execute("SELECT * FROM jobs WHERE corpus_id=? AND id=?", (manifest["corpus_id"], job_id)).fetchone()
        if row is None:
            raise ValueError("Unknown uncertain job")
        basis = _hold_basis(dict(row), manifest)
        matches = [(i, raw) for i, raw in enumerate(raw_lines, 1)
                   if json.loads(raw).get("request_sha256") == basis["payload_sha256"]]
        if len(matches) != 1:
            raise ValueError("Uncertain request requires exactly one pinned provider receipt")
        index, raw = matches[0]
        receipt = json.loads(raw)
        if receipt.get("http_status") != 503 or receipt.get("usage") is not None or receipt.get("response_id"):
            raise ValueError("Only HTTP503 without usage/response ID supports this hold policy")
        count = sum(digest(json.loads(job[0])) == basis["payload_sha256"] for job in
                    db.execute("SELECT payload FROM jobs WHERE corpus_id=?", (manifest["corpus_id"],)))
        if count != 1:
            raise ValueError("Uncertain request payload is not unique")
        journal = read(path) if path.exists() else {"version": "held-http503-v1", "corpus_id": manifest["corpus_id"], "holds": []}
        journal["holds"].append({**basis, "acknowledged_at": now(), "receipt_line": index,
            "receipt_sha256": hashlib.sha256(raw).hexdigest(), "policy": "retain full worst-case reservation; continue NEW jobs only; never replay"})
        write(path, journal)
    return validated_uncertain_holds(pipeline, manifest)


def load_selection(pipeline, identifier):
    source = pipeline.runtime / "sampling" / identifier / "half-v1"
    summary = read(source / "summary.json")
    manifest = read(source / "selection.json")
    if hashlib.sha256((source / "selection.json").read_bytes()).hexdigest() != summary["private_artifacts"]["selection_sha256"]:
        raise ValueError("Sampling manifest changed")
    ids = set(read(source / "selected_ids.json"))
    if ids != {row["id"] for row in manifest["documents"] if row["selected"]} or len(ids) != summary["selected_unique"]:
        raise ValueError("Selected ID frame changed")
    for filename, field in (("selected_unique.jsonl", "selected_unique"), ("adaptive_context_plans.jsonl", "context_plans")):
        info = summary["private_artifacts"][field]
        raw = (source / filename).read_bytes()
        if len(raw) != info["bytes"] or hashlib.sha256(raw).hexdigest() != info["sha256"]:
            raise ValueError("Frozen sample/context artifact changed")
    documents = {row["id"]: row for row in (json.loads(line) for line in (source / "selected_unique.jsonl").read_text(encoding="utf-8").splitlines())}
    plans = [json.loads(line) for line in (source / "adaptive_context_plans.jsonl").read_text(encoding="utf-8").splitlines()]
    if set(documents) != ids or len(plans) != len(ids) or {row["document"]["id"] for row in plans} != ids:
        raise ValueError("Sample and planned documents disagree")
    for plan in plans:
        original = documents[plan["document"]["id"]]
        for span in plan["document"]["excerpts"]:
            if not 0 <= span["start"] < span["end"] <= len(original["text"]) or original["text"][span["start"]:span["end"]] != span["text"]:
                raise ValueError("Planned excerpt does not match its source")
    return summary, manifest, documents, plans


def setup(pipeline, identifier, *, budget=70.):
    if budget != 70.:
        raise ValueError("This authorized run has a frozen $70 limit")
    corpus = pipeline.get_corpus(identifier)
    summary, selection, documents, plans = load_selection(pipeline, identifier)
    if corpus["source_sha256"] != summary["source_sha256"] or corpus["id"] != summary["corpus_id"]:
        raise ValueError("Selection belongs to another corpus")
    path = directory(pipeline, identifier) / "manifest.json"
    if path.exists():
        manifest = read(path)
        if manifest["budget_cap_usd"] != 70. or manifest["selection_sha256"] != summary["private_artifacts"]["selection_sha256"]:
            raise ValueError("Cannot rebase budget or change selection on resume")
        if not corpus["jobs"] and corpus["config"].get("budget_limit_usd") is None:
            pipeline.configure(identifier, {"budget_limit_usd": 68.})
        check(pipeline, manifest)
        return manifest, documents, plans
    if corpus["jobs"] or corpus["usage"]["estimated_usd"] or corpus["usage"]["uncertain_reserved_usd"]:
        raise ValueError("Initial setup requires this previously unprocessed corpus")
    manifest = {"version": "half-70-v1", "corpus_id": identifier, "source_sha256": corpus["source_sha256"],
                "created_at": now(), "selected_count": len(documents), "budget_cap_usd": 70.,
                "extraction_stage_cap_usd": 68., "start_spent_usd": 0.,
                "selection_sha256": summary["private_artifacts"]["selection_sha256"],
                "selected_ids_sha256": digest(sorted(documents)),
                "initial_recipes": {stage: pipeline._recipe(stage, corpus["config"]) for stage in ("relevance", "extract", "describe")},
                "routes": corpus["config"]["routes"], "pricing": corpus["config"]["pricing"],
                "authorization": "User requested this selected 12205-publication run with $70 total cap.",
                "notes": ["$2 within the authorized total is reserved for descriptor discovery.",
                          "Raw posts, source excerpts, IDs and model receipts stay private.",
                          "Uncertain paid requests are never automatically replayed; no final politician scores."]}
    write(path, manifest)
    pipeline.configure(identifier, {"budget_limit_usd": 68.})
    return manifest, documents, plans


def check(pipeline, manifest):
    corpus = pipeline.get_corpus(manifest["corpus_id"])
    if corpus["source_sha256"] != manifest["source_sha256"] or corpus["config"]["budget_limit_usd"] not in {68., 70.}:
        raise ValueError("Corpus source or authorized cap changed")
    if corpus["config"]["routes"] != manifest["routes"] or corpus["config"]["pricing"] != manifest["pricing"]:
        raise ValueError("Models or tariffs changed during this frozen run")
    if any(pipeline._recipe(stage, corpus["config"]) != recipe for stage, recipe in manifest["initial_recipes"].items()):
        raise ValueError("Processing recipe changed")
    return corpus


def enqueue_relevance(pipeline, manifest, documents, plans):
    identifier = manifest["corpus_id"]
    config = check(pipeline, manifest)["config"]
    protected = {identifier for identifier, doc in documents.items() if set(doc["cleaning"]["matched_actor_ids"]) - {"paul-kagame"}}
    # Protect comparative coverage even if the total cap eventually stops work.
    plans = sorted(plans, key=lambda row: (row["document"]["id"] not in protected, digest(row["document"]["id"])))
    recipe = pipeline._recipe("relevance", config)
    with pipeline.connection() as db:
        seen = {row[0] for row in db.execute("SELECT unit_id FROM jobs WHERE corpus_id=? AND stage='relevance' AND recipe=?", (identifier, recipe))}
        created = 0
        for number, plan in enumerate(plans, 1):
            document = plan["document"]
            unit = document["id"] + ":excerpts"
            if unit in seen:
                continue
            context = {"document": document, "config": pipeline._task_config("relevance", config)}
            context["config"]["objects"] = compact_objects(config["objects"], document)
            payload = make_task("relevance", context, model=config["routes"]["relevance"], max_output_tokens=MAX_OUTPUT["relevance"])
            job_id = "job-" + digest({"corpus": identifier, "recipe": recipe, "unit": unit, "context": context})[:32]
            db.execute("INSERT INTO jobs(id,corpus_id,stage,unit_id,recipe,context,payload,status,created_at) VALUES(?,?,?,?,?,?,?,'pending',?)",
                       (job_id, identifier, "relevance", unit, recipe, packed(context), packed(payload), now()))
            seen.add(unit)
            created += 1
            if number % 1000 == 0:
                emit({"phase": "enqueue_relevance", "prepared": number, "total": len(plans)})
    return created


def guard_queue(pipeline, manifest, stage, selected_ids):
    check(pipeline, manifest)
    held = validated_uncertain_holds(pipeline, manifest)["job_ids"]
    with pipeline.connection() as db:
        for row in db.execute("SELECT id,stage,status,recipe,context FROM jobs WHERE corpus_id=?", (manifest["corpus_id"],)):
            if row["status"] == "running" or (row["status"] == "uncertain" and row["id"] not in held):
                raise ValueError("Reconcile a running/uncertain request before resuming")
            if row["status"] != "pending":
                continue
            held_extraction = stage == "describe" and row["stage"] == "extract"
            if (row["stage"] != stage and not held_extraction) or row["recipe"] != manifest["initial_recipes"].get(row["stage"]):
                raise ValueError("Unexpected pending stage/recipe")
            if row["stage"] in {"relevance", "extract"} and json.loads(row["context"])["document"]["id"] not in selected_ids:
                raise ValueError("Pending request outside selected half")


def enqueue_extraction(pipeline, manifest, documents):
    """Reuse each frozen relevance context once, without reselecting passages.

    Contexts and IDs match ordinary extract preparation. The compact roster is
    unchanged from the same source passages; no additional actors are inferred.
    """
    identifier = manifest["corpus_id"]
    config = check(pipeline, manifest)["config"]
    recipe = manifest["initial_recipes"]["extract"]
    with pipeline.connection() as db:
        if db.execute("SELECT COUNT(*) FROM jobs WHERE corpus_id=? AND stage='relevance' AND status IN ('pending','running','uncertain','error')", (identifier,)).fetchone()[0]:
            raise ValueError("Finish classification before extraction")
        rows = db.execute("SELECT id,unit_id,context,result FROM jobs WHERE corpus_id=? AND stage='relevance' AND status='done' AND recipe=? ORDER BY created_at,id",
                          (identifier, manifest["initial_recipes"]["relevance"])).fetchall()
        seen = {row[0] for row in db.execute("SELECT unit_id FROM jobs WHERE corpus_id=? AND stage='extract' AND recipe=?", (identifier, recipe))}
        created = 0
        for row in rows:
            if row["unit_id"] in seen:
                continue
            result = json.loads(row["result"])
            if result.get("status") != "relevant":
                continue
            upstream = json.loads(row["context"])
            original = documents.get(upstream["document"]["id"])
            if original is None:
                raise ValueError("Relevant request outside the frozen half")
            document = dict(upstream["document"], **original["source_context"])
            context = {"document": document, "relevance": result, "relevance_job_id": row["id"], "config": upstream["config"]}
            payload = make_task("extract", context, model=config["routes"]["extract"], max_output_tokens=MAX_OUTPUT["extract"])
            job_id = "job-" + digest({"corpus": identifier, "recipe": recipe, "unit": row["unit_id"], "context": context})[:32]
            db.execute("INSERT INTO jobs(id,corpus_id,stage,unit_id,recipe,context,payload,status,created_at) VALUES(?,?,?,?,?,?,?,'pending',?)",
                       (job_id, identifier, "extract", row["unit_id"], recipe, packed(context), packed(payload), now()))
            seen.add(row["unit_id"])
            created += 1
            if created % 1000 == 0:
                emit({"phase": "enqueue_extraction", "created": created})
    return created


def snapshot(pipeline, manifest, stage=None, run=None):
    corpus = pipeline.get_corpus(manifest["corpus_id"])
    jobs = []
    with pipeline.connection() as db:
        for row in db.execute("SELECT * FROM jobs WHERE corpus_id=? ORDER BY created_at,id", (manifest["corpus_id"],)):
            job = dict(row)
            job["active_recipe"] = job["recipe"] == manifest["initial_recipes"].get(job["stage"])
            for key in ("context", "payload", "result"):
                job[key] = json.loads(job[key]) if job[key] else None
            jobs.append(job)
    holds = validated_uncertain_holds(pipeline, manifest)
    result = {"manifest": manifest, "corpus": corpus, "jobs": jobs,
              "uncertain_hold_summary": {"validated_job_count": len(holds["job_ids"]), "retained_reserved_usd": holds["reserved_usd"], "automatic_retry": False},
              "facts": pipeline._facts(manifest["corpus_id"], corpus["config"]),
              "phase_trace": {"stage": stage, "run": run}, "frozen_at": now()}
    write(directory(pipeline, manifest["corpus_id"]) / "results_private.json", result)
    return result


def require_completed_describe(pipeline, manifest):
    """Release the intermediate $2 reserve only after real describe work joins."""
    check(pipeline, manifest)
    held = validated_uncertain_holds(pipeline, manifest)["job_ids"]
    with pipeline.connection() as db:
        unresolved = db.execute("SELECT id,status FROM jobs WHERE corpus_id=? AND status IN ('running','uncertain')", (manifest["corpus_id"],)).fetchall()
        if any(row["status"] == "running" or row["id"] not in held for row in unresolved):
            raise ValueError("Reconcile running/uncertain requests before extraction tail")
        statuses = dict(db.execute("SELECT status,COUNT(*) FROM jobs WHERE corpus_id=? AND stage='describe' AND recipe=? GROUP BY status",
                                   (manifest["corpus_id"], manifest["initial_recipes"]["describe"])))
    if not statuses or set(statuses) - {"done", "review"}:
        raise ValueError("Extraction tail requires a nonempty fully terminal describe stage (done/review)")
    return statuses


def run_stage(pipeline, manifest, stage, documents, *, workers=8, max_jobs=None, extraction_tail=False):
    from src.corpus_pipeline.bounded_workers import run_workers
    from src.corpus_pipeline.receipt_client import ReceiptingClient
    identifier = manifest["corpus_id"]
    guard_queue(pipeline, manifest, stage, set(documents))
    tail_describe_statuses = None
    if extraction_tail:
        if stage != "extract":
            raise ValueError("Extraction tail applies only to extract")
        tail_describe_statuses = require_completed_describe(pipeline, manifest)
        if pipeline._row(identifier)["config"]["budget_limit_usd"] != 70.:
            pipeline.configure(identifier, {"budget_limit_usd": 70.})
    elif stage in {"relevance", "extract"} and pipeline._row(identifier)["config"]["budget_limit_usd"] != 68.:
        # A previous describe preparation may have exposed the final $2.
        # Restoring the stage cap never resets recorded spend or the manifest.
        pipeline.configure(identifier, {"budget_limit_usd": 68.})
    ledger = directory(pipeline, identifier) / "api_receipts_private.jsonl"
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise ValueError("OPENAI_API_KEY must be configured locally")
    trace = run_workers(pipeline.root, identifier, workers=workers, max_jobs=max_jobs,
                        stage=stage,
                        client_factory=lambda: ReceiptingClient(key, ledger),
                        stop_file=directory(pipeline, identifier) / "STOP", emit=emit)
    phase_name = "extract-tail" if extraction_tail else stage
    if extraction_tail:
        trace["budget_policy"] = {"phase": phase_name, "authorized_total_cap_usd": manifest["budget_cap_usd"],
            "intermediate_extraction_cap_usd": 68., "actual_spend_rebased": False,
            "describe_terminal_statuses_at_tail_start": tail_describe_statuses,
            "describe_automatically_refreshed": False,
            "dictionary_evidence_timing": "Describe preceded extraction tail; new tail evidence is not automatically incorporated in its dictionary."}
    write(directory(pipeline, identifier) / ("run-" + phase_name + "-latest.json"), trace)
    snapshot(pipeline, manifest, stage, trace)
    return trace


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--phase", choices=["setup", "relevance", "extract", "describe", "extract-tail", "snapshot"], default="snapshot")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--fact-ids-file", type=Path)
    parser.add_argument("--hold-uncertain-job", action="append", default=[], metavar="JOB_ID",
                        help="Explicitly retain a verified HTTP503 worst-case reservation; never retry this job")
    args = parser.parse_args(argv)
    pipeline = CorpusPipeline(args.root)
    manifest, documents, plans = setup(pipeline, args.corpus_id)
    for job_id in args.hold_uncertain_job:
        holds = acknowledge_uncertain_job(pipeline, manifest, job_id)
        emit({"phase": "uncertain_reservation_held", "held_count": len(holds["job_ids"]), "retained_reserved_usd": holds["reserved_usd"], "retried": False})
    if args.phase == "relevance":
        emit({"phase": "relevance_prepared", "created": enqueue_relevance(pipeline, manifest, documents, plans)})
    elif args.phase in {"extract", "extract-tail"}:
        if args.phase == "extract-tail":
            require_completed_describe(pipeline, manifest)
        emit({"phase": "extract_prepared", "created": enqueue_extraction(pipeline, manifest, documents)})
    elif args.phase == "describe":
        if not args.fact_ids_file:
            raise ValueError("Describe requires a balanced, explicitly selected fact_ids file")
        facts = read(args.fact_ids_file)
        # Spend at most the user-authorized $70 total, not another $70.
        pipeline.configure(args.corpus_id, {"budget_limit_usd": 70.})
        emit(pipeline.prepare(args.corpus_id, stage="describe", limit=5000, fact_ids=facts))
    if args.execute:
        if args.phase not in {"relevance", "extract", "describe", "extract-tail"}:
            raise ValueError("--execute requires an explicit paid phase")
        from dotenv import load_dotenv
        load_dotenv(args.root / ".env", override=False)
        stage = "extract" if args.phase == "extract-tail" else args.phase
        emit({"phase": "worker_complete", "result": run_stage(pipeline, manifest, stage, documents,
              workers=args.workers, max_jobs=args.max_jobs, extraction_tail=args.phase == "extract-tail")})
    else:
        snapshot(pipeline, manifest, args.phase)
        emit({"phase": "prepared_only", "api_calls": 0})


if __name__ == "__main__":
    main()
