"""Execute the frozen half-sample under one durable, non-rebased API cap.

Source material, credentials, queues and receipts remain local. Explicit
--execute is required to send anything. Resume cannot increase the frozen cap.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_full_corpus import write
from src.corpus_pipeline.models import make_task
from src.corpus_pipeline.service import CorpusPipeline, compact_objects, digest, now, packed, MAX_OUTPUT


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def emit(value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def directory(pipeline, identifier):
    return pipeline.runtime / "pilots" / identifier / "half-70-v1"


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
    with pipeline.connection() as db:
        for row in db.execute("SELECT stage,status,recipe,context FROM jobs WHERE corpus_id=?", (manifest["corpus_id"],)):
            if row["status"] in {"running", "uncertain"}:
                raise ValueError("Reconcile a running/uncertain request before resuming")
            if row["status"] != "pending":
                continue
            if row["stage"] != stage or row["recipe"] != manifest["initial_recipes"][stage]:
                raise ValueError("Unexpected pending stage/recipe")
            if stage in {"relevance", "extract"} and json.loads(row["context"])["document"]["id"] not in selected_ids:
                raise ValueError("Pending request outside selected half")


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
    result = {"manifest": manifest, "corpus": corpus, "jobs": jobs,
              "facts": pipeline._facts(manifest["corpus_id"], corpus["config"]),
              "phase_trace": {"stage": stage, "run": run}, "frozen_at": now()}
    write(directory(pipeline, manifest["corpus_id"]) / "results_private.json", result)
    return result


def run_stage(pipeline, manifest, stage, documents, *, workers=8, max_jobs=None):
    from src.corpus_pipeline.bounded_workers import run_workers
    from src.corpus_pipeline.receipt_client import ReceiptingClient
    identifier = manifest["corpus_id"]
    guard_queue(pipeline, manifest, stage, set(documents))
    ledger = directory(pipeline, identifier) / "api_receipts_private.jsonl"
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise ValueError("OPENAI_API_KEY must be configured locally")
    trace = run_workers(pipeline.root, identifier, workers=workers, max_jobs=max_jobs,
                        client_factory=lambda: ReceiptingClient(key, ledger),
                        stop_file=directory(pipeline, identifier) / "STOP", emit=emit)
    write(directory(pipeline, identifier) / ("run-" + stage + "-latest.json"), trace)
    snapshot(pipeline, manifest, stage, trace)
    return trace


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--phase", choices=["setup", "relevance", "extract", "describe", "snapshot"], default="snapshot")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--fact-ids-file", type=Path)
    args = parser.parse_args(argv)
    pipeline = CorpusPipeline(args.root)
    manifest, documents, plans = setup(pipeline, args.corpus_id)
    if args.phase == "relevance":
        emit({"phase": "relevance_prepared", "created": enqueue_relevance(pipeline, manifest, documents, plans)})
    elif args.phase == "extract":
        with pipeline.connection() as db:
            if db.execute("SELECT COUNT(*) FROM jobs WHERE corpus_id=? AND stage='relevance' AND status IN ('pending','running','uncertain','error')", (args.corpus_id,)).fetchone()[0]:
                raise ValueError("Finish classification before extraction")
        # Existing preparation reuses bounded source passages and relevant IDs.
        for _ in range(3):
            prepared = pipeline.prepare(args.corpus_id, stage="extract", limit=5000, document_ids=list(documents))
            emit({"phase": "extract_prepared", "created": prepared["created"]})
            if prepared["created"] < 5000:
                break
    elif args.phase == "describe":
        if not args.fact_ids_file:
            raise ValueError("Describe requires a balanced, explicitly selected fact_ids file")
        facts = read(args.fact_ids_file)
        # Spend at most the user-authorized $70 total, not another $70.
        pipeline.configure(args.corpus_id, {"budget_limit_usd": 70.})
        emit(pipeline.prepare(args.corpus_id, stage="describe", limit=5000, fact_ids=facts))
    if args.execute:
        if args.phase not in {"relevance", "extract", "describe"}:
            raise ValueError("--execute requires an explicit paid phase")
        from dotenv import load_dotenv
        load_dotenv(args.root / ".env", override=False)
        emit({"phase": "worker_complete", "result": run_stage(pipeline, manifest, args.phase, documents, workers=args.workers, max_jobs=args.max_jobs)})
    else:
        snapshot(pipeline, manifest, args.phase)
        emit({"phase": "prepared_only", "api_calls": 0})


if __name__ == "__main__":
    main()
