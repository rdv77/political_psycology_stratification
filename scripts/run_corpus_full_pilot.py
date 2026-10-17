"""Resumable full-corpus pilot with a frozen additional budget and private receipts.

Importing this module never loads credentials or calls an API. Only an explicit
paid phase with --execute can run prepared tasks. Earlier pilot files are intact.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.corpus_pipeline.service import CorpusPipeline, chunks, digest

PHASES = ("setup", "relevance", "extract", "describe", "summary")


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    temporary.replace(path)


def _positive_budget(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("Additional budget must be a positive finite number")
    return float(value)


def _same(a, b):
    return type(a) in (int, float) and type(b) in (int, float) and abs(a - b) <= 1e-12


def directory(pipeline, identifier):
    if not isinstance(identifier, str) or not re.fullmatch(r"corpus-[A-Za-z0-9_-]+", identifier):
        raise ValueError("Invalid corpus ID")
    return pipeline.runtime / "pilots" / identifier / "full-v2"


def _jobs(pipeline, identifier):
    with pipeline.connection() as db:
        return [dict(row) for row in db.execute("SELECT * FROM jobs WHERE corpus_id=? ORDER BY created_at,id", (identifier,))]


def _active(jobs, pipeline, config):
    return [job for job in jobs if job["recipe"] == pipeline._recipe(job["stage"], config)]


def _unit_coverage(expected, jobs, terminal):
    statuses = {}
    for job in jobs:
        statuses.setdefault(job["unit_id"], []).append(job["status"])
    prepared = expected & statuses.keys()
    finished = {unit for unit in prepared if len(statuses[unit]) == 1 and statuses[unit][0] in terminal}
    unprepared = expected - statuses.keys()
    duplicates = sum(len(values) > 1 for values in statuses.values())
    unexpected = statuses.keys() - expected
    return {"expected": len(expected), "prepared": len(prepared), "finished": len(finished),
            "unprepared": len(unprepared), "unfinished": len(expected - finished),
            "duplicate_units": duplicates, "unexpected_units": len(unexpected),
            "ready": len(finished) == len(expected) and not duplicates and not unexpected,
            "unprepared_unit_ids": sorted(unprepared), "unexpected_unit_ids": sorted(unexpected)}


def preparation_coverage(pipeline, identifier, *, config=None, jobs=None):
    """Actual eligible chunk frame versus unique current-recipe relevance jobs.

    A terminal review counts as finished classification, while remaining unknown
    for extraction. Empty prepared queues alone never establish corpus coverage.
    """
    config = config or pipeline.get_corpus(identifier)["config"]
    expected = set()
    with pipeline.connection() as db:
        for row in db.execute("SELECT data FROM documents WHERE corpus_id=? AND status='unique'", (identifier,)):
            document = json.loads(row[0])
            if pipeline._rule_exclusion(document, config) is None:
                if config.get("text_selection", {}).get("mode", "full") == "full":
                    expected.update(f"{document['id']}:{start}" for start, end, text in chunks(document["text"]))
                else:
                    expected.add(f"{document['id']}:excerpts")
    recipe = pipeline._recipe("relevance", config)
    if jobs is None:
        with pipeline.connection() as db:
            relevance = [dict(row) for row in db.execute(
                "SELECT unit_id,status FROM jobs WHERE corpus_id=? AND stage='relevance' AND recipe=?", (identifier, recipe))]
    else:
        relevance = [j for j in jobs if j["stage"] == "relevance" and j["recipe"] == recipe]
    return _unit_coverage(expected, relevance, {"done", "review"})


def extraction_coverage(pipeline, identifier, *, config=None, jobs=None):
    """Every current done/relevant classification unit needs a done extraction.

    A rejected/refused extraction is not a completed valid fact yield, even if
    billed. Classification review never creates an expected extraction unit.
    """
    config = config or pipeline.get_corpus(identifier)["config"]
    relevance_recipe = pipeline._recipe("relevance", config)
    extraction_recipe = pipeline._recipe("extract", config)
    if jobs is None:
        with pipeline.connection() as db:
            jobs = [dict(row) for row in db.execute(
                "SELECT stage,unit_id,status,recipe,CASE WHEN stage='relevance' THEN result END AS result FROM jobs "
                "WHERE corpus_id=? AND ((stage='relevance' AND recipe=?) OR (stage='extract' AND recipe=?))",
                (identifier, relevance_recipe, extraction_recipe))]
    expected, extractions = set(), []
    for job in jobs:
        if job["stage"] == "relevance" and job["recipe"] == relevance_recipe and job["status"] == "done":
            result = job.get("result")
            if isinstance(result, str):
                result = json.loads(result)
            if isinstance(result, dict) and result.get("status") == "relevant":
                expected.add(job["unit_id"])
        elif job["stage"] == "extract" and job["recipe"] == extraction_recipe:
            extractions.append(job)
    return _unit_coverage(expected, extractions, {"done"})


def _compact_readiness(value):
    return {key: _compact_readiness(item) if isinstance(item, dict) else item
            for key, item in value.items() if not key.endswith("_unit_ids")}


def setup(pipeline, identifier, additional_budget=5., *, allow_budget_increase=False):
    """Freeze start_spent once; resume never rebases cap on later expenditure."""
    budget = _positive_budget(additional_budget)
    path = directory(pipeline, identifier) / "manifest.json"
    corpus = pipeline.get_corpus(identifier)
    current_cap = corpus["config"].get("budget_limit_usd")
    if path.exists():
        manifest = read(path)
        _check_identity(manifest, corpus, budget)
        if manifest["setup_state"] == "ready":
            _check_cap(manifest, corpus)
            return manifest
        if manifest["setup_state"] != "planned":
            raise ValueError("Unknown setup receipt state")
    else:
        start = corpus["usage"]["estimated_usd"]
        cap = start + budget
        if current_cap is not None and cap > current_cap + 1e-12 and not allow_budget_increase:
            raise ValueError("Increasing the corpus cap requires setup --allow-budget-increase; no cap changed")
        jobs = _jobs(pipeline, identifier)
        with pipeline.connection() as db:
            docs = [json.loads(row[0]) for row in db.execute("SELECT data FROM documents WHERE corpus_id=? AND status='unique' ORDER BY source_row", (identifier,))]
        eligible = [d for d in docs if pipeline._rule_exclusion(d, corpus["config"]) is None]
        manifest = {"version": "full-v2", "corpus_id": identifier,
                    "created_at": datetime.now(timezone.utc).isoformat(), "setup_state": "planned",
                    "source_sha256": corpus["source_sha256"], "additional_budget_usd": budget,
                    "start_spent_usd": start, "start_outstanding_reserved_usd": corpus["usage"]["uncertain_reserved_usd"],
                    "budget_cap_usd": cap, "previous_budget_limit_usd": current_cap,
                    "budget_increase_authorized": bool(allow_budget_increase),
                    "starting_job_ids": [j["id"] for j in jobs], "initial_routes": corpus["config"]["routes"],
                    "initial_recipes": {s: pipeline._recipe(s, corpus["config"]) for s in ("relevance", "extract", "describe")},
                    "population": {"raw": corpus["profile"]["rows_total"], "unique": len(docs), "eligible_unique": len(eligible),
                                   "rule_exclusions": len(docs) - len(eligible)},
                    "scope": "all unique documents through current country/date rules; reuse exact completed active tasks",
                    "notes": ["The total cap is frozen at recorded starting spend plus the new additional budget.",
                              "Existing outstanding reservations reduce available room and are not treated as paid zero.",
                              "No automatic retries; review is traceable and excluded from extraction; describe is a separate explicit phase."]}
        write(path, manifest)
    cap = manifest["budget_cap_usd"]
    if not _same(current_cap, cap):
        if current_cap is not None and cap > current_cap + 1e-12 and not allow_budget_increase:
            raise ValueError("Finishing this planned setup requires --allow-budget-increase; its baseline is unchanged")
        pipeline.configure(identifier, {"budget_limit_usd": cap})
    manifest["setup_state"] = "ready"
    manifest["budget_increase_authorized"] = bool(manifest.get("budget_increase_authorized") or allow_budget_increase)
    write(path, manifest)
    return manifest


def _check_identity(manifest, corpus, budget):
    if manifest.get("version") != "full-v2" or manifest.get("corpus_id") != corpus["id"] or manifest.get("source_sha256") != corpus["source_sha256"]:
        raise ValueError("Full-pilot manifest does not match this corpus")
    if not _same(manifest.get("additional_budget_usd"), budget):
        raise ValueError("Requested additional budget differs from the frozen manifest; no rebasing or expansion allowed")
    if not _same(manifest.get("budget_cap_usd"), manifest.get("start_spent_usd", 0) + budget):
        raise ValueError("Invalid frozen budget arithmetic")


def _check_cap(manifest, corpus):
    if manifest.get("setup_state") != "ready" or not _same(corpus["config"].get("budget_limit_usd"), manifest["budget_cap_usd"]):
        raise ValueError("Stored corpus cap differs from the frozen ready manifest; ordinary phases cannot change it")


def load_manifest(pipeline, identifier, additional_budget=5.):
    path = directory(pipeline, identifier) / "manifest.json"
    if not path.exists():
        raise ValueError("Run explicit setup first; no budget is created during a processing phase")
    manifest = read(path)
    corpus = pipeline.get_corpus(identifier)
    _check_identity(manifest, corpus, _positive_budget(additional_budget))
    _check_cap(manifest, corpus)
    return manifest


def snapshot(pipeline, manifest, *, phase_trace=None):
    identifier = manifest["corpus_id"]
    corpus = pipeline.get_corpus(identifier)
    rows = _jobs(pipeline, identifier)
    jobs = []
    for original in rows:
        job = dict(original)
        job["active_recipe"] = job["recipe"] == pipeline._recipe(job["stage"], corpus["config"])
        for key in ("context", "payload", "result"):
            job[key] = json.loads(job[key]) if job[key] else None
        jobs.append(job)
    active = [j for j in jobs if j["active_recipe"]]
    relevance = [j for j in active if j["stage"] == "relevance"]
    excluded = [{"job_id": j["id"], "document_id": j["context"]["document"]["id"], "status": j["status"],
                 "reason": (j.get("result") or {}).get("reason") or j.get("error")}
                for j in relevance if j["status"] == "review"]
    usage = corpus["usage"]
    classification = preparation_coverage(pipeline, identifier, config=corpus["config"], jobs=jobs)
    extraction = extraction_coverage(pipeline, identifier, config=corpus["config"], jobs=jobs)
    output = {"manifest": manifest, "corpus": corpus, "jobs": jobs,
              "facts": pipeline._facts(identifier, corpus["config"]),
              "readiness": {"classification": classification, "extraction": extraction,
                            "ready": classification["ready"] and extraction["ready"]},
              "accounting": {"all_recipes_recorded_cost_usd": usage["estimated_usd"],
                             "outstanding_reserved_usd": usage["uncertain_reserved_usd"],
                             "start_spent_usd": manifest["start_spent_usd"],
                             "new_recorded_cost_delta_usd": usage["estimated_usd"] - manifest["start_spent_usd"],
                             "remaining_total_cap_usd": manifest["budget_cap_usd"] - usage["estimated_usd"] - usage["uncertain_reserved_usd"]},
              "active_stage_statuses": {s: dict(Counter(j["status"] for j in active if j["stage"] == s)) for s in ("relevance", "extract", "describe", "score")},
              "extraction_trace": {"done_relevant_chunks": sum(j["status"] == "done" and isinstance(j.get("result"), dict) and j["result"].get("status") == "relevant" for j in relevance),
                                   "excluded_review_chunks": excluded},
              "phase_trace": phase_trace}
    write(directory(pipeline, identifier) / "results_private.json", output)
    return output


def _emit_default(value):
    print(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False), flush=True)


def run_phase(pipeline, manifest, stage, *, execute=False, max_jobs=5000, batch_size=5,
              fact_ids=None, client=None, client_factory=None, workers=1, emit=None):
    if stage not in {"relevance", "extract", "describe"}:
        raise ValueError("Processing phase must be relevance, extract or describe")
    if type(max_jobs) is not int or not 1 <= max_jobs <= 5000 or type(batch_size) is not int or not 1 <= batch_size <= 5:
        raise ValueError("max_jobs1..5000 and per-worker batch_size1..5 required")
    if type(workers) is not int or not 1 <= workers <= 3 or (workers > 1 and client is not None) or (client is not None and client_factory is not None):
        raise ValueError("workers1..3; parallel tests require separate clients via client_factory")
    if fact_ids is not None and (stage != "describe" or not isinstance(fact_ids, list) or not fact_ids
                                 or any(not isinstance(x, str) or not x for x in fact_ids) or len(set(fact_ids)) != len(fact_ids)):
        raise ValueError("Nonempty unique fact IDs apply only to describe")
    emit = emit or _emit_default
    identifier = manifest["corpus_id"]
    corpus = pipeline.get_corpus(identifier)
    _check_identity(manifest, corpus, manifest["additional_budget_usd"])
    _check_cap(manifest, corpus)
    active = _active(_jobs(pipeline, identifier), pipeline, corpus["config"])
    if stage in {"extract", "describe"}:
        readiness = preparation_coverage(pipeline, identifier, config=corpus["config"], jobs=active)
        if not readiness["ready"]:
            raise ValueError("Classification coverage is incomplete: " + json.dumps(_compact_readiness(readiness), separators=(",", ":")))
        if stage == "describe":
            extraction = extraction_coverage(pipeline, identifier, config=corpus["config"], jobs=active)
            if not extraction["ready"]:
                raise ValueError("Extraction coverage is incomplete: " + json.dumps(_compact_readiness(extraction), separators=(",", ":")))
    upstream = "relevance" if stage == "extract" else "extract" if stage == "describe" else None
    if upstream:
        upstream_jobs = [j for j in active if j["stage"] == upstream]
        if not upstream_jobs or any(j["status"] in {"pending", "running", "uncertain", "error"} for j in upstream_jobs):
            raise ValueError("Finish or reconcile the active " + upstream + " phase first; review remains excluded and traceable")
    kwargs = {"fact_ids": fact_ids} if fact_ids is not None else {}
    prepared = pipeline.prepare(identifier, stage=stage, limit=5000, **kwargs)
    trace = {"phase": stage, "execute": bool(execute), "attempted": 0, "stopped_reason": "prepared_only", "workers": workers,
             "fact_selection_sha256": digest(fact_ids) if fact_ids is not None else None,
             "fact_selection_count": len(fact_ids) if fact_ids is not None else None,
             "prepared": prepared}
    write(directory(pipeline, identifier) / ("prepare-" + stage + ".json"), trace)
    emit({"phase": stage, "prepared": prepared, "execute": bool(execute)})
    if not execute:
        return snapshot(pipeline, manifest, phase_trace=trace)
    while trace["attempted"] < max_jobs:
        corpus = pipeline.get_corpus(identifier)
        _check_cap(manifest, corpus)
        all_jobs = _jobs(pipeline, identifier)
        active = _active(all_jobs, pipeline, corpus["config"])
        if any(j["status"] in {"running", "uncertain"} for j in all_jobs):
            trace["stopped_reason"] = "unreconciled_or_running_request"
            break
        if any(j["status"] == "error" and j["stage"] == stage for j in active):
            trace["stopped_reason"] = "existing_stage_error"
            break
        pending = [j for j in active if j["status"] == "pending"]
        if any(j["stage"] != stage for j in pending):
            trace["stopped_reason"] = "other_active_stage_pending"
            break
        if not pending:
            coverage = (preparation_coverage if stage == "relevance" else extraction_coverage if stage == "extract" else None)
            readiness = coverage(pipeline, identifier, config=corpus["config"], jobs=active) if coverage else None
            trace["stopped_reason"] = ("incomplete_preparation" if readiness and readiness["unprepared"] else
                                       "incomplete_processing" if readiness and not readiness["ready"] else "active_queue_empty")
            break
        usage = corpus["usage"]
        remaining = manifest["budget_cap_usd"] - usage["estimated_usd"] - usage["uncertain_reserved_usd"]
        if remaining <= 0:
            trace["stopped_reason"] = "frozen_budget_exhausted"
            break
        available = min(max_jobs - trace["attempted"], len(pending))
        allocations = []
        for _ in range(workers):
            if available <= 0:
                break
            allocations.append(min(batch_size, available))
            available -= allocations[-1]

        def worker(count):
            # Every parallel run owns its pipeline/client. Atomic DB claiming
            # and the common corpus cap account for other in-flight reserves.
            worker_pipeline = CorpusPipeline(pipeline.root) if workers > 1 else pipeline
            worker_client = client_factory() if client_factory is not None else client
            try:
                return worker_pipeline.run(identifier, max_jobs=count, budget_usd=remaining,
                                           **({"client": worker_client} if worker_client is not None else {}))
            finally:
                if client_factory is not None and worker_client is not None and hasattr(worker_client, "close"):
                    worker_client.close()

        worker_errors = []
        if workers == 1:
            try:
                runs = [worker(allocations[0])]
            except Exception as exc:
                runs = []
                worker_errors.append(type(exc).__name__)
        else:
            # Join every bounded worker before a single parent snapshot/write.
            with ThreadPoolExecutor(max_workers=len(allocations)) as executor:
                futures = [executor.submit(worker, count) for count in allocations]
                runs = []
                for future in futures:
                    try:
                        runs.append(future.result())
                    except Exception as exc:
                        worker_errors.append(type(exc).__name__)
        trace["attempted"] += sum(run["attempted"] for run in runs)
        for run in runs:
            write(directory(pipeline, identifier) / (run["run_id"] + ".json"), run)
        if worker_errors:
            trace["worker_exception_types"] = worker_errors
        snap = snapshot(pipeline, manifest, phase_trace=trace)
        emit({"phase": stage, "runs": runs, "worker_exception_types": worker_errors,
              "accounting": snap["accounting"], "active_statuses": snap["active_stage_statuses"][stage],
              "readiness": _compact_readiness(snap["readiness"])})
        stop = next((run for run in runs if run["failed"] or run["uncertain"] or run["stopped_reason"] not in {"job_limit", "queue_empty"}), None)
        if worker_errors or stop is not None or sum(run["attempted"] for run in runs) == 0:
            trace["stopped_reason"] = ("worker_exception_reconcile" if worker_errors else
                                       "request_error" if stop is not None and stop["failed"] else
                                       stop["stopped_reason"] if stop is not None else "no_progress")
            break
    else:
        trace["stopped_reason"] = "invocation_job_limit"
    return snapshot(pipeline, manifest, phase_trace=trace)


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--phase", choices=PHASES, default="summary")
    parser.add_argument("--budget-usd", "--additional-budget-usd", type=float, default=5., help="Frozen NEW additional budget, not total historic cap")
    parser.add_argument("--allow-budget-increase", action="store_true", help="Setup only: explicitly authorize the one-time corpus cap increase")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, choices=range(1, 6), default=5)
    parser.add_argument("--workers", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--fact-ids-file", type=Path, help="Describe only: JSON list or {fact_ids:[...]} of curated active facts")
    args = parser.parse_args(argv)
    if args.allow_budget_increase and args.phase != "setup":
        parser.error("--allow-budget-increase applies only to explicit setup")
    if args.execute and args.phase not in {"relevance", "extract", "describe"}:
        parser.error("--execute applies only to relevance, extract or describe")
    if args.fact_ids_file and args.phase != "describe":
        parser.error("--fact-ids-file applies only to describe")
    try:
        if args.execute:
            try:
                from dotenv import load_dotenv
                load_dotenv(args.root / ".env", override=False)
            except ImportError:
                pass
        pipeline = CorpusPipeline(args.root)
        if args.phase == "setup":
            manifest = setup(pipeline, args.corpus_id, args.budget_usd, allow_budget_increase=args.allow_budget_increase)
            results = snapshot(pipeline, manifest, phase_trace={"phase": "setup", "api_calls": 0})
        else:
            manifest = load_manifest(pipeline, args.corpus_id, args.budget_usd)
            if args.phase == "summary":
                results = snapshot(pipeline, manifest, phase_trace={"phase": "summary", "api_calls": 0})
            else:
                fact_ids = read(args.fact_ids_file) if args.fact_ids_file else None
                if isinstance(fact_ids, dict):
                    if "fact_ids" not in fact_ids:
                        raise ValueError("Curation JSON requires fact_ids; cannot silently select all facts")
                    fact_ids = fact_ids["fact_ids"]
                if args.fact_ids_file and fact_ids is None:
                    raise ValueError("Curation file must contain an explicit nonempty fact ID list")
                results = run_phase(pipeline, manifest, args.phase, execute=args.execute, max_jobs=args.max_jobs,
                                    batch_size=args.batch_size, fact_ids=fact_ids, workers=args.workers)
        _emit_default({"corpus_id": args.corpus_id, "phase": args.phase, "accounting": results["accounting"],
                       "active_stage_statuses": results["active_stage_statuses"], "phase_trace": results["phase_trace"],
                       "readiness": _compact_readiness(results["readiness"]),
                       "private_results": str(directory(pipeline, args.corpus_id) / "results_private.json")})
    except (ValueError, FileNotFoundError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
