"""Bounded concurrent queue consumption; billing/claiming stays in the service.

Each worker owns one persistent pipeline and HTTP client. No request is retried.
The caller prepares one stage and verifies its document scope before starting.
Progress reads aggregate SQL only; source texts and payloads are never loaded.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
import math
import os
from pathlib import Path
import threading
import time

from .models import OpenAIResponsesHTTP
from .service import CorpusPipeline


def _aggregate(pipeline, identifier):
    with pipeline.connection() as db:
        statuses = dict(db.execute("SELECT status,COUNT(*) FROM jobs WHERE corpus_id=? GROUP BY status", (identifier,)))
        usage = dict(db.execute("""SELECT COALESCE(SUM(input_tokens),0) input_tokens,
            COALESCE(SUM(output_tokens),0) output_tokens, COALESCE(SUM(cost_usd),0) cost_usd,
            COALESCE(SUM(CASE WHEN status IN ('running','uncertain') THEN reserved_usd ELSE 0 END),0) reserved_usd
            FROM jobs WHERE corpus_id=?""", (identifier,)).fetchone())
    return {"statuses": statuses, **usage}


def run_workers(root, identifier, *, workers=8, max_jobs=None, client_factory=None,
                emit=None, stop_file=None):
    """Drain a prepared stage subject to a fixed, existing corpus budget.

    ``max_jobs`` bounds total attempts across workers, including paid failures.
    ``client_factory()`` is called once inside each worker and its returned
    client is closed there. Default construction reads OPENAI_API_KEY only;
    loading .env and authorising API spending belong to the calling CLI.

    An error, uncertain bill, exception or stop file prevents subsequent work;
    already claimed HTTP requests finish before return. Near the cap, a blocked
    reservation stops conservatively; it does not raise/rebase the cap or retry.
    An empty pending queue does not mean preparation or the research is complete.
    """
    if type(workers) is not int or not 1 <= workers <= 64:
        raise ValueError("workers must be an integer from 1 to 64")
    if max_jobs is not None and (type(max_jobs) is not int or max_jobs < 1):
        raise ValueError("max_jobs must be a positive integer or null")
    if client_factory is not None and not callable(client_factory):
        raise ValueError("client_factory must be callable")
    pipeline = CorpusPipeline(root)
    config = pipeline._row(identifier)["config"]
    cap = config.get("budget_limit_usd")
    if type(cap) not in (int, float) or not math.isfinite(cap) or cap <= 0:
        raise ValueError("An existing positive corpus budget_limit_usd is required")
    stop_path = Path(stop_file) if stop_file is not None else None
    before = _aggregate(pipeline, identifier)
    event, lock = threading.Event(), threading.Lock()
    counters = {"attempted": 0, "completed": 0, "review": 0, "failed": 0, "uncertain": 0,
                "run_invocations": 0}
    errors, reasons = [], set()
    started = time.monotonic()

    def stop(reason):
        with lock:
            reasons.add(reason)
        event.set()

    def snapshot(final=False):
        aggregate = _aggregate(pipeline, identifier)
        with lock:
            counts, stop_reasons, failures = dict(counters), sorted(reasons), list(errors)
        priority = ("worker_exception", "uncertain_request", "error", "stop_file", "configuration_changed",
                    "corpus_budget_reservation", "budget_reservation", "max_jobs", "queue_empty")
        reason = next((value for value in priority if value in stop_reasons), "queue_empty" if final else "running")
        total, reserved = aggregate["cost_usd"], aggregate["reserved_usd"]
        return {**counts, "workers": workers, "max_jobs": max_jobs,
                "stopped_reason": reason, "stop_reasons": stop_reasons, "errors": failures,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "statuses": aggregate["statuses"],
                "accounting": {"budget_limit_usd": cap, "cost_before_usd": before["cost_usd"],
                    "cost_usd_total": total, "recorded_cost_delta_usd": total - before["cost_usd"],
                    "outstanding_reserved_usd": reserved, "available_usd": max(0., cap - total - reserved),
                    "input_tokens": aggregate["input_tokens"], "output_tokens": aggregate["output_tokens"],
                    "input_tokens_delta": aggregate["input_tokens"] - before["input_tokens"],
                    "output_tokens_delta": aggregate["output_tokens"] - before["output_tokens"]}}

    if stop_path is not None and stop_path.exists():
        stop("stop_file")
    elif not before["statuses"].get("pending", 0):
        stop("queue_empty")
    if event.is_set():
        result = snapshot(final=True)
        if emit:
            emit(result)
        return result
    if client_factory is None:
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise ValueError("OPENAI_API_KEY must be configured by the calling CLI")
        client_factory = lambda: OpenAIResponsesHTTP(key)

    def worker(number):
        client = None
        try:
            local = CorpusPipeline(root)
            client = client_factory()
            while not event.is_set():
                if stop_path is not None and stop_path.exists():
                    stop("stop_file")
                    break
                state = _aggregate(local, identifier)
                if not state["statuses"].get("pending", 0):
                    with lock:
                        reasons.add("queue_empty")
                    break
                remaining = cap - state["cost_usd"] - state["reserved_usd"]
                if remaining <= 1e-12:
                    stop("corpus_budget_reservation")
                    break
                with lock:
                    if event.is_set():
                        break
                    if max_jobs is not None and counters["run_invocations"] >= max_jobs:
                        reasons.add("max_jobs")
                        break
                    counters["run_invocations"] += 1
                result = local.run(identifier, max_jobs=1, budget_usd=remaining, client=client)
                with lock:
                    for key in ("completed", "review", "failed", "uncertain", "attempted"):
                        counters[key] += result.get(key, 0)
                if result.get("uncertain"):
                    stop("uncertain_request")
                elif result.get("failed"):
                    stop("error")
                elif result.get("stopped_reason") in {"corpus_budget_reservation", "budget_reservation", "configuration_changed"}:
                    stop(result["stopped_reason"])
                elif not result.get("attempted"):
                    with lock:
                        reasons.add("queue_empty")
                    break
        except Exception as exc:
            # Exception strings may contain source data, HTTP bodies or secrets.
            with lock:
                errors.append({"worker": number, "exception_type": type(exc).__name__})
            stop("worker_exception")
        finally:
            if client is not None and hasattr(client, "close"):
                try:
                    client.close()
                except Exception as exc:
                    with lock:
                        errors.append({"worker": number, "exception_type": type(exc).__name__, "during": "close"})
                    stop("worker_exception")

    last_time, last_count = started, 0
    with ThreadPoolExecutor(max_workers=min(workers, max_jobs or workers)) as pool:
        futures = [pool.submit(worker, number) for number in range(min(workers, max_jobs or workers))]
        pending = set(futures)
        try:
            while pending:
                _, pending = wait(pending, timeout=.25)
                if stop_path is not None and stop_path.exists():
                    stop("stop_file")
                with lock:
                    count = counters["attempted"]
                current = time.monotonic()
                if emit and pending and (current - last_time >= 30 or count - last_count >= 100):
                    emit(snapshot())
                    last_time, last_count = current, count
        except BaseException:
            stop("interrupted")
            raise
    result = snapshot(final=True)
    if emit:
        emit(result)
    return result
