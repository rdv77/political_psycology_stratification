"""Real SQLite claims with synthetic clients; no network or canonical data."""
import csv
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import time
import unittest

from src.corpus_pipeline.bounded_workers import run_workers, _safe_exception
from src.corpus_pipeline.models import ModelResponseError, estimate_cost, payload_input_bound, make_task
from src.corpus_pipeline.service import CorpusPipeline


class BoundedWorkersTests(unittest.TestCase):
    def fixture(self, count=20):
        base = Path(__file__).resolve().parents[1] / ".service_runtime/bounded_workers_tests"
        base.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=base)
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        source = root / "synthetic.csv"
        with source.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["text"])
            writer.writerows([[f"Alpha North proposes synthetic measure {index}. PRIVATE_SOURCE"] for index in range(count)])
        pipeline = CorpusPipeline(root)
        corpus = pipeline.import_csv(source, config={"scope": "Synthetic", "budget_limit_usd": 70.,
            "objects": [{"id": "alpha", "name": "Alpha North"}]})
        pipeline.prepare(corpus["id"], limit=count)
        return root, pipeline, corpus["id"]

    def factory(self, seen, clients, lock, *, barrier=None, stop_file=None, failure=None, delay=.005, check=None):
        class Client:
            def __init__(self):
                self.calls, self.closed = 0, False
                with lock:
                    clients.append(self)
                if barrier:
                    barrier.wait(timeout=5)
            def call(self, payload):
                context = json.loads(payload["input"][0]["content"][0]["text"])
                with lock:
                    seen.append(context["document"]["id"])
                    self.calls += 1
                if check:
                    check()
                if stop_file:
                    stop_file.write_text("stop", encoding="utf-8")
                time.sleep(delay)
                if failure:
                    raise failure
                data = {"status": "review" if self.calls == 1 else "relevant", "actor_ids": ["alpha"],
                        "language": "en", "reason": "Synthetic"}
                return {"data": data, "usage": {"input_tokens": 100, "output_tokens": 20}, "response_id": "synthetic"}
            def close(self):
                self.closed = True
        return Client

    def test_parallel_distinct_claims_global_max_and_persistent_clients(self):
        root, pipeline, identifier = self.fixture()
        seen, clients, lock, emitted = [], [], threading.Lock(), []
        result = run_workers(root, identifier, workers=3, max_jobs=11,
            client_factory=self.factory(seen, clients, lock, barrier=threading.Barrier(3)), emit=emitted.append)
        self.assertEqual(result["attempted"], 11)
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(len(clients), 3)
        self.assertTrue(all(c.closed for c in clients))
        self.assertTrue(any(c.calls > 1 for c in clients))
        self.assertEqual(result["review"], 3)  # Review does not stop the workers.
        self.assertEqual(result["completed"], 8)
        self.assertEqual(result["statuses"]["pending"], 9)
        self.assertEqual(result["stopped_reason"], "max_jobs")
        self.assertNotIn("PRIVATE_SOURCE", json.dumps(emitted))
        self.assertLessEqual(result["accounting"]["cost_usd_total"], 70)

    def test_shared_corpus_cap_includes_inflight_reservations(self):
        root, pipeline, identifier = self.fixture()
        with pipeline.connection() as db:
            payload = json.loads(db.execute("SELECT payload FROM jobs LIMIT 1").fetchone()[0])
        config = pipeline._row(identifier)["config"]
        reserve = estimate_cost(payload["model"], payload_input_bound(payload), payload["max_output_tokens"], pricing=config["pricing"])
        cap = reserve * 1.1
        pipeline.configure(identifier, {"budget_limit_usd": cap})
        seen, clients, lock, commitments = [], [], threading.Lock(), []
        def check():
            with pipeline.connection() as db:
                value = db.execute("SELECT SUM(cost_usd)+SUM(CASE WHEN status IN ('running','uncertain') THEN reserved_usd ELSE 0 END) FROM jobs").fetchone()[0]
            with lock:
                commitments.append(value)
        result = run_workers(root, identifier, workers=3, client_factory=self.factory(
            seen, clients, lock, barrier=threading.Barrier(3), delay=.1, check=check))
        self.assertTrue(commitments)
        self.assertLessEqual(max(commitments), cap + 1e-12)
        self.assertLessEqual(result["accounting"]["cost_usd_total"], cap)
        self.assertIn(result["stopped_reason"], {"budget_reservation", "corpus_budget_reservation"})
        self.assertGreater(result["statuses"].get("pending", 0), 0)

    def test_stop_file_stops_new_work_without_replaying_or_changing_config(self):
        root, pipeline, identifier = self.fixture()
        config = pipeline._row(identifier)["config"]
        stop = root / "STOP"
        seen, clients, lock = [], [], threading.Lock()
        result = run_workers(root, identifier, workers=1, stop_file=stop,
            client_factory=self.factory(seen, clients, lock, stop_file=stop))
        self.assertEqual(result["attempted"], 1)
        self.assertEqual(result["stopped_reason"], "stop_file")
        self.assertEqual(pipeline._row(identifier)["config"], config)
        again = run_workers(root, identifier, workers=1, stop_file=stop,
            client_factory=lambda: self.fail("Existing stop file must not construct a client"))
        self.assertEqual(again["attempted"], 0)

    def test_error_and_unknown_billing_stop_without_retry(self):
        for uncertain in (False, True):
            with self.subTest(uncertain=uncertain):
                root, pipeline, identifier = self.fixture()
                seen, clients, lock = [], [], threading.Lock()
                failure = ModelResponseError("transport_failure" if uncertain else "http_error", uncertain=uncertain,
                                             status=None if uncertain else 400)
                result = run_workers(root, identifier, workers=1,
                    client_factory=self.factory(seen, clients, lock, failure=failure))
                self.assertEqual(result["attempted"], 1)
                self.assertEqual(len(seen), 1)
                self.assertEqual(result["uncertain" if uncertain else "failed"], 1)
                self.assertEqual(result["stopped_reason"], "uncertain_request" if uncertain else "error")
                if uncertain:
                    self.assertGreater(result["accounting"]["outstanding_reserved_usd"], 0)
                self.assertEqual(result["statuses"]["pending"], 19)

    def test_stage_selection_preserves_held_extract_and_shared_68_70_cap(self):
        root, pipeline, identifier = self.fixture(count=4)
        config = pipeline._row(identifier)["config"]
        context = {"config": config, "facts": []}
        payload = make_task("describe", context, model=config["routes"]["describe"])
        with pipeline.connection() as db:
            db.execute("UPDATE jobs SET stage='extract',recipe=? WHERE corpus_id=?",
                       (pipeline._recipe("extract", config), identifier))
            first = db.execute("SELECT id FROM jobs ORDER BY id LIMIT 1").fetchone()[0]
            db.execute("UPDATE jobs SET status='done',cost_usd=67.99 WHERE id=?", (first,))
            for index in range(6):
                db.execute("""INSERT INTO jobs(id,corpus_id,stage,unit_id,recipe,context,payload,status,created_at)
                    VALUES(?,?,'describe',?,?,?,?,'pending','2099-01-01')""",
                    ("describe-" + str(index), identifier, "desc-unit-" + str(index),
                     pipeline._recipe("describe", config), json.dumps(context), json.dumps(payload)))
        called = []
        class DescribeOnly:
            def call(self, actual):
                called.append(actual["text"]["format"]["name"])
                return {"data": {"scales": []}, "usage": {"input_tokens": 100, "output_tokens": 20},
                        "response_id": "synthetic-description"}
            def close(self):
                pass
        pipeline.configure(identifier, {"budget_limit_usd": 68.})
        blocked = pipeline.run(identifier, stage="describe", client=DescribeOnly(), budget_usd=1.)
        self.assertEqual(blocked["attempted"], 0)
        self.assertEqual(blocked["stopped_reason"], "corpus_budget_reservation")
        pipeline.configure(identifier, {"budget_limit_usd": 70.})
        direct = pipeline.run(identifier, stage="describe", client=DescribeOnly(), max_jobs=1, budget_usd=1.)
        self.assertEqual(direct["completed"], 1)
        result = run_workers(root, identifier, stage="describe", workers=3, max_jobs=2, client_factory=DescribeOnly)
        self.assertEqual(result["attempted"], 2)
        self.assertEqual(result["stage_totals"]["stage"], "describe")
        self.assertEqual(result["stage_totals"]["statuses"]["pending"], 3)
        self.assertEqual(result["statuses"]["pending"], 6)
        self.assertEqual(called, ["corpus_describe"] * 3)
        with pipeline.connection() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM jobs WHERE stage='extract' AND status='pending'").fetchone()[0], 3)
        self.assertLess(result["accounting"]["cost_usd_total"], 70)

    def test_invalid_stage_rejected_before_any_client_call(self):
        root, pipeline, identifier = self.fixture(count=2)
        with self.assertRaisesRegex(ValueError, "stage"):
            pipeline.run(identifier, stage="invented", client=object())
        with self.assertRaisesRegex(ValueError, "stage"):
            run_workers(root, identifier, stage="invented", client_factory=lambda: self.fail("No client expected"))

    def test_connection_guard_shared_by_path_reentrant_and_other_db_independent(self):
        root, first, _ = self.fixture(count=2)
        second = CorpusPipeline(root / "." )
        independent = CorpusPipeline(root / "independent")
        self.assertIs(first._db_lock, second._db_lock)
        self.assertIsNot(first._db_lock, independent._db_lock)
        trying, entered, other_entered = threading.Event(), threading.Event(), threading.Event()
        def same_db():
            trying.set()
            with second.connection() as db:
                db.execute("SELECT 1")
                entered.set()
        def other_db():
            with independent.connection() as db:
                db.execute("SELECT 1")
                other_entered.set()
        same, other = threading.Thread(target=same_db), threading.Thread(target=other_db)
        try:
            with first.connection() as db:
                self.assertEqual(db.execute("PRAGMA synchronous").fetchone()[0], 2)  # FULL remains.
                with first.connection() as nested:
                    self.assertEqual(nested.execute("SELECT 1").fetchone()[0], 1)
                same.start()
                self.assertTrue(trying.wait(2))
                self.assertFalse(entered.wait(.05))
                other.start()
                self.assertTrue(other_entered.wait(2))
            self.assertTrue(entered.wait(2))
        finally:
            if same.ident is not None:
                same.join(timeout=3)
            if other.ident is not None:
                other.join(timeout=3)

    def test_32_workers_500_jobs_have_no_sqlite_failure_or_duplicate_claim(self):
        root, pipeline, identifier = self.fixture(count=500)
        seen, clients, lock = [], [], threading.Lock()
        result = run_workers(root, identifier, workers=32, max_jobs=500,
            client_factory=self.factory(seen, clients, lock, barrier=threading.Barrier(32), delay=.0001))
        self.assertEqual(result["attempted"], 500)
        self.assertEqual(result["completed"] + result["review"], 500)
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["uncertain"], 0)
        self.assertEqual(result["statuses"].get("pending", 0), 0)
        self.assertTrue(all(client.closed for client in clients))

    def test_sqlite_diagnostics_are_allowlisted_without_other_exception_text(self):
        error = sqlite3.OperationalError("database is locked")
        error.sqlite_errorcode, error.sqlite_errorname = 5, "SQLITE_BUSY"
        safe = _safe_exception(error, 1)
        self.assertEqual(safe["sqlite_errorcode"], 5)
        self.assertEqual(safe["sqlite_errorname"], "SQLITE_BUSY")
        self.assertEqual(safe["message"], "database is locked")
        for exc in (sqlite3.OperationalError("sk-PRIVATE_KEY PRIVATE_SOURCE"), RuntimeError("sk-PRIVATE_KEY")):
            self.assertNotIn("PRIVATE", json.dumps(_safe_exception(exc, 1)))

    def test_scope_anchor_keeps_wal_between_worker_transactions_and_closes_after_join(self):
        root, pipeline, identifier = self.fixture(count=3)
        wal = Path(str(pipeline.db_path) + "-wal")
        observations, closed = [], []
        test = self
        class Client:
            def call(self, payload):
                # The worker claim transaction/connection already closed; no
                # additional connection is opened by this check.
                observations.append(wal.exists() and wal.stat().st_size > 0)
                return {"data": {"status": "relevant", "actor_ids": ["alpha"], "language": "en", "reason": "Synthetic"},
                        "usage": {"input_tokens": 100, "output_tokens": 20}, "response_id": "synthetic"}
            def close(self):
                closed.append(wal.exists())
        result = run_workers(root, identifier, workers=1, client_factory=Client)
        self.assertEqual(result["attempted"], 3)
        self.assertEqual(observations, [True] * 3)
        self.assertEqual(closed, [True])
        self.assertFalse(wal.exists())
        # A leaked persistent connection prevents this rename on Windows.
        renamed = pipeline.db_path.with_suffix(".closed")
        pipeline.db_path.replace(renamed)
        renamed.replace(pipeline.db_path)

    def test_accounting_queries_use_covering_index_without_loading_wide_payloads(self):
        root, pipeline, identifier = self.fixture(count=2)
        queries = [
            ("SELECT SUM(cost_usd)+SUM(CASE WHEN status IN ('running','uncertain') THEN reserved_usd ELSE 0 END) FROM jobs WHERE corpus_id=?", (identifier,)),
            ("SELECT SUM(input_tokens),SUM(output_tokens),SUM(cost_usd),SUM(CASE WHEN status IN ('running','uncertain') THEN reserved_usd ELSE 0 END) FROM jobs WHERE corpus_id=? AND stage=?", (identifier, "relevance")),
        ]
        with pipeline.connection() as db:
            for query, arguments in queries:
                plan = " ".join(row[3] for row in db.execute("EXPLAIN QUERY PLAN " + query, arguments))
                self.assertIn("COVERING INDEX job_accounting", plan)
                self.assertIsNotNone(db.execute(query, arguments).fetchone())


if __name__ == "__main__":
    unittest.main()
