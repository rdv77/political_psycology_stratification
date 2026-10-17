"""Real SQLite claims with synthetic clients; no network or canonical data."""
import csv
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from src.corpus_pipeline.bounded_workers import run_workers
from src.corpus_pipeline.models import ModelResponseError, estimate_cost, payload_input_bound
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


if __name__ == "__main__":
    unittest.main()
