"""Frozen additional cap, resumability and stage scope; synthetic API only."""
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from scripts.run_corpus_full_pilot import directory, load_manifest, main, read, run_phase, setup, snapshot
from src.corpus_pipeline.models import ModelResponseError
from src.corpus_pipeline.service import CorpusPipeline


class SyntheticClient:
    def __init__(self, failure=None):
        self.calls = 0
        self.failure = failure

    def call(self, payload):
        self.calls += 1
        if self.failure:
            raise self.failure
        context = json.loads(payload["input"][0]["content"][0]["text"])
        stage = payload["text"]["format"]["name"].replace("corpus_", "")
        if stage == "relevance":
            text = context["document"]["text"]
            status = "relevant" if "tax reform" in text else "review" if "services" in text else "irrelevant"
            data = {"status": status, "actor_ids": ["alpha"] if status == "relevant" else ["beta"] if status == "review" else [],
                    "language": "en", "reason": "Synthetic scope test."}
        elif stage == "extract":
            data = {"facts": [{"actor_id": "alpha", "quote": context["document"]["text"], "claim": "A tax proposal.",
                                "kind": "public_statement", "event_date": None, "date_evidence": None, "topic": "taxation",
                                "trait_candidates": ["Policy revision"], "capacity": "personal",
                                "attribution_basis": "direct_statement", "attribution_evidence": context["document"]["text"]},
                               {"actor_id": "alpha", "quote": context["document"]["text"], "claim": "Role context only.",
                                "kind": "institutional_context", "event_date": None, "date_evidence": None, "topic": "context",
                                "trait_candidates": [], "capacity": "official", "attribution_basis": "context",
                                "attribution_evidence": context["document"]["text"]}]}
        else:
            raise AssertionError("No automatic describe/score calls")
        return {"data": data, "usage": {"input_tokens": 100, "output_tokens": 40, "input_tokens_details": {"cached_tokens": 0}},
                "response_id": "synthetic-response", "model": payload["model"]}


class AllRelevantClient(SyntheticClient):
    """Scope-positive documents; successful extraction may have zero facts."""
    def call(self, payload):
        self.calls += 1
        context = json.loads(payload["input"][0]["content"][0]["text"])
        stage = payload["text"]["format"]["name"].replace("corpus_", "")
        data = ({"status": "relevant", "actor_ids": ["beta" if "Beta South" in context["document"]["text"] else "alpha"],
                 "language": "en", "reason": "Synthetic approved actor mention."} if stage == "relevance" else {"facts": []})
        return {"data": data, "usage": {"input_tokens": 100, "output_tokens": 40},
                "response_id": "synthetic-all-relevant", "model": payload["model"]}


class FullCorpusPilotTests(unittest.TestCase):
    def setUp(self):
        temp_base = Path(__file__).resolve().parents[1] / ".service_runtime/corpus_full_tests"
        temp_base.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temp_base)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = self.root / "synthetic.csv"
        with source.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(["Заголовок", "Сообщение", "Дата", "Ссылка"])
            for text in ("Alpha North proposes Rwanda tax reform.", "Beta South discusses Rwanda services.",
                         "Alpha North discusses Rwanda environment.", "Alpha North discusses elsewhere.",
                         "Alpha North proposes Rwanda tax reform."):
                writer.writerow(["", text, "12.10.2025", "https://example.test/"])
        self.pipeline = CorpusPipeline(self.root)
        corpus = self.pipeline.import_csv(source, config={"scope": "Synthetic positions", "country_terms": ["Rwanda"],
            "date_from": "2021-01-01", "date_to": "2026-12-31", "budget_limit_usd": 2.,
            "objects": [{"id": "alpha", "name": "Alpha North"}, {"id": "beta", "name": "Beta South"}]})
        self.identifier = corpus["id"]
        self.first = self.pipeline.runtime / "pilots" / self.identifier
        self.first.mkdir(parents=True)
        (self.first / "manifest.json").write_bytes(b'first-pilot-manifest')
        (self.first / "results_private.json").write_bytes(b'first-pilot-results')
        (self.root / "reports").mkdir()
        (self.root / "reports/corpus_pilot_2026_10_17.json").write_bytes(b'first-public-report')
        self.emit = lambda value: None

    def configure(self):
        return setup(self.pipeline, self.identifier, allow_budget_increase=True)

    def seed_spend(self, amount):
        self.pipeline.prepare(self.identifier, limit=1)
        with self.pipeline.connection() as db:
            db.execute("UPDATE jobs SET status='done',cost_usd=?,input_tokens=100,output_tokens=10,result=? WHERE corpus_id=?",
                       (amount, json.dumps({"status": "relevant", "actor_ids": ["alpha"], "language": "en", "reason": "Synthetic"}), self.identifier))

    def test_setup_increase_needs_explicit_flag_and_leaves_old_pilot_untouched(self):
        self.seed_spend(.4)
        with self.assertRaises(ValueError):
            setup(self.pipeline, self.identifier)
        self.assertFalse((directory(self.pipeline, self.identifier) / "manifest.json").exists())
        self.assertEqual(self.pipeline.get_corpus(self.identifier)["config"]["budget_limit_usd"], 2.)
        manifest = self.configure()
        self.assertEqual(manifest["start_spent_usd"], .4)
        self.assertEqual(manifest["budget_cap_usd"], 5.4)
        self.assertEqual(manifest["population"], {"raw": 5, "unique": 4, "eligible_unique": 3, "rule_exclusions": 1})
        self.assertEqual((self.first / "manifest.json").read_bytes(), b'first-pilot-manifest')
        self.assertEqual((self.first / "results_private.json").read_bytes(), b'first-pilot-results')
        self.assertEqual((self.root / "reports/corpus_pilot_2026_10_17.json").read_bytes(), b'first-public-report')

    def test_resume_never_rebases_and_rejects_new_budget_or_external_cap(self):
        self.seed_spend(.4)
        manifest = self.configure()
        frozen = (directory(self.pipeline, self.identifier) / "manifest.json").read_bytes()
        with self.pipeline.connection() as db:
            db.execute("UPDATE jobs SET cost_usd=.6 WHERE corpus_id=?", (self.identifier,))
        self.assertEqual(setup(self.pipeline, self.identifier)["budget_cap_usd"], 5.4)
        self.assertEqual((directory(self.pipeline, self.identifier) / "manifest.json").read_bytes(), frozen)
        self.assertAlmostEqual(snapshot(self.pipeline, manifest)["accounting"]["new_recorded_cost_delta_usd"], .2)
        with self.assertRaises(ValueError):
            setup(self.pipeline, self.identifier, 6, allow_budget_increase=True)
        self.pipeline.configure(self.identifier, {"budget_limit_usd": 9})
        with self.assertRaises(ValueError):
            load_manifest(self.pipeline, self.identifier)
        with self.assertRaises(ValueError):
            setup(self.pipeline, self.identifier, allow_budget_increase=True)
        self.assertEqual(self.pipeline.get_corpus(self.identifier)["config"]["budget_limit_usd"], 9)

    def test_interrupted_setup_preserves_planned_baseline_on_explicit_recovery(self):
        self.seed_spend(.4)
        with patch.object(self.pipeline, "configure", side_effect=ValueError("synthetic interruption")), self.assertRaises(ValueError):
            self.configure()
        path = directory(self.pipeline, self.identifier) / "manifest.json"
        self.assertEqual(read(path)["setup_state"], "planned")
        with self.pipeline.connection() as db:
            db.execute("UPDATE jobs SET cost_usd=.6 WHERE corpus_id=?", (self.identifier,))
        with self.assertRaises(ValueError):
            setup(self.pipeline, self.identifier)
        manifest = self.configure()
        self.assertEqual(manifest["start_spent_usd"], .4)
        self.assertEqual(manifest["budget_cap_usd"], 5.4)
        self.assertEqual(manifest["setup_state"], "ready")

    def test_prepare_full_scope_without_ids_no_automatic_describe_or_api(self):
        manifest = self.configure()
        with patch.object(self.pipeline, "prepare", wraps=self.pipeline.prepare) as prepare, patch.object(self.pipeline, "run", side_effect=AssertionError("No API")):
            result = run_phase(self.pipeline, manifest, "relevance", emit=self.emit)
        self.assertNotIn("document_ids", prepare.call_args.kwargs)
        self.assertEqual(result["active_stage_statuses"]["relevance"], {"pending": 3})
        self.assertEqual(result["active_stage_statuses"]["describe"], {})
        self.assertEqual(result["phase_trace"]["attempted"], 0)

    def test_reviews_are_allowed_excluded_from_extract_and_only_eligible_facts_describe(self):
        manifest = self.configure()
        client = SyntheticClient()
        relevance = run_phase(self.pipeline, manifest, "relevance", execute=True, batch_size=2, client=client, emit=self.emit)
        self.assertEqual(client.calls, 3)
        self.assertEqual(relevance["active_stage_statuses"]["relevance"], {"done": 2, "review": 1})
        self.assertEqual(len(relevance["extraction_trace"]["excluded_review_chunks"]), 1)
        extraction = run_phase(self.pipeline, manifest, "extract", execute=True, client=client, emit=self.emit)
        self.assertEqual(client.calls, 4)
        self.assertEqual(extraction["active_stage_statuses"]["extract"], {"done": 1})
        self.assertEqual(len(extraction["facts"]), 2)
        self.assertEqual(extraction["active_stage_statuses"]["describe"], {})
        description = run_phase(self.pipeline, manifest, "describe", emit=self.emit)
        job = next(j for j in description["jobs"] if j["stage"] == "describe")
        self.assertEqual([f["kind"] for f in job["context"]["facts"]], ["public_statement"])
        self.assertEqual(client.calls, 4)

    def test_uncertain_request_stops_and_is_not_automatically_replayed(self):
        manifest = self.configure()
        client = SyntheticClient(ModelResponseError("transport_failure", uncertain=True))
        result = run_phase(self.pipeline, manifest, "relevance", execute=True, client=client, emit=self.emit)
        self.assertEqual(client.calls, 1)
        self.assertEqual(result["phase_trace"]["attempted"], 1)
        self.assertGreater(result["accounting"]["outstanding_reserved_usd"], 0)
        self.assertEqual(result["accounting"]["new_recorded_cost_delta_usd"], 0)
        repeated = run_phase(self.pipeline, manifest, "relevance", execute=True, client=client, emit=self.emit)
        self.assertEqual(client.calls, 1)
        self.assertEqual(repeated["phase_trace"]["stopped_reason"], "unreconciled_or_running_request")

    def test_known_error_stops_and_no_implicit_retry(self):
        manifest = self.configure()
        client = SyntheticClient(ModelResponseError("http_error", status=400))
        first = run_phase(self.pipeline, manifest, "relevance", execute=True, client=client, emit=self.emit)
        self.assertEqual(first["phase_trace"]["stopped_reason"], "request_error")
        initial_calls = client.calls  # service may finish other distinct jobs in its bounded batch.
        self.assertLessEqual(initial_calls, 5)
        repeated = run_phase(self.pipeline, manifest, "relevance", execute=True, client=client, emit=self.emit)
        self.assertEqual(client.calls, initial_calls)
        self.assertEqual(repeated["phase_trace"]["stopped_reason"], "existing_stage_error")

    def test_historical_pending_alone_does_not_trigger_run(self):
        manifest = self.configure()
        run_phase(self.pipeline, manifest, "relevance", execute=True, client=SyntheticClient(), emit=self.emit)
        with self.pipeline.connection() as db:
            row = dict(db.execute("SELECT * FROM jobs LIMIT 1").fetchone())
            row.update(id="job-old-pending", recipe="inactive-recipe", status="pending", cost_usd=0, input_tokens=0, output_tokens=0)
            db.execute("INSERT INTO jobs (" + ",".join(row) + ") VALUES (" + ",".join("?" for _ in row) + ")", tuple(row.values()))
        with patch.object(self.pipeline, "run", side_effect=AssertionError("No active work")):
            result = run_phase(self.pipeline, manifest, "relevance", execute=True, emit=self.emit)
        self.assertEqual(result["phase_trace"]["stopped_reason"], "active_queue_empty")
        self.assertTrue(any(j["id"] == "job-old-pending" and not j["active_recipe"] for j in result["jobs"]))

    def test_other_active_stage_cannot_be_accidentally_billed(self):
        manifest = self.configure()
        self.pipeline.prepare(self.identifier)
        config = self.pipeline.get_corpus(self.identifier)["config"]
        with self.pipeline.connection() as db:
            row = dict(db.execute("SELECT * FROM jobs LIMIT 1").fetchone())
            row.update(id="job-other-stage", stage="extract", recipe=self.pipeline._recipe("extract", config))
            db.execute("INSERT INTO jobs (" + ",".join(row) + ") VALUES (" + ",".join("?" for _ in row) + ")", tuple(row.values()))
        with patch.object(self.pipeline, "run", side_effect=AssertionError("Wrong stage must not run")):
            result = run_phase(self.pipeline, manifest, "relevance", execute=True, emit=self.emit)
        self.assertEqual(result["phase_trace"]["stopped_reason"], "other_active_stage_pending")

    def test_upstream_pending_blocks_extraction_and_invalid_curation_never_means_all(self):
        manifest = self.configure()
        run_phase(self.pipeline, manifest, "relevance", emit=self.emit)
        with self.assertRaises(ValueError):
            run_phase(self.pipeline, manifest, "extract", emit=self.emit)
        for value in ([], ["x", "x"], "x"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                run_phase(self.pipeline, manifest, "describe", fact_ids=value, emit=self.emit)
        bad = self.root / "bad-curation.json"
        bad.write_text('{"wrong_field":[]}', encoding="utf-8")
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            status = main(["--root", str(self.root), "--corpus-id", self.identifier, "--phase", "describe", "--fact-ids-file", str(bad)])
        self.assertEqual(status, 2)

    def test_tiny_frozen_budget_stops_before_any_model_request(self):
        manifest = setup(self.pipeline, self.identifier, 1e-10)
        client = SyntheticClient()
        result = run_phase(self.pipeline, manifest, "relevance", execute=True, client=client, emit=self.emit)
        self.assertEqual(client.calls, 0)
        self.assertEqual(result["phase_trace"]["stopped_reason"], "budget_reservation")

    def test_parallel_workers_own_clients_claim_distinct_jobs_and_parent_writes_snapshots(self):
        manifest = self.configure()
        barrier = threading.Barrier(3)
        clients, seen, snapshot_threads = [], [], []
        lock = threading.Lock()
        parent_thread = threading.get_ident()

        class WaitingClient(SyntheticClient):
            def call(inner, payload):
                context = json.loads(payload["input"][0]["content"][0]["text"])
                with lock:
                    seen.append(context["document"]["id"])
                barrier.wait(timeout=10)
                return super().call(payload)

            def close(inner):
                inner.closed = True

        def factory():
            item = WaitingClient()
            with lock:
                clients.append(item)
            return item

        real_snapshot = snapshot

        def checked_snapshot(*args, **kwargs):
            snapshot_threads.append(threading.get_ident())
            return real_snapshot(*args, **kwargs)

        with patch("scripts.run_corpus_full_pilot.snapshot", side_effect=checked_snapshot):
            result = run_phase(self.pipeline, manifest, "relevance", execute=True, workers=3, batch_size=1,
                               client_factory=factory, emit=self.emit)
        self.assertEqual(len(clients), 3)
        self.assertTrue(all(c.calls == 1 and c.closed for c in clients))
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(result["phase_trace"]["attempted"], 3)
        self.assertEqual(set(snapshot_threads), {parent_thread})
        self.assertLessEqual(result["accounting"]["new_recorded_cost_delta_usd"], manifest["additional_budget_usd"])
        self.assertEqual(result["active_stage_statuses"]["relevance"], {"done": 2, "review": 1})
        with self.assertRaises(ValueError):
            run_phase(self.pipeline, manifest, "relevance", workers=3, client=SyntheticClient(), emit=self.emit)

    def test_truncated_preparation_is_incomplete_until_explicit_relevance_repeats(self):
        manifest = self.configure()
        real_prepare = self.pipeline.prepare

        def limited(identifier, **kwargs):
            kwargs["limit"] = 1  # Emulate a source frame larger than the prepare cap.
            return real_prepare(identifier, **kwargs)

        client = SyntheticClient()
        with patch.object(self.pipeline, "prepare", side_effect=limited):
            first = run_phase(self.pipeline, manifest, "relevance", execute=True, client=client, emit=self.emit)
            self.assertEqual(first["phase_trace"]["stopped_reason"], "incomplete_preparation")
            self.assertEqual({k: first["readiness"]["classification"][k] for k in ("expected", "prepared", "finished", "unprepared", "ready")},
                             {"expected": 3, "prepared": 1, "finished": 1, "unprepared": 2, "ready": False})
            for stage in ("extract", "describe"):
                with self.subTest(stage=stage), self.assertRaises(ValueError):
                    run_phase(self.pipeline, manifest, stage, emit=self.emit)
            run_phase(self.pipeline, manifest, "relevance", execute=True, client=client, emit=self.emit)
            final = run_phase(self.pipeline, manifest, "relevance", execute=True, client=client, emit=self.emit)
        self.assertEqual(client.calls, 3)
        self.assertTrue(final["readiness"]["classification"]["ready"])
        self.assertEqual(final["readiness"]["classification"]["finished"], 3)  # One terminal review is still finished classification.
        self.assertEqual(final["readiness"]["classification"]["unprepared_unit_ids"], [])
        result = run_phase(self.pipeline, manifest, "extract", emit=self.emit)
        self.assertEqual(result["active_stage_statuses"]["extract"], {"pending": 1})

    def test_invocation_limit_keeps_prepared_but_unfinished_coverage_visible(self):
        manifest = self.configure()
        result = run_phase(self.pipeline, manifest, "relevance", execute=True, max_jobs=1, client=SyntheticClient(), emit=self.emit)
        self.assertEqual(result["phase_trace"]["stopped_reason"], "invocation_job_limit")
        self.assertEqual(result["readiness"]["classification"]["prepared"], 3)
        self.assertEqual(result["readiness"]["classification"]["finished"], 1)
        self.assertEqual(result["readiness"]["classification"]["unprepared"], 0)
        self.assertEqual(result["readiness"]["classification"]["unfinished"], 2)
        self.assertFalse(result["readiness"]["ready"])
        with self.assertRaises(ValueError):
            run_phase(self.pipeline, manifest, "extract", emit=self.emit)

    def test_truncated_extraction_preparation_requires_explicit_repeats_before_describe(self):
        manifest = self.configure()
        client = AllRelevantClient()
        run_phase(self.pipeline, manifest, "relevance", execute=True, client=client, emit=self.emit)
        real_prepare = self.pipeline.prepare

        def limited(identifier, **kwargs):
            kwargs["limit"] = 1
            return real_prepare(identifier, **kwargs)

        with patch.object(self.pipeline, "prepare", side_effect=limited):
            first = run_phase(self.pipeline, manifest, "extract", execute=True, client=client, emit=self.emit)
            self.assertEqual(first["phase_trace"]["stopped_reason"], "incomplete_preparation")
            self.assertTrue(first["readiness"]["classification"]["ready"])
            self.assertEqual({k: first["readiness"]["extraction"][k] for k in ("expected", "prepared", "finished", "unprepared", "ready")},
                             {"expected": 3, "prepared": 1, "finished": 1, "unprepared": 2, "ready": False})
            with self.assertRaises(ValueError):
                run_phase(self.pipeline, manifest, "describe", emit=self.emit)
            run_phase(self.pipeline, manifest, "extract", execute=True, client=client, emit=self.emit)
            final = run_phase(self.pipeline, manifest, "extract", execute=True, client=client, emit=self.emit)
        self.assertEqual(client.calls, 6)
        self.assertTrue(final["readiness"]["extraction"]["ready"])
        self.assertTrue(final["readiness"]["ready"])
        self.assertEqual(final["active_stage_statuses"]["describe"], {})

    def test_extraction_invocation_limit_and_review_remain_incomplete_for_describe(self):
        manifest = self.configure()
        client = AllRelevantClient()
        run_phase(self.pipeline, manifest, "relevance", execute=True, client=client, emit=self.emit)
        partial = run_phase(self.pipeline, manifest, "extract", execute=True, max_jobs=1, client=client, emit=self.emit)
        self.assertEqual(partial["phase_trace"]["stopped_reason"], "invocation_job_limit")
        self.assertEqual({k: partial["readiness"]["extraction"][k] for k in ("prepared", "finished", "unprepared", "unfinished", "ready")},
                         {"prepared": 3, "finished": 1, "unprepared": 0, "unfinished": 2, "ready": False})
        with self.assertRaises(ValueError):
            run_phase(self.pipeline, manifest, "describe", emit=self.emit)
        final = run_phase(self.pipeline, manifest, "extract", execute=True, client=client, emit=self.emit)
        self.assertTrue(final["readiness"]["extraction"]["ready"])
        with self.pipeline.connection() as db:
            db.execute("UPDATE jobs SET status='review' WHERE id=(SELECT id FROM jobs WHERE stage='extract' LIMIT 1)")
        reviewed = snapshot(self.pipeline, manifest)
        self.assertFalse(reviewed["readiness"]["extraction"]["ready"])
        with self.assertRaises(ValueError):
            run_phase(self.pipeline, manifest, "describe", emit=self.emit)


if __name__ == "__main__":
    unittest.main()
