"""Durability, provenance and budget checks with synthetic documents and no API."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

from src.corpus_pipeline.service import CorpusPipeline, chunks, digest
from src.corpus_pipeline.models import ModelResponseError


class FixedClient:
    def __init__(self, data):
        self.data, self.calls = data, 0

    def call(self, payload):
        self.calls += 1
        return {"data": self.data, "model": payload["model"], "response_id": "test-response",
                "usage": {"input_tokens": 100, "output_tokens": 40,
                          "input_tokens_details": {"cached_tokens": 0}}}


class CorpusServiceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.file = self.root / "input.csv"
        with self.file.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream, delimiter=";")
            writer.writerow(["Заголовок", "Сообщение", "Ссылка", "Дата", "Id сообщения"])
            writer.writerow(["", "Alpha North proposes a specific tax reform.", "https://example.org/1", "12.10.2025 12:00", "1,51E+11"])
            writer.writerow(["", "Alpha North proposes a specific tax reform.", "https://example.org/2", "12.10.2025 12:01", "1,51E+11"])
            writer.writerow(["", "Beta South discusses public services.", "https://example.org/3", "13.10.2025 10:00", "99999999999999999999999"])
            writer.writerow(["", "", "", "", ""])
        self.config = {"scope": "Synthetic public policies", "objects": [
            {"id": "alpha", "name": "Alpha North"}, {"id": "beta", "name": "Beta South"}]}
        self.pipeline = CorpusPipeline(self.root)
        self.corpus = self.pipeline.import_csv(self.file, config=self.config)
        self.identifier = self.corpus["id"]

    def relevant(self):
        return {"status": "relevant", "actor_ids": ["alpha"], "language": "en", "reason": "Explicit synthetic name"}

    def test_import_preserves_original_and_is_idempotent(self):
        counts = self.corpus["counts"]
        self.assertEqual(counts["documents"], 4)
        self.assertEqual(counts["unique_documents"], 2)
        self.assertEqual(counts["exact_duplicates"], 1)
        self.assertEqual(counts["empty_documents"], 1)
        self.assertEqual(self.pipeline.import_csv(self.file)["id"], self.identifier)
        original = next((self.root / ".corpus_runtime/originals").glob("*.csv"))
        self.assertEqual(original.read_bytes(), self.file.read_bytes())
        documents = self.pipeline.documents(self.identifier)["documents"]
        self.assertEqual(documents[0]["source_id"], "1,51E+11")
        self.assertEqual(documents[2]["source_id"], "99999999999999999999999")
        self.assertEqual(documents[1]["duplicate_of"], documents[0]["id"])

    def test_prepare_never_calls_api_and_does_not_repeat_queued_tasks(self):
        first = self.pipeline.prepare(self.identifier, limit=1)
        self.assertEqual(first["created"], 1)
        self.assertEqual(first["api_calls"], 0)
        self.assertGreater(first["total_estimated_upper_usd"], 0)
        self.assertEqual(self.pipeline.prepare(self.identifier, limit=1)["created"], 1)
        self.assertEqual(self.pipeline.prepare(self.identifier)["created"], 0)

    def test_tiny_budget_keeps_tasks_pending(self):
        self.pipeline.prepare(self.identifier)
        client = FixedClient(self.relevant())
        result = self.pipeline.run(self.identifier, budget_usd=0.00000001, client=client)
        self.assertEqual(result["stopped_reason"], "budget_reservation")
        self.assertEqual(client.calls, 0)
        self.assertEqual(self.pipeline.get_corpus(self.identifier)["jobs"]["pending"], 2)

    def test_restart_reuses_completed_tasks_and_retains_usage(self):
        self.pipeline.prepare(self.identifier, limit=1)
        client = FixedClient(self.relevant())
        self.assertEqual(self.pipeline.run(self.identifier, budget_usd=1, client=client)["completed"], 1)
        restarted = CorpusPipeline(self.root)
        result = restarted.run(self.identifier, budget_usd=1, client=client)
        self.assertEqual(result["attempted"], 0)
        self.assertEqual(client.calls, 1)
        self.assertEqual(restarted.get_corpus(self.identifier)["usage"]["input_tokens"], 100)

    def test_new_rubric_does_not_invalidate_relevance(self):
        before = self.pipeline._recipe("relevance", self.corpus["config"])
        configured = self.pipeline.configure(self.identifier, {"scales": [
            {"id": "reform", "name": "Reform", "definition": "Specific change"}]})
        self.assertEqual(self.pipeline._recipe("relevance", configured["config"]), before)
        self.assertFalse(configured["config"]["scales_approved"])
        with self.assertRaises(ValueError):
            self.pipeline.prepare(self.identifier, stage="score")

    def test_changed_scope_marks_pending_tasks_obsolete(self):
        self.pipeline.prepare(self.identifier)
        result = self.pipeline.configure(self.identifier, {"scope": "A different public space"})
        self.assertEqual(result["jobs"]["obsolete"], 2)
        self.assertEqual(self.pipeline.prepare(self.identifier)["created"], 2)

    def test_paid_invalid_output_is_review_and_still_accounted(self):
        self.pipeline.prepare(self.identifier, limit=1)
        result = self.pipeline.run(self.identifier, budget_usd=1, client=FixedClient({"invented": True}))
        self.assertEqual(result["review"], 1)
        corpus = self.pipeline.get_corpus(self.identifier)
        self.assertEqual(corpus["jobs"]["review"], 1)
        self.assertGreater(corpus["usage"]["estimated_usd"], 0)

    def test_language_allowlist_is_applied_after_classification(self):
        self.pipeline.configure(self.identifier, {"languages": ["French", "Kinyarwanda"]})
        self.pipeline.prepare(self.identifier, limit=1)
        self.pipeline.run(self.identifier, budget_usd=1, client=FixedClient(self.relevant()))
        with self.pipeline.connection() as db:
            result = json.loads(db.execute("SELECT result FROM jobs WHERE status='done'").fetchone()[0])
        self.assertEqual(result["status"], "irrelevant")
        self.assertEqual(self.pipeline.prepare(self.identifier, stage="extract")["created"], 0)

    def test_export_is_private_and_cannot_modify_research(self):
        result = self.pipeline.export(self.identifier)
        for item in result["files"]:
            self.assertTrue(Path(item["path"]).is_relative_to(self.root / ".corpus_runtime"))
        provenance = json.loads((Path(result["directory"]) / "provenance.json").read_text(encoding="utf-8"))
        self.assertFalse(provenance["canonical_study_modified"])
        self.assertFalse((self.root / "studies").exists())

    def test_credentials_and_missing_prices_cannot_enter_config(self):
        with self.assertRaises(ValueError):
            self.pipeline.configure(self.identifier, {"OPENAI_API_KEY": "synthetic-secret"})
        config = dict(self.corpus["config"])
        config["routes"] = dict(config["routes"], relevance="unknown-model")
        with self.assertRaises(ValueError):
            self.pipeline.configure(self.identifier, config)

    def test_all_text_is_covered_by_chunks(self):
        text = "word " * 10000
        ranges = list(chunks(text))
        covered = set()
        for start, end, value in ranges:
            self.assertEqual(value, text[start:end])
            covered.update(range(start, end))
        self.assertEqual(len(covered), len(text))

    def test_chunk_payload_never_repeats_full_raw_document(self):
        other = self.root / "long.csv"
        with other.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["text"])
            writer.writerow(["Alpha North " + "Long synthetic sentence. " * 1200])
        corpus = self.pipeline.import_csv(other, config=self.config)
        self.pipeline.prepare(corpus["id"], limit=10)
        with self.pipeline.connection() as db:
            rows = db.execute("SELECT context FROM jobs WHERE corpus_id=?", (corpus["id"],)).fetchall()
        self.assertGreater(len(rows), 1)
        for row in rows:
            document = json.loads(row[0])["document"]
            self.assertNotIn("raw_text", document)
            self.assertNotIn("metadata", document)
            self.assertLessEqual(len(document["text"]), 9000)

    def test_completed_response_with_unknown_charge_is_not_retried(self):
        self.pipeline.prepare(self.identifier, limit=1)
        class RefusedClient:
            def call(self, payload):
                raise ModelResponseError("refusal", model=payload["model"],
                                         response_id="test-refusal", status="completed", usage=None)
        result = self.pipeline.run(self.identifier, client=RefusedClient(), budget_usd=1)
        self.assertEqual(result["uncertain"], 1)
        self.assertGreater(result["accounted_or_reserved_usd"], 0)
        self.assertEqual(self.pipeline.retry_errors(self.identifier)["retry_queued"], 0)
        self.assertGreater(self.pipeline.get_corpus(self.identifier)["usage"]["uncertain_reserved_usd"], 0)

    def test_explicit_crash_recovery_never_replays_a_request(self):
        self.pipeline.prepare(self.identifier, limit=1)
        with self.pipeline.connection() as db:
            db.execute("UPDATE jobs SET status='running', reserved_usd=.1 WHERE corpus_id=?", (self.identifier,))
        restarted = CorpusPipeline(self.root)
        self.assertEqual(restarted.recover_interrupted(self.identifier)["marked_uncertain"], 1)
        self.assertEqual(restarted.retry_errors(self.identifier)["retry_queued"], 0)
        self.assertAlmostEqual(restarted.get_corpus(self.identifier)["usage"]["uncertain_reserved_usd"], .1)

    def test_end_to_end_mock_scoring_and_stale_evidence_invalidation(self):
        self.pipeline.prepare(self.identifier, limit=1)
        self.pipeline.run(self.identifier, client=FixedClient(self.relevant()), budget_usd=1)
        self.pipeline.prepare(self.identifier, stage="extract")
        fact = {"actor_id": "alpha", "quote": "Alpha North proposes a specific tax reform.",
                "claim": "A specific tax proposal", "kind": "public_statement",
                "event_date": "2025-10-12", "topic": "tax", "trait_candidates": ["Tax reform"],
                "capacity": "public statement"}
        self.pipeline.run(self.identifier, client=FixedClient({"facts": [fact]}), budget_usd=1)
        self.assertEqual(self.pipeline.get_corpus(self.identifier)["counts"]["facts"], 1)
        cfg = self.pipeline.get_corpus(self.identifier)["config"]
        facts = self.pipeline._facts(self.identifier, cfg)
        scale = {"id": "tax-change", "name": "Tax change", "definition": "Explicit change in tax rules",
                 "aliases": [], "anchor0": "Explicitly absent in a relevant context", "anchor3": "Specific concrete proposal",
                 "anchor5": "A strong explicit central commitment", "evidence_ids": [facts[0]["id"]]}
        first_export = self.pipeline.export(self.identifier)
        self.pipeline.prepare(self.identifier, stage="describe")
        self.pipeline.run(self.identifier, client=FixedClient({"scales": [scale]}), budget_usd=1)
        second_export = self.pipeline.export(self.identifier)
        self.assertNotEqual(first_export["directory"], second_export["directory"])
        self.assertEqual(json.loads((Path(first_export["directory"]) / "draft_scales.json").read_text(encoding="utf-8")), [])
        self.pipeline.configure(self.identifier, {"scales": [scale], "scales_approved": True})
        self.pipeline.prepare(self.identifier, stage="score")
        rating = {"actor_id": "alpha", "scale_id": "tax-change", "status": "observed", "value": 3,
                  "evidence_ids": [facts[0]["id"]], "reason": "Explicit concrete synthetic proposal"}
        self.pipeline.run(self.identifier, client=FixedClient({"ratings": [rating]}), budget_usd=1)
        self.assertEqual(self.pipeline.get_corpus(self.identifier)["counts"]["ratings"], 1)
        with self.pipeline.connection() as db:
            row = db.execute("SELECT * FROM facts WHERE id=?", (facts[0]["id"],)).fetchone()
            added = dict(facts[0], id="additional-synthetic-fact", claim="Additional independent synthetic context")
            db.execute("INSERT INTO facts VALUES(?,?,?,?,?,?)", (added["id"], self.identifier, row["job_id"],
                       "alpha", row["document_id"], json.dumps(added)))
        self.assertEqual(self.pipeline.get_corpus(self.identifier)["counts"]["ratings"], 0)
        exported = self.pipeline.export(self.identifier)
        self.assertEqual(json.loads((Path(exported["directory"]) / "ratings.json").read_text(encoding="utf-8")), [])


if __name__ == "__main__":
    unittest.main()
