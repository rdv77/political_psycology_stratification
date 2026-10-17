"""Frozen selection, resume and paid queue scope; no real API."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.run_half_corpus import (enqueue_extraction, enqueue_relevance, guard_queue, setup, run_stage,
    require_completed_describe, acknowledge_uncertain_job, validated_uncertain_holds)
from src.corpus_pipeline.models import estimate_cost, payload_input_bound
from src.corpus_pipeline.service import digest
from scripts.prepare_half_corpus import write_jsonl
from scripts.prepare_full_corpus import write
from src.corpus_pipeline.service import CorpusPipeline, source_context


class HalfCorpusExecutionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        source = self.root / "synthetic.csv"
        with source.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["text", "date"])
            writer.writerows([["Paul Kagame proposes reform.", "2026-10-17"],
                              ["Diane Rwigara proposes transparency.", "2026-10-17"],
                              ["Paul Kagame discusses agriculture.", "2026-10-16"],
                              ["Paul Kagame discusses education.", "2026-10-15"]])
        self.pipeline = CorpusPipeline(self.root)
        corpus = self.pipeline.import_csv(source, config={"scope": "Synthetic public positions", "languages": ["en"],
            "objects": [{"id": "paul-kagame", "name": "Paul Kagame"}, {"id": "diane-rwigara", "name": "Diane Rwigara"}],
            "text_selection": {"mode": "mentions", "max_tokens": 1500, "compact_roster": True,
                               "adaptive": True, "expanded_max_tokens": 6000}})
        self.identifier = corpus["id"]
        docs = self.pipeline.documents(self.identifier)["documents"][:2]
        directory = self.pipeline.runtime / "sampling" / self.identifier / "half-v1"
        directory.mkdir(parents=True)
        selected = []
        plans = []
        for doc in docs:
            actor = "diane-rwigara" if "Diane" in doc["text"] else "paul-kagame"
            selected.append({**doc, "cleaning": {"matched_actor_ids": [actor]}, "source_context": source_context(doc)})
            fragment = {key: doc.get(key, "") for key in ("id", "title", "url", "published_at", "source_name", "author", "language_hint")}
            fragment["excerpts"] = [{"start": 0, "end": len(doc["text"]), "text": doc["text"]}]
            plans.append({"document": fragment, "actor_ids": [actor], "token_budget": 1500})
        info = {"selected_unique": write_jsonl(directory / "selected_unique.jsonl", selected),
                "context_plans": write_jsonl(directory / "adaptive_context_plans.jsonl", plans)}
        ids = [doc["id"] for doc in docs]
        write(directory / "selected_ids.json", ids)
        write(directory / "selection.json", {"documents": [{"id": identifier, "selected": True} for identifier in ids]})
        info["selection_sha256"] = hashlib.sha256((directory / "selection.json").read_bytes()).hexdigest()
        write(directory / "summary.json", {"corpus_id": self.identifier, "source_sha256": corpus["source_sha256"],
                                           "private_artifacts": info, "selected_unique": 2})
        self.directory = directory

    def test_reuses_frozen_half_without_api_and_cannot_rebase_cap(self):
        manifest, documents, plans = setup(self.pipeline, self.identifier)
        self.assertEqual(manifest["budget_cap_usd"], 70)
        self.assertEqual(self.pipeline.get_corpus(self.identifier)["config"]["budget_limit_usd"], 68)
        self.assertEqual(enqueue_relevance(self.pipeline, manifest, documents, plans), 2)
        self.assertEqual(enqueue_relevance(self.pipeline, manifest, documents, plans), 0)
        guard_queue(self.pipeline, manifest, "relevance", set(documents))
        resumed, _, _ = setup(self.pipeline, self.identifier)
        self.assertEqual(manifest, resumed)
        self.assertEqual(self.pipeline.get_corpus(self.identifier)["usage"]["estimated_usd"], 0)
        with self.assertRaisesRegex(ValueError, "frozen"):
            setup(self.pipeline, self.identifier, budget=71)
        with self.assertRaisesRegex(ValueError, "Unexpected pending stage"):
            guard_queue(self.pipeline, manifest, "extract", set(documents))

    def test_changed_ids_and_extra_pending_documents_block_execution(self):
        manifest, documents, plans = setup(self.pipeline, self.identifier)
        self.pipeline.prepare(self.identifier, "relevance", limit=100)
        with self.assertRaisesRegex(ValueError, "outside selected half"):
            guard_queue(self.pipeline, manifest, "relevance", set(documents))
        write(self.directory / "selected_ids.json", ["unselected-or-invalid-id"])
        with self.assertRaisesRegex(ValueError, "Selected ID frame changed"):
            setup(self.pipeline, self.identifier)

    def test_cached_extract_matches_ordinary_preparation_and_preserves_passages(self):
        manifest, documents, plans = setup(self.pipeline, self.identifier)
        enqueue_relevance(self.pipeline, manifest, documents, plans)
        with self.assertRaisesRegex(ValueError, "Finish classification"):
            enqueue_extraction(self.pipeline, manifest, documents)
        with self.pipeline.connection() as db:
            for row in db.execute("SELECT id,context FROM jobs WHERE corpus_id=?", (self.identifier,)).fetchall():
                context = json.loads(row["context"])
                result = {"status": "relevant", "actor_ids": [actor["id"] for actor in context["config"]["objects"]],
                          "language": "en", "reason": "Synthetic source check"}
                db.execute("UPDATE jobs SET status='done',result=? WHERE id=?", (json.dumps(result), row["id"]))
        self.assertEqual(enqueue_extraction(self.pipeline, manifest, documents), 2)
        self.assertEqual(enqueue_extraction(self.pipeline, manifest, documents), 0)
        self.assertEqual(self.pipeline.prepare(self.identifier, stage="extract", limit=100,
                                              document_ids=list(documents))["created"], 0)
        with self.pipeline.connection() as db:
            rows = db.execute("SELECT context FROM jobs WHERE corpus_id=? AND stage='extract'", (self.identifier,)).fetchall()
        self.assertEqual(len(rows), 2)
        planned = {plan["document"]["id"]: plan["document"]["excerpts"] for plan in plans}
        for row in rows:
            context = json.loads(row["context"])
            self.assertEqual(context["document"]["excerpts"], planned[context["document"]["id"]])
            self.assertFalse(context["document"]["author_is_verified_actor"])

    def test_resuming_extract_restores_68_after_describe_without_rebasing_spend(self):
        manifest, documents, plans = setup(self.pipeline, self.identifier)
        enqueue_relevance(self.pipeline, manifest, documents, plans)
        with self.pipeline.connection() as db:
            db.execute("UPDATE jobs SET status='done',cost_usd=.5 WHERE corpus_id=?", (self.identifier,))
        self.pipeline.configure(self.identifier, {"budget_limit_usd": 70.})
        manifest_file = self.pipeline.runtime / "pilots" / self.identifier / "half-70-v1/manifest.json"
        original_manifest = manifest_file.read_bytes()
        before = self.pipeline._row(self.identifier)["config"]
        def workers(root, identifier, **kwargs):
            self.assertEqual(self.pipeline._row(identifier)["config"]["budget_limit_usd"], 68.)
            self.assertEqual(kwargs["stage"], "extract")
            return {"attempted": 0, "stopped_reason": "queue_empty"}
        with patch.dict("os.environ", {"OPENAI_API_KEY": "synthetic-offline-key"}), \
                patch("src.corpus_pipeline.bounded_workers.run_workers", side_effect=workers):
            run_stage(self.pipeline, manifest, "extract", documents)
        after = self.pipeline._row(self.identifier)["config"]
        self.assertEqual(after, dict(before, budget_limit_usd=68.))
        self.assertEqual(manifest_file.read_bytes(), original_manifest)
        with self.pipeline.connection() as db:
            self.assertEqual(db.execute("SELECT SUM(cost_usd) FROM jobs WHERE corpus_id=?", (self.identifier,)).fetchone()[0], 1.)

    def seed_describe(self, manifest, status="done"):
        config = self.pipeline._row(self.identifier)["config"]
        with self.pipeline.connection() as db:
            db.execute("""INSERT INTO jobs(id,corpus_id,stage,unit_id,recipe,context,payload,status,created_at,cost_usd)
                VALUES('describe-fixture',?,'describe','describe-unit',?,'{}','{}',?,'2026-10-17',.25)""",
                (self.identifier, manifest["initial_recipes"]["describe"], status))

    def test_extract_tail_blocked_before_describe_and_for_nonterminal_describe(self):
        manifest, documents, _ = setup(self.pipeline, self.identifier)
        with self.assertRaisesRegex(ValueError, "nonempty fully terminal"):
            require_completed_describe(self.pipeline, manifest)
        self.seed_describe(manifest, "pending")
        for status in ("pending", "error", "running", "uncertain"):
            with self.pipeline.connection() as db:
                db.execute("UPDATE jobs SET status=? WHERE id='describe-fixture'", (status,))
            with self.subTest(status=status), self.assertRaises(ValueError):
                require_completed_describe(self.pipeline, manifest)
        self.assertEqual(self.pipeline._row(self.identifier)["config"]["budget_limit_usd"], 68.)

    def test_extract_tail_releases_only_unused_total_cap_without_spend_or_manifest_reset(self):
        manifest, documents, plans = setup(self.pipeline, self.identifier)
        enqueue_relevance(self.pipeline, manifest, documents, plans)
        config = self.pipeline._row(self.identifier)["config"]
        with self.pipeline.connection() as db:
            db.execute("UPDATE jobs SET stage='extract',recipe=? WHERE corpus_id=?",
                       (manifest["initial_recipes"]["extract"], self.identifier))
        self.seed_describe(manifest, "review")
        manifest_file = self.pipeline.runtime / "pilots" / self.identifier / "half-70-v1/manifest.json"
        original = manifest_file.read_bytes()
        def workers(root, identifier, **kwargs):
            self.assertEqual(self.pipeline._row(identifier)["config"]["budget_limit_usd"], 70.)
            self.assertEqual(kwargs["stage"], "extract")
            return {"attempted": 0, "stopped_reason": "queue_empty"}
        with patch.dict("os.environ", {"OPENAI_API_KEY": "synthetic-offline-key"}), \
                patch("src.corpus_pipeline.bounded_workers.run_workers", side_effect=workers):
            trace = run_stage(self.pipeline, manifest, "extract", documents, extraction_tail=True)
        self.assertEqual(trace["budget_policy"]["phase"], "extract-tail")
        self.assertEqual(trace["budget_policy"]["describe_terminal_statuses_at_tail_start"], {"review": 1})
        self.assertFalse(trace["budget_policy"]["actual_spend_rebased"])
        self.assertFalse(trace["budget_policy"]["describe_automatically_refreshed"])
        self.assertTrue((manifest_file.parent / "run-extract-tail-latest.json").exists())
        self.assertEqual(manifest_file.read_bytes(), original)
        with self.pipeline.connection() as db:
            self.assertEqual(db.execute("SELECT SUM(cost_usd) FROM jobs WHERE corpus_id=?", (self.identifier,)).fetchone()[0], .25)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM jobs WHERE stage='extract' AND status='pending'").fetchone()[0], 2)
        self.assertEqual(self.pipeline._row(self.identifier)["config"], dict(config, budget_limit_usd=70.))

    def test_extract_tail_blocked_by_uncertain_request_in_other_stage(self):
        manifest, documents, plans = setup(self.pipeline, self.identifier)
        self.seed_describe(manifest)
        enqueue_relevance(self.pipeline, manifest, documents, plans)
        with self.pipeline.connection() as db:
            db.execute("UPDATE jobs SET status='uncertain',reserved_usd=.1 WHERE stage='relevance'")
        with self.assertRaisesRegex(ValueError, "running/uncertain"):
            require_completed_describe(self.pipeline, manifest)

    def held_fixture(self):
        manifest, documents, plans = setup(self.pipeline, self.identifier)
        enqueue_relevance(self.pipeline, manifest, documents, plans)
        with self.pipeline.connection() as db:
            row = db.execute("SELECT id,payload FROM jobs ORDER BY id LIMIT 1").fetchone()
            job_id, payload = row["id"], json.loads(row["payload"])
            reserve = estimate_cost(payload["model"], payload_input_bound(payload), payload["max_output_tokens"], pricing=manifest["pricing"])
            db.execute("UPDATE jobs SET status='uncertain',reserved_usd=?,started_at='start',finished_at='finish' WHERE id=?", (reserve, job_id))
        directory = self.pipeline.runtime / "pilots" / self.identifier / "half-70-v1"
        receipt = {"request_sha256": digest(payload), "http_status": 503, "usage": None, "response_id": None,
                   "model": payload["model"], "timestamp": "2026-10-17"}
        ledger = directory / "api_receipts_private.jsonl"
        ledger.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
        return manifest, documents, job_id, reserve, ledger

    def test_explicit_503_hold_keeps_full_reserve_and_never_requeues_over_cap(self):
        from src.corpus_pipeline.bounded_workers import run_workers
        manifest, documents, job_id, reserve, ledger = self.held_fixture()
        with self.assertRaisesRegex(ValueError, "running/uncertain"):
            guard_queue(self.pipeline, manifest, "relevance", set(documents))
        held = acknowledge_uncertain_job(self.pipeline, manifest, job_id)
        self.assertEqual(held, {"job_ids": {job_id}, "reserved_usd": reserve})
        guard_queue(self.pipeline, manifest, "relevance", set(documents))
        with self.pipeline.connection() as db:
            db.execute("""INSERT INTO jobs(id,corpus_id,stage,unit_id,recipe,context,payload,status,created_at,cost_usd)
                VALUES('old-spend',?,'relevance','old-unit',?,'{}','{}','done','2026-10-17',?)""",
                       (self.identifier, manifest["initial_recipes"]["relevance"], 68 - reserve - 1e-8))
        class NoPaidCall:
            def call(self, payload):
                raise AssertionError("Held reserve must block a further call over cap")
            def close(self):
                pass
        result = run_workers(self.root, self.identifier, workers=1, stage="relevance", client_factory=NoPaidCall)
        self.assertEqual(result["attempted"], 0)
        self.assertEqual(result["accounting"]["outstanding_reserved_usd"], reserve)
        with self.pipeline.connection() as db:
            row = db.execute("SELECT status,reserved_usd FROM jobs WHERE id=?", (job_id,)).fetchone()
        self.assertEqual(tuple(row), ("uncertain", reserve))

    def test_hold_rejects_tampered_ack_pinned_receipt_and_duplicate_unknown_call(self):
        manifest, documents, job_id, reserve, ledger = self.held_fixture()
        acknowledge_uncertain_job(self.pipeline, manifest, job_id)
        journal = ledger.parent / "uncertain_holds.json"
        original = journal.read_bytes()
        changed = json.loads(original)
        changed["holds"][0]["reserved_usd"] += .001
        journal.write_text(json.dumps(changed), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "changed"):
            validated_uncertain_holds(self.pipeline, manifest)
        journal.write_bytes(original)
        receipt = ledger.read_bytes()
        ledger.write_bytes(receipt + receipt)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            validated_uncertain_holds(self.pipeline, manifest)
        ledger.write_bytes(receipt.replace(b'503', b'500'))
        with self.assertRaisesRegex(ValueError, "receipt changed"):
            validated_uncertain_holds(self.pipeline, manifest)

    def test_hold_allows_tail_but_new_unacknowledged_uncertain_still_blocks(self):
        manifest, documents, job_id, reserve, ledger = self.held_fixture()
        acknowledge_uncertain_job(self.pipeline, manifest, job_id)
        with self.pipeline.connection() as db:
            db.execute("UPDATE jobs SET status='done' WHERE id!=?", (job_id,))
        self.seed_describe(manifest)
        self.assertEqual(require_completed_describe(self.pipeline, manifest), {"done": 1})
        with self.pipeline.connection() as db:
            db.execute("UPDATE jobs SET status='uncertain',reserved_usd=.1 WHERE id!=? AND stage='relevance'", (job_id,))
        with self.assertRaisesRegex(ValueError, "running/uncertain"):
            require_completed_describe(self.pipeline, manifest)


if __name__ == "__main__":
    unittest.main()
