"""Offline synthetic rule cases plus actual append-only service integration."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.review_corpus_facts import POLICY, REVIEWER, build_proposals, review
from src.corpus_pipeline.service import CorpusPipeline, digest, packed


class FactReviewRuleTests(unittest.TestCase):
    def proposal(self, claim, source="article", kind="action"):
        facts = [{"id": "f1", "kind": kind, "claim": claim, "extraction_job_id": "j1"}]
        jobs = [{"id": "j1", "context": {"document": {"source_type": source}}}]
        before = copy.deepcopy((facts, jobs))
        result = build_proposals(facts, jobs)
        self.assertEqual((facts, jobs), before)
        return result

    def test_planned_travel_is_context_not_completed_action(self):
        for phrase in ("is scheduled to travel", "plans to visit", "will travel", "will visit", "is due to visit", "se rendra", "prévoit de visiter"):
            with self.subTest(phrase=phrase):
                result = self.proposal("Alpha North " + phrase + " another country.")
                annotation = result["decisions"][0]["annotation"]
                self.assertEqual(annotation["kind"], "institutional_context")
                self.assertEqual(annotation["temporal_status"], "planned")
                self.assertEqual(annotation["measurement_status"], "context_only")

    def test_completed_announcement_is_not_the_future_trip(self):
        for claim in ("Alpha North announced plans to visit another country.",
                      "Alpha North a annoncé qu'il se rendra dans un autre pays."):
            self.assertEqual(self.proposal(claim)["annotations_proposed"], 0)

    def test_post_expert_attribution_regression_not_own_action(self):
        claim = "Alpha North was identified by a UN Group of Experts as coordinating with another movement."
        for source in ("post", "comment", "repost", "unknown"):
            result = self.proposal(claim, source)
            annotation = result["decisions"][0]["annotation"]
            self.assertEqual(annotation["kind"], "attributed_image")
            self.assertEqual(annotation["attribution_basis"], "third_party_claim")
            self.assertNotEqual(annotation.get("measurement_status"), "context_only")

    def test_article_accusation_or_sanction_is_external_action(self):
        for claim in ("Alpha North was accused of misconduct.", "Alpha North has been convicted.",
                      "Alpha North was sanctioned by another country.", "Experts named Alpha North as a coordinator.",
                      "Alpha North was identified by experts as a coordinator."):
            with self.subTest(claim=claim):
                result = self.proposal(claim)
                self.assertEqual(result["decisions"][0]["annotation"]["kind"], "attributed_image")

    def test_completed_own_action_and_own_accusation_not_flagged_by_passive_rule(self):
        for claim in ("Alpha North attended a meeting.", "Alpha North accused another official of misconduct."):
            self.assertEqual(self.proposal(claim)["annotations_proposed"], 0)

    def test_never_upgrades_already_context_or_image(self):
        for kind in ("attributed_image", "institutional_context", "public_statement"):
            self.assertEqual(self.proposal("Alpha North plans to visit.", "post", kind)["annotations_proposed"], 0)


class FactReviewIntegrationTests(unittest.TestCase):
    def setUp(self):
        parent = Path(__file__).resolve().parents[1] / ".service_runtime"
        parent.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = self.root / "input.csv"
        with source.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream, delimiter=";")
            writer.writerow(["id", "text"])
            writer.writerow(["1", "Alpha North is scheduled to travel."])
        self.pipeline = CorpusPipeline(self.root)
        self.corpus = self.pipeline.import_csv(source, config={"scope": "Synthetic", "objects": [{"id": "alpha", "name": "Alpha North"}]})
        self.identifier = self.corpus["id"]
        with self.pipeline.connection() as db:
            document = json.loads(db.execute("SELECT data FROM documents WHERE corpus_id=?", (self.identifier,)).fetchone()[0])
            for n, (claim, source_type) in enumerate([("Alpha North is scheduled to travel.", "article"),
                                                    ("Alpha North was identified by experts as a coordinator.", "post")]):
                job_id = "j" + str(n)
                fact_id = "f" + str(n)
                fact = {"id": fact_id, "actor_id": "alpha", "document_id": document["id"], "kind": "action",
                        "claim": claim, "quote": claim, "event_date": None, "attribution_basis": "reported_action",
                        "extraction_job_id": job_id, "source_url": "https://example.invalid/synthetic"}
                context = {"document": {"id": document["id"], "source_type": source_type, "text": claim}}
                db.execute("INSERT INTO jobs(id,corpus_id,stage,unit_id,recipe,context,payload,status,created_at,result) "
                           "VALUES(?,?,?,?,?,?,?,'done',?,?)", (job_id, self.identifier, "extract", document["id"] + str(n),
                           self.pipeline._recipe("extract", self.corpus["config"]), packed(context), "{}", "2026-10-17T00:00:00Z", packed({"facts": [fact]})))
                db.execute("INSERT INTO facts VALUES(?,?,?,?,?,?)", (fact_id, self.identifier, job_id, "alpha", document["id"], packed(fact)))

    def raw_snapshot(self):
        with self.pipeline.connection() as db:
            return {table: [tuple(row) for row in db.execute("SELECT * FROM " + table + " ORDER BY id")]
                    for table in ("facts", "jobs")}

    def test_dryrun_then_apply_is_append_only_and_idempotent(self):
        raw = self.raw_snapshot()
        before = self.pipeline._facts(self.identifier, self.corpus["config"])
        with patch("src.corpus_pipeline.models.OpenAIResponsesHTTP.call", side_effect=AssertionError("No API")):
            result = review(self.root, self.identifier)
            self.assertEqual(result["annotations_proposed"], 2)
            self.assertFalse(result["apply_requested"])
            self.assertEqual(self.raw_snapshot(), raw)
            self.assertEqual(self.pipeline._facts(self.identifier, self.corpus["config"]), before)
            result = review(self.root, self.identifier, apply=True)
            self.assertEqual(result["application"]["annotations_added"], 2)
            after = self.pipeline._facts(self.identifier, self.corpus["config"])
            self.assertNotEqual(digest(before), digest(after))
            self.assertEqual(self.raw_snapshot(), raw)
            self.assertEqual({f["kind"] for f in after}, {"institutional_context", "attributed_image"})
            result = review(self.root, self.identifier, apply=True)
            self.assertEqual(result["annotations_proposed"], 0)
            self.assertEqual(result["application"]["annotations_added"], 0)
            self.assertEqual(self.raw_snapshot(), raw)

    def test_same_review_receipt_is_idempotent_and_revision_changes_snapshot(self):
        facts = self.pipeline._facts(self.identifier, self.corpus["config"])
        with self.pipeline.connection() as db:
            jobs = [dict(r) for r in db.execute("SELECT id,context FROM jobs WHERE corpus_id=?", (self.identifier,))]
        decisions = build_proposals(facts, jobs)["decisions"]
        self.assertEqual(self.pipeline.review_facts(self.identifier, decisions, reviewer=REVIEWER, policy=POLICY)["annotations_added"], 2)
        first = digest(self.pipeline._facts(self.identifier, self.corpus["config"]))
        self.assertEqual(self.pipeline.review_facts(self.identifier, decisions, reviewer=REVIEWER, policy=POLICY)["annotations_added"], 0)
        self.assertEqual(digest(self.pipeline._facts(self.identifier, self.corpus["config"])), first)
        revision = [{"fact_id": "f1", "annotation": {"measurement_status": "context_only", "review_note": "Further conservative source review"}}]
        self.pipeline.review_facts(self.identifier, revision, reviewer=REVIEWER, policy=POLICY + "-revision")
        self.assertNotEqual(digest(self.pipeline._facts(self.identifier, self.corpus["config"])), first)
        with self.pipeline.connection() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM fact_reviews").fetchone()[0], 3)


if __name__ == "__main__":
    unittest.main()
