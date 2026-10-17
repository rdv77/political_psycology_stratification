"""Bounded source passages remain auditable through queue, guards and storage."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

from src.corpus_pipeline.evidence import normalize_extraction
from src.corpus_pipeline.identity import ground_relevance
from src.corpus_pipeline.models import make_task, validate_result
from src.corpus_pipeline.service import CorpusPipeline, compact_objects
from scripts.run_corpus_full_pilot import preparation_coverage


class Client:
    def __init__(self, data):
        self.data = data

    def call(self, payload):
        return {"data": self.data, "model": payload["model"], "response_id": "synthetic",
                "usage": {"input_tokens": 100, "output_tokens": 40,
                          "input_tokens_details": {"cached_tokens": 0}}}


class ExcerptIntegrationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.quote = "Alpha North said: we support the proposal."
        self.text = "Background without any actor. " * 160 + self.quote
        self.source = self.root / "input.csv"
        with self.source.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["title", "text", "url"])
            writer.writerow(["Background", self.text, "https://example.org/a"])
        self.pipeline = CorpusPipeline(self.root)
        self.config = {"scope": "Synthetic policies", "objects": [
            {"id": "alpha", "name": "Alpha North", "aliases": ["Alpha", "Альфа"]},
            {"id": "beta", "name": "Beta South", "aliases": ["Бета"]}]}
        self.corpus = self.pipeline.import_csv(self.source, config=self.config)
        self.identifier = self.corpus["id"]

    def fact(self):
        return {"actor_id": "alpha", "quote": self.quote, "claim": "Supports proposal",
                "kind": "public_statement", "event_date": None, "topic": "policy",
                "trait_candidates": ["cooperation"], "capacity": "official",
                "attribution_basis": "direct_statement", "attribution_evidence": self.quote,
                "date_evidence": None}

    def test_default_full_preserves_legacy_recipe(self):
        current = self.corpus["config"]
        legacy = {k: v for k, v in current.items() if k != "text_selection"}
        for stage in ("relevance", "extract", "describe", "score"):
            self.assertEqual(self.pipeline._recipe(stage, legacy), self.pipeline._recipe(stage, current))

    def test_selection_changes_downstream_cache_and_pending_queue(self):
        self.pipeline.prepare(self.identifier)
        old = self.corpus["config"]
        configured = self.pipeline.configure(self.identifier, {"text_selection": {
            "mode": "mentions", "max_tokens": 500, "compact_roster": True}})
        self.assertEqual(configured["jobs"]["obsolete"], 1)
        for stage in ("relevance", "extract", "describe", "score"):
            self.assertNotEqual(self.pipeline._recipe(stage, old), self.pipeline._recipe(stage, configured["config"]))

    def test_late_actor_is_selected_once_and_quote_keeps_original_offset(self):
        self.pipeline.configure(self.identifier, {"text_selection": {
            "mode": "mentions", "max_tokens": 500, "compact_roster": True}})
        prepared = self.pipeline.prepare(self.identifier)
        self.assertEqual(prepared["created"], 1)
        self.assertEqual(prepared["api_calls"], 0)
        coverage = preparation_coverage(self.pipeline, self.identifier)
        self.assertEqual(coverage["expected"], 1)
        self.assertEqual(coverage["prepared"], 1)
        self.assertEqual(coverage["unexpected_units"], 0)
        with self.pipeline.connection() as db:
            stored = db.execute("SELECT context,payload FROM jobs").fetchone()
            context = json.loads(stored[0])
            wire_context = json.loads(json.loads(stored[1])["input"][0]["content"][0]["text"])
            original = json.loads(db.execute("SELECT data FROM documents").fetchone()[0])["text"]
        doc = context["document"]
        self.assertNotIn("text", doc)
        self.assertIn("text_selection", doc)
        self.assertNotIn("text_selection", wire_context["document"])
        self.assertEqual([a["id"] for a in context["config"]["objects"]], ["alpha"])
        self.assertTrue(any(self.quote in s["text"] for s in doc["excerpts"]))
        for span in doc["excerpts"]:
            self.assertEqual(span["text"], original[span["start"]:span["end"]])
        # This isolated temporary store has no tokenizer cache: fallback still
        # obeys a stricter total byte cap, not 500 bytes for every window.
        self.assertLessEqual(sum(len(s["text"].encode("utf-8")) for s in doc["excerpts"]), 500)
        self.pipeline.run(self.identifier, budget_usd=1, client=Client({
            "status": "relevant", "actor_ids": ["alpha"], "language": "en", "reason": "Named actor"}))
        self.assertEqual(self.pipeline.prepare(self.identifier, "extract")["created"], 1)
        result = self.pipeline.run(self.identifier, budget_usd=1, client=Client({"facts": [self.fact()]}))
        self.assertEqual(result["completed"], 1)
        with self.pipeline.connection() as db:
            fact = json.loads(db.execute("SELECT data FROM facts").fetchone()[0])
        self.assertEqual(fact["quote_start"], original.index(self.quote))
        self.assertEqual(original[fact["quote_start"]:fact["quote_end"]], self.quote)

    def test_no_name_fallback_keeps_roster_and_does_not_invent_aliases(self):
        actors = self.config["objects"]
        self.assertEqual(compact_objects(actors, {"text": "A nameless statement"}), actors)
        self.assertEqual(compact_objects(actors, {"text": "Альфа supports cooperation."}),
                         [{"id": "alpha", "name": "Alpha North", "aliases": ["Альфа"]}])
        self.assertEqual(compact_objects(actors, {"text": "SuperAlpha supports cooperation."}), actors)

    def test_quoted_passages_cannot_bridge_gaps_or_borrow_another_speaker(self):
        context = {"config": self.config, "document": {"excerpts": [
            {"start": 10, "end": 10 + len(self.quote), "text": self.quote},
            {"start": 900, "end": 924, "text": "Beta South says nothing."}]}}
        payload = make_task("extract", context)
        self.assertIn("ONE excerpt", payload["instructions"])
        self.assertEqual(len(validate_result("extract", {"facts": [self.fact()]}, context)["facts"]), 1)
        borrowed = dict(self.fact(), attribution_evidence="Beta South says nothing.")
        with self.assertRaises(ValueError):
            validate_result("extract", {"facts": [borrowed]}, context)
        normalized = normalize_extraction({"facts": [borrowed]}, context, validate_result)
        self.assertEqual(normalized["facts"], [])
        invented = dict(self.fact(), quote=self.quote + "Beta South says nothing.")
        with self.assertRaises(ValueError):
            validate_result("extract", {"facts": [invented]}, context)

    def test_identity_cannot_be_composed_from_separate_excerpts(self):
        context = {"config": self.config, "document": {"excerpts": [
            {"start": 0, "end": 5, "text": "Alpha"}, {"start": 100, "end": 105, "text": "North"}]}}
        context["config"] = copy.deepcopy(self.config)
        context["config"]["objects"][0]["aliases"] = []
        result = ground_relevance({"status": "relevant", "actor_ids": ["alpha"], "reason": "Synthetic"}, context)
        self.assertEqual(result["status"], "review")

    def test_compact_roster_matches_identity_unicode_and_whitespace_rules(self):
        actors = [{"id": "emile", "name": "Émile Nord", "aliases": ["Émile"]},
                  {"id": "alpha", "name": "Alpha North", "aliases": []},
                  {"id": "beta", "name": "Beta South", "aliases": []}]
        text = "E\u0301mile Nord speaks. Alpha\nNorth replies. Beta South answers."
        compact = compact_objects(actors, {"text": text})
        self.assertEqual([a["id"] for a in compact], ["emile", "alpha", "beta"])
        self.assertEqual(compact[0]["aliases"], ["Émile"])

    def test_invalid_selection_is_rejected_before_queue(self):
        for policy in ({"mode": "mentions", "max_tokens": True}, {"mode": "mentions", "max_tokens": 0},
                       {"mode": "other", "max_tokens": 500}, {"mode": "head", "max_tokens": 500, "secret": "x"}):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                self.pipeline.configure(self.identifier, {"text_selection": policy})


if __name__ == "__main__":
    unittest.main()
