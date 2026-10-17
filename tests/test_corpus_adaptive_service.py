"""Adaptive queue preparation with full-frame counts and zero model calls."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.corpus_pipeline.context_policy import select_document_context
from src.corpus_pipeline.service import CorpusPipeline, packed


class AdaptiveServiceTests(unittest.TestCase):
    def setUp(self):
        writable = Path(__file__).resolve().parents[1] / ".service_runtime"
        writable.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=writable)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        api = patch("src.corpus_pipeline.models.OpenAIResponsesHTTP.call", side_effect=AssertionError("No API calls allowed"))
        api.start()
        self.addCleanup(api.stop)
        self.alpha = "Rwanda Alpha North " + "ordinary source details " * 130
        self.betas = ["Rwanda " + "background details " * 110 + f" Beta South speaks in example {i}. "
                      + "further context " * 90 for i in range(6)]
        source = self.root / "input.csv"
        with source.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream, delimiter=";")
            writer.writerow(["id", "text", "date", "url", "Тип"])
            writer.writerow(["alpha", self.alpha, "2026-10-16", "https://example.invalid/a", "Статья"])
            for i, text in enumerate(self.betas):
                writer.writerow(["beta" + str(i), text, "2026-10-16", f"https://example.invalid/b{i}", "Статья"])
            # The same source body at another URL is not an extra unique doc.
            writer.writerow(["alpha-copy", self.alpha, "2026-10-16", "https://example.invalid/copy", "Статья"])
            writer.writerow(["alpha-no-country", "Alpha North " + "excluded details " * 150,
                             "2026-10-16", "https://example.invalid/no-country", "Статья"])
            writer.writerow(["alpha-outside-period", "Rwanda Alpha North " + "historical details " * 150,
                             "2020-10-16", "https://example.invalid/old", "Статья"])
            writer.writerow(["empty", "", "", "", ""])
        self.pipeline = CorpusPipeline(self.root)
        self.policy = {"mode": "mentions", "max_tokens": 1500, "compact_roster": True,
                       "adaptive": True, "expanded_max_tokens": 6000}
        self.config = {"scope": "Synthetic public positions in Rwanda", "objects": [
            {"id": "alpha", "name": "Alpha North", "aliases": []},
            {"id": "beta", "name": "Beta South", "aliases": []}],
            "country_terms": ["Rwanda"], "date_from": "2021-10-15", "date_to": "2026-10-17",
            "text_selection": self.policy}
        corpus = self.pipeline.import_csv(source, config=self.config)
        self.identifier = corpus["id"]
        with self.pipeline.connection() as db:
            self.documents = {json.loads(row["data"])["source_id"]: dict(json.loads(row["data"]), status=row["status"])
                              for row in db.execute("SELECT data,status FROM documents WHERE corpus_id=?", (self.identifier,))}
        self.assertFalse((self.pipeline.runtime / "tokenizer_cache").exists())

    def jobs(self, stage="relevance"):
        with self.pipeline.connection() as db:
            values = [dict(row) for row in db.execute("SELECT * FROM jobs WHERE corpus_id=? AND stage=? ORDER BY created_at,id",
                                                      (self.identifier, stage))]
        for value in values:
            value["context"] = json.loads(value["context"])
            value["payload"] = json.loads(value["payload"])
        return values

    def test_document_subset_does_not_make_common_actor_rare(self):
        selected = self.documents["beta0"]["id"]
        with patch("src.corpus_pipeline.service.select_document_context", wraps=select_document_context) as selector:
            prepared = self.pipeline.prepare(self.identifier, limit=1, document_ids=[selected])
        self.assertEqual((prepared["created"], prepared["api_calls"]), (1, 0))
        args = selector.call_args.args
        self.assertEqual(args[2], {"alpha": 1, "beta": 6})
        self.assertEqual(args[3], 7)
        document = self.jobs()[0]["context"]["document"]
        meta = document["text_selection"]
        self.assertEqual(document["id"], selected)
        self.assertEqual(meta["rare_actor_ids"], [])
        self.assertFalse(meta["expanded"])
        self.assertEqual(meta["effective_max_tokens"], 1500)
        self.assertFalse(meta["token_count_exact"])
        self.assertLessEqual(sum(len(s["text"].encode("utf-8")) for s in document["excerpts"]), 1500)

    def test_limit_one_still_uses_all_eligible_unique_documents(self):
        with patch("src.corpus_pipeline.service.select_document_context", wraps=select_document_context) as selector:
            prepared = self.pipeline.prepare(self.identifier, limit=1)
        self.assertEqual(prepared["created"], 1)
        self.assertGreater(selector.call_count, 0)
        for call in selector.call_args_list:
            self.assertEqual(call.args[2], {"alpha": 1, "beta": 6})
            self.assertEqual(call.args[3], 7)
        self.assertEqual(len(self.jobs()), 1)

    def test_rare_actor_expands_using_private_store_fallback_and_exact_coordinates(self):
        original = self.documents["alpha"]
        self.assertGreater(len(original["text"].encode("utf-8")), 1500)
        self.assertLess(len(original["text"].encode("utf-8")), 6000)
        self.pipeline.prepare(self.identifier, document_ids=[original["id"]])
        doc = self.jobs()[0]["context"]["document"]
        self.assertTrue(doc["text_selection"]["expanded"])
        self.assertEqual(doc["text_selection"]["rare_actor_ids"], ["alpha"])
        self.assertEqual(doc["text_selection"]["effective_max_tokens"], 6000)
        self.assertEqual(doc["text_selection"]["budget_unit"], "utf8_bytes_upper_bound")
        self.assertEqual(len(doc["excerpts"]), 1)
        self.assertEqual(doc["excerpts"][0], {"start": 0, "end": len(original["text"]), "text": original["text"]})

    def test_late_name_windows_keep_global_document_coordinates(self):
        original = self.documents["beta0"]
        self.pipeline.prepare(self.identifier, document_ids=[original["id"]])
        doc = self.jobs()[0]["context"]["document"]
        self.assertNotIn("text", doc)
        self.assertTrue(any(span["start"] > 0 for span in doc["excerpts"]))
        self.assertTrue(any("Beta South" in span["text"] for span in doc["excerpts"]))
        for span in doc["excerpts"]:
            self.assertEqual(span["text"], original["text"][span["start"]:span["end"]])

    def test_diagnostics_stay_private_in_relevance_and_extraction_wire_payloads(self):
        original = self.documents["alpha"]
        self.pipeline.prepare(self.identifier, document_ids=[original["id"]])
        relevance = self.jobs()[0]
        self.assertIn("text_selection", relevance["context"]["document"])
        wire = json.loads(relevance["payload"]["input"][0]["content"][0]["text"])
        self.assertNotIn("text_selection", wire["document"])
        self.assertEqual([a["id"] for a in wire["config"]["objects"]], ["alpha"])
        for private_key in ("rare_actor_ids", "rarity_basis", "longform_signals", "effective_max_tokens"):
            self.assertNotIn(private_key, relevance["payload"]["input"][0]["content"][0]["text"])
        with self.pipeline.connection() as db:
            db.execute("UPDATE jobs SET status='done',result=? WHERE id=?", (packed({"status": "relevant", "actor_ids": ["alpha"],
                       "language": "en", "reason": "Synthetic offline fixture"}), relevance["id"]))
        self.assertEqual(self.pipeline.prepare(self.identifier, stage="extract", document_ids=[original["id"]])["created"], 1)
        extract = self.jobs("extract")[0]
        self.assertTrue(extract["context"]["document"]["text_selection"]["expanded"])
        wire = json.loads(extract["payload"]["input"][0]["content"][0]["text"])
        self.assertNotIn("text_selection", wire["document"])
        self.assertEqual(wire["document"]["excerpts"], extract["context"]["document"]["excerpts"])

    def test_adaptive_lineage_changes_all_stages_but_explicit_false_preserves_bounded_recipe(self):
        base = copy.deepcopy(self.config)
        base["text_selection"] = {k: v for k, v in self.policy.items() if k not in {"adaptive", "expanded_max_tokens"}}
        old = self.pipeline._config(base)
        unchanged = self.pipeline._config(dict(base, text_selection=dict(base["text_selection"], adaptive=False, expanded_max_tokens=6000)))
        adaptive = self.pipeline._config(self.config)
        smaller = self.pipeline._config(dict(self.config, text_selection=dict(self.policy, expanded_max_tokens=5000)))
        for stage in ("relevance", "extract", "describe", "score"):
            with self.subTest(stage=stage):
                self.assertEqual(self.pipeline._recipe(stage, old), self.pipeline._recipe(stage, unchanged))
                self.assertNotEqual(self.pipeline._recipe(stage, old), self.pipeline._recipe(stage, adaptive))
                self.assertNotEqual(self.pipeline._recipe(stage, adaptive), self.pipeline._recipe(stage, smaller))

    def test_full_adaptive_false_default_preserves_legacy_full_recipe(self):
        old = self.pipeline._config(dict(self.config, text_selection={"mode": "full", "max_tokens": 1000, "compact_roster": False}))
        new = self.pipeline._config(dict(self.config, text_selection={"mode": "full", "max_tokens": 1000, "compact_roster": False,
                                                                  "adaptive": False, "expanded_max_tokens": 6000}))
        for stage in ("relevance", "extract", "describe", "score"):
            self.assertEqual(self.pipeline._recipe(stage, old), self.pipeline._recipe(stage, new))

    def test_invalid_adaptive_modes_caps_and_types_are_rejected_before_queue(self):
        bad = [dict(self.policy, mode="full"), dict(self.policy, mode="head"),
               dict(self.policy, expanded_max_tokens=6001), dict(self.policy, expanded_max_tokens=1499),
               dict(self.policy, expanded_max_tokens=True), dict(self.policy, max_tokens=7000),
               dict(self.policy, adaptive="true")]
        for policy in bad:
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                self.pipeline.configure(self.identifier, {"text_selection": policy})
        self.assertEqual(self.jobs(), [])


if __name__ == "__main__":
    unittest.main()
