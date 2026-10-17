"""Local source preparation remains private and never invokes model clients."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.prepare_full_corpus import prepare, repair_split_dates
from src.corpus_pipeline.service import CorpusPipeline, source_context


class ByteEncoding:
    name = "synthetic-byte-encoding"

    def encode(self, text, **kwargs):
        return list(text.encode("utf-8"))


class FullPreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        source = self.root / "synthetic.csv"
        with source.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream, delimiter=";")
            writer.writerow(["Дата публикации", "Время публикации", "Текст сообщения", "Тип", "Кто пишет"])
            for day, time, text, kind, author in [
                ("17.10.2026", "15:41", "Alpha North supports Rwanda policy. PRIVATE_BODY", "Пост", "PRIVATE_AUTHOR"),
                ("16.10.2026", "12:00", "Alpha North supports Rwanda policy. PRIVATE_BODY", "Репосты с дополнением", "PRIVATE_COPY"),
                ("15.10.2026", "10:00", "An unidentified Rwanda speaker comments. PRIVATE_REVIEW", "Статья", "PRIVATE_AUTHOR"),
                ("14.10.2026", "11:00", "Alpha North comments without a country name.", "Пост", "PRIVATE_AUTHOR"),
                ("13.10.2026", "09:00", "", "", "")]:
                writer.writerow([day, time, text, kind, author])
        self.pipeline = CorpusPipeline(self.root)
        config = {"scope": "Synthetic public positions", "objects": [{"id": "alpha", "name": "Alpha North"}],
                  "date_from": "2021-10-15", "date_to": "2026-10-17", "country_terms": []}
        self.corpus = self.pipeline.import_csv(source, config=config)
        self.identifier = self.corpus["id"]
        self.pilot = self.root / "pilot.json"
        self.pilot.write_text(json.dumps({"corpus": {"config": {"country_terms": ["Rwanda"]}}}), encoding="utf-8")
        # Provenance hashes cover the implementation that the calling task runs.
        for rel in ("scripts/prepare_full_corpus.py", "src/corpus_pipeline/context_policy.py",
                    "src/corpus_pipeline/cleaning.py", "src/corpus_pipeline/budget_forecast.py",
                    "src/corpus_pipeline/preparation_cache.py"):
            path = self.root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic version", encoding="utf-8")

    def test_export_deduplicates_text_but_keeps_unknowns_and_origin_index(self):
        with patch("scripts.prepare_full_corpus.offline_encoding", return_value=ByteEncoding()), \
             patch("scripts.prepare_full_corpus.forecast", return_value={"variants": {}}) as estimate, \
             patch("src.corpus_pipeline.models.OpenAIResponsesHTTP.call", side_effect=AssertionError("No API")):
            result = prepare(self.root, self.identifier, self.pilot)
        self.assertEqual(result["raw_records"], 5)
        self.assertEqual(result["import_status_counts"], {"unique": 3, "exact_duplicate": 1, "empty": 1})
        self.assertEqual(result["kept_for_semantic_screening"], 3)
        self.assertEqual(result["api_calls"], 0)
        self.assertEqual(result["api_cost_usd"], 0)
        self.assertNotIn("PRIVATE_", json.dumps(result))
        directory = self.pipeline.runtime / "cleaning" / self.identifier
        sources = [json.loads(line) for line in (directory / "cleaned_unique.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(sources), 3)
        self.assertTrue(any("PRIVATE_BODY" in doc["text"] for doc in sources))
        self.assertTrue(any(doc["cleaning"]["status"] == "needs_semantic_review" for doc in sources))
        index = [json.loads(line) for line in (directory / "cleaning_index.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(index), 5)
        self.assertTrue(any(doc.get("duplicate_of") for doc in index))
        plans = estimate.call_args.args[0]
        self.assertEqual({name: len(values) for name, values in plans.items()}, {"base1500": 3, "adaptive6000": 3})
        for plan in plans["adaptive6000"]:
            self.assertIn("author_is_verified_actor", plan["source_context"])
            self.assertIn(plan["token_budget"], (1500, 6000))
        with patch("scripts.prepare_full_corpus.offline_encoding", return_value=ByteEncoding()), \
             patch("scripts.prepare_full_corpus.forecast", return_value={"variants": {}}), \
             patch("scripts.prepare_full_corpus.annotate_document", side_effect=AssertionError("Use the verified cleaning checkpoint")):
            resumed = prepare(self.root, self.identifier, self.pilot)
        self.assertEqual(resumed["cleaning"], result["cleaning"])
        self.assertEqual(resumed["source_type_counts"], result["source_type_counts"])
        self.assertEqual(resumed["context"], result["context"])

    def test_split_date_repair_is_idempotent_and_does_not_change_original_cells(self):
        with self.pipeline.connection() as db:
            for row in db.execute("SELECT id,data FROM documents WHERE corpus_id=?", (self.identifier,)).fetchall():
                value = json.loads(row["data"])
                value.update(published_at=value["metadata"]["Время публикации"], published_at_raw=value["metadata"]["Время публикации"])
                db.execute("UPDATE documents SET data=? WHERE id=?", (json.dumps(value), row["id"]))
        self.assertEqual(repair_split_dates(self.pipeline, self.identifier), 5)
        self.assertEqual(repair_split_dates(self.pipeline, self.identifier), 0)
        document = self.pipeline.documents(self.identifier)["documents"][0]
        self.assertEqual(document["published_at"], "2026-10-17T15:41:00")
        self.assertEqual(document["metadata"]["Дата публикации"], "17.10.2026")
        self.assertEqual(document["metadata"]["Время публикации"], "15:41")

    def test_repost_with_added_commentary_retains_the_repost_channel(self):
        context = source_context({"metadata": {"Тип": "Репосты с дополнением", "Тип площадки": "Соцсеть"}})
        self.assertEqual(context["source_type"], "repost")
        self.assertFalse(context["author_is_verified_actor"])


if __name__ == "__main__":
    unittest.main()
