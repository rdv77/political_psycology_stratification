"""Frozen selection, resume and paid queue scope; no real API."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.run_half_corpus import enqueue_relevance, guard_queue, setup
from scripts.prepare_half_corpus import write_jsonl
from scripts.prepare_full_corpus import write
from src.corpus_pipeline.service import CorpusPipeline


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
            selected.append({**doc, "cleaning": {"matched_actor_ids": [actor]}})
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


if __name__ == "__main__":
    unittest.main()
