"""Synthetic private-file checkpoint tests: no source corpus, SQLite or API."""
from collections import Counter
import copy
import json
from pathlib import Path
import tempfile
import unittest

from src.corpus_pipeline.preparation_cache import CHECKPOINT, create_checkpoint, load_checkpoint


class PreparationCacheTests(unittest.TestCase):
    def setUp(self):
        runtime = Path(__file__).resolve().parents[1] / ".service_runtime"
        runtime.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=runtime)
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.source, self.config = "a" * 64, "b" * 64
        self.annotation = {"status": "needs_semantic_review", "reason": "unknown_language_hint",
                           "matched_actor_ids": ["actor-a"], "flags": ["unknown_language_hint"]}
        self.context = {"source_type": "comment", "source_platform": "social", "author_is_verified_actor": False}
        self.cleaned = [{"id": "doc-2", "source_row": 2, "title": "Synthetic title", "text": "Synthetic private text",
                         "url": "https://example.test/a", "published_at": "2026-01-01", "source_name": "Test",
                         "author": "Unverified author", "language_hint": "", "cleaning": self.annotation,
                         "source_context": self.context}]
        excluded = dict(self.annotation, status="rule_excluded", reason="outside_period", flags=["outside_period"])
        self.index = [{"document_id": "doc-2", "source_row": 2, **self.annotation},
                      {"document_id": "doc-3", "source_row": 3, "status": "exact_duplicate", "duplicate_of": "doc-2"},
                      {"document_id": "doc-4", "source_row": 4, **excluded},
                      {"document_id": "doc-5", "source_row": 5, "status": "empty", "duplicate_of": None}]
        self.write_rows()

    def write_rows(self):
        for name, rows in (("cleaned_unique.jsonl", self.cleaned), ("cleaning_index.jsonl", self.index)):
            (self.directory / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    def test_verified_round_trip_keeps_all_index_statuses_and_recovers_source_context(self):
        before = copy.deepcopy(self.cleaned)
        manifest = create_checkpoint(self.directory, self.source, self.config)
        frames, annotations, statuses = load_checkpoint(self.directory, self.source, self.config)
        self.assertEqual(len(frames), 1)
        self.assertEqual(len(annotations), 2)  # Includes the unique date-excluded row.
        self.assertEqual(statuses, Counter(unique=2, exact_duplicate=1, empty=1))
        document, annotation = frames[0]
        self.assertEqual(document["_source_context"], self.context)
        self.assertEqual(annotation, self.annotation)
        self.assertNotIn("cleaning", document)
        self.assertNotIn("source_context", document)
        self.assertNotIn("metadata", document)
        self.assertEqual(self.cleaned, before)
        self.assertEqual(manifest["counts"]["raw_records"], 4)
        self.assertNotIn("Synthetic", json.dumps(manifest))
        self.assertEqual((self.directory / CHECKPOINT).read_text(encoding="utf-8").count("private"), 0)

    def test_source_config_bytes_and_manifest_count_changes_are_cache_misses(self):
        manifest = create_checkpoint(self.directory, self.source, self.config)
        self.assertIsNone(load_checkpoint(self.directory, "c" * 64, self.config))
        self.assertIsNone(load_checkpoint(self.directory, self.source, "c" * 64))
        manifest["counts"]["raw_records"] += 1
        (self.directory / CHECKPOINT).write_text(json.dumps(manifest), encoding="utf-8")
        self.assertIsNone(load_checkpoint(self.directory, self.source, self.config))
        create_checkpoint(self.directory, self.source, self.config)
        path = self.directory / "cleaned_unique.jsonl"
        path.write_bytes(path.read_bytes().replace(b"Synthetic private text", b"Changed private text"))
        self.assertIsNone(load_checkpoint(self.directory, self.source, self.config))

    def test_missing_extra_and_disagreeing_frames_cannot_be_certified(self):
        variants = [("missing", []),
                    ("duplicate", self.cleaned * 2),
                    ("wrong_identity", [dict(self.cleaned[0], id="other")]),
                    ("wrong_source_row", [dict(self.cleaned[0], source_row=9)]),
                    ("different_annotation", [dict(self.cleaned[0], cleaning=dict(self.annotation, flags=[]))]),
                    ("raw_metadata", [dict(self.cleaned[0], metadata={"sensitive": "not_allowed"})])]
        for name, rows in variants:
            with self.subTest(name=name):
                original = self.cleaned
                self.cleaned = rows
                self.write_rows()
                with self.assertRaises(ValueError):
                    create_checkpoint(self.directory, self.source, self.config)
                self.assertFalse((self.directory / CHECKPOINT).exists())
                self.cleaned = original

    def test_incomplete_generation_and_corrupt_or_missing_files_are_cache_misses(self):
        self.assertIsNone(load_checkpoint(self.directory, self.source, self.config))
        create_checkpoint(self.directory, self.source, self.config)
        pending = self.directory / "cleaning_index.jsonl.tmp"
        pending.write_text("", encoding="utf-8")
        self.assertIsNone(load_checkpoint(self.directory, self.source, self.config))
        with self.assertRaises(ValueError):
            create_checkpoint(self.directory, self.source, self.config)
        pending.unlink()
        (self.directory / "cleaning_index.jsonl").write_text("{broken\n", encoding="utf-8")
        self.assertIsNone(load_checkpoint(self.directory, self.source, self.config))
        (self.directory / "cleaning_index.jsonl").unlink()
        self.assertIsNone(load_checkpoint(self.directory, self.source, self.config))

    def test_index_identity_and_duplicate_representative_must_be_consistent(self):
        for change in (dict(self.index[1], duplicate_of="missing"),
                       dict(self.index[1], duplicate_of="doc-3"),
                       dict(self.index[1], source_row=2)):
            with self.subTest(change=change):
                self.index[1] = change
                self.write_rows()
                with self.assertRaises(ValueError):
                    create_checkpoint(self.directory, self.source, self.config)


if __name__ == "__main__":
    unittest.main()
