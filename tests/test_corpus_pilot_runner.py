"""Restarting a pilot must never silently ignore a lower requested budget."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.run_rwanda_corpus_pilot import setup


class CorpusPilotRunnerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        values = {
            "studies/rwanda-politicians-2021-2026/study.json": {"objects": [{"id": "alpha"}], "date_from": "2021-10-15"},
            "research/rwanda_politicians_2021_2026/corpus_search/actor_aliases.json": {
                "actors": [{"id": "alpha", "canonical_latin": "Alpha North", "aliases": []}]},
            "research/rwanda_politicians_2021_2026/corpus_search/verified_language_aliases.json": {"aliases": []},
            ".corpus_runtime/pilots/corpus-test/manifest.json": {"corpus_id": "corpus-test", "budget_cap_usd": 2},
        }
        for name, value in values.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value), encoding="utf-8")
        class FakePipeline:
            runtime = self.root / ".corpus_runtime"
            saved_cap = 2

            def import_csv(self, *args, **kwargs):
                return {"id": "corpus-test", "config": {"budget_limit_usd": self.saved_cap}}
        self.pipeline = FakePipeline()

    def test_restarting_with_lower_budget_is_rejected_before_any_phase(self):
        with patch("scripts.run_rwanda_corpus_pilot.ROOT", self.root):
            with self.assertRaisesRegex(ValueError, "requested=0.01"):
                setup(self.pipeline, Path("input.csv"), 48, .01)

    def test_matching_saved_budget_resumes(self):
        with patch("scripts.run_rwanda_corpus_pilot.ROOT", self.root):
            self.assertEqual(setup(self.pipeline, Path("input.csv"), 48, 2)["budget_cap_usd"], 2)

    def test_changed_database_cap_cannot_disagree_with_manifest(self):
        self.pipeline.saved_cap = .01
        with patch("scripts.run_rwanda_corpus_pilot.ROOT", self.root):
            with self.assertRaisesRegex(ValueError, "config=0.01"):
                setup(self.pipeline, Path("input.csv"), 48, 2)


if __name__ == "__main__":
    unittest.main()
