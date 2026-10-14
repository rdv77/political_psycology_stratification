"""Bounded coverage completion changes only targeted unknown fixture cells."""
from __future__ import annotations

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts.complete_pilot_coverage import complete_coverage
from src.semantic_service.store import StudyStore, fingerprint


TEMP_PARENT = Path(__file__).resolve().parents[1] / ".service_runtime"


class CoverageCompletionTests(unittest.TestCase):
    def setUp(self):
        TEMP_PARENT.mkdir(exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix="coverage-completion-test-", dir=TEMP_PARENT))
        self.addCleanup(self.cleanup)
        self.store = StudyStore(self.root / "studies")
        study = self.store.create({"title": "Synthetic coverage fixture", "target_count": 12})
        self.identifier = study["id"]
        objects = [{"id": f"o{i}", "name": f"Synthetic object {i}", "sources": []} for i in range(12)]
        scales = [{"id": f"s{i}", "name": f"Synthetic scale {i}", "positive": f"Synthetic A {i}",
                   "negative": f"Synthetic B {i}", "definition": "Synthetic rubric only"} for i in range(15)]
        self.store.edit(self.identifier, "objects", objects)
        self.store.approve(self.identifier, "objects", actor="test_fixture")
        self.store.edit(self.identifier, "scales", scales)
        self.store.approve(self.identifier, "scales", actor="test_fixture")
        self.study = self.store.update(self.identifier, status="complete")
        self.directory = self.store.directory(self.identifier)
        self.ratings_file = self.directory / "ratings.json"
        self.metadata_file = self.directory / "output/metadata.json"
        self.metadata_file.parent.mkdir()
        self.ratings = []
        for obj in objects:
            for scale in scales:
                unknown = (obj["id"] == "o0" and scale["id"] != "s0") or (obj["id"] == "o1" and scale["id"] == "s1")
                self.ratings.append({"object_id": obj["id"], "scale_id": scale["id"],
                                     "status": "unknown" if unknown else "observed",
                                     "positive": None if unknown else 3., "negative": None if unknown else 1.,
                                     "reason": f"Original immutable fixture {obj['id']}/{scale['id']}",
                                     "fixture_extra_field": [obj["id"], scale["id"]]})
        self.metadata = {"loading_table": [{"id": f"s{i}"} for i in range(14)],
                         "diagnostics": {"excluded_objects": [{"id": "o0", "reason": "insufficient_coverage"}]}}
        self.write(self.ratings_file, self.ratings)
        self.write(self.metadata_file, self.metadata)
        evidence = self.directory / "evidence"
        evidence.mkdir()
        self.base = evidence / "o0.json"
        self.supplement = evidence / "dossier_supplement-o0.json"
        self.write(self.base, {"basis": "base-fixture", "facts": [
            {"text": "Supported base fact", "source_url": "https://fixture.example/base"},
            {"text": "Unretrieved base claim", "source_url": "https://invented.invalid/base"}],
            "sources": [{"title": "Base", "url": "https://fixture.example/base"}],
            "retrieval_citations": ["https://fixture.example/base"]})
        self.write(self.supplement, {"basis": "supplement-fixture", "scales_basis": fingerprint(self.study["scales"]),
            "facts": [{"text": "Supported base fact", "source_url": "https://fixture.example/base"},
                      {"text": "Supported first supplement fact", "source_url": "https://fixture.example/supplement"}],
            "sources": [{"title": "Supplement", "url": "https://fixture.example/supplement"}],
            "supplement_provenance": {"retrieval_citations": ["https://fixture.example/supplement"]}})
        self.original_bytes = {file: file.read_bytes() for file in (self.ratings_file, self.metadata_file, self.base, self.supplement)}
        self.prompts = []
        self.keep_unknown = {"s1"}
        self.search_hook = None
        self.provider = SimpleNamespace(model="mock-rating-model", search_model="mock-search-model",
                                        request=Mock(side_effect=self.request))

    def cleanup(self):
        target = self.root.resolve()
        if target.parent != TEMP_PARENT.resolve() or not target.name.startswith("coverage-completion-test-"):
            raise AssertionError("Unsafe fixture cleanup")
        shutil.rmtree(target)

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

    def request(self, identifier, task, prompt, schema, **options):
        self.assertEqual(identifier, self.identifier)
        self.prompts.append((task, prompt, options))
        if task.startswith("coverage-completion-search-"):
            if self.search_hook:
                self.search_hook()
            return {"data": {"facts": [
                {"text": "Supported final retrieval", "source_url": "https://fixture.example/final", "period": "2024"},
                {"text": "Unretrieved fabricated final claim", "source_url": "https://invented.invalid/final", "period": "2024"}],
                "sources": [{"title": "Final", "url": "https://fixture.example/final"},
                            {"title": "Invented", "url": "https://invented.invalid/final"}], "limitations": "Synthetic fixture"},
                "citations": ["https://fixture.example/final"], "model": "mock-search-model", "at": "test-time"}
        self.assertTrue(task.startswith("coverage-completion-ratings-"))
        scales = json.loads(prompt.split("Шкалы: ", 1)[1].split("\nФакты: ", 1)[0])
        facts = json.loads(prompt.split("\nФакты: ", 1)[1])
        self.assertEqual(len(facts), 3)
        self.assertTrue(all("invented.invalid" not in fact["source_url"] for fact in facts))
        return {"data": {"ratings": [{"scale_id": scale["id"],
                    "status": "unknown" if scale["id"] in self.keep_unknown else "observed",
                    "positive": None if scale["id"] in self.keep_unknown else 4.,
                    "negative": None if scale["id"] in self.keep_unknown else 1.,
                    "confidence": .8, "reason": "Synthetic final rating", "evidence_indices": [] if scale["id"] in self.keep_unknown else [2]}
                   for scale in scales]}, "model": "mock-rating-model", "at": "test-time"}

    def run_apply(self):
        return complete_coverage(self.root, self.identifier, apply=True, workers=1, provider=self.provider)

    def test_default_dry_run_has_no_provider_calls_or_backups(self):
        with patch("scripts.complete_pilot_coverage.ModelProvider", side_effect=AssertionError("Dry-run constructed a provider")):
            summary = complete_coverage(self.root, self.identifier)
        self.assertFalse(summary["apply"])
        self.assertEqual(summary["target_objects"], 1)
        self.assertEqual(summary["target_cells"], 13)
        self.assertEqual(summary["targets"], [{"object_id": "o0", "scale_ids": [f"s{i}" for i in range(1, 14)]}])
        self.assertFalse((self.directory / "history").exists())
        self.assertFalse((self.directory / ".coverage_completion.lock").exists())
        self.assertEqual(self.ratings_file.read_bytes(), self.original_bytes[self.ratings_file])
        self.provider.request.assert_not_called()

    def test_only_targeted_unknowns_change_and_all_prior_evidence_is_immutable(self):
        summary = self.run_apply()
        self.assertEqual(summary["newly_observed"], 12)
        self.assertEqual(summary["remaining_target_unknown"], 1)
        rows = json.loads(self.ratings_file.read_text(encoding="utf-8"))
        original = {(row["object_id"], row["scale_id"]): row for row in self.ratings}
        for row in rows:
            key = row["object_id"], row["scale_id"]
            if key[0] != "o0" or key[1] not in {f"s{i}" for i in range(1, 14)}:
                self.assertEqual(row, original[key])
        unknown = next(row for row in rows if row["object_id"] == "o0" and row["scale_id"] == "s1")
        self.assertEqual(unknown["status"], "unknown")
        self.assertIsNone(unknown["positive"])
        self.assertIsNone(unknown["negative"])
        for file in (self.base, self.supplement, self.metadata_file):
            self.assertEqual(file.read_bytes(), self.original_bytes[file])
        history = self.directory / "history/pre-coverage-completion"
        self.assertEqual((history / "ratings.json").read_bytes(), self.original_bytes[self.ratings_file])
        self.assertEqual((history / "metadata.json").read_bytes(), self.original_bytes[self.metadata_file])
        enriched = json.loads((self.directory / "evidence/coverage_completion-o0.json").read_text(encoding="utf-8"))
        self.assertEqual(len(enriched["completion_facts"]), 1)
        self.assertEqual(len(enriched["facts"]), 3)
        searches = [call for call in self.prompts if call[0].startswith("coverage-completion-search-")]
        ratings = [call for call in self.prompts if call[0].startswith("coverage-completion-ratings-")]
        self.assertEqual(len(searches), 1)
        self.assertEqual(searches[0][2], {"search": True, "max_tokens": 8000})
        self.assertEqual(len(ratings), 2)
        self.assertEqual([len(json.loads(prompt.split("Шкалы: ", 1)[1].split("\nФакты: ", 1)[0])) for _, prompt, _ in ratings], [12, 1])
        self.assertFalse((self.directory / ".coverage_completion.lock").exists())
        self.assertEqual(self.store.get(self.identifier)["approvals"], self.study["approvals"])

    def test_ledger_prevents_repeated_search_when_unknown_cells_remain(self):
        self.run_apply()
        count = self.provider.request.call_count
        ratings_bytes = self.ratings_file.read_bytes()
        backup = (self.directory / "history/pre-coverage-completion/ratings.json").read_bytes()
        summary = self.run_apply()
        self.assertEqual(self.provider.request.call_count, count)
        self.assertEqual(summary["newly_observed"], 0)
        self.assertEqual(summary["remaining_target_unknown"], 1)
        self.assertEqual(self.ratings_file.read_bytes(), ratings_bytes)
        self.assertEqual((self.directory / "history/pre-coverage-completion/ratings.json").read_bytes(), backup)

    def test_normalized_duplicate_facts_do_not_trigger_rerating_or_new_observations(self):
        def duplicates_only(identifier, task, prompt, schema, **options):
            self.assertEqual(identifier, self.identifier)
            self.assertTrue(task.startswith("coverage-completion-search-"))
            return {"data": {"facts": [
                {"text": "  SUPPORTED   BASE FACT  ", "source_url": "https://fixture.example/base/", "period": "2024"},
                {"text": "Supported first supplement fact", "source_url": "https://fixture.example/supplement", "period": "2024"},
                {"text": "A fabricated new claim", "source_url": "https://invented.invalid/unretrieved", "period": "2024"}],
                "sources": [], "limitations": "No genuinely new grounded facts"},
                "citations": ["https://fixture.example/base", "https://fixture.example/supplement"],
                "model": "mock-search-model", "at": "test-time"}
        self.provider.request.side_effect = duplicates_only
        summary = self.run_apply()
        self.provider.request.assert_called_once()
        self.assertEqual(summary["newly_observed"], 0)
        self.assertEqual(summary["remaining_target_unknown"], 13)
        self.assertEqual(self.ratings_file.read_bytes(), self.original_bytes[self.ratings_file])
        ledger = json.loads((self.directory / "evidence/coverage_completion_ledger.json").read_text(encoding="utf-8"))
        record = ledger["objects"]["o0"]
        self.assertEqual(record["status"], "completed_no_new_evidence")
        self.assertEqual(record["updates"], [])
        self.assertEqual(record["accepted_fact_count"], 2)
        self.assertEqual(record["added_fact_count"], 0)
        evidence = json.loads((self.directory / "evidence/coverage_completion-o0.json").read_text(encoding="utf-8"))
        self.assertEqual(evidence["completion_facts"], [])
        self.assertEqual(len(evidence["facts"]), 2)

    def test_old_only_fact_references_cannot_convert_unknown_to_observed(self):
        def old_references_only(identifier, task, prompt, schema, **options):
            result = self.request(identifier, task, prompt, schema, **options)
            if task.startswith("coverage-completion-ratings-"):
                for row in result["data"]["ratings"]:
                    row.update(status="observed", positive=4., negative=1., evidence_indices=[0, 1],
                               reason="Synthetic attempted reassessment using only prior evidence")
            return result
        self.provider.request.side_effect = old_references_only
        summary = self.run_apply()
        self.assertEqual(summary["newly_observed"], 0)
        self.assertEqual(summary["remaining_target_unknown"], 13)
        rows = json.loads(self.ratings_file.read_text(encoding="utf-8"))
        targets = [row for row in rows if row["object_id"] == "o0" and row["scale_id"] in {f"s{i}" for i in range(1, 14)}]
        self.assertEqual(len(targets), 13)
        for row in targets:
            self.assertEqual(row["status"], "unknown")
            self.assertIsNone(row["positive"])
            self.assertIsNone(row["negative"])
            self.assertEqual(row["reason"], "Новое основание для изменения оценки не найдено")
        ledger = json.loads((self.directory / "evidence/coverage_completion_ledger.json").read_text(encoding="utf-8"))
        self.assertEqual(ledger["objects"]["o0"]["added_fact_count"], 1)
        self.assertEqual(ledger["objects"]["o0"]["observed_updates"], 0)
        calls_before_repeat = self.provider.request.call_count
        repeated = self.run_apply()
        self.assertEqual(self.provider.request.call_count, calls_before_repeat)
        self.assertEqual(repeated["newly_observed"], 0)
        self.assertEqual(repeated["remaining_target_unknown"], 13)

    def test_unapproved_running_or_over_budget_study_is_rejected_before_calls(self):
        original = copy.deepcopy(self.study)
        mutations = (
            {"status": "running", "progress": {"stage": "ratings"}},
            {"budget_limit_usd": 21},
            {"approvals": {"objects": {"approved": False}, "scales": {"approved": False}}},
        )
        for fields in mutations:
            self.store.save(dict(copy.deepcopy(original), **fields))
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                self.run_apply()
        self.provider.request.assert_not_called()
        self.assertEqual(self.ratings_file.read_bytes(), self.original_bytes[self.ratings_file])

    def test_running_coverage_completion_stage_can_resume(self):
        self.store.update(self.identifier, status="running", progress={"stage": "coverage_completion"})
        self.assertEqual(self.run_apply()["newly_observed"], 12)

    def test_concurrently_changed_ratings_are_not_overwritten(self):
        changed = copy.deepcopy(self.ratings)
        changed[-1]["reason"] = "Concurrent fixture edit must survive"
        self.search_hook = lambda: self.write(self.ratings_file, changed)
        with self.assertRaisesRegex(ValueError, "параллельно"):
            self.run_apply()
        self.assertEqual(json.loads(self.ratings_file.read_text(encoding="utf-8")), changed)
        self.assertTrue((self.directory / "evidence/coverage_completion_ledger.json").exists())
        self.assertFalse((self.directory / ".coverage_completion.lock").exists())

    def test_failed_atomic_commit_preserves_original_ratings_and_keeps_backup(self):
        from scripts.complete_pilot_coverage import atomic_json
        def fail_final(path, value):
            if Path(path) == self.ratings_file:
                raise OSError("Synthetic final commit failure")
            return atomic_json(path, value)
        with patch("scripts.complete_pilot_coverage.atomic_json", side_effect=fail_final), \
             self.assertRaisesRegex(OSError, "Synthetic final commit failure"):
            self.run_apply()
        self.assertEqual(self.ratings_file.read_bytes(), self.original_bytes[self.ratings_file])
        self.assertEqual((self.directory / "history/pre-coverage-completion/ratings.json").read_bytes(), self.original_bytes[self.ratings_file])
        self.assertFalse((self.directory / ".coverage_completion.lock").exists())


if __name__ == "__main__":
    unittest.main()
