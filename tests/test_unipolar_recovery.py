"""Offline tests: historical candidates never become accepted pairs implicitly."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.recover_rwanda_independent_poles import audit_fresh, inventory, recover, safe, STUDY_ID
from src.semantic_service.store import fingerprint


class RecoveryTests(unittest.TestCase):
    def example(self, value=2):
        data = {"study_id": STUDY_ID, "basis": {"scales": "fixed"}, "input_sha256": {"source": "frozen"},
            "objects": [{"id": "actor"}], "scales": [{"id": "scale", "positive": "A", "negative": "B"}],
            "evidence_snapshots": {"evidence/a.json": {"facts": [{"text": "Specific action", "source_url": "https://official.example/action"}]}},
            "candidates": [{"candidate_id": "c", "object_id": "actor", "scale_id": "scale", "pole": "positive",
                "value": value, "evidence_indices": [0], "evidence_file": "evidence/a.json", "validation_errors": [],
                "rationale": "Original", "provenance": {"file": "raw.json"}, "source_basis": "b", "facts_sha256": "f"}]}
        reviews = {"status": "completed_per_pole_semantic_review", "inventory_basis": fingerprint({"basis": data["basis"], "input_sha256": data["input_sha256"]}),
            "decisions": [{"object_id": "actor", "scale_id": "scale", "pole": "positive", "decision": "retain", "candidate_id": "c", "reason": "Directly supported"}]}
        return data, reviews

    def test_known_pole_survives_unknown_opposite_without_new_number(self):
        data, reviews = self.example()
        result = recover(data, reviews)
        scores = result["ratings"][0]["pole_scores"]
        self.assertEqual(scores["positive"]["value"], 2)
        self.assertEqual(scores["positive"]["status"], "observed")
        self.assertEqual(scores["negative"]["status"], "unknown")
        self.assertIsNone(scores["negative"]["value"])
        self.assertNotIn("difference", result["ratings"][0])

    def test_unfinished_review_and_changed_input_fail_closed(self):
        data, reviews = self.example()
        pending = dict(reviews, status="pending")
        with self.assertRaisesRegex(ValueError, "completed"):
            recover(data, pending)
        data["input_sha256"]["source"] = "changed"
        with self.assertRaisesRegex(ValueError, "exact inventory"):
            recover(data, reviews)

    def test_zero_needs_explicit_semantic_authorization(self):
        data, reviews = self.example(value=0)
        with self.assertRaisesRegex(ValueError, "explicit"):
            recover(data, reviews)

    def test_out_of_range_or_repaired_foreign_index_cannot_be_accepted(self):
        data, reviews = self.example()
        reviews["decisions"][0]["evidence_indices"] = [3]
        with self.assertRaisesRegex(ValueError, "subset"):
            recover(data, reviews)
        reviews["decisions"][0].pop("evidence_indices")
        data["candidates"][0]["validation_errors"] = ["initial_dossier_basis_mismatch"]
        with self.assertRaisesRegex(ValueError, "invalid"):
            recover(data, reviews)

    def test_fresh_packet_flags_repeated_same_event_and_preserves_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root / "studies" / STUDY_ID
            evidence = directory / "evidence"
            evidence.mkdir(parents=True)
            study = {"objects": [{"id": "actor", "name": "Actor"}], "scales": [{"id": "scale", "positive": "A", "negative": "B", "definition": "Exact definition"}],
                     "date_from": "2021-10-15", "date_to": "2026-10-15"}
            dossier = {"object_id": "actor", "basis": "fixed", "facts": [
                {"text": "First retelling", "source_url": "https://official.example/1", "period": "2024-03-01", "event_id": "one-event"},
                {"text": "Second retelling", "source_url": "https://official.example/2", "period": "2024-03-01", "event_id": "one-event"}]}
            row = {"object_id": "actor", "scale_id": "scale", "evidence_file": "evidence/independent-dossier-actor.json", "evidence_basis": "fixed",
                   "pole_scores": {"positive": {"status": "observed", "value": 3, "support_type": "positive_manifestation", "evidence_indices": [0, 1],
                        "source_urls": ["https://official.example/1", "https://official.example/2"], "rationale": "Action"},
                        "negative": {"status": "unknown", "value": None}}}
            paths = {directory / "study.json": study, evidence / "independent-dossier-actor.json": dossier, evidence / "independent-ratings-actor.json": [row]}
            for path, value in paths.items():
                path.write_text(json.dumps(value), encoding="utf-8")
            before = {path: path.read_bytes() for path in paths}
            result = audit_fresh(root, 0, 1)
            self.assertEqual(result["counts"]["observed"], 1)
            self.assertEqual(result["counts"]["unknown"], 1)
            self.assertIn("value_above_2_requires_distinct_recurring_events", result["cells"][0]["review_warnings"])
            self.assertEqual(result["cells"][0]["validation_errors"], [])
            self.assertEqual(before, {path: path.read_bytes() for path in paths})
            row["pole_scores"]["negative"]["value"] = 0
            (evidence / "independent-ratings-actor.json").write_text(json.dumps([row]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "null"):
                audit_fresh(root, 0, 1)

    def test_paths_cannot_escape_study(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "within"):
                safe(Path(temp), "../outside.json")

    def test_inventory_uses_frozen_pairs_after_live_matrix_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root / "studies" / STUDY_ID
            frozen = directory / "attempts/paired_difference_2026-10-15"
            frozen.mkdir(parents=True)
            (directory / "evidence").mkdir()
            (root / "research/rwanda_politicians_2021_2026").mkdir(parents=True)
            study = {"objects": [{"id": "actor"}], "scales": [{"id": "scale", "positive": "A", "negative": "B"}],
                     "date_from": "2021-10-15", "date_to": "2026-10-15"}
            dossier = {"object_id": "actor", "basis": "d", "facts": [{"text": "Action", "source_url": "https://official.example/action", "period": "2024-01-01"}]}
            ratings = [{"object_id": "actor", "scale_id": "scale", "positive": 2, "negative": 0, "status": "observed",
                        "evidence_indices": [0], "evidence_file": "evidence/actor.json", "evidence_basis": "d"}]
            for path, value in {frozen / "study.json": study, frozen / "ratings.json": ratings,
                                directory / "study.json": dict(study, measurement_model="independent_poles"),
                                directory / "ratings.json": [], directory / "evidence/model_response_index.json": {"responses": []},
                                directory / "evidence/actor.json": dossier}.items():
                path.write_text(json.dumps(value), encoding="utf-8")
            result = inventory(root)
            self.assertEqual(result["counts"]["candidate_proposals"], 2)
            self.assertEqual(result["counts"]["distinct_nonzero_valid_poles"], 1)
            self.assertIn("attempts/paired_difference_2026-10-15/ratings.json", result["candidates"][0]["provenance"]["file"])
            self.assertNotIn("studies/" + STUDY_ID + "/ratings.json", result["input_sha256"])


if __name__ == "__main__":
    unittest.main()
