"""Rerunning factor settings must preserve an explicitly bound audited matrix.

Every study is disposable; model, analysis and export boundaries are mocked.
"""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.semantic_service.server import Application
from src.semantic_service.store import fingerprint, ratings_basis


ROOT = Path(__file__).resolve().parents[1]


class RatingsCacheTests(unittest.TestCase):
    def setUp(self):
        temp_parent = ROOT / ".service_runtime"
        temp_parent.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="ratings-cache-test-", dir=temp_parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.provider = SimpleNamespace(configured=True, rate_all=Mock(),
            choose_clusters=Mock(return_value={"recommended_k": 3, "reason": "Synthetic profile review",
                                               "discrete_structure_supported": True}),
            interpret=Mock(side_effect=lambda _study, analysis: analysis))
        with patch("src.semantic_service.server.ModelProvider", return_value=self.provider):
            self.app = Application(root=self.root)
        study = self.app.store.create({"title": "Disposable audited matrix", "target_count": 12})
        self.identifier = study["id"]
        objects = [{"id": f"o{i}", "name": f"Synthetic object {i}"} for i in range(12)]
        scales = [{"id": f"s{i}", "name": f"Synthetic scale {i}", "positive": f"A{i}",
                   "negative": f"B{i}", "definition": "Synthetic observable contrast"} for i in range(6)]
        self.app.store.edit(self.identifier, "objects", objects)
        self.app.store.approve(self.identifier, "objects", actor="unit_test_fixture")
        self.app.store.edit(self.identifier, "scales", scales)
        self.app.store.approve(self.identifier, "scales", actor="unit_test_fixture")
        self.baseline = [{"object_id": obj["id"], "scale_id": scale["id"], "status": "unknown",
                          "positive": None, "negative": None, "reason": "No original evidence"}
                         for obj in objects for scale in scales]
        self.audited = copy.deepcopy(self.baseline)
        for index, row in enumerate(self.audited[:38]):
            row.update(status="observed", positive=4, negative=1, difference=3, mass=5,
                       reason=f"Audited supplementary evidence {index}", evidence_indices=[7])
        self.provider.rate_all.return_value = copy.deepcopy(self.baseline)
        self.ratings_path = self.app.store.directory(self.identifier) / "ratings.json"
        self.ratings_path.write_text(json.dumps(self.audited, ensure_ascii=False, indent=2), encoding="utf-8")
        self.receipt = {"basis": ratings_basis(self.app.store.get(self.identifier)),
                        "ratings_sha256": fingerprint(self.audited), "mode": "audited_fixture"}
        self.app.store.update(self.identifier, ratings_cache=self.receipt)

    def run_analysis(self, options=None, expect_error=False):
        def fit(_study, _ratings, output, factor_count=None, cluster_count=None):
            output.mkdir(parents=True, exist_ok=True)
            return {"factor_count": factor_count or 2, "cluster_count": cluster_count or 3,
                    "diagnostics": {"cluster_selection": {}}, "warnings": []}
        analyze = Mock(side_effect=fit)
        export = Mock(return_value=[])
        with patch.dict(sys.modules, {
            "src.semantic_service.analysis": SimpleNamespace(analyze_study=analyze),
            "src.semantic_service.reports": SimpleNamespace(export_report=export),
        }):
            self.app.run_analysis(self.identifier, options or {})
            self.app.jobs[self.identifier].join(timeout=3)
        self.assertFalse(self.app.jobs[self.identifier].is_alive())
        current = self.app.store.get(self.identifier)
        self.assertEqual(current["status"], "error" if expect_error else "complete", current.get("error"))
        return current, analyze, export

    def test_valid_cache_keeps_all_audited_rows_and_still_reviews_and_interprets(self):
        before = self.ratings_path.read_bytes()
        current, analyze, export = self.run_analysis({"factor_count": 4, "cluster_count": 3})
        self.provider.rate_all.assert_not_called()
        self.provider.choose_clusters.assert_called_once()
        self.provider.interpret.assert_called_once()
        analyze.assert_called_once()
        self.assertEqual(analyze.call_args.args[1], self.audited)
        self.assertEqual(analyze.call_args.kwargs, {"factor_count": 4, "cluster_count": 3})
        self.assertEqual(export.call_args.args[2], self.audited)
        self.assertEqual(self.ratings_path.read_bytes(), before)
        self.assertEqual(current["ratings_cache"], self.receipt)
        self.assertEqual(sum(row["status"] == "observed" for row in analyze.call_args.args[1]), 38)

    def test_changed_research_requirements_invalidate_measurement_cache(self):
        self.app.store.update(self.identifier, research_requirements={
            "required_scales": ["s0"], "instructions": ["Use only personally attributed dated actions"]})
        _, analyze, _ = self.run_analysis()
        self.provider.rate_all.assert_called_once()
        self.assertEqual(analyze.call_args.args[1], self.baseline)

    def test_preview_stage_never_runs_clusters_or_full_export(self):
        self.app.store.update(self.identifier, delivery_mode="factor_preview")
        result = {"models": [{"q": 2}], "russia_review": {"status": "insufficient_measurement"}}
        preview, full, export = Mock(return_value=result), Mock(), Mock()
        with patch.dict(sys.modules, {
            "src.semantic_service.factor_preview": SimpleNamespace(preview_factors=preview),
            "src.semantic_service.analysis": SimpleNamespace(analyze_study=full),
            "src.semantic_service.reports": SimpleNamespace(export_report=export),
        }):
            self.app.run_analysis(self.identifier, {"factor_count": 4})
            self.app.jobs[self.identifier].join(timeout=3)
        self.assertFalse(self.app.jobs[self.identifier].is_alive())
        current = self.app.store.get(self.identifier)
        self.assertEqual(current["status"], "await_factor_selection", current.get("error"))
        self.assertEqual(current["analysis"], result)
        self.assertEqual([a["format"] for a in current["artifacts"]], ["md", "json"])
        preview.assert_called_once()
        full.assert_not_called()
        export.assert_not_called()
        self.provider.choose_clusters.assert_not_called()
        self.provider.interpret.assert_not_called()
        self.provider.rate_all.assert_not_called()

    def test_stale_checksum_falls_back_and_binds_the_new_matrix(self):
        altered = copy.deepcopy(self.audited)
        altered[0]["reason"] = "Content changed after receipt"
        self.ratings_path.write_text(json.dumps(altered), encoding="utf-8")
        current, analyze, _ = self.run_analysis()
        self.provider.rate_all.assert_called_once()
        self.assertEqual(analyze.call_args.args[1], self.baseline)
        saved = json.loads(self.ratings_path.read_text(encoding="utf-8"))
        self.assertEqual(saved, self.baseline)
        self.assertEqual(current["ratings_cache"]["ratings_sha256"], fingerprint(saved))
        self.assertEqual(current["ratings_cache"]["basis"], ratings_basis(current))

    def test_changed_aspect_invalidates_basis_without_reusing_old_rows(self):
        self.app.store.update(self.identifier, aspect="Another public aspect")
        current, analyze, _ = self.run_analysis()
        self.provider.rate_all.assert_called_once()
        self.assertEqual(analyze.call_args.args[1], self.baseline)
        self.assertNotEqual(current["ratings_cache"]["basis"], self.receipt["basis"])

    def test_force_rerate_true_bypasses_a_valid_cache(self):
        current, analyze, _ = self.run_analysis({"force_rerate": True})
        self.provider.rate_all.assert_called_once()
        self.assertEqual(analyze.call_args.args[1], self.baseline)
        self.assertEqual(current["ratings_cache"]["ratings_sha256"], fingerprint(self.baseline))

    def test_malformed_or_missing_matrix_cannot_use_a_receipt(self):
        for content in ("not JSON", None):
            with self.subTest(content=content):
                if content is None:
                    self.ratings_path.unlink()
                else:
                    self.ratings_path.write_text(content, encoding="utf-8")
                self.app.store.update(self.identifier, ratings_cache=self.receipt)
                self.provider.rate_all.reset_mock()
                self.run_analysis()
                self.provider.rate_all.assert_called_once()

    def test_both_approvals_remain_required_even_with_a_matching_cache(self):
        for field in ("objects", "scales"):
            with self.subTest(field=field):
                study = self.app.store.get(self.identifier)
                original = copy.deepcopy(study["approvals"])
                study["approvals"][field] = {"approved": False}
                self.app.store.save(study)
                with self.assertRaisesRegex(ValueError, "утвердить"):
                    self.app.run_analysis(self.identifier, {})
                self.assertNotIn(self.identifier, self.app.jobs)
                self.app.store.update(self.identifier, approvals=original)
        self.provider.rate_all.assert_not_called()

    def test_imported_branch_remains_first_and_clears_previous_provider_receipt(self):
        self.app.import_ratings(self.identifier, self.audited)
        self.assertIsNone(self.app.store.get(self.identifier)["ratings_cache"])
        _, analyze, _ = self.run_analysis({"use_imported_ratings": True, "force_rerate": True})
        self.assertEqual(analyze.call_args.args[1], self.audited)
        self.provider.rate_all.assert_not_called()
        self.provider.choose_clusters.assert_not_called()
        self.provider.interpret.assert_not_called()

    def test_edit_clears_cache_but_noop_edit_preserves_it(self):
        study = self.app.store.get(self.identifier)
        self.assertEqual(self.app.store.edit(self.identifier, "scales", study["scales"])["ratings_cache"], self.receipt)
        changed = copy.deepcopy(study["scales"])
        changed[0]["definition"] += "; reviewed definition"
        self.assertNotIn("ratings_cache", self.app.store.edit(self.identifier, "scales", changed))

    def test_failed_rerating_cannot_leave_an_old_receipt_on_partial_results(self):
        def fail(_study, progress):
            self.ratings_path.write_text(json.dumps(self.baseline[:1]), encoding="utf-8")
            raise ValueError("Synthetic rating interruption")
        self.provider.rate_all.side_effect = fail
        current, analyze, export = self.run_analysis({"force_rerate": True}, expect_error=True)
        self.assertIsNone(current["ratings_cache"])
        analyze.assert_not_called()
        export.assert_not_called()

    def test_basis_binds_every_context_and_definition_field(self):
        original = self.app.store.get(self.identifier)
        for field in ("scope", "aspect", "date_from", "date_to", "objects", "scales"):
            with self.subTest(field=field):
                changed = copy.deepcopy(original)
                changed[field] = "Synthetic changed value"
                self.assertNotEqual(ratings_basis(changed), ratings_basis(original))


if __name__ == "__main__":
    unittest.main()
