"""Semantic review must respect numerical eligibility and the user's k.

Only disposable fixture studies and mocked model/export boundaries are used.
The running pilot, server process, credentials, and external APIs are untouched.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.semantic_service.providers import ModelProvider
from src.semantic_service.server import Application
from src.semantic_service.store import StudyStore


ROOT = Path(__file__).resolve().parents[1]
TEMP_PARENT = ROOT / ".service_runtime"


def candidate_fixture():
    candidates = []
    # k=3 has the best silhouette but fails stability; k=5 has a tiny group.
    # k=2 is exactly on both stability thresholds and must remain eligible.
    for k, sizes, silhouette, ari, jaccard in (
        (2, [10, 10], .25, .65, .60),
        (3, [6, 7, 7], .90, .64, .90),
        (4, [5, 5, 5, 5], .35, .80, .75),
        (5, [2, 3, 4, 5, 6], .95, .90, .90),
    ):
        groups, offset = [], 0
        for index, size in enumerate(sizes):
            members = [f"o{i}" for i in range(offset, offset + size)]
            offset += size
            groups.append({"id": f"C{index + 1}", "size": size, "member_ids": members,
                           "representatives": [{"id": members[0], "name": f"Synthetic {members[0]}", "distance": .1}],
                           "top_factors": [{"factor_id": "F1", "label": "Synthetic factor", "coordinate": float(index),
                                            "direction": "positive" if index else "neutral"}],
                           "distinguishing_scales": [{"id": "s0", "name": "Synthetic feature", "positive": "A",
                                                       "negative": "B", "difference": .75, "group_mean": 1.,
                                                       "overall_mean": .25, "direction": "positive"}]})
        candidates.append({"k": k, "silhouette": silhouette, "min_size": min(sizes), "max_size": max(sizes),
                           "stability": {"median_ari": ari, "min_cluster_median_jaccard": jaccard},
                           "semantic_profiles": groups})
    return {"n_objects": 20, "factor_count": 2, "cluster_count": 2,
            "factors": [{"id": "F1", "label": "Synthetic factor"}], "cluster_candidates": candidates,
            "diagnostics": {"cluster_selection": {"recommended_k": 2, "minimum_size_preference": 3}}}


class SemanticClusterTests(unittest.TestCase):
    def setUp(self):
        TEMP_PARENT.mkdir(exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix="semantic-review-test-", dir=TEMP_PARENT))
        self.addCleanup(self.remove_fixture)
        self.environment = patch.dict(os.environ, {"OPENAI_API_KEY": "test-key-never-sent", "PPLX_API_KEY": ""})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.dotenv = patch("src.semantic_service.providers.load_dotenv")
        self.dotenv.start()
        self.addCleanup(self.dotenv.stop)
        self.network = patch("src.semantic_service.providers.httpx.Client",
                             side_effect=AssertionError("External model calls are forbidden in semantic-review tests"))
        self.client = self.network.start()
        self.addCleanup(self.network.stop)
        self.addCleanup(self.client.assert_not_called)
        self.store = StudyStore(self.root / "studies")
        self.study = self.store.create({"title": "Synthetic cluster review fixture", "target_count": 20})
        self.provider = ModelProvider(self.root, self.store)

    def remove_fixture(self):
        target = self.root.resolve()
        if target.parent != TEMP_PARENT.resolve() or not target.name.startswith("semantic-review-test-"):
            raise AssertionError("Unsafe fixture cleanup destination")
        shutil.rmtree(target)

    @staticmethod
    def response(k=2, discrete=True):
        return {"data": {"recommended_k": k, "reason": "Synthetic substantive comparison of computed profiles",
                         "alternatives": [{"k": value, "interpretation": f"Synthetic description for k={value}"}
                                          for value in (2, 3, 4, 5)],
                         "discrete_structure_supported": discrete},
                "model": "mock-review-model", "at": "synthetic-test-time"}

    def test_review_only_allows_size_and_stability_eligible_k_and_keeps_all_profiles(self):
        analysis = candidate_fixture()
        with patch.object(self.provider, "request", return_value=self.response(4)) as request:
            choice = self.provider.choose_clusters(self.study, analysis)
        args = request.call_args.args
        material = json.loads(args[2].split("Данные:\n", 1)[1])
        schema = args[3]
        self.assertEqual(schema["properties"]["recommended_k"]["enum"], [2, 4])
        alternatives_schema = schema["properties"]["alternatives"]
        self.assertEqual(alternatives_schema["minItems"], 4)
        self.assertEqual(alternatives_schema["maxItems"], 4)
        self.assertEqual(alternatives_schema["items"]["properties"]["k"]["enum"], [2, 3, 4, 5])
        self.assertEqual(material["eligible_k"], [2, 4])
        # Ineligible partitions still supply the comparison evidence, rather than
        # being silently removed or replaced with the selected partition's groups.
        self.assertEqual(material["candidates"], analysis["cluster_candidates"])
        for candidate in material["candidates"]:
            self.assertEqual(len(candidate["semantic_profiles"]), candidate["k"])
            members = [identity for group in candidate["semantic_profiles"] for identity in group["member_ids"]]
            self.assertEqual(set(members), {f"o{i}" for i in range(20)})
            self.assertEqual(len(members), 20)
        self.assertEqual(choice["recommended_k"], 4)
        self.assertEqual({row["k"] for row in choice["alternatives"]}, {2, 3, 4, 5})
        self.assertTrue(choice["review_complete"])
        self.assertEqual(choice["quality_warnings"], [])
        self.assertIsNone(choice["quality_warning"])
        self.assertEqual(choice["provenance"]["model"], "mock-review-model")

    def test_partial_secondary_explanations_remain_explicitly_incomplete(self):
        response = self.response(4)
        response["data"]["alternatives"] = [
            {"k": 2, "interpretation": "Available substantive explanation"},
            {"k": 3, "interpretation": "   "},
        ]
        with patch.object(self.provider, "request", return_value=response):
            choice = self.provider.choose_clusters(self.study, candidate_fixture())
        self.assertEqual(choice["recommended_k"], 4)
        self.assertFalse(choice["review_complete"])
        self.assertTrue(choice["quality_warnings"])
        self.assertIn("неполно", choice["quality_warning"])
        # Preserve the actual evidence, and do not invent descriptions for the
        # missing candidates, including the selected k whose main reason exists.
        self.assertEqual([row["k"] for row in choice["alternatives"]], [2, 3, 4, 5])
        self.assertEqual(choice["alternatives"][0]["interpretation"], "Available substantive explanation")
        for row in choice["alternatives"][1:]:
            self.assertEqual(row["interpretation"], "Содержательное объяснение для данного k модель не вернула")
        self.assertIn(choice["quality_warning"], choice["reason"])

    def test_duplicate_and_unknown_explanations_are_not_presented_as_complete(self):
        response = self.response(2)
        response["data"]["alternatives"] += [
            {"k": 2, "interpretation": "Conflicting repeated description"},
            {"k": 999, "interpretation": "Unknown candidate must not enter the report"},
        ]
        with patch.object(self.provider, "request", return_value=response):
            choice = self.provider.choose_clusters(self.study, candidate_fixture())
        self.assertEqual([row["k"] for row in choice["alternatives"]], [2, 3, 4, 5])
        self.assertEqual(choice["alternatives"][0]["interpretation"], "Synthetic description for k=2")
        self.assertFalse(choice["review_complete"])
        self.assertTrue(any("повторяющиеся" in warning for warning in choice["quality_warnings"]))
        self.assertTrue(any("неизвестные" in warning for warning in choice["quality_warnings"]))

    def test_selected_k_requires_a_nonempty_primary_reason(self):
        for reason in ("", " \n ", None):
            response = self.response(2)
            response["data"]["reason"] = reason
            with self.subTest(reason=reason), \
                 patch.object(self.provider, "request", return_value=response), \
                 self.assertRaisesRegex(ValueError, "обоснование выбранного k"):
                self.provider.choose_clusters(self.study, candidate_fixture())

    def test_out_of_eligible_recommendation_is_rejected_even_if_it_is_a_candidate(self):
        # k=3 exists and has a larger silhouette, but is not a stable alternative.
        with patch.object(self.provider, "request", return_value=self.response(3)), \
             self.assertRaisesRegex(ValueError, "недопустимое k"):
            self.provider.choose_clusters(self.study, candidate_fixture())

    def test_fallback_first_preserves_size_eligibility_then_all_candidates(self):
        for all_tiny, expected in ((False, [2, 3, 4]), (True, [2, 3, 4, 5])):
            analysis = candidate_fixture()
            for candidate in analysis["cluster_candidates"]:
                candidate["stability"]["median_ari"] = .1
                if all_tiny:
                    candidate["min_size"] = 1
            with self.subTest(all_tiny=all_tiny), \
                 patch.object(self.provider, "request", return_value=self.response(2, discrete=False)) as request:
                choice = self.provider.choose_clusters(self.study, analysis)
            self.assertEqual(request.call_args.args[3]["properties"]["recommended_k"]["enum"], expected)
            self.assertFalse(choice["discrete_structure_supported"])

    def approved_application(self):
        app = Application(root=self.root, studies_dir=self.store.root)
        identifier = self.study["id"]
        objects = [{"id": f"o{i}", "name": f"Synthetic object {i}", "sources": []} for i in range(20)]
        scales = [{"id": f"s{i}", "name": f"Synthetic scale {i}", "positive": f"Synthetic A {i}",
                   "negative": f"Synthetic B {i}", "definition": "Synthetic rubric only"} for i in range(6)]
        app.store.edit(identifier, "objects", objects)
        app.store.approve(identifier, "objects", actor="unit_test_fixture")
        app.store.edit(identifier, "scales", scales)
        app.store.approve(identifier, "scales", actor="unit_test_fixture")
        ratings = [{"object_id": obj["id"], "scale_id": scale["id"],
                    "status": "observed", "positive": 3., "negative": 1.}
                   for obj in objects for scale in scales]
        return app, ratings

    def run_fixture(self, options, recommendation, imported=False, discrete=True):
        app, ratings = self.approved_application()
        identifier = self.study["id"]
        if imported:
            app.import_ratings(identifier, ratings)
        choice = self.response(recommendation, discrete)["data"]
        choice["provenance"] = {"model": "mock-review-model"}
        provider = SimpleNamespace(configured=True, rate_all=Mock(return_value=ratings),
                                   choose_clusters=Mock(return_value=choice),
                                   interpret=Mock(side_effect=lambda _study, analysis: analysis))
        app.provider = provider

        def fit(_study, supplied_ratings, output, factor_count=None, cluster_count=None):
            self.assertEqual(supplied_ratings, ratings)
            output.mkdir(parents=True, exist_ok=True)
            result = candidate_fixture()
            result["factor_count"] = factor_count or 2
            result["cluster_count"] = cluster_count or 2
            result["warnings"] = []
            return result

        analyze = Mock(side_effect=fit)
        export = Mock(return_value=[{"format": "md", "label": "Synthetic export fixture",
                                     "url": f"/api/studies/{identifier}/files/report.md"}])
        # Lazy imports in the worker are replaced only in this test process.
        # The separate real server and running pilot are not imported or touched.
        with patch.dict(sys.modules, {
            "src.semantic_service.analysis": SimpleNamespace(analyze_study=analyze),
            "src.semantic_service.reports": SimpleNamespace(export_report=export),
        }):
            app.run_analysis(identifier, options)
            app.jobs[identifier].join(timeout=3)
        self.assertFalse(app.jobs[identifier].is_alive())
        study = app.store.get(identifier)
        self.assertEqual(study["status"], "complete", study.get("error"))
        self.assertIsNone(study["error"])
        return app, study, analyze, export, provider

    def test_automatic_semantic_choice_refits_once_without_rerating_and_preserves_q(self):
        app, study, analyze, export, provider = self.run_fixture({"factor_count": 4}, recommendation=3, discrete=False)
        self.assertEqual(analyze.call_count, 2)
        self.assertEqual(analyze.call_args_list[0].kwargs, {"factor_count": 4, "cluster_count": None})
        self.assertEqual(analyze.call_args_list[1].kwargs, {"factor_count": 4, "cluster_count": 3})
        provider.rate_all.assert_called_once()
        provider.choose_clusters.assert_called_once()
        provider.interpret.assert_called_once()
        self.assertEqual(provider.interpret.call_args.args[1]["cluster_count"], 3)
        self.assertEqual(export.call_args.args[1]["cluster_count"], 3)
        self.assertEqual(study["analysis"]["cluster_count"], 3)
        selection = study["analysis"]["diagnostics"]["cluster_selection"]
        self.assertEqual(selection["selection_authority"], "semantic_and_numerical_review")
        self.assertEqual(selection["content_recommended_k"], 3)
        self.assertTrue(any("не подтверждает естественные" in warning for warning in study["analysis"]["warnings"]))
        metadata = json.loads((app.store.directory(study["id"]) / "output/metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["cluster_count"], 3)
        self.assertFalse(metadata["cluster_content_review"]["discrete_structure_supported"])

    def test_user_cluster_override_is_preserved_despite_different_semantic_recommendation(self):
        _, study, analyze, export, provider = self.run_fixture({"factor_count": 4, "cluster_count": 4}, recommendation=3)
        analyze.assert_called_once()
        self.assertEqual(analyze.call_args.kwargs, {"factor_count": 4, "cluster_count": 4})
        self.assertEqual(study["analysis"]["cluster_count"], 4)
        selection = study["analysis"]["diagnostics"]["cluster_selection"]
        self.assertEqual(selection["selection_authority"], "user_override")
        self.assertEqual(selection["content_recommended_k"], 3)
        self.assertIn("k=4", study["analysis"]["cluster_selection_reason"])
        self.assertEqual(provider.interpret.call_args.args[1]["cluster_count"], 4)
        self.assertEqual(export.call_args.args[1]["cluster_count"], 4)

    def test_identical_semantic_and_numerical_k_does_not_trigger_an_extra_fit(self):
        _, study, analyze, _, provider = self.run_fixture({}, recommendation=2)
        analyze.assert_called_once()
        provider.choose_clusters.assert_called_once()
        self.assertEqual(study["analysis"]["cluster_count"], 2)

    def test_imported_offline_measurements_do_not_trigger_new_semantic_model_calls(self):
        _, study, analyze, export, provider = self.run_fixture({"use_imported_ratings": True}, recommendation=3, imported=True)
        analyze.assert_called_once()
        export.assert_called_once()
        provider.rate_all.assert_not_called()
        provider.choose_clusters.assert_not_called()
        provider.interpret.assert_not_called()
        self.assertEqual(study["analysis"]["cluster_count"], 2)
        self.assertNotIn("cluster_content_review", study["analysis"])


if __name__ == "__main__":
    unittest.main()
