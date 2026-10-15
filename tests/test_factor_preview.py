"""Narrow offline checks of thematic screening and q-comparison visibility."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.semantic_service.factor_preview import (
    _factor_meaning_rows, _russia_factor_review, _theme_configuration, preview_factors,
)
from src.semantic_service.analysis import InsufficientMeasurementError, _factor_descriptions


def fixture(n=16, p=8):
    """Explicit synthetic test measurements, unrelated to the Rwanda study."""
    rng = np.random.default_rng(83)
    objects = [{"id": f"test-o{i}", "name": f"Тестовый объект {i}"} for i in range(n)]
    scales = [{"id": f"test-s{j}", "name": f"Тестовая шкала {j}", "positive": f"Первый полюс {j}",
               "negative": f"Второй полюс {j}", "definition": "Синтетическое измерение для теста"} for j in range(p)]
    latent = rng.normal(size=(n, 2))
    matrix = np.clip(latent @ rng.normal(size=(2, p)) + rng.normal(scale=.5, size=(n, p)), -5, 5)
    ratings = [{"object_id": objects[i]["id"], "scale_id": scales[j]["id"],
                "value": float(matrix[i, j]), "status": "observed", "reason": "Synthetic test observation", "sources": []}
               for i in range(n) for j in range(p)]
    return {"objects": objects, "scales": scales}, ratings


def configuration(scales, count=3):
    return _theme_configuration({"scales": scales, "research_requirements": {"required_scales": scales[:count]}})


class FactorPreviewTests(unittest.TestCase):
    def test_insufficient_measurements_export_counts_without_fitting_or_fake_variance(self):
        for failure in ("minimum_retained_dimensions", "empty_matrix", "minimum_retained_coverage"):
            with self.subTest(failure=failure):
                n, p = (16, 8) if failure == "minimum_retained_coverage" else (42, 24)
                study, ratings = fixture(n=n, p=p)
                study["research_requirements"] = {"required_scales": study["scales"][:3]}
                for row in ratings:
                    i = int(row["object_id"].split("o")[1])
                    j = int(row["scale_id"].removeprefix("test-s"))
                    known = (j < 5 or (j == 5 and i < 10)) if p == 24 else (i + j) % 8 >= 3
                    if failure == "empty_matrix":
                        known = False
                    if not known:
                        row["status"] = "unknown"
                expected_count = sum(row["status"] == "observed" for row in ratings)
                before = deepcopy(study), deepcopy(ratings)
                with tempfile.TemporaryDirectory() as directory, \
                     patch("src.semantic_service.factor_preview._dimension_analysis") as fit, \
                     patch("src.semantic_service.factor_preview._draw_loadings") as draw:
                    result = preview_factors(study, ratings, Path(directory))
                    saved = json.loads((Path(directory) / "factor_preview.json").read_text(encoding="utf-8"))
                    markdown = (Path(directory) / "factor_preview.md").read_text(encoding="utf-8")
                    self.assertEqual(sorted(path.name for path in Path(directory).iterdir()),
                                     ["factor_preview.json", "factor_preview.md"])
                self.assertEqual(result, saved)
                self.assertEqual(result["measurement_status"], "insufficient_data")
                self.assertEqual(result["models"], [])
                self.assertEqual(result["dimension_candidates"], [])
                self.assertIsNone(result["recommended_q"])
                self.assertIsNone(result["dimension_selection"]["recommended_q"])
                self.assertEqual(result["russia_review"]["status"], "insufficient_measurement")
                self.assertNotIn("variance_total", result)
                self.assertNotIn('"variance_share"', json.dumps(result))
                self.assertNotIn("clusters", result)
                diagnostics = result["diagnostics"]
                self.assertEqual(diagnostics["failure_code"], "minimum_retained_dimensions" if failure == "empty_matrix" else failure)
                if failure == "empty_matrix":
                    self.assertIsNone(diagnostics["retained_coverage"])
                    self.assertEqual((result["n_objects"], result["n_scales"]), (0, 0))
                self.assertFalse(diagnostics["factor_analysis_started"])
                self.assertEqual(diagnostics["observed_input_ratings"], expected_count)
                self.assertEqual(diagnostics["unknown_input_cells"], n * p - expected_count)
                self.assertEqual(len(diagnostics["input_object_coverage"]), n)
                self.assertEqual(len(diagnostics["input_scale_coverage"]), p)
                self.assertEqual(sum(item["observed_scales"] for item in diagnostics["input_object_coverage"]), expected_count)
                self.assertEqual(sum(item["observed_objects"] for item in diagnostics["input_scale_coverage"]), expected_count)
                self.assertIn(study["objects"][-1]["name"], markdown)
                self.assertIn(study["scales"][-1]["name"], markdown)
                self.assertIn("FA не запускался", markdown)
                self.assertIn("не равны нулю", markdown)
                fit.assert_not_called()
                draw.assert_not_called()
                self.assertEqual((study, ratings), before)

    def test_invalid_duplicate_rating_is_not_swallowed_as_insufficient_measurement(self):
        study, ratings = fixture()
        for row in ratings:
            row["status"] = "unknown"
        ratings.append(deepcopy(ratings[0]))
        with tempfile.TemporaryDirectory() as directory, \
             patch("src.semantic_service.factor_preview._dimension_analysis") as fit:
            output = Path(directory) / "should-not-exist"
            with self.assertRaisesRegex(ValueError, "Повторная оценка") as caught:
                preview_factors(study, ratings, output)
            self.assertNotIsInstance(caught.exception, InsufficientMeasurementError)
            self.assertFalse(output.exists())
            fit.assert_not_called()

    def test_mixed_signed_three_family_candidate_is_numeric_not_validated(self):
        study, _ = fixture(p=6)
        vector = np.array([[.65], [-.60], [.50], [.80], [.1], [0.]])
        row = _russia_factor_review(vector, study["scales"], 6., configuration(study["scales"]))[0]
        self.assertEqual(row["status"], "numeric_candidate")
        self.assertEqual(row["evidence_audit"], "pending")
        self.assertEqual(row["polarity"], "mixed_signs")
        self.assertEqual(row["distinct_family_count"], 3)
        self.assertEqual(row["strong_russia_indicators"][1]["loading"], -.6)
        self.assertEqual(row["strong_russia_indicators"][1]["high_factor_property"], "Второй полюс 1")
        self.assertEqual(row["strong_non_russia_indicators"][0]["id"], "test-s3")
        self.assertEqual(row["scope_hint"], "mixed_with_other_strong_scales")
        self.assertTrue(row["mandatory_report"])

    def test_global_variance_gate_and_two_indicator_provisional_status(self):
        study, _ = fixture(p=20)
        vector = np.zeros((20, 1)); vector[:3, 0] = .35
        row = _russia_factor_review(vector, study["scales"], 20., configuration(study["scales"]))[0]
        self.assertAlmostEqual(row["variance_share"], 3 * .35 ** 2 / 20)
        self.assertLess(row["variance_share"], .02)
        self.assertEqual(row["status"], "below_reporting_threshold")
        self.assertFalse(row["mandatory_report"])
        vector[:3, 0] = [.7, -.6, .1]
        row = _russia_factor_review(vector, study["scales"], 20., configuration(study["scales"]))[0]
        self.assertEqual(row["status"], "two_indicator_candidate")
        self.assertTrue(row["mandatory_report"])

    def test_synonyms_in_one_declared_family_do_not_create_three_indicators(self):
        study, _ = fixture(p=6)
        for scale in study["scales"][:3]:
            scale["metadata"] = {"research_domain": "russia_relations", "semantic_family_id": "one-family"}
        config = _theme_configuration(study)
        self.assertEqual(len(config["configured_ids"]), 3)
        row = _russia_factor_review(np.array([[.7], [.6], [-.5], [0.], [0.], [0.]]), study["scales"], 6., config)[0]
        self.assertEqual(row["distinct_family_count"], 1)
        self.assertEqual(row["status"], "single_indicator_only")
        self.assertFalse(row["mandatory_report"])

    def test_required_scales_accept_strings_objects_and_metadata_markers(self):
        study, _ = fixture(p=6)
        study["scales"][2]["research_domain"] = "russia_relations"
        study["research_requirements"] = {"required_scales": ["test-s0", {**study["scales"][1], "semantic_family_id": "security"}],
                                           "instructions": ["Do not turn unknown into neutral"]}
        config = _theme_configuration(study)
        self.assertEqual(config["configured_ids"], ["test-s0", "test-s1", "test-s2"])
        self.assertEqual(config["family_by_id"]["test-s1"], "security")
        self.assertEqual(config["instructions"], study["research_requirements"]["instructions"])
        study["research_requirements"]["minimum_indicators"] = 2
        with self.assertRaisesRegex(ValueError, "fixes minimum_indicators"):
            _theme_configuration(study)

    def test_compact_semantic_table_uses_approved_poles_for_signed_loadings(self):
        study, _ = fixture(p=6)
        study["scales"][0].update(positive="Твёрдый", negative="Мягкий")
        study["scales"][1].update(positive="Доверчивый", negative="Недоверчивый")
        loadings = np.zeros((6, 2)); loadings[:2, 0] = [.8, -.7]
        factors = _factor_descriptions(loadings, study["scales"], 6.)
        table = [{"id": scale["id"], "name": scale["name"],
                  "loadings": {"F1": float(loadings[i, 0]), "F2": float(loadings[i, 1])}}
                 for i, scale in enumerate(study["scales"])]
        rows = _factor_meaning_rows({"factors": factors, "loading_table": table})
        self.assertEqual(rows[0][1], "Твёрдый; Недоверчивый")
        self.assertEqual(rows[0][2], "Мягкий; Доверчивый")
        self.assertEqual(rows[1][0], "F2 *")
        self.assertIn("Нет сильных нагрузок", rows[1][1])

    def test_higher_q_candidate_cannot_be_hidden_by_parsimony_q2(self):
        study, ratings = fixture()
        study["research_requirements"] = {"required_scales": study["scales"][:3]}
        raw = np.array([r["value"] for r in ratings]).reshape(16, 8)
        matrices = {2: np.zeros((8, 2)), 3: np.zeros((8, 3))}
        matrices[2][3:, :2] = [.6, .4]
        matrices[3][3:, :2] = [.6, .4]
        matrices[3][:3, 2] = [.6, -.5, .4]
        models = {q: {"training_imputation": raw.mean(axis=0), "training_mean": raw.mean(axis=0),
                      "training_scale": raw.std(axis=0), "loadings": values, "rotation": np.eye(q)}
                  for q, values in matrices.items()}
        dimensions = [{"q": q, "heldout_log_likelihood_mean": -10., "heldout_log_likelihood_sd": .2,
                       "heldout_log_likelihood_splits": [-10.1, -10., -9.9],
                       "residual_rms_off_diagonal": .1, "convergence_warnings": 0}
                      for q in matrices]
        before = deepcopy(study), deepcopy(ratings)
        with tempfile.TemporaryDirectory() as directory, patch("src.semantic_service.factor_preview._dimension_analysis",
                    return_value=(2, dimensions, models, {"recommended_q": 2})):
            result = preview_factors(study, ratings, Path(directory), make_charts=False)
            markdown = (Path(directory) / "factor_preview.md").read_text(encoding="utf-8")
            saved = json.loads((Path(directory) / "factor_preview.json").read_text(encoding="utf-8"))
        self.assertIn({"q": 3, "factor_id": "F3", "status": "numeric_candidate"}, result["russia_review"]["mandatory_report_refs"])
        self.assertIn("q3/F3", markdown)
        self.assertIn("## Смыслы всех факторов по решениям", markdown)
        self.assertIn("### q=2", markdown)
        # Ordinary non-Russia F1 is interpretable without deciphering JSON.
        self.assertIn("Первый полюс 3", markdown)
        self.assertIn("Второй полюс 3", markdown)
        self.assertIn("Тестовая шкала 3 +0.60", markdown)
        self.assertIn("Номера и знаки факторов локальны", markdown)
        self.assertEqual(result, saved)
        self.assertEqual(result["dimension_selection"]["recommended_q"], 2)
        self.assertNotIn("clusters", result)
        self.assertEqual((study, ratings), before)

    def test_real_offline_fitting_exports_all_admissible_q_and_source_unknowns(self):
        study, ratings = fixture(p=8)
        study["scales"][0]["research_domain"] = "russia_relations"
        for row in ratings:
            if row["scale_id"] == "test-s0" and int(row["object_id"].split("o")[1]) >= 8:
                row["status"] = "unknown"
        with tempfile.TemporaryDirectory() as directory, patch("src.semantic_service.analysis.PARALLEL_PERMUTATIONS", 10):
            result = preview_factors(study, ratings, Path(directory), make_charts=True)
            self.assertTrue((Path(directory) / "factor_preview.json").is_file())
            self.assertTrue(all((Path(directory) / chart["file"]).is_file() for chart in result["charts"]))
        self.assertEqual([m["q"] for m in result["models"]], [2, 3])
        self.assertEqual(result["n_scales"], 7)
        self.assertEqual(result["russia_review"]["excluded_ids"], ["test-s0"])
        self.assertEqual(result["russia_review"]["status"], "insufficient_measurement")
        self.assertEqual(result["russia_review"]["scale_coverage"][0]["observed_count"], 8)
        self.assertEqual(result["russia_review"]["scale_coverage"][0]["observed_input_count"], 8)
        self.assertFalse(result["russia_review"]["scale_coverage"][0]["retained"])
        self.assertEqual(len(result["charts"]), 2)


if __name__ == "__main__":
    unittest.main()
