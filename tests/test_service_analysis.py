"""Offline MVP analysis checks use measured synthetic fixtures, never paid calls."""
from copy import deepcopy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from scripts.latent_space_benchmark import fit_model, transform_new_objects
from src.semantic_service.analysis import (
    _dimension_analysis, _factor_descriptions, _prepare, _tsne_initialization, analyze_study,
)


def fixture(n=16, p=8):
    rng = np.random.default_rng(20261015)
    objects = [{"id": f"o{i}", "name": f"Объект {i}"} for i in range(n)]
    scales = [{"id": f"s{j}", "name": f"Шкала {j}", "positive": f"Полюс +{j}",
               "negative": f"Полюс -{j}", "definition": "Измеренная тестовая шкала"} for j in range(p)]
    latent = rng.normal(size=(n, 2)) + np.repeat([-1., 1.], n // 2)[:, None]
    weights = rng.normal(size=(2, p))
    matrix = np.clip(latent @ weights + rng.normal(scale=.35, size=(n, p)), -5, 5)
    ratings = [{"object_id": objects[i]["id"], "scale_id": scales[j]["id"],
                "positive": float(max(matrix[i, j], 0)), "negative": float(max(-matrix[i, j], 0)),
                "status": "observed", "reason": "Зафиксированная синтетическая тестовая оценка", "sources": []}
               for i in range(n) for j in range(p)]
    return {"objects": objects, "scales": scales, "focus_object_ids": ["o0"]}, ratings


class ServiceAnalysisTests(unittest.TestCase):
    def test_independent_poles_keep_known_property_despite_unknown_root_and_opposite(self):
        study, ratings = fixture(p=6)
        study["measurement_model"] = "independent_poles"
        for row in ratings:
            row["pole_scores"] = {pole: {"status": "observed", "value": row[pole]}
                                  for pole in ("positive", "negative")}
            row["status"] = "partial"
        ratings[0]["status"] = "unknown"
        ratings[0]["pole_scores"] = {"positive": {"status": "observed", "value": 4},
                                     "negative": {"status": "unknown", "value": None}}
        before = deepcopy(study), deepcopy(ratings)
        matrix, mass, _, scales, diagnostics = _prepare(study, ratings)
        self.assertEqual(matrix.shape, (16, 12))
        self.assertEqual(matrix.loc["o0", "s0__positive"], 4)
        self.assertTrue(np.isnan(matrix.loc["o0", "s0__negative"]))
        self.assertEqual(scales["s0__negative"]["name"], "Полюс -0")
        self.assertEqual(scales["s0__negative"]["parent_scale_id"], "s0")
        self.assertEqual(scales["s0__negative"]["negative"], "")
        self.assertTrue(mass.isna().all().all())
        self.assertEqual(diagnostics["observed_input_ratings"], 16 * 12 - 1)
        self.assertEqual(diagnostics["input_scale_definitions"], 6)
        self.assertEqual(diagnostics["measurement_model"], "independent_poles")
        self.assertIsNone(diagnostics["zero_differences"])
        self.assertEqual((study, ratings), before)

    def test_unipolar_values_are_direct_and_low_expression_is_not_an_antonym(self):
        study, ratings = fixture(p=6)
        for scale in study["scales"]:
            scale["kind"] = "unipolar"
            scale.pop("positive"); scale.pop("negative")
        for row in ratings:
            row["value"] = 2.5 + .4 * (row.pop("positive") - row.pop("negative"))
        matrix, _, _, scale_map, diagnostics = _prepare(study, ratings)
        self.assertEqual(matrix.columns.tolist(), [f"s{i}" for i in range(6)])
        self.assertEqual(diagnostics["measurement_model"], "independent_poles")
        loadings = np.zeros((6, 2)); loadings[:2, 0] = [.8, -.7]
        factor = _factor_descriptions(loadings, list(scale_map.values()), 6.)[0]
        negative = factor["negative_traits"][0]
        self.assertEqual(negative["high_factor_property"], "Меньшая выраженность: Шкала 1")
        self.assertEqual(negative["low_factor_property"], "Большая выраженность: Шкала 1")
        self.assertIn("не означает выраженность его антонима", factor["interpretation"])
        self.assertNotIn("↔", factor["label"])
        self.assertAlmostEqual(factor["variance_share"], (.8 ** 2 + .7 ** 2) / 6.)
        ratings[0]["value"] = -1
        with self.assertRaisesRegex(ValueError, "от 0 до 5"):
            _prepare(study, ratings)

    def test_paired_alias_and_unipolar_name_match_provider_contract(self):
        study, ratings = fixture(p=6)
        for scale in study["scales"]:
            scale["kind"] = "paired"
        matrix, _, _, _, _ = _prepare(study, ratings)
        self.assertEqual(matrix.shape, (16, 6))
        study["measurement_model"] = "independent_poles"
        for scale in study["scales"]:
            scale["kind"] = "unipolar"
        for row in ratings:
            row["value"] = 2.5 + .4 * (row.pop("positive") - row.pop("negative"))
        _, _, _, scale_map, _ = _prepare(study, ratings)
        # A UI conversion can leave old pole labels; canonical meaning is name.
        self.assertEqual(scale_map["s0"]["positive"], "Шкала 0")
        self.assertEqual(scale_map["s0"]["negative"], "")

    def test_independent_pole_observations_must_be_finite_in_range(self):
        study, ratings = fixture(p=6)
        study["measurement_model"] = "independent_poles"
        for row in ratings:
            row["pole_scores"] = {pole: {"status": "observed", "value": row[pole]}
                                  for pole in ("positive", "negative")}
        for invalid in (6, -1, float("nan"), float("inf"), True):
            with self.subTest(invalid=invalid):
                ratings[0]["pole_scores"]["positive"]["value"] = invalid
                with self.assertRaises(ValueError):
                    _prepare(study, ratings)

    def test_unknown_is_missing_and_mass_preserves_ambivalence(self):
        study, ratings = fixture()
        ratings[0].update(positive=4., negative=4.)
        ratings[1].update(status="unknown", positive=0., negative=0.)
        matrix, mass, _, _, diag = _prepare(study, ratings)
        self.assertEqual(matrix.iloc[0, 0], 0.)
        self.assertEqual(mass.iloc[0, 0], 8.)
        self.assertTrue(np.isnan(matrix.iloc[0, 1]))
        self.assertEqual(diag["zero_difference_with_mass"], 1)

    def test_fully_unknown_object_does_not_receive_coordinates(self):
        study, ratings = fixture()
        for rating in ratings:
            if rating["object_id"] == "o0":
                rating["status"] = "unknown"
        matrix, _, _, _, diag = _prepare(study, ratings)
        self.assertNotIn("o0", matrix.index)
        self.assertEqual(diag["excluded_objects"][0]["reason"], "no_observations")

    def test_sparse_excluded_scales_do_not_remove_objects_measured_on_the_core(self):
        study, ratings = fixture(n=16, p=16)
        for rating in ratings:
            if int(rating["scale_id"][1:]) >= 8 and int(rating["object_id"][1:]) >= 4:
                rating["status"] = "unknown"
        matrix, _, _, _, diag = _prepare(study, ratings)
        # These 12 objects have only 8/16 original scales, but all 8 valid core
        # scales. A row-first cutoff would discard them before removing noise.
        self.assertEqual(matrix.index.tolist(), [f"o{i}" for i in range(16)])
        self.assertEqual(matrix.columns.tolist(), [f"s{j}" for j in range(8)])
        self.assertEqual(int(matrix.loc["o10"].notna().sum()), 8)
        self.assertEqual(diag["excluded_objects"], [])
        self.assertEqual({item["id"] for item in diag["excluded_scales"]}, {f"s{j}" for j in range(8, 16)})
        self.assertEqual(diag["preprocessing_order"], "scales_then_objects_until_stable")
        self.assertEqual(diag["preprocessing_iterations"][0]["required_objects_per_scale"], 10)
        self.assertEqual(diag["preprocessing_iterations"][0]["required_scales_per_object"], 5)
        self.assertEqual(diag["preprocessing_iterations"][-1]["excluded_scale_ids"], [])
        self.assertEqual(diag["retained_coverage"], 1.)

    def test_insufficient_data_and_ambiguous_import_are_rejected(self):
        study, ratings = fixture()
        with self.assertRaisesRegex(ValueError, "Недостаточно"):
            _prepare(study, [{**rating, "status": "unknown"} for rating in ratings])
        ratings[0]["value"] = -5. if ratings[0]["positive"] else 5.
        with self.assertRaisesRegex(ValueError, "Противоречивые"):
            _prepare(study, ratings)

    def test_factor_interpretation_never_invents_negative_pole(self):
        _, _, _, scales, _ = _prepare(*fixture())
        factors = _factor_descriptions(np.full((8, 2), .5), list(scales.values()), 8.)
        self.assertEqual(factors[0]["negative_traits"], [])
        self.assertIn("второй полюс отдельно не именуется", factors[0]["interpretation"])
        self.assertAlmostEqual(sum(item["variance_share"] for item in factors), .5)

    def test_single_independent_dimension_is_rejected_and_degenerate_tsne_is_safe(self):
        raw = np.repeat(np.arange(16, dtype=float)[:, None], 8, axis=1)
        with self.assertRaisesRegex(ValueError, "ранг < 2"):
            _dimension_analysis(raw)
        self.assertEqual(_tsne_initialization(np.ones((6, 2))), "random")
        self.assertEqual(_tsne_initialization(np.array([[0., 0.], [1., 0.], [0., 1.]])), "pca")

    def test_negative_nominal_df_candidates_are_never_fitted(self):
        raw = np.random.default_rng(583).normal(size=(24, 6))
        with patch("src.semantic_service.analysis.PARALLEL_PERMUTATIONS", 5), \
             patch("src.semantic_service.analysis.fit_model", wraps=fit_model) as fit:
            _, candidates, models, selection = _dimension_analysis(raw)
        self.assertEqual([item["q"] for item in candidates], [2, 3])
        self.assertEqual(set(models), {2, 3})
        self.assertEqual({call.args[1] for call in fit.call_args_list}, {2, 3})
        self.assertEqual([(item["parameter_count"], item["covariance_moments"], item["nominal_df"])
                          for item in candidates], [(17, 21, 4), (21, 21, 0)])
        guard = selection["parameter_count_constraint"]
        self.assertTrue(guard["necessary_not_sufficient"])
        self.assertEqual([(item["q"], item["nominal_df"]) for item in guard["excluded_candidates"]],
                         [(4, -3), (5, -5)])

    def test_seventeen_scales_still_allow_q2_through_q8(self):
        raw = np.random.default_rng(584).normal(size=(24, 17))
        with patch("src.semantic_service.analysis.PARALLEL_PERMUTATIONS", 5):
            _, candidates, _, selection = _dimension_analysis(raw)
        self.assertEqual([item["q"] for item in candidates], list(range(2, 9)))
        self.assertEqual(selection["parameter_count_constraint"]["excluded_candidates"], [])
        self.assertEqual(candidates[-1]["covariance_moments"], 153)
        self.assertEqual(candidates[-1]["parameter_count"], 125)
        self.assertEqual(candidates[-1]["nominal_df"], 28)

    def test_factor_property_orientation_follows_loading_sign(self):
        scales = [{"id": "s1", "name": "Общее имя 1", "positive": "Твёрдый", "negative": "Мягкий"},
                  {"id": "s2", "name": "Общее имя 2", "positive": "Доверчивый", "negative": "Недоверчивый"}]
        factor = _factor_descriptions(np.array([[.8], [-.7]]), scales, 2.)[0]
        self.assertEqual(factor["label"], "Твёрдый ↔ Доверчивый")
        self.assertEqual(factor["negative_traits"][0]["name"], "Общее имя 2")
        self.assertEqual(factor["negative_traits"][0]["high_factor_property"], "Недоверчивый")
        self.assertIn("Высокая координата связана со свойствами: Твёрдый, Недоверчивый", factor["interpretation"])
        self.assertIn("Низкая координата связана со свойствами: Доверчивый, Мягкий", factor["interpretation"])

    def test_end_to_end_overrides_and_reproducible_exports(self):
        study, ratings = fixture()
        # Reduce repeated diagnostics only; keep the real fitting and charts.
        with tempfile.TemporaryDirectory() as temporary, \
             patch("src.semantic_service.analysis.PARALLEL_PERMUTATIONS", 15), \
             patch("src.semantic_service.analysis.STABILITY_REPEATS", 3):
            output = Path(temporary)
            result = analyze_study(study, ratings, output, factor_count=2, cluster_count=2)
            self.assertEqual((result["factor_count"], result["cluster_count"]), (2, 2))
            self.assertEqual(sum(item["count"] for item in result["clusters"]), 16)
            self.assertEqual(len(result["charts"]), 7)
            self.assertTrue(result["diagnostics"]["dimension_selection"]["overridden"])
            self.assertTrue(result["diagnostics"]["cluster_selection"]["overridden"])
            for chart in result["charts"]:
                self.assertTrue((output / chart["file"]).is_file())
            params = joblib.load(output / "parameters.joblib")
            matrix = pd.read_csv(output / "matrix.csv", index_col="object_id")
            scores = pd.read_csv(output / "scores.csv", index_col="object_id")
            loadings = pd.read_csv(output / "loadings.csv", index_col="scale_id")
            self.assertEqual(matrix.columns.tolist(), params["trait_order"])
            self.assertEqual(loadings.index.tolist(), params["trait_order"])
            np.testing.assert_allclose(transform_new_objects(matrix.to_numpy(), params),
                                       scores[["F1", "F2"]].to_numpy(), atol=1e-10)
            self.assertAlmostEqual(result["variance_total"], sum(f["variance_share"] for f in result["factors"]))
            saved = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["n_objects"], 16)
            self.assertEqual(len(saved["object_profiles"]), 16)
            self.assertNotIn("factor_scores", saved["diagnostics"]["excluded_objects"])

    def test_every_candidate_has_complete_semantic_groups_and_numeric_centers(self):
        study, ratings = fixture()
        with tempfile.TemporaryDirectory() as temporary, \
             patch("src.semantic_service.analysis.PARALLEL_PERMUTATIONS", 10), \
             patch("src.semantic_service.analysis.STABILITY_REPEATS", 3), \
             patch("src.semantic_service.analysis._save_charts", return_value=([], {})):
            output = Path(temporary)
            result = analyze_study(study, ratings, output, factor_count=2)
            scores = pd.read_csv(output / "scores.csv", index_col="object_id")
        expected_ids = {item["id"] for item in study["objects"]}
        for candidate in result["cluster_candidates"]:
            with self.subTest(k=candidate["k"]):
                groups = candidate["semantic_profiles"]
                self.assertEqual(len(groups), candidate["k"])
                member_ids = [identity for group in groups for identity in group["member_ids"]]
                self.assertEqual(len(member_ids), len(set(member_ids)))
                self.assertEqual(set(member_ids), expected_ids)
                for group in groups:
                    self.assertEqual(group["size"], len(group["member_ids"]))
                    self.assertEqual(len(group["representatives"]), min(5, group["size"]))
                    self.assertTrue({item["id"] for item in group["representatives"]} <= set(group["member_ids"]))
                    for factor in group["top_factors"]:
                        actual_center = scores.loc[group["member_ids"], factor["factor_id"]].mean()
                        self.assertAlmostEqual(actual_center, factor["coordinate"], places=10)
                    for scale in group["distinguishing_scales"]:
                        self.assertIn(scale["direction"], ("positive", "negative", "neutral"))
                        self.assertIn("positive", scale)
                        self.assertIn("negative", scale)

    def test_invalid_factor_override_fails_clearly(self):
        study, ratings = fixture()
        with tempfile.TemporaryDirectory() as temporary, \
             patch("src.semantic_service.analysis.PARALLEL_PERMUTATIONS", 5), \
             self.assertRaisesRegex(ValueError, "Число факторов"):
            analyze_study(study, ratings, Path(temporary), factor_count=50)


if __name__ == "__main__":
    unittest.main()
