"""Leakage, measurement-policy and bounded posterior-imputation checks."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.semantic_service.imputation_review import (
    coverage_sensitivity, factor_imputation_sensitivity, impute_from_training, masked_validation,
    review_imputation, score_masked_trial,
)


def fixture(n=20, p=8):
    rng = np.random.default_rng(641)
    latent = rng.normal(size=(n, 2))
    raw = np.clip(2.5 + latent @ rng.normal(scale=.5, size=(2, p)) + rng.normal(scale=.25, size=(n, p)), 0, 5)
    study = {"measurement_model": "independent_poles",
             "objects": [{"id": f"o{i}", "name": f"Synthetic object {i}"} for i in range(n)],
             "scales": [{"id": f"s{j}", "name": f"Synthetic property {j}", "kind": "unipolar",
                         "definition": "Synthetic test variable, not a real political rating"} for j in range(p)]}
    ratings = [{"object_id": f"o{i}", "scale_id": f"s{j}", "status": "observed", "value": float(raw[i, j])}
               for i in range(n) for j in range(p)]
    return study, ratings, raw


class ImputationReviewTests(unittest.TestCase):
    def test_strict_scenarios_use_current_flags_and_both_followup_methods(self):
        study, ratings, _ = fixture()
        for row in ratings:
            if row["scale_id"] == "s0":
                row["source_audit"] = {"decision": "flag"}
            if row["scale_id"] == "s1":
                row["provenance"] = {"method": "coverage_only_assistant_completion_before_factor_fit"}
            if row["scale_id"] == "s2":
                row["provenance"] = {"method": "targeted_coverage_expansion"}
            if row["scale_id"] == "s3":
                row["previous_unknown_measurement"] = {"provenance": {"method": "completion"}}
        before = deepcopy(ratings)
        result = coverage_sensitivity(study, ratings)
        self.assertEqual([x["removed_count"] for x in result["scenarios"]], [20, 40, 60])
        self.assertEqual([x["diagnostics"]["retained_scales"] for x in result["scenarios"]], [7, 6, 5])
        self.assertEqual(result["scenarios"][2]["measurement_status"], "insufficient_data_no_FA_fitted")
        self.assertTrue(all(not x["FA_fitted"] for x in result["scenarios"]))
        self.assertEqual(ratings, before)

    def test_masked_truth_never_changes_any_method_fit_or_prediction(self):
        _, _, raw = fixture()
        train, validation = raw[:15], raw[15:].copy()
        hidden = np.zeros_like(validation, dtype=bool)
        hidden[0, 0] = hidden[2, 3] = True
        changed = validation.copy(); changed[hidden] = [0., 5.]
        for method in ["mean", "bayesian_ridge", "knn"]:
            first = score_masked_trial(train, validation, hidden, method=method, seed=9)
            second = score_masked_trial(train, changed, hidden, method=method, seed=9)
            np.testing.assert_array_equal(first["predicted"], second["predicted"])
            self.assertNotEqual(first["truth"], second["truth"])
            self.assertEqual(first["fit"]["fit_rows"], 15)

    def test_observed_zero_preserved_posterior_varies_only_missing_and_is_bounded(self):
        _, _, raw = fixture()
        raw[0, 0] = 0.; raw[1, 2] = np.nan; raw[3, 4] = np.nan
        first, _ = impute_from_training(raw, raw, method="bayesian_ridge", posterior=True, seed=12)
        repeated, _ = impute_from_training(raw, raw, method="bayesian_ridge", posterior=True, seed=12)
        other, _ = impute_from_training(raw, raw, method="bayesian_ridge", posterior=True, seed=13)
        np.testing.assert_array_equal(first, repeated)
        np.testing.assert_array_equal(first[np.isfinite(raw)], raw[np.isfinite(raw)])
        self.assertEqual(first[0, 0], 0.)
        self.assertTrue(np.all((first >= 0) & (first <= 5)))
        self.assertFalse(np.array_equal(first[~np.isfinite(raw)], other[~np.isfinite(raw)]))

    def test_entirely_unmeasured_feature_or_object_is_not_zero_filled(self):
        _, _, raw = fixture()
        empty_feature = raw.copy(); empty_feature[:, 1] = np.nan
        with self.assertRaisesRegex(ValueError, "entirely unmeasured variable"):
            impute_from_training(empty_feature, raw, method="mean")
        with self.assertRaisesRegex(ValueError, "entirely unmeasured object"):
            impute_from_training(raw, np.full((1, 8), np.nan), method="knn")

    def test_validation_methods_share_identical_masks_and_train_rows_are_disjoint(self):
        _, _, raw = fixture()
        raw[0, 1] = np.nan
        result = masked_validation(raw, repeats=3, mask_fraction=.12, seed=58)
        self.assertEqual(result["successful_repeats"], 3)
        for trial in result["trials"]:
            self.assertTrue(set(trial["train_rows"]).isdisjoint(trial["validation_rows"]))
            for row, column in trial["hidden_cells"]:
                self.assertIn(row, trial["validation_rows"])
                self.assertTrue(np.isfinite(raw[row, column]))
            truths = [x["truth"] for x in trial["methods"].values()]
            self.assertTrue(all(x == truths[0] for x in truths))

    def test_factor_comparison_matches_reference_and_obeys_parameter_count(self):
        _, _, raw = fixture(p=6)
        raw[0, 0] = np.nan; raw[4, 2] = np.nan
        result = factor_imputation_sensitivity(raw, [f"s{j}" for j in range(6)], posterior_draws=2)
        self.assertEqual([x["q"] for x in result["models"]], [2, 3])
        self.assertEqual([x["q"] for x in result["excluded_dimensions"]], [4])
        for model in result["models"]:
            reference = model["deterministic"][0]
            np.testing.assert_allclose(reference["congruence_to_mean_reference"], 1., atol=1e-10)
            self.assertAlmostEqual(reference["variance_total"], sum(reference["aligned_factor_shares"]))
            self.assertAlmostEqual(reference["variance_denominator"], 6.)
            self.assertEqual(len(model["posterior"]["draws"]), 2)
        self.assertTrue(all(not x["estimate_is_observation"] and x["measurement_status"] == "unknown"
                            for x in result["missing_cell_estimates"]))

    def test_policy_excludes_sparse_theme_and_unknown_object_before_imputation(self):
        study, ratings, _ = fixture()
        study["objects"].append({"id": "empty", "name": "No measured properties"})
        study["scales"].append({"id": "russia-test", "name": "Synthetic thematic property", "kind": "unipolar",
                                "research_domain": "russia_relations", "definition": "Synthetic sparse property"})
        ratings.append({"object_id": "o0", "scale_id": "russia-test", "status": "observed", "value": 1.})
        before = deepcopy(study), deepcopy(ratings)
        with tempfile.TemporaryDirectory() as directory:
            result = review_imputation(study, ratings, Path(directory), repeats=2, posterior_draws=2, q_values=(2,))
            json.dumps(result, allow_nan=False)
        self.assertEqual(result["n_objects"], 20)
        self.assertNotIn("empty", result["object_ids"])
        self.assertNotIn("russia-test", result["variable_ids"])
        self.assertEqual(result["russia_block"]["observed_input_properties"], 1)
        self.assertEqual(result["russia_block"]["retained_variables"], [])
        self.assertEqual((study, ratings), before)

    def test_insufficient_measurement_is_reported_but_invalid_input_is_not_swallowed(self):
        study, ratings, _ = fixture(p=5)
        with tempfile.TemporaryDirectory() as directory:
            result = review_imputation(study, ratings, Path(directory), repeats=2, posterior_draws=2)
            self.assertEqual(result["measurement_status"], "insufficient_data")
            self.assertIsNone(result["factor_sensitivity"])
            self.assertIsNone(result["validation"])
            with self.assertRaisesRegex(ValueError, "Повторная оценка"):
                review_imputation(study, ratings + ratings[:1], Path(directory), repeats=2, posterior_draws=2)


if __name__ == "__main__":
    unittest.main()
