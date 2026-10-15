"""Ward invariants, conditional denominators and source-state preservation."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
from sklearn.metrics import adjusted_rand_score

from scripts.latent_space_benchmark import fit_model, transform_new_objects
from src.semantic_service.analysis import _prepare, SEED
from src.semantic_service.ward_review import (
    DISPLAY_SIGNS, _object_boundaries, group_profiles, review_ward_q3,
    summarize_partition_variants, ward_partition,
)


class WardReviewTests(unittest.TestCase):
    def test_reflection_and_orthogonal_axis_permutation_preserve_partition(self):
        scores = np.random.default_rng(43).normal(size=(24, 3))
        for k in [2, 3, 5]:
            reference = ward_partition(scores, k)
            self.assertEqual(adjusted_rand_score(reference, ward_partition(scores * DISPLAY_SIGNS, k)), 1.)
            self.assertEqual(adjusted_rand_score(reference, ward_partition(scores[:, [2, 0, 1]], k)), 1.)

    def test_shared_pair_denominators_and_permuted_group_names(self):
        reference = np.array([0, 0, 1, 1])
        variants = [{"indices": [0, 1, 2], "labels": [1, 1, 0]},
                    {"indices": [1, 2, 3], "labels": [1, 0, 0]}]
        result = summarize_partition_variants(reference, variants, ["a", "b", "c", "d"], 2)
        self.assertEqual(result["median_adjusted_rand"], 1.)
        self.assertEqual(result["pair_jointly_present_count"][0][1], 1)
        self.assertEqual(result["pair_co_membership_frequency"][0][1], 1.)
        self.assertEqual(result["pair_jointly_present_count"][0][3], 0)
        self.assertIsNone(result["pair_co_membership_frequency"][0][3])
        self.assertTrue(all(x["same_cluster_fraction"] == 1. for x in result["objects"]))
        self.assertTrue(all(x["median"] == 1. for x in result["per_cluster_jaccard"]))

    def test_unknown_not_zero_and_boundary_never_reassigns_ward_member(self):
        scores = np.array([[0., 0., 0.], [.1, 0., 0.], [100., 0., 0.], [2., 0., 0.], [2.1, 0., 0.]])
        labels = np.array([0, 0, 0, 1, 1])
        ids = list("abcde"); names = {x: x for x in ids}
        boundary, _ = _object_boundaries(scores, labels, ids, names)
        self.assertLess(boundary[0]["centroid_margin"], 0)
        self.assertEqual(boundary[0]["cluster_id"], "C1")
        raw = np.array([[4.], [np.nan], [np.nan], [1.], [1.]])
        profile = group_profiles(raw, scores, labels, ids, names, ["s"], {"s": "Measured property"})[0]
        self.assertEqual(profile["observed_trait_profiles"][0]["observed_mean"], 4.)
        self.assertEqual(profile["observed_trait_profiles"][0]["observed_n"], 1)
        self.assertEqual(profile["distinguishing_traits"], [])

    def test_review_reproduces_partial_coordinate_posterior_and_preserves_inputs(self):
        rng = np.random.default_rng(87)
        raw = np.clip(2.5 + rng.normal(size=(20, 3)) @ rng.normal(scale=.5, size=(3, 8)) + rng.normal(scale=.3, size=(20, 8)), 0, 5)
        raw[0, 0] = np.nan; raw[4, 2] = np.nan
        study = {"measurement_model": "independent_poles", "objects": [{"id": f"o{i}", "name": f"Object {i}"} for i in range(20)],
                 "scales": [{"id": f"s{j}", "name": f"Property {j}", "kind": "unipolar", "definition": "Synthetic variable"} for j in range(8)]}
        ratings = [{"object_id": f"o{i}", "scale_id": f"s{j}", "status": "observed" if np.isfinite(raw[i, j]) else "unknown",
                    "value": float(raw[i, j]) if np.isfinite(raw[i, j]) else None} for i in range(20) for j in range(8)]
        before = deepcopy((study, ratings))
        matrix, _, _, _, diagnostics = _prepare(study, ratings)
        diagnostics.update(training_object_order=matrix.index.tolist(), trait_order=matrix.columns.tolist())
        parameters, _ = fit_model(matrix.to_numpy(), 3, seed=SEED)
        loadings = parameters["loadings"] @ parameters["rotation"]
        preview = {"diagnostics": diagnostics, "models": [{"q": 3, "variance_total": .5, "factors": [],
            "loading_table": [{"id": sid, "loadings": {f"F{j+1}": float(row[j]) for j in range(3)}} for sid, row in zip(matrix.columns, loadings)]}]}
        with tempfile.TemporaryDirectory() as tmp:
            result = review_ward_q3(study, ratings, preview, Path(tmp), k_values=(2, 3), subsample_repeats=3, imputation_draws=2)
            actual = np.array([list(x["raw_scores"].values()) for x in result["objects"]])
            np.testing.assert_array_equal(actual, transform_new_objects(matrix.to_numpy(), parameters, rotated=True))
            self.assertIsNone(result["selected_k"])
            self.assertIsNone(result["objects"][0]["property_measurements"][0]["value"])
            for c in result["candidates"]:
                self.assertEqual(sum(c["sizes"]), 20)
                self.assertEqual({x["id"] for g in c["clusters"] for x in g["members"]}, set(result["object_order"]))
        self.assertEqual((study, ratings), before)


if __name__ == "__main__":
    unittest.main()
