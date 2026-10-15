"""Meaningful invariants of cross-dimensional model comparisons, no APIs."""
import unittest
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.compare_factor_models import _assert_same_rectangle, _snapshot_baseline, compare_partitions, factor_congruence, freeze_input_files


class FactorModelComparisonTests(unittest.TestCase):
    def test_signed_permutation_is_recovered_without_modifying_loadings(self):
        rng = np.random.default_rng(24)
        a = rng.normal(size=(17, 2))
        b = np.column_stack([-a[:, 1], rng.normal(size=17), a[:, 0]])
        original = b.copy()
        result = factor_congruence(a, b)
        self.assertEqual([(x["target_factor"], x["sign"]) for x in result["matched_factors"]], [("F3", 1), ("F1", -1)])
        self.assertEqual(result["unmatched_target_factors"], ["F2"])
        self.assertTrue(all(abs(x["absolute_congruence"] - 1) < 1e-12 for x in result["matched_factors"]))
        self.assertTrue(np.max(result["principal_angles_degrees"]) < 1e-10)
        np.testing.assert_array_equal(b, original)

    def test_arbitrary_rotation_is_distinguished_from_axis_identity(self):
        a = np.array([[1., 0.], [0., 1.], [1., 1.], [1., -1.]])
        rotation = np.array([[1., -1.], [1., 1.]]) / np.sqrt(2)
        result = factor_congruence(a, a @ rotation)
        self.assertTrue(all(x["absolute_congruence"] < .8 for x in result["matched_factors"]))
        self.assertLess(result["procrustes"]["relative_loading_residual"], 1e-12)
        self.assertTrue(np.max(result["principal_angles_degrees"]) < 1e-10)

    def test_label_permutation_is_not_reported_as_object_movement(self):
        index = list("abcdef")
        first = pd.Series(["C1", "C1", "C2", "C2", "C3", "C3"], index=index)
        second = first.map({"C1": "C3", "C2": "C1", "C3": "C2"}).sample(frac=1, random_state=3)
        result = compare_partitions(first, second)
        self.assertEqual(result["ari"], 1.)
        self.assertEqual(result["nmi"], 1.)
        self.assertEqual(result["moved_count"], 0)
        self.assertEqual(result["unchanged_count"], 6)
        self.assertEqual([x["object_id"] for x in result["transitions"]], index)

    def test_actual_boundary_changes_are_complete_and_labels_cannot_hide_them(self):
        first = pd.Series(["C1", "C1", "C1", "C2", "C2", "C2", "C3", "C3", "C3"], index=list("abcdefghi"))
        second = pd.Series(["C2", "C2", "C1", "C1", "C1", "C1", "C3", "C3", "C3"], index=first.index)
        result = compare_partitions(first, second)
        self.assertEqual(result["moved_count"], 1)
        self.assertEqual([x["object_id"] for x in result["transitions"] if x["changed"]], ["c"])
        self.assertEqual(sum(map(sum, result["contingency_aligned"]["values"])), 9)
        self.assertLess(result["ari"], 1.)
        with self.assertRaisesRegex(ValueError, "same objects"):
            compare_partitions(first, second.iloc[:-1])

    def test_frozen_rectangle_guard_includes_missingness_order_and_values(self):
        frame = pd.DataFrame([[0., np.nan], [1., 2.]], index=["a", "b"], columns=["s1", "s2"])
        _assert_same_rectangle(frame.copy(), frame)
        for altered in [frame.iloc[::-1], frame[["s2", "s1"]], frame.fillna(0.), frame + 1.]:
            with self.assertRaises(ValueError):
                _assert_same_rectangle(altered, frame)

    def test_existing_baseline_is_reused_without_recursive_comparison_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            primary, snapshot = root / "output", root / "output" / "models" / "q2"
            primary.mkdir()
            (primary / "metadata.json").write_text('{"factor_count":2}', encoding="utf-8")
            for name in ["matrix.csv", "scores.csv", "loadings.csv", "cluster_labels.csv"]:
                (primary / name).write_text("unchanged numeric contents", encoding="utf-8")
            original = _snapshot_baseline(primary, snapshot)
            (primary / "metadata.json").write_text(json.dumps({"factor_count": 2, "model_comparison": {"models": []}}), encoding="utf-8")
            self.assertEqual(_snapshot_baseline(primary, snapshot), original)
            self.assertNotIn("model_comparison", json.loads((snapshot / "metadata.json").read_text(encoding="utf-8")))
            (primary / "scores.csv").write_text("changed numeric data", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no longer match"):
                _snapshot_baseline(primary, snapshot)

    def test_frozen_inputs_preserve_fit_basis_but_reject_changed_measurements(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            study_path, ratings_path = root / "study.json", root / "ratings.json"
            study = {"scope": "public image", "objects": [{"id": "a"}], "scales": [{"id": "s"}], "analysis": None}
            study_path.write_text(json.dumps(study), encoding="utf-8")
            ratings_path.write_text('[{"value":0}]', encoding="utf-8")
            frozen, _ = freeze_input_files(study_path, ratings_path, root / "models")
            original = frozen.read_bytes()
            study["analysis"] = {"model_comparison": {"models": []}}
            study_path.write_text(json.dumps(study), encoding="utf-8")
            self.assertEqual(freeze_input_files(study_path, ratings_path, root / "models")[0].read_bytes(), original)
            study["scope"] = "changed measurement scope"
            study_path.write_text(json.dumps(study), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "measurement basis changed"):
                freeze_input_files(study_path, ratings_path, root / "models")


if __name__ == "__main__":
    unittest.main()
