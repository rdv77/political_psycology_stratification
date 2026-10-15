"""Guard interpretation exports against misleading membership comparisons."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from scripts.finalize_factor_comparison import partition_overlap, reviewed_metadata


class ComparisonExportTests(unittest.TestCase):
    def test_unequal_group_counts_show_intersections_without_false_movement_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            first, second = Path(temporary) / "a.csv", Path(temporary) / "b.csv"
            pd.DataFrame({"object_id": list("abcdef"), "cluster_id": ["A"] * 3 + ["B"] * 3}).to_csv(first, index=False)
            pd.DataFrame({"object_id": list("fedcba"), "cluster_id": ["Z", "Z", "Y", "X", "X", "X"]}).to_csv(second, index=False)
            result = partition_overlap(first, second)
            self.assertEqual((result["k_reference"], result["k_target"]), (2, 3))
            self.assertEqual(result["overlap"]["values"], [[3, 0, 0], [0, 1, 2]])
            self.assertNotIn("moved_count", result)
            self.assertLess(result["ari"], 1)
            pd.read_csv(second).iloc[:-1].to_csv(second, index=False)
            with self.assertRaisesRegex(ValueError, "exactly the same"):
                partition_overlap(first, second)

    def test_review_cannot_change_computed_membership_or_silently_mismatch_size(self):
        metadata = {"factors": [{"id": "F1", "label": "numeric", "variance_share": .3}],
                    "clusters": [{"id": "C1", "name": "numeric", "count": 2,
                                  "members": [{"id": "a"}, {"id": "b"}]}]}
        original = deepcopy(metadata)
        review = {"factors": [{"id": "F1", "label": "axis", "interpretation": "profile"}],
                  "clusters": [{"id": "C1", "label": "group", "interpretation": "profile", "count": 2}]}
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "review.json"
            source.write_text(json.dumps(review), encoding="utf-8")
            with patch("scripts.finalize_factor_comparison.REVIEW", source), patch(
                    "scripts.finalize_factor_comparison.ROOT", source.parent):
                enriched = reviewed_metadata(metadata, review)
                self.assertEqual(metadata, original)
                self.assertEqual(enriched["clusters"][0]["members"], original["clusters"][0]["members"])
                self.assertEqual(enriched["factors"][0]["variance_share"], .3)
                review["clusters"][0]["count"] = 3
                with self.assertRaisesRegex(ValueError, "cluster size"):
                    reviewed_metadata(metadata, review)
                review["clusters"][0]["id"] = "C2"
                with self.assertRaisesRegex(ValueError, "all numerical"):
                    reviewed_metadata(metadata, review)


if __name__ == "__main__":
    unittest.main()
