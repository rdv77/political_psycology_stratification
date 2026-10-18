import tempfile
import unittest
from pathlib import Path

from src.corpus_pipeline.native_mass import checkpoint_changes, parse_annotations, stopping_decision


class NativeMassTests(unittest.TestCase):
    def test_missing_is_not_zero(self):
        before = [{"actor_id": "a", "scale_id": "s", "value": None}]
        after = [{"actor_id": "a", "scale_id": "s", "value": 0}]
        result = checkpoint_changes(before, after)
        self.assertEqual(result["new_known"], 1)
        self.assertEqual(result["substantive_changes"], 1)

    def test_half_point_is_substantive(self):
        before = [{"actor_id": "a", "scale_id": "s", "value": 3}]
        after = [{"actor_id": "a", "scale_id": "s", "value": 3.5}]
        self.assertEqual(checkpoint_changes(before, after)["substantive_changes"], 1)

    def test_no_early_stop_without_two_full_aggregated_blocks(self):
        block = {"reviewed_cases_in_block": 500, "substantive_changes": 0, "profiles_aggregated": True}
        self.assertFalse(stopping_decision([block], rare_actor_queue_exhausted=True))
        self.assertFalse(stopping_decision([block, block], rare_actor_queue_exhausted=False))
        self.assertTrue(stopping_decision([block, block], rare_actor_queue_exhausted=True))
        self.assertFalse(stopping_decision([block, {**block, "profiles_aggregated": False}], rare_actor_queue_exhausted=True))

    def test_held_case_cannot_have_a_score(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.tsv"
            path.write_text("1|s=3|hold|Неясная атрибуция\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                parse_annotations([path], [{"case_id": "B00001"}], {"scales": [{"id": "s"}]})
