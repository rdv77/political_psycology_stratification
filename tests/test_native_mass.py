import tempfile
import unittest
from pathlib import Path

from src.corpus_pipeline.native_mass import checkpoint_changes, parse_annotations, stopping_decision, extend_literal_reuse, source_intensity_bounds, bound_changes, protocol_only_screen


class NativeMassTests(unittest.TestCase):
    def test_protocol_filter_preserves_substantive_actions(self):
        def entry(text):
            return {"matched_scale_ids": [], "fact": {"kind": "action", "attribution_basis": "reported_action",
                    "quote": text, "attribution_evidence": text, "claim": text}}
        self.assertIsNotNone(protocol_only_screen(entry("The president received letters of credence from the ambassador.")))
        self.assertIsNone(protocol_only_screen(entry("He joined the ceremony after signing a peace agreement.")))
        self.assertIsNone(protocol_only_screen(entry("The minister joined residents and worked on a road to help farmers.")))
        self.assertIsNone(protocol_only_screen(entry("He joined officials to announce a policy.")))
        self.assertIsNone(protocol_only_screen(entry("He met Russia's minister.")))

    def test_bounds_preserve_missing_and_do_not_pretend_midpoints_observed(self):
        cells = [{"actor_id": "a", "scale_id": "s", "value": None,
                  "episode_ids": ["e1", "e2"], "case_ids": ["b1", "b2"]},
                 {"actor_id": "a", "scale_id": "t", "value": None,
                  "episode_ids": [], "case_ids": []}]
        episodes = [{"episode_id": "e1", "ratings": [{"scale_id": "s", "value": 3}]},
                    {"episode_id": "e2", "ratings": [{"scale_id": "s", "value": 5}]}]
        result = source_intensity_bounds({"episodes": episodes, "scenarios": {"loose": {"cells": cells}}})["loose"]
        self.assertEqual(result["informative_cells"], 1)
        self.assertEqual(result["cells"][0]["lower"], 3)
        self.assertEqual(result["cells"][0]["upper"], 5)
        self.assertFalse(result["cells"][0]["midpoint_is_observed"])
        self.assertIsNone(result["cells"][1]["midpoint_scenario"])

    def test_future_annotation_does_not_fill_earlier_checkpoint(self):
        fact = {"actor_id": "a", "quote": "text", "attribution_evidence": "a said text"}
        index = {"one": {"fact": fact}, "two": {"fact": dict(fact)}}
        queue = [{"case_id": "B00001", "fact_id": "one"}, {"case_id": "B00002", "fact_id": "two"}]
        annotations = {2: {**queue[1], "verdict": "retain", "ratings": [], "reason_ru": "test"}}
        self.assertNotIn(1, extend_literal_reuse(annotations, queue, index, through=2))

    def test_reuse_requires_same_actor_and_attribution(self):
        fact = {"actor_id": "a", "quote": "support policy", "attribution_evidence": "a said support policy"}
        index = {"one": {"fact": fact}, "two": {"fact": dict(fact)},
                 "three": {"fact": {**fact, "actor_id": "b"}},
                 "four": {"fact": {**fact, "attribution_evidence": "critic said support policy"}}}
        queue = [{"case_id": f"B{i:05d}", "fact_id": fid} for i, fid in enumerate(index, 1)]
        annotations = {1: {**queue[0], "verdict": "retain", "ratings": [], "reason_ru": "test"}}
        result = extend_literal_reuse(annotations, queue, index, through=4)
        self.assertEqual(set(result), {1, 2})
        self.assertEqual(result[2]["literal_reuse_of"], "B00001")

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
