"""Regression: one intense statement differs from many copied statements."""
import unittest
from src.corpus_pipeline.rubrics import intensity_rubric_issues, normalize_discovery


class CorpusRubricTests(unittest.TestCase):
    def test_frequency_and_visibility_cannot_raise_intensity(self):
        for phrase in ["repeatedly champions", "consistently promotes", "systematically denies", "frequently criticizes", "регулярно выступает", "многократно повторяет", "prominently centers"]:
            self.assertTrue(intensity_rubric_issues({"anchor5": phrase}), phrase)
        self.assertFalse(intensity_rubric_issues({"anchor5": "Categorically rejects the accusation in this statement."}))

    def test_bad_proposal_does_not_discard_independent_good_proposal(self):
        good = {"id": "direct", "anchor5": "Explicitly commits resources to the project."}
        bad = {"id": "copies", "anchor5": "Regularly expresses support."}
        out = normalize_discovery({"scales": [bad, good]}, {"config": {}}, lambda stage, data, ctx: data)
        self.assertEqual(out["scales"], [good])
        self.assertEqual(out["quality_control"]["skipped_scales"], 1)


if __name__ == "__main__":
    unittest.main()
