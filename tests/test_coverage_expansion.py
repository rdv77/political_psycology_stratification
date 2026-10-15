import copy
import hashlib
import unittest

from scripts.apply_rwanda_coverage_expansion import collect_proposals, supported_indices
from src.semantic_service.store import fingerprint


class CoverageExpansionTests(unittest.TestCase):
    def setUp(self):
        self.study = {"objects": [{"id": "p"}], "scales": [{"id": "s"}], "date_from": "2021-10-15", "date_to": "2026-10-15",
                      "scope": "Synthetic actor's public actions", "measurement_model": "independent_poles"}
        self.ratings = [{"object_id": "p", "scale_id": "s", "pole_scores": {
            "positive": {"status": "unknown", "value": None}, "negative": {"status": "observed", "value": 2}}}]
        self.plan = {"input_ratings_fingerprint": fingerprint(self.ratings),
                     "input_study_scope": {key: self.study[key] for key in ("scope", "date_from", "date_to", "measurement_model")},
                     "input_objects_fingerprint": fingerprint(self.study["objects"]),
                     "input_scales_fingerprint": fingerprint(self.study["scales"]),
                     "checked_unknown_properties": [{"object_id": "p", "scale_id": "s", "pole": "positive"}]}
        self.report = {"batch": "A", "plan_fingerprint": fingerprint(self.plan), "checked_properties": [["p", "s", "positive"]],
                       "observations": [{"object_id": "p", "scale_id": "s", "pole": "positive", "value": 1,
                                         "rationale": "A narrow direct manifestation", "new_facts": [{
                                             "actor_id": "p", "source_url": "https://example.org/speech", "text": "An attributed proposal",
                                             "period": "2024-05-01", "event_id": "speech", "capacity": "A public role"}]}]}

    def test_new_evidence_preserves_original_input_and_event(self):
        original = copy.deepcopy(self.ratings)
        proposals, facts = collect_proposals(self.study, self.ratings, self.plan, [self.report])
        self.assertEqual(self.ratings, original)
        self.assertEqual(proposals[0]["value"], 1)
        self.assertEqual(facts[0]["event_id"], "speech")
        self.assertEqual(facts[0]["topic_ids"], ["s"])

    def test_proposals_cannot_overwrite_known_pole_or_create_zero(self):
        for update in ({"pole": "negative"}, {"value": 0}):
            report = copy.deepcopy(self.report)
            report["observations"][0].update(update)
            with self.assertRaises(ValueError):
                collect_proposals(self.study, self.ratings, self.plan, [report])

    def test_source_actor_date_and_complete_review_are_required(self):
        for update in ({"actor_id": "someone-else"}, {"period": "2019-01-01"}):
            report = copy.deepcopy(self.report)
            report["observations"][0]["new_facts"][0].update(update)
            with self.assertRaises(ValueError):
                collect_proposals(self.study, self.ratings, self.plan, [report])
        report = copy.deepcopy(self.report)
        report["checked_properties"] = []
        with self.assertRaises(ValueError):
            collect_proposals(self.study, self.ratings, self.plan, [report])

    def test_frozen_scope_is_enforced_even_with_identical_objects_and_ratings(self):
        for field, value in (("scope", "Other public space"), ("date_to", "2027-10-15"), ("measurement_model", "paired_difference")):
            changed = dict(self.study, **{field: value})
            with self.assertRaisesRegex(ValueError, "scope, period or measurement model"):
                collect_proposals(changed, self.ratings, self.plan, [self.report])

    def test_expansion_cannot_reuse_context_only_registry_excluded_or_foreign_support(self):
        proposal = dict(self.report["observations"][0], evidence_indices=[0], new_facts=[])
        fact = self.report["observations"][0]["new_facts"][0]
        for update in ({"measurement_status": "context_only"}, {"measurement_status": "excluded_known_source_issue"}, {"actor_id": "other"}):
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                supported_indices(proposal, [dict(fact, **update)], self.study)
        study = dict(self.study, independent_evidence={"additional_excluded_facts": [{"object_id": "p", "source_url": fact["source_url"],
            "text_sha256": hashlib.sha256(fact["text"].encode()).hexdigest()}]})
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            supported_indices(proposal, [fact], study)
        report = copy.deepcopy(self.report)
        report["observations"][0]["new_facts"][0]["measurement_status"] = "context_only"
        with self.assertRaisesRegex(ValueError, "context-only"):
            collect_proposals(self.study, self.ratings, self.plan, [report])


if __name__ == "__main__":
    unittest.main()
