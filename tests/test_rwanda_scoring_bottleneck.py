"""The forensic funnel must count units separately and keep private text local."""
import json
import unittest

from scripts.audit_rwanda_scoring_bottleneck import audit


class ScoringBottleneckAuditTests(unittest.TestCase):
    def test_candidate_and_rating_counts_are_not_equated(self):
        facts = [
            {"id": "private-fact-1", "actor_id": "a", "kind": "public_statement",
             "attribution_basis": "direct_statement", "quote": "PRIVATE QUOTE",
             "document_id": "private-doc-1", "literal_material_group_id": "private-group-1",
             "source_url": "https://private.invalid/1", "event_date": None, "source_date": "2026-01-01"},
            {"id": "private-fact-2", "actor_id": "a", "kind": "public_statement",
             "attribution_basis": "direct_statement", "document_id": "private-doc-2",
             "literal_material_group_id": "private-group-1", "event_date": None},
            {"id": "private-fact-3", "actor_id": "b", "kind": "attributed_image",
             "attribution_basis": "third_party_claim", "document_id": "private-doc-3"},
        ]
        cases = [{"actor_id": "a", "sample_kind": "selected_unit", "verdict": "retain",
                  "event_date": {"status": "publication_only"}, "ratings": [{"scale_id": "g01", "value": 3}]}]
        strict = {"cells": [{"actor_id": "a", "scale_id": "g01", "status": "unknown"}],
                  "counts": {"numeric_case_ratings_before_episode_deduplication": 0,
                             "episode_property_values_after_deduplication": 0,
                             "observed_cells": 0, "cells_held_for_event_independence": 0}}
        loose = {"cells": [{"actor_id": "a", "scale_id": "g01", "status": "observed"}],
                 "counts": {"numeric_case_ratings_before_episode_deduplication": 1,
                            "episode_property_values_after_deduplication": 1,
                            "observed_cells": 1, "cells_held_for_event_independence": 0}}
        profiles = {"approved_window": {"from": "2021-10-15", "to": "2026-10-15"},
                    "counts": {"reviewed_episode_groups": 1}, "scales": [{"id": "g01"}],
                    "scenarios": {"strict_dated": strict, "content_supported_unknown_time_sensitivity": loose}}
        result = audit(facts, {"annotations": [{"source_fact_ids": ["private-fact-1"]}]},
                       {"cases": cases}, profiles, [{"id": "a", "name": "Approved A"}, {"id": "b", "name": "Approved B"}],
                       {"date_from": "2021-10-15", "date_to": "2026-10-17", "aspect": "public_positions"})
        self.assertEqual(result["counts"]["all_extracted_facts"], 3)
        self.assertEqual(result["counts"]["eligible_candidates_after_general_checks"], 2)
        self.assertEqual(result["counts"]["candidate_literal_groups"], 1)
        self.assertEqual(result["counts"]["eligible_candidates_not_semantically_reviewed"], 1)
        self.assertEqual(result["counts"]["strict_actor_property_values"], 0)
        self.assertEqual(result["counts"]["unknown_time_sensitivity_actor_property_values"], 1)
        public = json.dumps(result)
        for private in ("private-fact", "private-doc", "private-group", "PRIVATE QUOTE", "private.invalid"):
            self.assertNotIn(private, public)
        self.assertFalse(result["ratings_generated"])
        self.assertEqual(result["api_calls"], 0)


if __name__ == "__main__":
    unittest.main()
