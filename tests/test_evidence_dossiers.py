"""Retrieval accounts for all evidence without creating measurements."""
from copy import deepcopy
import json
import unittest

from src.corpus_pipeline.evidence_dossiers import build_dossiers


def fixture():
    actors = [{"id": "a", "name": "Approved Actor"}, {"id": "b", "name": "No Material"}]
    dictionary = {"approved": True, "stage_status": "approved", "scales": [
        {"id": "m01", "name": "Technology"}, {"id": "m10", "name": "Cooperation"},
        {"id": "russia-security-cooperation-positive", "name": "Support"},
        {"id": "russia-security-cooperation-negative", "name": "Restriction"}]}
    fact = {"id": "PRIVATE-F1", "actor_id": "a", "kind": "public_statement",
            "attribution_basis": "direct_statement", "quote": "We reject technological cooperation.",
            "claim": "PRIVATE_MACHINE_CLAIM", "attribution_evidence": "PRIVATE_ATTRIBUTION",
            "trait_candidates": [], "source_date": "2026-05-01", "event_date": None,
            "document_id": "PRIVATE_DOCUMENT", "source_url": "https://private.invalid/"}
    config = {"aspect": "public_positions", "date_from": "2021-01-01", "date_to": "2026-10-18"}
    return [fact], dictionary, actors, config


class EvidenceDossiersTests(unittest.TestCase):
    def test_all_cells_and_all_facts_are_preserved_and_no_score_is_created(self):
        facts, dictionary, actors, config = fixture()
        facts += [dict(facts[0], id="PRIVATE-F2", quote="No lexical retrieval cue here."),
                  dict(facts[0], id="PRIVATE-F3", kind="attributed_image", attribution_basis="third_party_claim")]
        before = deepcopy((facts, dictionary, actors, config))
        result = build_dossiers(facts, dictionary, actors, config)
        self.assertEqual((facts, dictionary, actors, config), before)
        self.assertEqual(len(result["private_fact_index"]), 3)
        self.assertEqual(len(result["private_cells"]), 8)
        self.assertEqual(result["private_actor_dossiers"][0]["unmatched_rescue_fact_ids"], ["PRIVATE-F2"])
        self.assertTrue(all(c["value"] is None and c["measurement_status"] == "not_scored" for c in result["private_cells"]))
        self.assertFalse(result["private_fact_index"]["PRIVATE-F3"]["retrieval_eligible"])
        self.assertEqual(result["public_summary"]["scores_generated"], 0)

    def test_negation_is_retrieved_but_not_turned_into_a_positive_value(self):
        result = build_dossiers(*fixture())
        cell = next(c for c in result["private_cells"] if c["actor_id"] == "a" and c["scale_id"] == "m01")
        self.assertEqual(cell["candidate_fact_count"], 1)
        self.assertIsNone(cell["value"])
        self.assertEqual(result["private_fact_index"]["PRIVATE-F1"]["fact"]["quote"], "We reject technological cooperation.")

    def test_russia_polarities_retrieve_same_passage_without_inferring_polarity(self):
        facts, dictionary, actors, config = fixture()
        facts[0]["quote"] = "Nous limitons la coopération militaire avec la Russie."
        result = build_dossiers(facts, dictionary, actors, config)
        cells = [c for c in result["private_cells"] if c["actor_id"] == "a" and c["scale_id"].startswith("russia-")]
        self.assertEqual([c["candidate_fact_count"] for c in cells], [1, 1])
        self.assertTrue(all(c["value"] is None for c in cells))

    def test_generic_security_without_russia_is_not_a_russia_candidate(self):
        facts, dictionary, actors, config = fixture()
        facts[0]["quote"] = "Our military security agreement."
        result = build_dossiers(facts, dictionary, actors, config)
        self.assertTrue(all(not c["candidates"] for c in result["private_cells"] if c["scale_id"].startswith("russia-")))

    def test_dates_and_literal_groups_do_not_become_invented_events(self):
        facts, dictionary, actors, config = fixture()
        facts += [dict(facts[0], id="PRIVATE-F2"), dict(facts[0], id="PRIVATE-F3", event_date="2026-04-30")]
        result = build_dossiers(facts, dictionary, actors, config)
        cell = next(c for c in result["private_cells"] if c["actor_id"] == "a" and c["scale_id"] == "m01")
        self.assertEqual(cell["literal_review_group_count"], 2)
        self.assertEqual(len(cell["shortlist"][0]["literal_member_fact_ids"]), 2)
        self.assertTrue(all(not c["episode_identity_verified"] and not c["source_independence_verified"] for c in cell["shortlist"]))
        self.assertIsNone(result["private_fact_index"]["PRIVATE-F1"]["fact"]["event_date"])

    def test_machine_proposal_is_marked_as_search_aid_and_original_is_preserved(self):
        facts, dictionary, actors, config = fixture()
        facts[0].update(quote="Source without topical cue.", claim="Supports artificial intelligence.")
        result = build_dossiers(facts, dictionary, actors, config)
        cell = next(c for c in result["private_cells"] if c["actor_id"] == "a" and c["scale_id"] == "m01")
        self.assertEqual(cell["candidates"][0]["source_search_cues"], [])
        self.assertEqual(cell["candidates"][0]["machine_aid_search_cues"], ["artificial intelligence"])
        self.assertIsNone(cell["value"])

    def test_public_allowlist_does_not_expose_quotes_identifiers_or_urls(self):
        result = build_dossiers(*fixture())
        public = json.dumps(result["public_summary"], ensure_ascii=False)
        for token in ("PRIVATE", "private.invalid", "We reject", "source_date", "fact_id"):
            self.assertNotIn(token, public)

    def test_stable_ranking_accounts_for_all_candidates_beyond_shortlist(self):
        facts, dictionary, actors, config = fixture()
        facts += [dict(facts[0], id="PRIVATE-F2", quote="Technological capacity."),
                  dict(facts[0], id="PRIVATE-F3", quote="Technological development.")]
        first = build_dossiers(facts, dictionary, actors, config, max_per_cell=1)
        second = build_dossiers(list(reversed(facts)), dictionary, actors, config, max_per_cell=1)
        self.assertEqual(first["private_cells"], second["private_cells"])
        cell = next(c for c in first["private_cells"] if c["actor_id"] == "a" and c["scale_id"] == "m01")
        self.assertEqual(len(cell["candidates"]), 3)
        self.assertEqual(len(cell["shortlist"]), 1)

    def test_query_validation_and_dictionary_approval_are_required(self):
        facts, dictionary, actors, config = fixture()
        with self.assertRaises(ValueError):
            build_dossiers(facts, dict(dictionary, approved=False), actors, config)
        with self.assertRaises(ValueError):
            build_dossiers(facts, dictionary, actors, config, cues={"unapproved": "anything"})
        with self.assertRaises(ValueError):
            build_dossiers(facts + facts, dictionary, actors, config)


if __name__ == "__main__":
    unittest.main()
