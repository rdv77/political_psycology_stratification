from copy import deepcopy
import json
import unittest

import numpy as np

from src.corpus_pipeline.text_structure import (
    analyze, compare_structures, fit_structure, prepare_units,
)


def roster():
    return [{"id": f"person-{i}", "name": f"Knownname{i} Familyname{i}",
             "name_original": f"Given{i} Surname{i}"} for i in range(8)] + [
                 {"id": "missing-person", "name": "Missing Person"}]


def fact(actor="person-0", index=0, claim="Supports digital education and healthcare technology.", **changes):
    item = {"id": f"private-fact-{actor}-{index}", "actor_id": actor,
            "kind": "public_statement", "attribution_basis": "direct_statement",
            "quote": f"Original source wording {actor} {index}",
            "claim": claim, "event_date": None, "document_id": f"private-document-{actor}-{index}",
            "trait_candidates": ["digital_innovation"], "correction_flags": []}
    item.update(changes)
    return item


def corpus():
    topics = ["digital technology innovation healthcare education",
              "judicial rights legal court constitution",
              "regional diplomatic military peace security",
              "climate economic financing investment adaptation"]
    result = []
    for i in range(8):
        for j in range(8):
            topic = topics[(i // 2 + j // 5) % len(topics)]
            result.append(fact(f"person-{i}", j, f"Given{i} Surname{i} says Rwanda supports {topic}.",
                               trait_candidates=[topic.replace(" ", "_")]))
    return result


class TextStructureTests(unittest.TestCase):
    def test_literal_duplicates_do_not_create_extra_model_units(self):
        source = fact()
        repeated = {**source, "id": "private-other-id", "document_id": "another-source"}
        result = prepare_units([source, repeated], roster())
        self.assertEqual(result["eligible_fact_count"], 2)
        self.assertEqual(result["literal_group_count"], 1)
        self.assertEqual(result["literal_repeat_excess"], 1)
        self.assertFalse(result["units"][0]["is_verified_independent_event"])

    def test_known_and_unknown_times_and_distinct_dates_remain_separate(self):
        source = fact()
        result = prepare_units([source, {**source, "event_date": "2026-01-01"},
                                {**source, "event_date": "2026-02-01"}], roster())
        self.assertEqual(result["literal_group_count"], 3)
        self.assertEqual(sum(u["has_known_event_time"] for u in result["units"]), 2)

    def test_nonown_rejected_and_outside_window_facts_are_not_text_inputs(self):
        data = [fact(), fact(index=1, kind="attributed_image", attribution_basis="third_party_claim"),
                fact(index=2, review_status="rejected"), fact(index=3, event_date="2018-01-01")]
        result = prepare_units(data, roster(), config={"aspect": "public_positions", "date_from": "2021-10-15"})
        self.assertEqual(result["eligible_fact_count"], 1)
        self.assertEqual(result["rejected"]["not_eligible_own_position_candidate"], 3)

    def test_names_aliases_and_reporting_formulae_are_removed_before_features(self):
        actors = [{"id": "alice-verdant", "name": "Alice Verdant", "name_original": "Alice Verdant"}]
        aliases = {"actors": [{"id": "alice-verdant", "aliases": [{"text": "Ally Verdante"}]}]}
        source = fact("alice-verdant", claim="President Alice Verdant, also Ally Verdante, said Rwanda supports Russian medical cooperation.",
                      trait_candidates=[])
        result = prepare_units([source], actors, aliases=aliases)
        self.assertEqual(result["units"][0]["text"], "russian medical cooperation")

    def test_all_roster_cards_retained_and_missing_material_is_unknown(self):
        result = analyze(corpus(), roster())
        cards = result["public"]["baseline"]["actors"]
        missing = next(c for c in cards if c["actor_id"] == "missing-person")
        self.assertEqual(len(cards), 9)
        self.assertIsNone(missing["text_coordinates"])
        self.assertEqual(missing["status"], "no_eligible_text_material")
        self.assertFalse(result["public"]["measurement_contract"]["approved_v2_ratings_generated"])
        self.assertFalse(result["public"]["baseline"]["factor_analysis_of_approved_scales"])

    def test_actor_profiles_have_equal_norm_and_variance_has_the_stated_denominator(self):
        prepared = prepare_units(corpus(), roster())
        result = fit_structure(prepared["units"], roster())
        matrix = np.asarray(result["private"]["actor_term_matrix"])
        scores = np.asarray(result["private"]["scores"])
        np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-12)
        ratio = np.var(scores[:, :3], axis=0).sum() / np.var(matrix, axis=0).sum()
        model = next(m for m in result["public"]["solutions"] if m["q"] == 3)
        self.assertAlmostEqual(model["centered_text_variance_share"], ratio, places=12)
        energy = np.square(scores[:, :3]).sum() / np.square(matrix).sum()
        self.assertAlmostEqual(model["uncentered_text_energy_share"], energy, places=12)

    def test_repeated_identical_sources_leave_profile_coordinates_unchanged(self):
        original = corpus()
        repeated = deepcopy(original)
        repeated.extend({**item, "id": item["id"] + "-reprint"} for item in original[:16])
        left = fit_structure(prepare_units(original, roster())["units"], roster())
        right = fit_structure(prepare_units(repeated, roster())["units"], roster())
        np.testing.assert_allclose(left["private"]["scores"], right["private"]["scores"], atol=1e-12)

    def test_public_outputs_have_no_claims_quotes_or_internal_source_identifiers(self):
        data = corpus()
        result = analyze(data, roster())
        public = json.dumps(result["public"], ensure_ascii=False)
        for source in data:
            self.assertNotIn(source["id"], public)
            self.assertNotIn(source["document_id"], public)
            self.assertNotIn(source["quote"], public)
            self.assertNotIn(source["claim"], public)
        self.assertEqual(result["public"]["api_calls"], 0)
        self.assertEqual(result["public"]["new_api_cost_usd"], 0)

    def test_component_comparison_is_exact_for_the_same_fit(self):
        fitted = fit_structure(prepare_units(corpus(), roster())["units"], roster())
        comparison = compare_structures(fitted, fitted)
        for match in comparison["matched_text_components"]:
            self.assertAlmostEqual(match["absolute_shared_vocabulary_cosine"], 1, places=12)
        for model in comparison["pairwise_distance_comparisons"]:
            self.assertAlmostEqual(model["shared_actor_pairwise_distance_spearman"], 1, places=12)

    def test_original_facts_are_not_mutated(self):
        data = corpus()
        frozen = deepcopy(data)
        analyze(data, roster())
        self.assertEqual(data, frozen)

    def test_claim_only_sensitivity_does_not_use_trait_labels(self):
        source = fact(trait_candidates=["irrelevant_secret_label"])
        default = prepare_units([source], roster())["units"][0]["text"]
        claims = prepare_units([source], roster(), include_trait_candidates=False)["units"][0]["text"]
        self.assertIn("irrelevant secret label", default)
        self.assertNotIn("irrelevant", claims)


if __name__ == "__main__":
    unittest.main()
