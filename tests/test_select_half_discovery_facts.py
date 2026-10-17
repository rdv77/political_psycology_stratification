"""Synthetic offline selection only; no private corpus, queue or model client."""
from collections import Counter
import copy
import random
import unittest

from scripts.select_half_discovery_facts import SEED, _digest, select_facts


def fact(identifier, actor, document=None, quote=None, **kwargs):
    value = {"id": identifier, "actor_id": actor, "document_id": document or "doc-" + identifier,
             "quote": quote or "Literal quote " + identifier, "kind": "public_statement",
             "attribution_basis": "direct_statement", "event_date": "2026-01-01", "topic": "education",
             "source_url": "https://example.test/" + identifier}
    value.update(kwargs)
    return value


def snapshot(facts, actors=("a", "b", "c")):
    return {"corpus": {"config": {"objects": [{"id": actor, "name": "Approved " + actor} for actor in actors],
                                  "aspect": "public_positions", "date_from": "2021-10-15", "date_to": "2026-10-17"}},
            "facts": facts}


class DiscoverySelectionTests(unittest.TestCase):
    def test_rare_actors_first_round_robin_is_balanced_and_order_independent(self):
        rows = [fact("a" + str(i), "a") for i in range(12)] + [fact("b" + str(i), "b") for i in range(9)] + [fact("c0", "c")]
        original = snapshot(rows)
        before = copy.deepcopy(original)
        selected, audit = select_facts(original, target=8)
        self.assertEqual(selected[0], "c0")
        counts = {row["actor_id"]: row["selected_facts"] for row in audit["actor_coverage"]}
        self.assertEqual(counts["c"], 1)
        self.assertLessEqual(abs(counts["a"] - counts["b"]), 1)
        self.assertEqual(original, before)
        shuffled = copy.deepcopy(rows)
        random.Random(42).shuffle(shuffled)
        again, audit_again = select_facts(snapshot(shuffled), target=8)
        self.assertEqual(selected, again)
        self.assertEqual(audit["facts_snapshot_sha256"], audit_again["facts_snapshot_sha256"])

    def test_same_actor_literal_quote_dedup_preserves_case_negation_and_other_actors(self):
        rows = [fact("a1", "a", quote="We cooperate."), fact("a2", "a", quote="We cooperate."),
                fact("a3", "a", quote="we cooperate."), fact("a4", "a", quote="We do not cooperate."),
                fact("b1", "b", quote="We cooperate.")]
        selected, audit = select_facts(snapshot(rows), target=10)
        self.assertEqual(len(selected), 4)
        self.assertEqual(audit["counts"]["literal_duplicate_records"], 1)
        self.assertIn("b1", selected)
        duplicates = [row for row in audit["selection_index"] if row["reason"] == "literal_same_actor_quote_duplicate"]
        self.assertEqual(len(duplicates), 1)
        self.assertIn(duplicates[0]["representative_fact_id"], selected)
        self.assertFalse(audit["source_records_changed"])

    def test_global_document_cap_produces_explicit_shortfall_without_relaxation(self):
        rows = [fact("a" + str(i), "a", document="shared") for i in range(4)] + [fact("b1", "b", document="shared")]
        selected, audit = select_facts(snapshot(rows), target=5)
        self.assertEqual(len(selected), 2)
        self.assertIn("b1", selected)  # The rare actor receives its turn first.
        self.assertEqual(audit["counts"]["shortfall"], 3)
        self.assertEqual(audit["counts"]["actual_max_facts_per_document"], 2)
        self.assertFalse(audit["capacity_relaxed"])
        self.assertTrue(audit["capacity_maximum_reached_when_short"])

    def test_alternate_quote_representative_can_free_document_instead_of_false_shortfall(self):
        # Force A's initial SHA preference to the shared document, then B needs
        # that sole slot. A's exact-quote representative must move to its spare.
        ids = sorted(("a1", "a2"), key=lambda identifier: _digest([SEED, identifier]))
        shared, spare = ids
        rows = [fact(shared, "a", document="shared", quote="Same literal A quote"),
                fact(spare, "a", document="spare", quote="Same literal A quote"),
                fact("b1", "b", document="shared", quote="Distinct B quote")]
        selected, audit = select_facts(snapshot(rows, ("a", "b")), target=2, max_per_document=1)
        self.assertEqual(set(selected), {spare, "b1"})
        self.assertTrue(audit["target_reached"])
        self.assertEqual(audit["counts"]["representative_swaps"], 1)
        self.assertEqual(audit["counts"]["actual_max_facts_per_document"], 1)

    def test_source_and_literal_topic_diversity_are_used_without_semantic_clusters(self):
        rows = [fact("x1", "a", topic="Education", source_url="https://one.test/x"),
                fact("x2", "a", topic="education", source_url="https://one.test/y"),
                fact("y1", "a", topic="Health", source_url="https://two.test/z")]
        selected, audit = select_facts(snapshot(rows), target=2)
        self.assertIn("y1", selected)
        self.assertEqual(audit["actor_coverage"][0]["selected_source_hosts"], 2)
        self.assertEqual(audit["actor_coverage"][0]["selected_literal_topics"], 2)

    def test_measurement_eligibility_and_roster_are_enforced_without_external_inference(self):
        rows = [fact("good", "a"), fact("image", "a", kind="attributed_image", attribution_basis="third_party_claim"),
                fact("old", "b", event_date="2019-01-01"), fact("excluded", "b", review_status="rejected"),
                fact("alien", "alien"), fact("context", "b", measurement_status="context_only")]
        selected, audit = select_facts(snapshot(rows), target=600)
        self.assertEqual(selected, ["good"])
        self.assertEqual(audit["counts"]["excluded_by_rule"], {"measurement_ineligible": 4, "unapproved_actor": 1})
        self.assertEqual(len(audit["actor_coverage"]), 3)
        self.assertEqual(audit["actor_coverage"][2]["selected_facts"], 0)
        self.assertEqual(audit["api_calls"], 0)
        self.assertFalse(audit["queue_changed"])

    def test_conflicting_ids_and_implicit_cap_relaxation_are_rejected(self):
        with self.assertRaises(ValueError):
            select_facts(snapshot([fact("same", "a"), fact("same", "b")]))
        with self.assertRaises(ValueError):
            select_facts(snapshot([]), max_per_document=3)

    def test_default_600_is_balanced_over_all_42_and_ignores_old_scores(self):
        actors = tuple("actor-" + str(i) for i in range(42))
        rows = [fact(actor + "-" + str(i), actor) for actor in actors for i in range(20)]
        selected, audit = select_facts(snapshot(rows, actors))
        self.assertEqual(len(selected), 600)
        self.assertEqual(audit["counts"]["actors_selected"], 42)
        counts = [row["selected_facts"] for row in audit["actor_coverage"]]
        self.assertLessEqual(max(counts) - min(counts), 1)
        changed = copy.deepcopy(rows)
        for row in changed:
            row.update(old_factor_score=99, relation_to_russia="untrusted previous label")
        again, _ = select_facts(snapshot(changed, actors))
        self.assertEqual(selected, again)

    def test_multistep_representative_swap_keeps_cap_and_all_selected_actors(self):
        aa = sorted(("a1", "a2"), key=lambda identifier: _digest([SEED, identifier]))
        bb = sorted(("b1", "b2"), key=lambda identifier: _digest([SEED, identifier]))
        rows = [fact(aa[0], "a", document="d1", quote="A quote"),
                fact(aa[1], "a", document="d2", quote="A quote"),
                fact(bb[0], "b", document="d2", quote="B quote"),
                fact(bb[1], "b", document="d3", quote="B quote"),
                fact("c1", "c", document="d1", quote="C quote")]
        selected, audit = select_facts(snapshot(rows), target=3, max_per_document=1)
        self.assertEqual(set(selected), {aa[1], bb[1], "c1"})
        self.assertEqual(audit["counts"]["actual_max_facts_per_document"], 1)
        self.assertEqual(audit["counts"]["representative_swaps"], 2)
        self.assertEqual(audit["shortfall_reasons"], [])


if __name__ == "__main__":
    unittest.main()
