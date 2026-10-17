"""Synthetic snapshots only; no API, private corpus, queue or publication."""
import copy
import json
import unittest

from scripts.summarize_half_processing import render, summarize


def job(identifier, stage="extract", status="done", active=True, document="doc-private", **kwargs):
    value = {"id": identifier, "stage": stage, "status": status, "active_recipe": active,
             "payload": {"model": "gpt-5.1", "private_payload": "SECRET_PAYLOAD"},
             "context": {"document": {"id": document, "source_type": "article", "author": "PRIVATE_AUTHOR",
                                      "text": "SECRET_FULL_TEXT", "url": "https://private.example/SECRET_URL"},
                         "relevance": {"language": "fr"}},
             "result": {"status": "relevant"}, "input_tokens": 100, "output_tokens": 10,
             "cost_usd": .1, "reserved_usd": 2.}
    value.update(kwargs)
    return value


def fact(identifier="fact-private", actor="a", **kwargs):
    value = {"id": identifier, "actor_id": actor, "extraction_job_id": "extract-1", "document_id": "doc-private",
             "kind": "public_statement", "attribution_basis": "direct_statement", "source_url": "https://private.example/SECRET_URL",
             "event_date": None, "quote": "SECRET_QUOTE", "claim": "SECRET_CLAIM", "trait_candidates": ["Cooperation", "cooperation"]}
    value.update(kwargs)
    return value


class HalfProcessingSummaryTests(unittest.TestCase):
    def setUp(self):
        self.snapshot = {"manifest": {"selected_count": 12205, "budget_cap_usd": 70., "start_spent_usd": 0.,
                                      "source_sha256": "a" * 64},
            "corpus": {"id": "synthetic", "config": {"objects": [{"id": "a", "name": "Approved Actor A"},
                         {"id": "b", "name": "Approved Actor B"}], "aspect": "public_positions",
                         "date_from": "2021-10-15", "date_to": "2026-10-17"}, "draft_scales": []},
            "jobs": [job("extract-1"), job("rel-1", stage="relevance")], "facts": [fact()]}

    def test_actor_cards_include_no_evidence_and_counts_are_not_intensity(self):
        self.snapshot["facts"] += [fact("f2", trait_candidates=["COOPERATION"], kind="attributed_image", attribution_basis="third_party_claim")]
        before = copy.deepcopy(self.snapshot)
        result = summarize(self.snapshot)
        self.assertEqual([card["actor_id"] for card in result["actor_cards"]], ["a", "b"])
        first, empty = result["actor_cards"]
        self.assertEqual((first["facts"], first["eligible_public_positions"]), (2, 1))
        self.assertEqual(first["trait_candidates"][0]["fact_count"], 2)  # Repeated alias inside one fact is counted once.
        self.assertEqual(first["trait_candidates"][0]["eligible_public_position_fact_count"], 1)
        self.assertEqual(first["source_coverage"]["eligible_public_position_documents"], 1)
        self.assertEqual(empty["facts"], 0)
        self.assertEqual(empty["trait_candidates"], [])
        self.assertNotIn("value", json.dumps(result))
        self.assertEqual(self.snapshot, before)

    def test_all_costs_include_review_errors_obsolete_and_uncertain_reservations(self):
        self.snapshot["jobs"] += [job("bad", status="review", cost_usd=.2),
            job("old", stage="describe", active=False, cost_usd=.3),
            job("unknown", status="uncertain", cost_usd=0., reserved_usd=1.5),
            job("error", status="error", cost_usd=.4)]
        result = summarize(self.snapshot)
        account = result["accounting"]
        self.assertEqual(account["all_receipts"]["total"]["estimated_usd"], 1.1)
        self.assertEqual(account["active_recipes"]["total"]["estimated_usd"], .8)
        self.assertEqual(account["all_receipts"]["total"]["outstanding_reserved_usd"], 1.5)
        self.assertEqual(account["uncommitted_remaining_usd"], 67.4)
        self.assertEqual(account["unresolved_requests"], 1)
        self.assertEqual(account["all_receipts"]["total"]["input_tokens"], 600)
        self.assertTrue(any(row["status"] == "review" for row in account["all_receipts"]["by_stage_model_status"]))

    def test_context_and_rejected_traits_do_not_reappear_as_properties(self):
        self.snapshot["facts"] += [
            fact("context", kind="institutional_context", trait_candidates=["officeholder"]),
            fact("old", measurement_status="context_only", trait_candidates=["historic_inference"]),
            fact("wrong-person", review_status="rejected", trait_candidates=["wrong_identity_property"])]
        result = summarize(self.snapshot)
        self.assertEqual(result["facts"]["active"], 4)  # Audit history remains visible.
        self.assertEqual(result["facts"]["eligible_public_positions"], 1)
        self.assertEqual([row["label"] for row in result["trait_candidates"]], ["Cooperation"])
        self.assertEqual(result["public_safety"]["context_or_rejected_trait_occurrences_omitted"], 3)
        for label in ("officeholder", "historic_inference", "wrong_identity_property"):
            self.assertNotIn(label, render(result))

    def test_partial_statuses_are_visible_and_never_claim_full_12205_completion(self):
        self.snapshot["jobs"] += [job("rel-2", stage="relevance", status="review", document="doc-2"),
                                  job("rel-3", stage="relevance", status="pending", document="doc-3")]
        result = summarize(self.snapshot)
        coverage = result["coverage"]
        self.assertEqual(coverage["selected_documents"], 12205)
        self.assertEqual(coverage["relevance"]["documents_not_prepared"], 12202)
        self.assertEqual(coverage["relevance"]["review_documents"], 1)
        self.assertEqual(coverage["relevance"]["pending_documents"], 1)
        self.assertFalse(coverage["complete_classification"])
        self.assertFalse(coverage["complete_extraction_of_relevant"])
        self.assertIn("не нулевая", render(result))

    def test_only_literal_declared_aliases_group_and_definitions_do_not_automerge(self):
        def scale(name, aliases, definition="A definition"):
            return {"name": name, "aliases": aliases, "definition": definition, "evidence_ids": ["fact-private"]}
        self.snapshot["corpus"]["draft_scales"] = [scale("Diplomatic cooperation", ["Cooperation"]),
            scale("COOPERATION", [], "Different definition"), scale("Not cooperation", []),
            scale("Education", []), scale("Training", [])]
        groups = summarize(self.snapshot)["draft_dictionary"]
        self.assertEqual(len(groups), 4)
        grouped = next(group for group in groups if group["draft_count"] == 2)
        self.assertTrue(grouped["definitions_differ"])
        self.assertFalse(grouped["approved"])
        self.assertEqual(grouped["supporting_facts"], 1)

    def test_public_whitelist_drops_sources_payloads_contacts_and_raw_ids(self):
        self.snapshot["facts"][0]["trait_candidates"] += ["contact person@example.test", "https://bad.example/path", "Call +12 345 678 901", "x" * 161]
        result = summarize(self.snapshot)
        public = json.dumps(result, ensure_ascii=False) + render(result)
        for private in ("SECRET_", "PRIVATE_AUTHOR", "doc-private", "fact-private", "private.example", "person@example.test", "+12 345"):
            self.assertNotIn(private, public)
        self.assertEqual(result["public_safety"]["unsafe_candidate_labels_omitted"], 4)
        self.assertEqual(result["actor_cards"][0]["source_coverage"]["source_hosts"], 1)

    def test_draft_loses_public_property_status_when_its_only_evidence_is_withdrawn(self):
        self.snapshot["corpus"]["draft_scales"] = [{"name": "Unsupported property", "aliases": [],
            "definition": "An unapproved proposal", "evidence_ids": ["fact-private"]}]
        self.snapshot["facts"][0]["measurement_status"] = "context_only"
        result = summarize(self.snapshot)
        self.assertEqual(result["draft_dictionary"], [])
        self.assertEqual(result["public_safety"]["drafts_without_eligible_evidence_omitted"], 1)
        self.assertNotIn("Unsupported property", render(result))

    def test_untrusted_labels_are_plain_markdown_text(self):
        self.snapshot["facts"][0]["trait_candidates"] = ["[Label](made-up-scheme:target)", "Invisible\u200bLabel"]
        result = summarize(self.snapshot)
        self.assertEqual(result["public_safety"]["unsafe_candidate_labels_omitted"], 1)
        self.assertIn("\\[Label\\]", render(result))

    def test_inactive_extraction_facts_are_not_counted_and_shared_job_is_not_billed_twice(self):
        self.snapshot["jobs"].append(copy.deepcopy(self.snapshot["jobs"][0]))
        self.snapshot["facts"] += [fact("f-old", extraction_job_id="inactive"), fact("f-alien", actor="alien")]
        result = summarize(self.snapshot)
        self.assertEqual(result["accounting"]["all_receipts"]["total"]["jobs"], 2)
        self.assertEqual(result["facts"]["active"], 1)
        self.assertEqual(result["facts"]["skipped_snapshot_rows"], {"inactive_or_incomplete_extraction": 1, "unapproved_actor": 1})

    def test_all_42_names_and_order_are_present_even_before_any_jobs(self):
        self.snapshot["corpus"]["config"]["objects"] = [{"id": str(i), "name": "Approved Actor " + str(i)} for i in range(42)]
        self.snapshot["jobs"], self.snapshot["facts"] = [], []
        result = summarize(self.snapshot)
        self.assertEqual(len(result["actor_cards"]), 42)
        self.assertEqual([card["actor_id"] for card in result["actor_cards"]], [str(i) for i in range(42)])
        self.assertEqual(render(result).count("### Approved Actor "), 42)
        self.assertEqual(result["warnings"], [])

    def audit(self):
        return {"audit_policy_version": "offline-fact-and-negative-review-v1", "corpus_id": "synthetic", "source_sha256": "a" * 64,
                "scope": {"selected_document_frame": 12205, "unfinished_active_jobs": 0, "snapshot_is_partial": True},
                "structural_census": {"active_accepted_facts_checked": 1, "facts_with_recheck_issues": 0, "jobs_with_qc": 1,
                    "original_normalization_totals": {"input_facts": 3, "skipped_facts": 1, "downgraded_facts": 1, "cleared_dates": 1},
                    "original_normalization_issue_counts": {"uncertain_date": 1}},
                "semantic_review": {"status": "pending_human_or_assistant_source_review", "facts_selected": 1,
                    "risk_packet_facts": 1, "negative_packet_documents": 2, "accuracy_estimate": None,
                    "note": "UNTRUSTED_PROSE_SHOULD_NOT_COPY"}}

    def test_quality_defaults_to_unchecked_and_whitelists_only_public_audit_counters(self):
        unchecked = summarize(self.snapshot)
        self.assertFalse(unchecked["quality_review"]["available"])
        self.assertIn("Содержательная точность не установлена", render(unchecked))
        result = summarize(self.snapshot, self.audit())
        quality = result["quality_review"]
        self.assertEqual(quality["structural_checks"]["active_accepted_facts_checked"], 1)
        self.assertEqual(quality["normalization"]["totals"]["cleared_dates"], 1)
        self.assertIsNone(quality["semantic_review"]["accuracy_estimate"])
        self.assertEqual(quality["semantic_review"]["status"], "pending_source_review")
        self.assertNotIn("UNTRUSTED_PROSE", json.dumps(result))
        self.assertIn("создание пакета не означает его содержательной проверки", render(result))

    def test_quality_rejects_private_wrapper_and_wrong_source_but_flags_stale_fact_counts(self):
        for audit in ({"public_aggregate": self.audit(), "semantic_review_packet": []},
                      dict(self.audit(), source_sha256="c" * 64)):
            with self.assertRaises(ValueError):
                summarize(self.snapshot, audit)
        audit = self.audit()
        audit["structural_census"]["active_accepted_facts_checked"] = 0
        self.assertFalse(summarize(self.snapshot, audit)["quality_review"]["snapshot_alignment"]["fact_count_matches"])


if __name__ == "__main__":
    unittest.main()
