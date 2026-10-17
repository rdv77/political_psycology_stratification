import copy
import json
import unittest

from scripts.summarize_half_quality import aggregate_quality, render_quality


class HalfQualitySummaryTests(unittest.TestCase):
    def fixture(self):
        seed = {"findings": [{"fact_id": "PRIVATE_FACT" + str(i), "document_id": "PRIVATE_DOC" + str(i),
            "actor_id": "PRIVATE_ACTOR" + str(i % 3), "language": "en", "judgment": "supported_recorded_channel",
            "source_support": "supported_as_source_assertion", "failure_codes": [],
            "original_fact": {"kind": "public_statement"}, "source_snippet": "PRIVATE_SOURCE", "source_url": "https://PRIVATE.invalid"}
            for i in range(80)]}
        seed["findings"][0]["proposed_annotation"] = {"measurement_status": "context_only", "review_status": "reviewed", "review_note": "PRIVATE_NOTE"}
        agency = {"findings": [{"fact_id": "PRIVATE_AGENCY" + str(i), "document_id": "PRIVATE_AGENCY_DOC" + str(i),
            "actor_id": "PRIVATE_ACTOR" + str(i % 2), "decision": "context_only" if i == 0 else "keep", "reason": "PRIVATE_REASON"}
            for i in range(20)], "review_facts_decisions": [{"fact_id": "PRIVATE_AGENCY0", "annotation": {"kind": "institutional_context", "review_note": "PRIVATE_NOTE"}}]}
        classification = {"findings": [{"document_id": "PRIVATE_CLASS_DOC", "original_decision": {"status": "review", "language": "fr", "actor_ids": ["PRIVATE_ACTOR"]},
            "judgment": {"status": "conservative_hold_of_useful_material", "identity": "supported", "language": "supported_for_supplied_passages"}, "reason": "PRIVATE_REASON"}]}
        return seed, agency, classification

    def test_exact_union_100_schema_categories_separate_and_input_immutable(self):
        args = self.fixture()
        original = copy.deepcopy(args)
        result = aggregate_quality(*args)
        self.assertTrue(result["fact_review_union"]["complete_expected_review"])
        self.assertEqual(result["fact_review_union"]["unique_facts"], 100)
        self.assertEqual(result["agency_fact_review"]["decision_counts"], {"context_only": 1, "keep": 19})
        self.assertEqual(result["classification_review"]["status_judgment_counts"], {"conservative_hold_of_useful_material": 1})
        self.assertIsNone(result["annotations"]["actual_applied"])
        self.assertIsNone(result["negative_document_review"])
        self.assertEqual(args, original)

    def test_overlap_and_partial_review_never_claims_100_or_accuracy(self):
        seed, agency, classification = self.fixture()
        agency["findings"][0]["fact_id"] = seed["findings"][0]["fact_id"]
        result = aggregate_quality(seed, agency, classification)
        self.assertEqual(result["fact_review_union"]["unique_facts"], 99)
        self.assertEqual(result["fact_review_union"]["overlap_facts"], 1)
        self.assertFalse(result["fact_review_union"]["complete_expected_review"])
        self.assertIsNone(result["population_accuracy_estimate"])
        self.assertIn("не соответствует", render_quality(result))
        seed["findings"].append(copy.deepcopy(seed["findings"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate fact ID"):
            aggregate_quality(seed, agency, classification)

    def test_annotations_proposed_and_actual_distinct_and_outside_review(self):
        args = self.fixture()
        applied = [{"fact_id": "PRIVATE_FACT0", "data": json.dumps({"measurement_status": "context_only", "review_note": "different note"})},
                   {"fact_id": "PRIVATE_FACT0", "data": {"review_status": "reviewed"}},
                   {"fact_id": "PRIVATE_OUTSIDE", "data": {"measurement_status": "context_only"}}]
        result = aggregate_quality(*args, applied_reviews=applied)
        actual = result["annotations"]["actual_applied"]
        self.assertEqual(result["annotations"]["proposed_distinct_facts"], 2)
        self.assertEqual(actual["annotation_rows"], 3)
        self.assertEqual(actual["distinct_annotated_facts"], 2)
        self.assertEqual(actual["annotated_facts_outside_review_union"], 1)
        self.assertEqual(actual["proposal_semantically_matched_facts"], 1)
        self.assertEqual(actual["proposed_not_matched_facts"], 1)
        self.assertEqual(actual["proposed_with_any_annotation_facts"], 1)
        self.assertEqual(actual["proposed_without_annotation_facts"], 1)
        applied.append({"fact_id": "PRIVATE_FACT0", "data": {"measurement_status": "unreviewed"}})
        current = aggregate_quality(*args, applied_reviews=applied)["annotations"]["actual_applied"]
        self.assertEqual(current["proposal_semantically_matched_facts"], 0)
        self.assertEqual(current["proposed_with_different_annotation_facts"], 1)
        args[0]["findings"][1]["proposed_annotation"] = {"review_note": "PRIVATE_NOTE_ONLY"}
        applied.append({"fact_id": "PRIVATE_FACT1", "data": {"review_note": "different text remains note-only"}})
        with_note = aggregate_quality(*args, applied_reviews=applied)
        self.assertEqual(with_note["annotations"]["proposed_note_only_facts"], 1)
        self.assertEqual(with_note["annotations"]["actual_applied"]["proposal_semantically_matched_facts"], 1)

    def test_strict_whitelist_no_free_text_identifiers_paths_or_secret_categories(self):
        seed, agency, classification = self.fixture()
        seed["findings"][0].update(judgment="PRIVATE_SECRET", language="https://PRIVATE.invalid", failure_codes=["PRIVATE_KEY"])
        classification["findings"][0]["judgment"]["identity"] = "C:/PRIVATE/path"
        receipts = {"receipt_lines": 1, "receipt_priced_cost_usd_known": .5, "receipt_recomputed_cost_usd": None,
            "fully_priced_matched_cost_difference_usd": 0., "issue_counts": {"receipt_usage_unknown": 1, "PRIVATE_SECRET": 1},
            "private": "PRIVATE_SOURCE", "request_sha256": "PRIVATE_HASH", "response_id": "PRIVATE_RESPONSE"}
        result = aggregate_quality(seed, agency, classification, receipt_reconciliation=receipts)
        public = json.dumps(result, ensure_ascii=False) + render_quality(result)
        for forbidden in ("PRIVATE", "https://", "C:/", "request_sha256", "response_id"):
            self.assertNotIn(forbidden, public)
        self.assertEqual(result["seed_fact_review"]["judgment_counts"]["unrecognized"], 1)
        self.assertIsNone(result["receipt_reconciliation"]["receipt_recomputed_cost_usd"])

    def test_negative_schema_preserves_barriers_and_partial_reading_extent(self):
        rows = []
        for index in range(20):
            status = "supported_irrelevant" if index < 12 else "useful_material_held_by_identity_guard" if index < 19 else "uncertain_language_ocr_overfilter"
            rows.append({"document_id": "PRIVATE_NEGATIVE" + str(index), "category": "irrelevant",
                "original_decision": {"status": "irrelevant", "language": "en", "actor_ids": []},
                "judgment": {"status": status, "identity": "supported_actor_in_full_source", "language": "supported"},
                "reading_mode": "full_supplied_source_passages" if index < 17 else "targeted_head_tail_and_actor_windows_plus_lexical_census"})
        result = aggregate_quality(*self.fixture(), negative={"findings": rows})
        summary = result["negative_document_review"]
        self.assertEqual(summary["targeted_summary_groups"], {"supported_conservative": 12, "potential_useful_material_loss_or_barrier": 7, "uncertain": 1})
        self.assertEqual(summary["distinct_documents"], 20)
        self.assertEqual(summary["reading_mode_counts"]["full_supplied_source_passages"], 17)
        self.assertEqual(summary["reading_mode_counts"]["targeted_head_tail_and_actor_windows_plus_lexical_census"], 3)
        self.assertIn("не означают полный перевод", render_quality(result))


if __name__ == "__main__":
    unittest.main()
