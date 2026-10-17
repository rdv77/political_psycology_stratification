"""Synthetic attribution/date regressions; no source corpus or API access."""
import copy
import unittest

from src.corpus_pipeline.evidence import EVIDENCE_POLICY_VERSION, normalize_extraction
from src.corpus_pipeline.models import validate_result


class CorpusEvidenceTests(unittest.TestCase):
    def context(self, text, source_type="article"):
        return {"document": {"text": text, "source_type": source_type,
                             "author_is_verified_actor": False, "published_at": "2026-10-17"},
                "config": {"objects": [{"id": "alpha", "name": "Alpha North", "aliases": ["Альфа Норт"]}],
                           "date_from": "2021-10-15", "date_to": "2026-10-17"},
                "relevance": {"actor_ids": ["alpha"]}}

    def fact(self, quote="Alpha North said: we support the proposal.", **changes):
        fact = {"actor_id": "alpha", "quote": quote, "claim": "Supports the proposal",
                "kind": "public_statement", "event_date": None, "topic": "policy",
                "trait_candidates": ["cooperation"], "capacity": "official",
                "attribution_basis": "direct_statement", "attribution_evidence": quote,
                "date_evidence": None}
        fact.update(changes)
        return fact

    def normalize(self, facts, context):
        return normalize_extraction({"facts": facts}, context, validate_result)

    def test_direct_reported_statements_and_reported_action(self):
        for basis, kind in [("direct_statement", "public_statement"), ("reported_statement", "public_statement"),
                            ("reported_action", "action")]:
            with self.subTest(basis=basis):
                fact = self.fact(attribution_basis=basis, kind=kind)
                result = self.normalize([fact], self.context(fact["quote"]))
                self.assertEqual(result["facts"], [fact])
                self.assertEqual(result["quality_control"]["issues"], [])

    def test_comment_repost_and_uncertain_are_never_own(self):
        for source_type, basis in [("comment", "direct_statement"), ("repost", "reported_action"),
                                    ("article", "third_party_claim"), ("article", "uncertain")]:
            for kind in ("action", "public_statement"):
                with self.subTest(source=source_type, basis=basis, kind=kind):
                    fact = self.fact(kind=kind, attribution_basis=basis)
                    context = self.context(fact["quote"], source_type)
                    context["document"]["author_is_verified_actor"] = True
                    result = self.normalize([fact], context)
                    self.assertEqual(result["facts"][0]["kind"], "attributed_image")
                    self.assertEqual(result["facts"][0]["capacity"], "third-party")
                    self.assertEqual(result["quality_control"]["downgraded_facts"], 1)

    def test_context_does_not_become_personal_position(self):
        fact = self.fact(attribution_basis="context", kind="action")
        result = self.normalize([fact], self.context(fact["quote"]))
        self.assertEqual(result["facts"][0]["kind"], "institutional_context")

    def test_invalid_fact_does_not_discard_valid_siblings(self):
        good = self.fact()
        bad = self.fact(quote="Invented quote", attribution_evidence=good["quote"])
        result = self.normalize([bad, good], self.context(good["quote"]))
        self.assertEqual(result["facts"], [good])
        self.assertEqual(result["quality_control"]["input_facts"], 2)
        self.assertEqual(result["quality_control"]["skipped_facts"], 1)
        self.assertEqual(result["quality_control"]["issues"], [{"index": 0, "code": "quote_not_exact"}])

    def test_actor_in_neighboring_attribution_can_ground_pronoun_quote(self):
        text = "Alpha North responded. He supports this plan."
        fact = self.fact(quote="He supports this plan.", attribution_evidence="Alpha North responded.")
        self.assertEqual(len(self.normalize([fact], self.context(text))["facts"]), 1)

    def test_alias_unicode_case_and_word_boundaries(self):
        quote = "АЛЬФА НОРТ поддерживает проект."
        self.assertEqual(len(self.normalize([self.fact(quote)], self.context(quote))["facts"]), 1)
        quote = "SuperAlpha North supports the proposal."
        self.assertEqual(self.normalize([self.fact(quote)], self.context(quote))["facts"], [])

    def test_other_actor_or_distant_name_does_not_ground_fact(self):
        quote = "Someone supports this plan."
        for attribution, text in [(quote, "Alpha North is mentioned. " + quote),
                                   ("Alpha North said:", "Alpha North said:" + " gap" * 100 + quote)]:
            with self.subTest(attribution=attribution):
                fact = self.fact(quote, attribution_evidence=attribution)
                self.assertEqual(self.normalize([fact], self.context(text))["facts"], [])

    def test_exact_attribution_evidence_is_required(self):
        fact = self.fact(attribution_evidence="Alpha North never said this.")
        result = self.normalize([fact], self.context(fact["quote"]))
        self.assertEqual(result["facts"], [])
        self.assertEqual(result["quality_control"]["issues"][0]["code"], "attribution_not_exact")

    def test_missing_remote_and_fabricated_date_evidence_clears_date_only(self):
        quote = "Alpha North supports the plan."
        for evidence in (None, "Unknown date", "2026-10-14"):
            with self.subTest(evidence=evidence):
                fact = self.fact(quote, event_date="2026-10-14", date_evidence=evidence)
                result = self.normalize([fact], self.context(quote + " Another event occurred 2026-10-14."))
                self.assertEqual(len(result["facts"]), 1)
                self.assertIsNone(result["facts"][0]["event_date"])
                self.assertIsNone(result["facts"][0]["date_evidence"])
                self.assertEqual(result["quality_control"]["cleared_dates"], 1)

    def test_year_is_required_no_publication_date_or_weekday_inference(self):
        for evidence in ("Tuesday", "4 December"):
            quote = f"Alpha North supported the plan on {evidence}."
            fact = self.fact(quote, event_date="2026-10-13", date_evidence=evidence)
            result = self.normalize([fact], self.context(quote))
            self.assertIsNone(result["facts"][0]["event_date"])
            self.assertIn({"index": 0, "code": "date_year_missing"}, result["quality_control"]["issues"])

    def test_numeric_date_must_match_both_iso_and_local_day_month(self):
        for evidence in ("2026-10-13", "13.10.2026"):
            quote = f"Alpha North supported the plan on {evidence}."
            good = self.fact(quote, event_date="2026-10-13", date_evidence=evidence)
            self.assertEqual(self.normalize([good], self.context(quote))["facts"][0]["event_date"], "2026-10-13")
            bad = dict(good, event_date="2026-10-14")
            result = self.normalize([bad], self.context(quote))
            self.assertIsNone(result["facts"][0]["event_date"])
            self.assertIn({"index": 0, "code": "date_value_mismatch"}, result["quality_control"]["issues"])

    def test_outside_period_is_skipped_not_rescued_by_clearing_date(self):
        quote = "Alpha North supported the plan on 2020-10-13."
        for evidence in ("2020-10-13", None):
            fact = self.fact(quote, event_date="2020-10-13", date_evidence=evidence)
            result = self.normalize([fact], self.context(quote))
            self.assertEqual(result["facts"], [])
            self.assertEqual(result["quality_control"]["skipped_facts"], 1)
            self.assertEqual(result["quality_control"]["cleared_dates"], 0)

    def test_qa_change_counts_include_only_final_accepted_facts(self):
        fact = self.fact(kind="action", attribution_basis="third_party_claim", event_date="2026-10-14")
        context = self.context(fact["quote"])
        context["relevance"]["actor_ids"] = ["someone-else"]
        result = self.normalize([fact], context)
        self.assertEqual(result["facts"], [])
        self.assertEqual(result["quality_control"]["downgraded_facts"], 0)
        self.assertEqual(result["quality_control"]["cleared_dates"], 0)

    def test_kind_basis_mismatch_is_skipped_not_promoted(self):
        fact = self.fact(kind="action", attribution_basis="direct_statement")
        self.assertEqual(self.normalize([fact], self.context(fact["quote"]))["facts"], [])

    def test_inputs_and_returned_facts_do_not_share_mutable_state(self):
        fact = self.fact()
        data, context = {"facts": [fact]}, self.context(fact["quote"])
        originals = copy.deepcopy((data, context))
        result = normalize_extraction(data, context, validate_result)
        result["facts"][0]["trait_candidates"].append("changed")
        self.assertEqual((data, context), originals)

    def test_malformed_enum_actor_kind_or_date_skips_only_bad_record(self):
        good = self.fact()
        bad_values = [{"attribution_basis": []}, {"actor_id": []}, {"kind": []},
                      {"event_date": []}, {"event_date": "2026-99-99"}]
        for changes in bad_values:
            with self.subTest(changes=changes):
                result = self.normalize([self.fact(**changes), good], self.context(good["quote"]))
                self.assertEqual(result["facts"], [good])
                self.assertEqual(result["quality_control"]["skipped_facts"], 1)

    def test_each_fact_uses_supplied_validator(self):
        good, bad = self.fact(), self.fact(claim="Reject this synthetic claim")
        calls = []
        def validator(stage, payload, context):
            calls.append((stage, len(payload["facts"])))
            if payload["facts"][0]["claim"].startswith("Reject"):
                raise ValueError("Synthetic semantic rejection")
            return copy.deepcopy(payload)
        result = normalize_extraction({"facts": [good, bad]}, self.context(good["quote"]), validator)
        self.assertEqual(calls, [("extract", 1), ("extract", 1)])
        self.assertEqual(result["facts"], [good])

    def test_policy_version_and_empty_input(self):
        self.assertEqual(EVIDENCE_POLICY_VERSION, "local-attribution-date-review-v1")
        result = self.normalize([], self.context("Synthetic text"))
        self.assertEqual(result["quality_control"]["input_facts"], 0)
        self.assertEqual(result["facts"], [])


if __name__ == "__main__":
    unittest.main()
