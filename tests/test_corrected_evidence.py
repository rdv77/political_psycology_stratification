"""No-API versioned corrections: provenance, time, agency and safe summaries."""
from copy import deepcopy
import json
import unittest

from src.corpus_pipeline.corrected_evidence import correct
from src.corpus_pipeline.scoring_preparation import digest, filter_eligible_fact


def fixture():
    text = "Approved Name said on 2 May 2026: we remember 1995 and support cooperation."
    fact = {"id": "fact-PRIVATE", "actor_id": "actor", "document_id": "doc-PRIVATE",
            "extraction_job_id": "job-PRIVATE", "quote": "we remember 1995 and support cooperation.",
            "claim": "PRIVATE CLAIM", "attribution_evidence": text, "kind": "public_statement",
            "attribution_basis": "direct_statement", "capacity": "official", "event_date": "2026-05-02",
            "date_evidence": "2 May 2026", "source_date": "2026-06-01T12:00:00"}
    snapshot = {"corpus": {"config": {"aspect": "public_positions", "date_from": "2021-10-15",
                 "date_to": "2026-10-15", "objects": [{"id": "actor", "name": "Approved Name", "aliases": []}]}},
                "facts": [fact], "jobs": [{"id": "job-PRIVATE", "stage": "extract", "context": {
                    "document": {"id": "doc-PRIVATE", "text": text, "source_type": "article"}}}]}
    annotation = {"case_id": "U01", "source_fact_ids": [fact["id"]], "original_actor_id": "actor",
                  "reviewed_attribution": {"status": "supported", "basis": "direct_statement",
                                            "explanation_ru": "Собственная речь."},
                  "reviewed_event_date": {"status": "corrected_exact", "date": "2026-05-02",
                                          "date_from": None, "date_to": None,
                                          "evidence_quote": "2 May 2026", "explanation_ru": "Дата речи."},
                  "verdict": "retain"}
    case = {"case_id": "U01", "actor_id": "actor", "member_facts": [deepcopy(fact)],
            "sources": [{"id": "doc-PRIVATE", "text": text}]}
    return snapshot, annotation, case


class CorrectedEvidenceTests(unittest.TestCase):
    def test_corrections_are_derived_complete_and_inputs_untouched(self):
        snapshot, annotation, case = fixture()
        originals = deepcopy((snapshot, annotation, case))
        result = correct(snapshot, {"annotations": [annotation]}, source_cases=[case])
        self.assertEqual((snapshot, annotation, case), originals)
        self.assertEqual(len(result["private_facts"]), len(snapshot["facts"]))
        self.assertEqual(result["private_facts"][0]["reviewed_case_id"], "U01")
        self.assertEqual(result["metadata"]["derived_facts_content_sha256"], digest(result["private_facts"]))
        self.assertFalse(result["public_summary"]["database_modified"])
        self.assertEqual(result["public_summary"]["api_calls"], 0)

    def test_relative_and_publication_only_times_stay_unknown(self):
        snapshot, annotation, case = fixture()
        annotation["reviewed_event_date"].update(status="publication_only", date=None, evidence_quote=None)
        result = correct(snapshot, [annotation], source_cases=[case])
        fact = result["private_facts"][0]
        self.assertIsNone(fact["event_date"])
        self.assertIsNone(fact["event_date_from"])
        self.assertEqual(fact["source_date"], "2026-06-01T12:00:00")
        self.assertEqual(fact["temporal_status"], "publication_only")

    def test_interval_is_not_promoted_to_first_day(self):
        snapshot, annotation, case = fixture()
        annotation["reviewed_event_date"].update(status="supported_interval", date=None,
                                                date_from="2025-10-01", date_to="2025-10-31")
        fact = correct(snapshot, [annotation], source_cases=[case])["private_facts"][0]
        self.assertIsNone(fact["event_date"])
        self.assertEqual(fact["event_date_to"], "2025-10-31")

    def test_old_year_inside_current_speech_is_not_historical_position(self):
        snapshot, _, _ = fixture()
        fact = correct(snapshot, [])["private_facts"][0]
        self.assertEqual(fact["event_date"], "2026-05-02")
        self.assertTrue(filter_eligible_fact(fact, snapshot["corpus"]["config"]))

    def test_unsupported_event_date_is_cleared_without_publication_fallback(self):
        snapshot, _, _ = fixture()
        snapshot["facts"][0]["date_evidence"] = None
        fact = correct(snapshot, [])["private_facts"][0]
        self.assertIsNone(fact["event_date"])
        self.assertIn("date_not_grounded", fact["correction_flags"])

    def test_month_or_year_is_never_kept_as_an_exact_first_day(self):
        for fragment in ("In February 2025", "in 2025", "March 2026"):
            snapshot, _, _ = fixture()
            original = snapshot["jobs"][0]["context"]["document"]["text"]
            text = original.replace("2 May 2026", fragment)
            snapshot["jobs"][0]["context"]["document"]["text"] = text
            snapshot["facts"][0].update(attribution_evidence=text, date_evidence=fragment,
                                          event_date="2025-02-01" if "2025" in fragment else "2026-03-01")
            fact = correct(snapshot, [])["private_facts"][0]
            self.assertIsNone(fact["event_date"])
            self.assertIn("date_precision_not_exact_day", fact["correction_flags"])

    def test_exact_date_precision_in_english_french_kinyarwanda_swahili(self):
        for fragment in ("May 2, 2026", "2 mai 2026", "2 Gicurasi 2026", "2 Mei 2026", "2/5/2026"):
            snapshot, _, _ = fixture()
            text = snapshot["jobs"][0]["context"]["document"]["text"].replace("2 May 2026", fragment)
            snapshot["jobs"][0]["context"]["document"]["text"] = text
            snapshot["facts"][0].update(attribution_evidence=text, date_evidence=fragment)
            self.assertEqual(correct(snapshot, [])["private_facts"][0]["event_date"], "2026-05-02")

    def test_full_text_date_mismatching_selected_day_is_cleared(self):
        snapshot, _, _ = fixture()
        snapshot["facts"][0]["event_date"] = "2026-05-03"
        fact = correct(snapshot, [])["private_facts"][0]
        self.assertIsNone(fact["event_date"])
        self.assertIn("date_value_mismatch", fact["correction_flags"])

    def test_not_own_homonym_and_hold_are_ineligible(self):
        for status, verdict in (("not_own", "exclude"), ("homonym", "exclude"), ("unresolved", "hold")):
            snapshot, annotation, case = fixture()
            annotation["reviewed_attribution"]["status"] = status
            annotation["verdict"] = verdict
            fact = correct(snapshot, [annotation], source_cases=[case])["private_facts"][0]
            self.assertFalse(filter_eligible_fact(fact, snapshot["corpus"]["config"]))

    def test_third_party_claim_and_comment_cannot_be_own_position(self):
        for update in ("basis", "comment", "repost"):
            snapshot, _, _ = fixture()
            if update == "basis": snapshot["facts"][0]["attribution_basis"] = "third_party_claim"
            else: snapshot["jobs"][0]["context"]["document"]["source_type"] = update
            fact = correct(snapshot, [])["private_facts"][0]
            self.assertEqual(fact["kind"], "attributed_image")
            self.assertFalse(filter_eligible_fact(fact, snapshot["corpus"]["config"]))

    def test_protocol_or_passive_cue_alone_does_not_invalidate_substantive_statement(self):
        snapshot, _, _ = fixture()
        old = "we remember 1995 and support cooperation."
        new = "we attended talks and oppose policies we are accused of supporting."
        snapshot["facts"][0]["quote"] = new
        snapshot["facts"][0]["attribution_evidence"] = snapshot["facts"][0]["attribution_evidence"].replace(old, new)
        snapshot["jobs"][0]["context"]["document"]["text"] = snapshot["facts"][0]["attribution_evidence"]
        fact = correct(snapshot, [])["private_facts"][0]
        self.assertTrue(filter_eligible_fact(fact, snapshot["corpus"]["config"]))
        self.assertIn("protocol_or_participation_cue_review_only", fact["correction_flags"])
        self.assertIn("passive_agency_or_accusation_cue_review_only", fact["correction_flags"])

    def test_literal_repeats_are_not_verified_independent_episodes(self):
        snapshot, _, _ = fixture()
        other = deepcopy(snapshot["facts"][0]); other["id"] = "fact-REPEAT"
        snapshot["facts"].append(other)
        result = correct(snapshot, [])
        one, two = result["private_facts"]
        self.assertEqual(one["literal_material_group_id"], two["literal_material_group_id"])
        self.assertFalse(one["episode_identity_verified"])
        self.assertFalse(one["source_independence_verified"])
        self.assertEqual(result["public_summary"]["counts"]["literal_repeat_excess"], 1)
        self.assertFalse(result["public_summary"]["frequency_used_as_intensity"])

    def test_missing_context_or_ungrounded_attribution_needs_review(self):
        for change in ("missing_context", "wrong_quote", "wrong_attribution"):
            snapshot, _, _ = fixture()
            if change == "missing_context": snapshot["jobs"] = []
            elif change == "wrong_quote": snapshot["facts"][0]["quote"] = "not actually in source"
            else: snapshot["facts"][0]["attribution_evidence"] = "invented name"
            fact = correct(snapshot, [])["private_facts"][0]
            self.assertEqual(fact["review_status"], "unreviewed")
            self.assertFalse(filter_eligible_fact(fact, snapshot["corpus"]["config"]))

    def test_stale_wrong_actor_missing_fact_or_duplicate_review_fails_closed(self):
        for change in ("actor", "missing", "duplicate", "stale"):
            snapshot, annotation, case = fixture()
            rows = [annotation]
            if change == "actor": annotation["original_actor_id"] = "other"
            elif change == "missing": annotation["source_fact_ids"] = ["missing"]
            elif change == "duplicate": rows.append(deepcopy(annotation))
            else: case["member_facts"][0]["claim"] = "changed after review"
            with self.assertRaises(ValueError): correct(snapshot, rows, source_cases=[case])

    def test_outside_and_cross_boundary_interval_do_not_enter_current_measurement(self):
        for start, end, status in (("2017-01-01", "2017-12-31", "outside_window"),
                                   ("2021-01-01", "2021-12-31", "supported_interval")):
            snapshot, annotation, case = fixture()
            annotation["reviewed_event_date"].update(status=status, date=None, date_from=start, date_to=end)
            fact = correct(snapshot, [annotation], source_cases=[case])["private_facts"][0]
            self.assertFalse(filter_eligible_fact(fact, snapshot["corpus"]["config"]))

    def test_public_summary_never_contains_private_content_or_source_identifiers(self):
        snapshot, annotation, case = fixture()
        output = json.dumps(correct(snapshot, [annotation], source_cases=[case])["public_summary"])
        for secret in ("PRIVATE CLAIM", "doc-PRIVATE", "job-PRIVATE", "fact-PRIVATE", "2 May 2026", "we remember"):
            self.assertNotIn(secret, output)

    def test_invalid_date_precision_and_foreign_document_are_rejected(self):
        snapshot, annotation, case = fixture()
        annotation["reviewed_event_date"]["date"] = "2026-05"
        with self.assertRaises(ValueError): correct(snapshot, [annotation], source_cases=[case])
        snapshot, _, _ = fixture()
        snapshot["jobs"][0]["context"]["document"]["id"] = "foreign"
        with self.assertRaises(ValueError): correct(snapshot, [])


if __name__ == "__main__":
    unittest.main()
