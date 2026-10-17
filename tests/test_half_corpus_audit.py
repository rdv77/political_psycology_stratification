import copy
import json
import unittest
from unittest.mock import patch

from scripts.audit_half_corpus import build_audit


ACTORS = [{"id": "paul-kagame", "name": "Paul Kagame", "aliases": ["Kagame"]},
          {"id": "alpha", "name": "Alpha North", "aliases": []},
          {"id": "beta", "name": "Beta South", "aliases": []},
          {"id": "zero", "name": "Zero East", "aliases": []}]


def fixture(counts=None):
    counts = counts or {"paul-kagame": 8, "alpha": 2, "beta": 1}
    config = {"scope": "Synthetic Rwanda", "aspect": "public_positions", "objects": ACTORS,
              "date_from": "2021-10-15", "date_to": "2026-10-17",
              "text_selection": {"mode": "mentions", "max_tokens": 1500}}
    snapshot = {"corpus": {"id": "corpus-synthetic", "source_sha256": "a" * 64, "config": config},
                "manifest": {"corpus_id": "corpus-synthetic", "source_sha256": "a" * 64},
                "jobs": [], "facts": []}
    documents = []
    for actor_id, count in counts.items():
        actor = next(actor for actor in ACTORS if actor["id"] == actor_id)
        for index in range(count):
            identifier = f"private-doc-{actor_id}-{index}"
            text = f"Private original Rwanda. {actor['name']} said: We support public access {index}."
            quote = f"We support public access {index}."
            attribution = f"{actor['name']} said:"
            doc = {"id": identifier, "text": text, "published_at": "2026-10-16T12:00:00",
                   "url": "https://source.invalid/" + identifier, "author": "PRIVATE ACCOUNT"}
            documents.append(doc)
            source = {"id": identifier, "url": doc["url"], "source_type": "article",
                      "excerpts": [{"start": 0, "end": len(text), "text": text}]}
            rel = {"id": "rel-" + identifier, "active_recipe": True, "stage": "relevance", "status": "done",
                   "context": {"config": config, "document": source},
                   "result": {"status": "relevant", "actor_ids": [actor_id], "language": "rw" if index % 2 else "en"}}
            ext = {"id": "ext-" + identifier, "active_recipe": True, "stage": "extract", "status": "done",
                   "context": {"config": config, "document": source, "relevance": rel["result"]},
                   "payload": {"never_publish": "PRIVATE PAYLOAD SECRET"},
                   "result": {"quality_control": {"input_facts": 1, "skipped_facts": 0,
                                                 "downgraded_facts": 0, "cleared_dates": 0, "issues": []}}}
            fact = {"id": "fact-" + identifier, "actor_id": actor_id, "quote": quote,
                    "claim": f"{actor['name']} supports public access.", "kind": "public_statement",
                    "event_date": None, "topic": "access", "trait_candidates": ["public access"],
                    "capacity": "official", "attribution_basis": "direct_statement",
                    "attribution_evidence": attribution, "date_evidence": None,
                    "document_id": identifier, "source_url": doc["url"],
                    "quote_start": text.index(quote), "quote_end": text.index(quote) + len(quote),
                    "extraction_job_id": ext["id"]}
            snapshot["jobs"].extend([rel, ext])
            snapshot["facts"].append(fact)
    selection = {"corpus_id": "corpus-synthetic", "source_sha256": "a" * 64,
                 "documents": [{"id": doc["id"], "selected": True} for doc in documents]}
    return snapshot, documents, selection


class HalfCorpusAuditTests(unittest.TestCase):
    def test_structural_census_and_actor_balance_do_not_claim_semantic_accuracy(self):
        snapshot, documents, selection = fixture()
        original = copy.deepcopy((snapshot, documents, selection))
        with patch("src.corpus_pipeline.models.OpenAIResponsesHTTP.call", side_effect=AssertionError("No API")):
            result = build_audit(snapshot, documents, selection=selection)
        public = result["public_aggregate"]
        self.assertEqual(public["structural_census"]["active_accepted_facts_checked"], 11)
        self.assertEqual(public["structural_census"]["facts_with_recheck_issues"], 0)
        actors = {row["actor_id"]: row for row in public["semantic_review"]["actors"]}
        self.assertEqual(actors["paul-kagame"]["selected_for_review"], 5)
        self.assertEqual(actors["alpha"]["selected_for_review"], 2)
        self.assertEqual(actors["beta"]["selected_for_review"], 1)
        self.assertEqual(public["semantic_review"]["zero_fact_actor_ids"], ["zero"])
        self.assertIsNone(public["semantic_review"]["accuracy_estimate"])
        self.assertTrue(all(row["review_status"] == "pending" for row in result["semantic_review_packet"]))
        self.assertEqual((snapshot, documents, selection), original)

    def test_public_aggregate_is_an_explicit_privacy_allowlist(self):
        snapshot, documents, selection = fixture()
        result = build_audit(snapshot, documents, selection=selection)
        public = json.dumps(result["public_aggregate"])
        for secret in ("private-doc-", "https://source.invalid", "PRIVATE ACCOUNT", "PRIVATE PAYLOAD SECRET",
                       "Private original Rwanda", "We support public access"):
            self.assertNotIn(secret, public)
        self.assertIn("We support public access", json.dumps(result["semantic_review_packet"]))

    def test_bad_offsets_wrong_speaker_and_cross_excerpt_evidence_are_flagged(self):
        snapshot, documents, selection = fixture({"alpha": 3})
        snapshot["facts"][0]["quote_start"] += 1
        snapshot["facts"][1]["actor_id"] = "beta"
        job = next(job for job in snapshot["jobs"] if job["id"] == snapshot["facts"][2]["extraction_job_id"])
        source = job["context"]["document"]
        text = documents[2]["text"]
        at_end = text.index("said:") + len("said:")
        q_start = text.index(snapshot["facts"][2]["quote"])
        source["excerpts"] = [{"start": 0, "end": at_end, "text": text[:at_end]},
                              {"start": q_start, "end": len(text), "text": text[q_start:]}]
        result = build_audit(snapshot, documents, selection=selection)
        codes = result["public_aggregate"]["structural_census"]["issue_counts"]
        self.assertEqual(codes["quote_global_coordinates_mismatch"], 1)
        self.assertEqual(codes["recheck_actor_not_local"], 1)
        self.assertEqual(codes["recheck_attribution_not_local"], 1)
        self.assertEqual(result["public_aggregate"]["structural_census"]["facts_with_recheck_issues"], 3)

    def test_source_provenance_mismatch_and_inactive_facts_not_silently_accepted(self):
        snapshot, documents, selection = fixture({"alpha": 2})
        snapshot["jobs"][1]["active_recipe"] = False
        snapshot["jobs"][3]["context"]["document"]["excerpts"][0]["text"] = "x" * len(documents[1]["text"])
        result = build_audit(snapshot, documents, selection=selection)
        census = result["public_aggregate"]["structural_census"]
        self.assertEqual(census["inactive_facts_ignored"], 1)
        self.assertEqual(census["active_accepted_facts_checked"], 1)
        self.assertEqual(census["issue_counts"]["source_span_original_mismatch"], 1)

    def test_negative_packets_include_completed_empty_extractions_and_reviews(self):
        snapshot, documents, selection = fixture({"alpha": 3})
        snapshot["facts"] = []
        # First document is relevant with a completed, empty extraction.
        snapshot["jobs"][0]["result"]["language"] = "sw"
        # Second is a completed review, third is not yet processed.
        snapshot["jobs"][2]["result"].update(status="review", language="rw")
        snapshot["jobs"][3]["active_recipe"] = False
        snapshot["jobs"][4]["status"] = "pending"
        snapshot["jobs"][4]["result"] = None
        snapshot["jobs"][5]["active_recipe"] = False
        result = build_audit(snapshot, documents, selection=selection)
        categories = {row["stratum"][0] for row in result["negative_review_packet"]}
        self.assertEqual(categories, {"relevant_no_facts", "review"})
        self.assertEqual({row["stratum"][1] for row in result["negative_review_packet"]}, {"sw", "rw"})
        self.assertTrue(result["public_aggregate"]["scope"]["snapshot_is_partial"])
        self.assertNotIn("Private original", json.dumps(result["public_aggregate"]))

    def test_original_qc_counts_remain_separate_from_recheck_and_semantic_review(self):
        snapshot, documents, selection = fixture({"alpha": 1})
        snapshot["jobs"][1]["result"]["quality_control"].update(
            input_facts=5, skipped_facts=4, downgraded_facts=1,
            issues=[{"index": 0, "code": "actor_not_local"}, {"index": 1, "code": "date_year_missing"}])
        result = build_audit(snapshot, documents, selection=selection)
        qc = result["public_aggregate"]["structural_census"]
        self.assertEqual(qc["original_normalization_totals"]["skipped_facts"], 4)
        self.assertEqual(qc["original_normalization_issue_counts"]["date_year_missing"], 1)
        self.assertEqual(qc["facts_with_recheck_issues"], 0)
        self.assertIsNone(result["public_aggregate"]["semantic_review"]["accuracy_estimate"])

    def test_input_order_determinism_and_candidate_families_not_events(self):
        snapshot, documents, selection = fixture()
        similar = [{"left_id": documents[0]["id"], "right_id": documents[1]["id"], "similarity": .95}]
        first = build_audit(snapshot, documents, selection=selection, similar=similar)
        other = copy.deepcopy(snapshot)
        other["jobs"].reverse()
        other["facts"].reverse()
        second = build_audit(other, list(reversed(documents)), selection=selection, similar=similar)
        self.assertEqual(first["semantic_review_packet"], second["semantic_review_packet"])
        self.assertEqual(first["public_aggregate"], second["public_aggregate"])
        self.assertEqual(first["public_aggregate"]["duplicate_context"]["candidate_links_supplied"], 1)
        self.assertNotIn("independent_events", first["public_aggregate"]["duplicate_context"])

    def test_invalid_or_unfrozen_inputs_fail_without_inference(self):
        snapshot, documents, selection = fixture({"alpha": 1})
        bad = copy.deepcopy(snapshot)
        del bad["jobs"][0]["active_recipe"]
        with self.assertRaises(ValueError):
            build_audit(bad, documents, selection=selection)
        bad = copy.deepcopy(selection)
        bad["source_sha256"] = "different"
        with self.assertRaises(ValueError):
            build_audit(snapshot, documents, selection=bad)
        with self.assertRaises(ValueError):
            build_audit(snapshot, [], selection=selection)
        with self.assertRaises(ValueError):
            build_audit(snapshot, documents, selection=selection, facts_per_actor=True)

    def test_global_document_cap_stops_repeated_facts_dominating_review(self):
        snapshot, documents, selection = fixture({"alpha": 1})
        for index in range(8):
            repeated = copy.deepcopy(snapshot["facts"][0])
            repeated["id"] += f"-copy-{index}"
            snapshot["facts"].append(repeated)
        result = build_audit(snapshot, documents, selection=selection, max_per_document=2)
        self.assertEqual(result["public_aggregate"]["structural_census"]["active_accepted_facts_checked"], 9)
        self.assertEqual(len(result["semantic_review_packet"]), 2)
        self.assertEqual(len(result["risk_review_packet"]), 0)
        self.assertEqual(result["public_aggregate"]["context_coverage"]["union_characters_in_supplied_relevance_contexts"],
                         len(documents[0]["text"]))

    def test_malformed_receipt_labels_cannot_inject_source_text_into_public_aggregate(self):
        snapshot, documents, selection = fixture({"alpha": 1})
        secret = "SECRET SOURCE QUOTE https://private.invalid"
        snapshot["facts"][0]["kind"] = secret
        snapshot["jobs"][1]["result"]["quality_control"]["issues"] = [{"index": 0, "code": secret}]
        result = build_audit(snapshot, documents, selection=selection)
        self.assertNotIn(secret, json.dumps(result["public_aggregate"]))
        self.assertEqual(result["public_aggregate"]["accepted_facts"]["by_kind"], {"unknown": 1})
        self.assertEqual(result["public_aggregate"]["structural_census"]["original_normalization_issue_counts"], {"unknown": 1})

    def test_missing_private_selection_metadata_is_unknown_not_zero(self):
        snapshot, documents, selection = fixture({"alpha": 3})
        snapshot["jobs"][0]["context"]["document"]["text_selection"] = {
            "expanded": True, "truncated": True, "no_mentions_fallback": False}
        snapshot["jobs"][2]["context"]["document"]["text_selection"] = {
            "expanded": False, "truncated": False, "no_mentions_fallback": False}
        result = build_audit(snapshot, documents, selection=selection)
        coverage = result["public_aggregate"]["context_coverage"]
        self.assertIsNone(coverage["expanded_documents"])
        self.assertIsNone(coverage["truncated_documents"])
        self.assertIsNone(coverage["no_mention_head_fallback_documents"])
        self.assertEqual(coverage["metadata_availability"]["expanded_documents"],
                         {"known_true": 1, "known_false": 1, "unknown": 1})
        self.assertEqual(coverage["metadata_availability"]["no_mention_head_fallback_documents"],
                         {"known_true": 0, "known_false": 2, "unknown": 1})
        self.assertEqual(coverage["union_characters_in_supplied_relevance_contexts"],
                         sum(len(document["text"]) for document in documents))

    def test_public_fact_fingerprint_matches_private_and_tracks_fact_revisions(self):
        snapshot, documents, selection = fixture({"alpha": 1})
        first = build_audit(snapshot, documents, selection=selection)
        self.assertEqual(first["public_aggregate"]["provenance"]["facts_snapshot_sha256"],
                         first["provenance"]["facts_snapshot_sha256"])
        snapshot["facts"][0]["claim"] += " Revised claim."
        second = build_audit(snapshot, documents, selection=selection)
        self.assertNotEqual(first["provenance"]["facts_snapshot_sha256"], second["provenance"]["facts_snapshot_sha256"])


if __name__ == "__main__":
    unittest.main()
