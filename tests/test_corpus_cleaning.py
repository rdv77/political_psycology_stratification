"""Pure synthetic offline-screening tests; no source corpus, store or API."""
import copy
import json
import unittest

from src.corpus_pipeline.cleaning import annotate_document, candidate_families, summarize_annotations


class CorpusCleaningTests(unittest.TestCase):
    def setUp(self):
        self.config = {"objects": [{"id": "alpha", "name": "Émile North", "aliases": ["Alpha"]}],
                       "country_terms": ["Rwanda"], "languages": ["rw", "fr", "en", "sw"],
                       "date_from": "2021-10-15", "date_to": "2026-10-15"}
        self.document = {"text": "Alpha comments on Rwanda.", "language_hint": "Английский",
                         "published_at": "2026-10-15T12:00:00", "metadata": {"private": "not_exported"}}

    def test_unknown_names_languages_dates_are_review_not_negative(self):
        for change, flag in [({"text": "Rwanda has a public discussion."}, "no_known_actor_name"),
                             ({"language_hint": ""}, "unknown_language_hint"),
                             ({"language_hint": "not-a-detected-language"}, "unknown_language_hint"),
                             ({"published_at": "2026"}, "unknown_publication_date"),
                             ({"published_at": "2026-02-30"}, "unknown_publication_date")]:
            with self.subTest(change=change):
                result = annotate_document(dict(self.document, **change), self.config)
                self.assertEqual(result["status"], "needs_semantic_review")
                self.assertIn(flag, result["flags"])

    def test_only_country_and_known_period_rules_exclude(self):
        result = annotate_document(dict(self.document, text="Alpha comments on an issue."), self.config)
        self.assertEqual(result["status"], "rule_excluded")
        self.assertEqual(result["reason"], "country_not_matched")
        self.assertIn("missing_country_with_actor_match", result["flags"])
        result = annotate_document(dict(self.document, published_at="2026-10-16"), self.config)
        self.assertEqual(result["reason"], "outside_period")
        # Country checks have the same priority as the existing service rule.
        result = annotate_document(dict(self.document, text="Alpha", published_at="2020-01-01"), self.config)
        self.assertEqual(result["reason"], "country_not_matched")
        self.assertIn("outside_period", result["flags"])

    def test_clearing_country_gate_keeps_known_actor_for_semantic_filter(self):
        config = dict(self.config, country_terms=[])
        result = annotate_document(dict(self.document, text="Alpha is a Rwandan politician."), config)
        self.assertEqual(result["status"], "ready_for_semantic_filter")
        self.assertEqual(result["matched_actor_ids"], ["alpha"])

    def test_missing_country_review_terms_retain_document_without_hard_exclusion(self):
        config = dict(self.config, country_terms=[], country_review_terms=["Rwanda", "u Rwanda"])
        result = annotate_document(dict(self.document, text="Alpha is a Rwandan politician."), config)
        self.assertEqual(result["status"], "needs_semantic_review")
        self.assertEqual(result["reason"], "country_not_matched")
        self.assertIn("missing_country_with_actor_match", result["flags"])
        self.assertEqual(result["matched_actor_ids"], ["alpha"])
        self.assertEqual(annotate_document(self.document, config)["status"], "ready_for_semantic_filter")
        # Review-only terms do not override the independent date rule.
        result = annotate_document(dict(self.document, text="Alpha", published_at="2026-10-16"), config)
        self.assertEqual(result["reason"], "outside_period")
        self.assertEqual(result["status"], "rule_excluded")

    def test_boundaries_are_inclusive_and_empty_text_remains_review_without_country_gate(self):
        for value in ("2021-10-15", "2026-10-15T23:59:59+02:00"):
            self.assertEqual(annotate_document(dict(self.document, published_at=value), self.config)["status"],
                             "ready_for_semantic_filter")
        result = annotate_document(dict(self.document, text=""), dict(self.config, country_terms=[]))
        self.assertEqual(result["status"], "needs_semantic_review")
        self.assertIn("empty_text", result["flags"])

    def test_unicode_whitespace_and_word_boundaries_do_not_invent_aliases(self):
        result = annotate_document(dict(self.document, text="E\u0301mile\nNorth speaks about Rwanda."), self.config)
        self.assertEqual(result["matched_actor_ids"], ["alpha"])
        result = annotate_document(dict(self.document, text="SuperAlpha speaks about Rwanda."), self.config)
        self.assertEqual(result["matched_actor_ids"], [])

    def test_shared_alias_requires_review_and_sources_are_not_changed_or_exported(self):
        config = copy.deepcopy(self.config)
        config["objects"].append({"id": "beta", "name": "Beta South", "aliases": ["Alpha"]})
        before_document, before_config = copy.deepcopy(self.document), copy.deepcopy(config)
        result = annotate_document(self.document, config)
        self.assertEqual(result["status"], "needs_semantic_review")
        self.assertEqual(result["matched_actor_ids"], ["alpha", "beta"])
        self.assertIn("ambiguous_approved_alias", result["flags"])
        self.assertEqual(self.document, before_document)
        self.assertEqual(config, before_config)
        self.assertEqual(set(result), {"status", "reason", "matched_actor_ids", "flags"})
        self.assertNotIn("not_exported", json.dumps(result))
        self.assertNotIn(self.document["text"], json.dumps(result))

    def test_longer_full_name_does_not_hide_another_actors_nested_alias(self):
        objects = [{"id": "alpha", "name": "Alpha North", "aliases": []},
                   {"id": "beta", "name": "Beta South", "aliases": ["North"]}]
        result = annotate_document(dict(self.document, text="Alpha North discusses Rwanda."),
                                   dict(self.config, objects=objects))
        self.assertEqual(result["matched_actor_ids"], ["alpha", "beta"])

    def test_summary_counts_documents_once_per_actor_and_keeps_review_losses_visible(self):
        ready = annotate_document(self.document, self.config)
        unknown = annotate_document(dict(self.document, language_hint=""), self.config)
        excluded = annotate_document(dict(self.document, text="Alpha without a country word."), self.config)
        summary = summarize_annotations(iter([ready, unknown, excluded]))
        self.assertEqual(summary["documents"], 3)
        self.assertEqual(summary["status_counts"], {"needs_semantic_review": 1, "ready_for_semantic_filter": 1,
                                                   "rule_excluded": 1})
        self.assertEqual(summary["actor_document_counts"], {"alpha": 3})
        self.assertEqual(summary["actor_document_counts_by_status"]["rule_excluded"], {"alpha": 1})
        self.assertEqual(summary["flag_counts"]["missing_country_with_actor_match"], 1)

    def test_near_components_are_deterministic_review_families_not_deletion(self):
        edges = [{"left_id": "b", "right_id": "a"}, {"left_id": "b", "right_id": "c"},
                 {"left_id": "a", "right_id": "b"}, {"left_id": "y", "right_id": "z"}]
        groups = candidate_families(iter(edges))
        self.assertEqual(groups, candidate_families(reversed(edges)))
        self.assertEqual(groups[0]["member_ids"], ["a", "b", "c"])
        self.assertEqual(groups[0]["pair_count"], 2)
        self.assertFalse(groups[0]["delete_automatically"])
        self.assertEqual(groups[0]["status"], "needs_duplicate_review")
        self.assertEqual(candidate_families([]), [])


if __name__ == "__main__":
    unittest.main()
