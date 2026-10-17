"""Synthetic lexical identity regressions; no API or corpus content required."""
import copy
import unittest

from src.corpus_pipeline.identity import IDENTITY_POLICY_VERSION, ground_relevance


class CorpusIdentityTests(unittest.TestCase):
    def context(self, text, objects=None):
        return {"document": {"text": text}, "config": {"objects": objects or [
            {"id": "james", "name": "James Kabarebe", "aliases": ["Джеймс Кабаребе"]},
            {"id": "paul", "name": "Paul Kagame", "aliases": []}]}}

    def result(self, actor_ids=None, status="relevant"):
        return {"status": status, "actor_ids": actor_ids or ["james"],
                "language": "en", "reason": "Original model attribution"}

    def test_absent_actor_is_review_not_replaced_with_another_name(self):
        result = self.result()
        grounded = ground_relevance(result, self.context("Alex Kagame and Muganga discussed the event."))
        self.assertEqual(grounded["status"], "review")
        self.assertEqual(grounded["actor_ids"], ["james"])
        self.assertTrue(grounded["reason"].startswith(result["reason"]))
        self.assertIn("james", grounded["reason"])
        self.assertIn("no approved name/alias mention", grounded["reason"])

    def test_full_name_matches_case_insensitively_with_boundaries(self):
        result = self.result()
        self.assertEqual(ground_relevance(result, self.context('He quoted "JAMES KABAREBE".')), result)
        for text in ("SuperJames Kabarebe", "James KabarebeExtra", "James_Kabarebe"):
            with self.subTest(text=text):
                self.assertEqual(ground_relevance(result, self.context(text))["status"], "review")

    def test_approved_alias_matches_without_inventing_transliteration(self):
        result = self.result()
        self.assertEqual(ground_relevance(result, self.context("Выступил джеймс кабаребе.")), result)
        self.assertEqual(ground_relevance(result, self.context("James Kabarébé appeared."))["status"], "review")

    def test_normalizes_unicode_and_layout_without_stripping_accents(self):
        objects = [{"id": "edouard", "name": "Édouard Ngirente", "aliases": []}]
        result = self.result(["edouard"])
        self.assertEqual(ground_relevance(result, self.context("E\u0301DOUARD\u00a0\nNgirente spoke.", objects)), result)
        self.assertEqual(ground_relevance(result, self.context("Edouard Ngirente spoke.", objects))["status"], "review")

    def test_regex_metacharacters_in_names_are_literal(self):
        objects = [{"id": "a", "name": "A. North", "aliases": []}]
        result = self.result(["a"])
        self.assertEqual(ground_relevance(result, self.context("A. North spoke.", objects)), result)
        self.assertEqual(ground_relevance(result, self.context("Ax North spoke.", objects))["status"], "review")

    def test_one_missing_actor_reviews_entire_result_preserving_all_ids(self):
        result = self.result(["james", "paul"])
        grounded = ground_relevance(result, self.context("James Kabarebe spoke."))
        self.assertEqual(grounded["status"], "review")
        self.assertEqual(grounded["actor_ids"], ["james", "paul"])
        self.assertIn("actor IDs: paul.", grounded["reason"])

    def test_surname_is_not_generated_from_full_name(self):
        result = self.result(["paul"])
        self.assertEqual(ground_relevance(result, self.context("Alex Kagame spoke."))["status"], "review")

    def test_shared_surname_alias_is_only_a_lexical_match_not_identity_proof(self):
        # Deliberate limitation: the caller/model must still distinguish homonyms.
        objects = [{"id": "paul", "name": "Paul Kagame", "aliases": ["Kagame"]}]
        result = self.result(["paul"])
        self.assertEqual(ground_relevance(result, self.context("Alex Kagame spoke.", objects)), result)

    def test_pronouns_and_author_metadata_do_not_replace_textual_name(self):
        context = self.context("He proposed an initiative.")
        context["document"]["author"] = "James Kabarebe"
        self.assertEqual(ground_relevance(self.result(), context)["status"], "review")

    def test_other_statuses_and_nested_history_are_copied_without_mutation(self):
        for status in ("irrelevant", "review", "relevant"):
            with self.subTest(status=status):
                result = self.result(status=status)
                result["history"] = [{"actor_ids": ["james"], "note": "Immutable original"}]
                context = self.context("James Kabarebe spoke.")
                before_result, before_context = copy.deepcopy(result), copy.deepcopy(context)
                grounded = ground_relevance(result, context)
                self.assertEqual(grounded, before_result)
                self.assertIsNot(grounded, result)
                grounded["actor_ids"].append("new")
                grounded["history"][0]["note"] = "Changed copy"
                self.assertEqual(result, before_result)
                self.assertEqual(context, before_context)

    def test_downgrade_does_not_mutate_result_or_context(self):
        result, context = self.result(), self.context("No actor is explicitly identified.")
        before_result, before_context = copy.deepcopy(result), copy.deepcopy(context)
        grounded = ground_relevance(result, context)
        self.assertEqual(grounded["status"], "review")
        self.assertEqual(result, before_result)
        self.assertEqual(context, before_context)

    def test_unknown_returned_id_fails_closed_to_review(self):
        grounded = ground_relevance(self.result(["unlisted"]), self.context("An unlisted person spoke."))
        self.assertEqual(grounded["status"], "review")
        self.assertEqual(grounded["actor_ids"], ["unlisted"])

    def test_policy_version_is_explicit_for_recipe_hashes(self):
        self.assertEqual(IDENTITY_POLICY_VERSION, "lexical-mention-review-v1")


if __name__ == "__main__":
    unittest.main()
