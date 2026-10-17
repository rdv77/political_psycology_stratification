"""Synthetic adaptive context windows; no APIs or copied private corpus text."""
import copy
import unittest

from src.corpus_pipeline.context_policy import actor_mentions, select_document_context
from src.corpus_pipeline.excerpts import offline_encoding


class CharacterEncoding:
    name = "synthetic_characters"
    def encode(self, text, **kwargs):
        return list(text)


class ContextPolicyTests(unittest.TestCase):
    def setUp(self):
        self.encoding = CharacterEncoding()
        self.objects = [{"id": "alpha", "name": "Alpha North", "aliases": ["Альфа Норт"]},
                        {"id": "beta", "name": "Beta South", "aliases": []}]

    def select(self, text, counts=None, title="", policy=None, encoding="default"):
        return select_document_context({"text": text, "title": title}, self.objects,
            {"alpha": 100} if counts is None else counts, 1000, policy or {},
            self.encoding if encoding == "default" else encoding)

    def check(self, text, result):
        spans = result["spans"]
        self.assertLessEqual(len(spans), 4)
        for span in spans:
            self.assertEqual(span["text"], text[span["start"]:span["end"]])
        for a, b in zip(spans, spans[1:]):
            self.assertLess(a["end"], b["start"])
        self.assertLessEqual(result["metadata"]["selected_budget_units"], result["metadata"]["effective_max_tokens"])

    def test_short_rare_source_stays_complete_at_base_cap(self):
        text = "Alpha North spoke. " * 30
        result = self.select(text, {"alpha": 2})
        self.assertFalse(result["metadata"]["expanded"])
        self.assertEqual(result["spans"][0]["text"], text)

    def test_rare_actor_expands_and_keeps_all_if_within_6000(self):
        text = "Alpha North " + "context words " * 300
        result = self.select(text, {"alpha": 5})
        self.assertTrue(result["metadata"]["expanded"])
        self.assertEqual(result["metadata"]["effective_max_tokens"], 6000)
        self.assertEqual(result["spans"][0]["text"], text)

    def test_missing_count_is_unknown_not_rare_zero(self):
        text = "Alpha North " + "context " * 1000
        result = self.select(text, {})
        self.assertFalse(result["metadata"]["expanded"])
        self.assertEqual(result["metadata"]["missing_actor_count_ids"], ["alpha"])

    def test_rare_name_near_end_is_not_lost_before_actual_tail_window(self):
        text = "Alpha North " + "context " * 4000 + " Beta South " + "ending " * 300
        result = self.select(text, {"alpha": 100, "beta": 1})
        self.assertTrue(result["metadata"]["expanded"])
        self.assertIn("beta", result["metadata"]["covered_actor_ids"])
        self.assertEqual(result["metadata"]["uncovered_rare_actor_ids"], [])
        self.check(text, result)

    def test_long_multilingual_interview_candidates_expand_without_language_inference(self):
        for title in ("Interview with Alpha North", "Entretien avec Alpha North", "Ikiganiro Alpha North", "Mahojiano Alpha North"):
            text = "Alpha North\n" + ("Interviewer: A question.\nAlpha North: " + "response words " * 90 + "\n") * 12
            result = self.select(text, title=title)
            self.assertTrue(result["metadata"]["expanded"])
            self.assertTrue(result["metadata"]["longform_signals"]["candidate"])
            self.assertFalse(result["metadata"]["longform_signals"]["language_inferred"])
            self.assertGreater(result["metadata"]["selected_budget_units"], 4500)
            self.check(text, result)

    def test_long_interview_preserves_head_middle_and_end_passages(self):
        text = "START Alpha North\n" + ("Interviewer: Question.\nAlpha North: " + "answer " * 250 + "\n") * 15 + "END CONCLUSION"
        result = self.select(text, title="Interview")
        self.check(text, result)
        self.assertEqual(result["spans"][0]["start"], 0)
        self.assertEqual(result["spans"][-1]["end"], len(text))
        self.assertTrue(any(s["start"] < len(text) * 2 // 3 and s["end"] > len(text) // 3 for s in result["spans"]))
        self.assertLessEqual(result["metadata"]["selected_budget_units"], 6000)

    def test_ordinary_common_actor_document_remains_base(self):
        text = "Alpha North " + "ordinary source " * 1000
        result = self.select(text)
        self.assertFalse(result["metadata"]["expanded"])
        self.assertLessEqual(result["metadata"]["selected_budget_units"], 1500)

    def test_custom_rarity_threshold_and_no_actor_invention(self):
        text = "Beta South " + "source " * 1000
        result = self.select(text, {"beta": 8}, policy={"rare_actor_threshold": 8})
        self.assertEqual(result["metadata"]["rare_actor_ids"], ["beta"])
        self.assertEqual(actor_mentions("SuperAlpha NorthExtra and Альфа Норт", self.objects), ["alpha"])

    def test_shared_or_overlapping_alias_does_not_hide_another_actor(self):
        objects = [{"id": "alpha", "name": "Alpha North", "aliases": []},
                   {"id": "other", "name": "Other Person", "aliases": ["North"]},
                   {"id": "accent", "name": "Édouard Ngirente", "aliases": []}]
        text = "ALPHA\nNORTH met E\u0301DOUARD\u00a0Ngirente."
        self.assertEqual(actor_mentions(text, objects), ["accent", "alpha", "other"])

    def test_fallback_bytes_never_claims_exact_tokens_or_exceeds_cap(self):
        text = "Alpha North العربية " * 1000
        result = self.select(text, {"alpha": 1}, encoding=None)
        self.assertFalse(result["metadata"]["token_count_exact"])
        self.assertIsNone(result["metadata"]["selected_tokens"])
        self.check(text, result)

    def test_pure_determinism_no_input_mutation(self):
        document = {"text": "Alpha North " + "source " * 1000, "title": "Interview"}
        counts = {"alpha": 1}
        before = copy.deepcopy((document, self.objects, counts))
        a = select_document_context(document, self.objects, counts, 1000, {}, self.encoding)
        b = select_document_context(document, self.objects, counts, 1000, {}, self.encoding)
        self.assertEqual(a, b)
        self.assertEqual((document, self.objects, counts), before)

    def test_exact_cached_token_budget_if_asset_available(self):
        encoding = offline_encoding()
        if encoding is None:
            self.skipTest("No local tokenizer asset")
        text = "Alpha North\n" + ("Interviewer: A question.\nAlpha North: " + " source words" * 400 + "\n") * 12 + "END"
        result = self.select(text, {"alpha": 2}, title="Interview", encoding=encoding)
        self.assertTrue(result["metadata"]["expanded"])
        self.assertGreater(sum(s["tokens"] for s in result["spans"]), 5000)
        self.assertLessEqual(sum(s["tokens"] for s in result["spans"]), 6000)
        self.check(text, result)

    def test_invalid_policy_or_counts_rejected(self):
        for policy in ({"expanded_max_tokens": 100000}, {"base_max_tokens": 7000}, {"rare_actor_threshold": -1}):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                self.select("Alpha North " * 1000, policy=policy)
        with self.assertRaises(ValueError):
            self.select("Alpha North", {"alpha": 1001})


if __name__ == "__main__":
    unittest.main()
