"""Source coordinates, document-level budgets and offline tokenizer checks."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.corpus_pipeline.excerpts import offline_encoding, select_spans, select_spans_with_metadata


class CharacterEncoding:
    name = "synthetic_character_encoder"
    def encode(self, text, **kwargs):
        return list(text)


class CorpusExcerptTests(unittest.TestCase):
    def setUp(self):
        self.encoding = CharacterEncoding()
        self.objects = [{"id": "alpha", "name": "Alpha North", "aliases": ["Альфа Норт"]},
                        {"id": "beta", "name": "Beta South", "aliases": []},
                        {"id": "gamma", "name": "Gamma East", "aliases": []}]

    def check_spans(self, text, result, budget=None):
        spans = result["spans"]
        for span in spans:
            self.assertEqual(span["text"], text[span["start"]:span["end"]])
            self.assertGreater(span["end"], span["start"])
        for left, right in zip(spans, spans[1:]):
            self.assertLess(left["end"], right["start"])
        if budget is not None:
            self.assertLessEqual(result["metadata"]["selected_budget_units"], budget)

    def test_full_is_one_exact_span_and_ignores_excerpt_cap(self):
        text = "A full source αβγ\n" * 100
        result = select_spans_with_metadata(text, self.objects, "full", 5, self.encoding)
        self.assertEqual(result["spans"], [{"start": 0, "end": len(text), "text": text, "tokens": len(text)}])
        self.assertFalse(result["metadata"]["budget_applied"])

    def test_head_is_exact_prefix_not_unicode_replacement(self):
        text = "Édouard العربية й 🇷🇼 text " * 100
        result = select_spans_with_metadata(text, self.objects, "head", 33, self.encoding)
        self.assertEqual(result["spans"][0]["text"], text[:33])
        self.check_spans(text, result, 33)

    def test_mentions_prioritize_different_actors_over_repeated_name(self):
        text = ("Alpha North " + "a" * 250 + " ") * 8 + "Beta South " + "b" * 800 + " Gamma East " + "c" * 800
        result = select_spans_with_metadata(text, self.objects, "mentions", 600, self.encoding)
        self.check_spans(text, result, 600)
        self.assertEqual(result["metadata"]["covered_actor_ids"], ["alpha", "beta", "gamma"])
        self.assertLessEqual(len(result["spans"]), 4)

    def test_overlapping_windows_are_merged_without_duplicate_text(self):
        text = "context " * 20 + "Alpha North and Beta South spoke." + " context" * 30
        result = select_spans_with_metadata(text, self.objects, "mentions", 500, self.encoding)
        self.check_spans(text, result, 500)
        self.assertEqual(len(result["spans"]), 1)
        self.assertEqual(result["metadata"]["covered_actor_ids"], ["alpha", "beta"])

    def test_mentions_preserves_whole_short_source_instead_of_gapped_windows(self):
        text = "context words " * 28 + "Alpha North spoke."
        self.assertTrue(300 <= len(text) <= 450)
        result = select_spans_with_metadata(text, self.objects, "mentions", 500, self.encoding)
        self.assertEqual(result["spans"], [{"start": 0, "end": len(text), "text": text, "tokens": len(text)}])
        self.assertFalse(result["metadata"]["truncated"])
        self.assertEqual(result["metadata"]["matched_actor_ids"], ["alpha"])
        self.assertEqual(result["metadata"]["covered_actor_ids"], ["alpha"])

    def test_unicode_aliases_case_whitespace_diacritic_and_word_boundaries(self):
        objects = [{"id": "edouard", "name": "Édouard Ngirente", "aliases": []}]
        text = "prefix " * 90 + "E\u0301DOUARD\nNGIRENTE speaks." + " suffix" * 90
        result = select_spans_with_metadata(text, objects, "mentions", 150, self.encoding)
        self.assertEqual(result["metadata"]["covered_actor_ids"], ["edouard"])
        self.check_spans(text, result, 150)
        result = select_spans_with_metadata("SuperAlpha NorthExtra", self.objects, "mentions", 100, self.encoding)
        self.assertTrue(result["metadata"]["no_mentions_fallback"])

    def test_no_mentions_falls_back_to_head_with_explicit_metadata(self):
        text = "No named actor here. " * 100
        result = select_spans_with_metadata(text, self.objects, "mentions", 100, self.encoding)
        self.assertTrue(result["metadata"]["no_mentions_fallback"])
        self.assertEqual(result["metadata"]["effective_mode"], "head")
        self.assertEqual(result["spans"][0]["text"], text[:100])

    def test_total_budget_not_per_window_for_500_1000_1500(self):
        text = ("Alpha North " + "a" * 1200 + " Beta South " + "b" * 1200 + " Gamma East " + "c" * 1200 + " ") * 3
        for budget in (500, 1000, 1500):
            with self.subTest(budget=budget):
                result = select_spans_with_metadata(text, self.objects, "mentions", budget, self.encoding)
                self.check_spans(text, result, budget)
                self.assertLessEqual(len(result["spans"]), 4)

    def test_fallback_is_explicit_and_uses_conservative_utf8_bytes(self):
        text = "العربية Édouard й " * 50
        with patch("src.corpus_pipeline.excerpts.offline_encoding", return_value=None):
            result = select_spans_with_metadata(text, self.objects, "head", 50)
        self.assertFalse(result["metadata"]["token_count_exact"])
        self.assertIsNone(result["spans"][0]["tokens"])
        self.assertEqual(result["metadata"]["budget_unit"], "utf8_bytes_upper_bound")
        self.assertLessEqual(len(result["spans"][0]["text"].encode("utf-8")), 50)
        self.check_spans(text, result, 50)

    def test_explicit_none_never_loads_another_workspaces_cached_encoder(self):
        text = "العربية Édouard й " * 100
        with patch("src.corpus_pipeline.excerpts.offline_encoding", side_effect=AssertionError("Explicit None must not auto-load")):
            result = select_spans_with_metadata(text, self.objects, "head", 500, encoding=None)
            spans = select_spans(text, self.objects, "head", 500, encoding=None)
        self.assertFalse(result["metadata"]["token_count_exact"])
        self.assertLessEqual(sum(len(s["text"].encode("utf-8")) for s in spans), 500)

    def test_omitted_encoding_auto_loads_local_encoder(self):
        with patch("src.corpus_pipeline.excerpts.offline_encoding", return_value=self.encoding) as loader:
            result = select_spans_with_metadata("Synthetic source", self.objects, "head", 10)
        loader.assert_called_once_with()
        self.assertTrue(result["metadata"]["token_count_exact"])
        self.assertEqual(result["spans"][0]["tokens"], 10)

    def test_default_cache_loader_never_calls_network_get_encoding(self):
        with tempfile.TemporaryDirectory() as name:
            with patch("tiktoken.get_encoding", side_effect=AssertionError("No network-capable loader")):
                self.assertIsNone(offline_encoding(Path(name)))

    def test_real_cached_o200k_budget_when_asset_is_available(self):
        with patch("tiktoken.get_encoding", side_effect=AssertionError("No network-capable loader")):
            encoding = offline_encoding()
        if encoding is None:
            self.skipTest("No locally cached tokenizer asset")
        text = "Rwanda " + "Édouard العربية не Alpha North spoke. " * 300
        result = select_spans_with_metadata(text, self.objects, "head", 500, encoding)
        self.assertTrue(result["metadata"]["token_count_exact"])
        self.assertLessEqual(sum(len(encoding.encode(s["text"])) for s in result["spans"]), 500)
        self.check_spans(text, result, 500)

    def test_real_300_to_450_token_source_keeps_full_context_in_mentions_mode(self):
        encoding = offline_encoding()
        if encoding is None:
            self.skipTest("No locally cached tokenizer asset")
        text = " source" * 380 + " Alpha North spoke."
        count = len(encoding.encode(text, disallowed_special=()))
        self.assertTrue(300 <= count <= 450)
        result = select_spans_with_metadata(text, self.objects, "mentions", 500, encoding)
        self.assertEqual(len(result["spans"]), 1)
        self.assertEqual(result["spans"][0]["text"], text)
        self.assertEqual(result["spans"][0]["tokens"], count)
        self.assertFalse(result["metadata"]["truncated"])

    def test_deterministic_no_input_mutation_and_empty_source(self):
        before = copy.deepcopy(self.objects)
        text = "Alpha North " + "a" * 1000 + " Beta South " + "b" * 1000
        first = select_spans(text, self.objects, "mentions", 500, self.encoding)
        self.assertEqual(first, select_spans(text, self.objects, "mentions", 500, self.encoding))
        self.assertEqual(self.objects, before)
        self.assertEqual(select_spans("", self.objects, "head", 100, self.encoding), [])

    def test_invalid_mode_and_budget_rejected(self):
        for kwargs in ({"mode": "invented"}, {"max_tokens": 0}, {"max_tokens": True}, {"max_windows": 5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                select_spans("Synthetic source", self.objects, encoding=self.encoding, **kwargs)


if __name__ == "__main__":
    unittest.main()
