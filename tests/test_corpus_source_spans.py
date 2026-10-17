"""Synthetic source-boundary regressions; no corpus, filesystem or API access."""
import copy
import unittest

from src.corpus_pipeline.source_spans import (
    document_spans, document_texts, fact_span, quote_location,
)


class CorpusSourceSpansTests(unittest.TestCase):
    def span(self, start, text):
        return {"start": start, "end": start + len(text), "text": text}

    def fact(self, **changes):
        value = {"quote": "We agree.", "attribution_evidence": "Alpha North said:",
                 "date_evidence": None}
        value.update(changes)
        return value

    def test_sorted_ranges_are_copies_with_original_character_coordinates(self):
        document = {"excerpts": [self.span(300, "Альфа 🟢"), self.span(20, "First.")]}
        original = copy.deepcopy(document)
        spans = document_spans(document)
        self.assertEqual([span["start"] for span in spans], [20, 300])
        self.assertEqual(spans[1]["end"], 307)
        self.assertEqual(quote_location(document, "🟢"), 306)
        spans[0]["start"] = 999
        self.assertEqual(document, original)

    def test_legacy_text_uses_absolute_chunk_start_and_actual_length(self):
        text = "Alpha North said: We agree."
        document = {"text": text, "chunk_start": 140, "chunk_end": 9999}
        self.assertEqual(document_spans(document), [self.span(140, text)])
        self.assertEqual(quote_location(document, "We agree."), 158)
        self.assertEqual(fact_span(document, self.fact()), self.span(140, text))
        self.assertEqual(document_spans({"text": "x"}), [self.span(0, "x")])
        self.assertEqual(document_spans({"text": ""}), [])

    def test_neither_quotes_nor_identity_passages_cross_even_adjacent_boundaries(self):
        document = {"excerpts": [self.span(0, "Alpha "), self.span(6, "North said:")]}
        self.assertEqual(document_texts(document), ["Alpha ", "North said:"])
        self.assertIsNone(quote_location(document, "Alpha North said:"))
        self.assertIsNone(fact_span(document, self.fact(
            quote="Alpha North said:", attribution_evidence="North said:")))

    def test_attribution_in_another_range_does_not_validate_quote(self):
        document = {"excerpts": [self.span(0, "Alpha North said:"),
                                   self.span(50, "We agree.")]}
        self.assertEqual(quote_location(document, "We agree."), 50)
        self.assertIsNone(fact_span(document, self.fact()))

    def test_date_evidence_must_share_the_quote_and_attribution_range(self):
        text = "Alpha North said: We agree."
        document = {"excerpts": [self.span(10, text), self.span(100, "2026-10-15")]}
        self.assertIsNone(fact_span(document, self.fact(date_evidence="2026-10-15")))
        for unknown in (None, ""):
            with self.subTest(date_evidence=unknown):
                self.assertEqual(fact_span(document, self.fact(date_evidence=unknown)),
                                 self.span(10, text))

    def test_all_evidence_in_one_range_preserves_absolute_offsets(self):
        text = "2026-10-15 Alpha North said: We agree."
        document = {"excerpts": [self.span(800, text)]}
        self.assertEqual(fact_span(document, self.fact(date_evidence="2026-10-15")),
                         self.span(800, text))
        self.assertEqual(quote_location(document, "We agree."), 829)

    def test_later_identical_quote_does_not_rescue_first_range(self):
        document = {"excerpts": [self.span(500, "Alpha North said: We agree."),
                                   self.span(100, "Unattributed: We agree.")]}
        self.assertEqual(quote_location(document, "We agree."), 114)
        self.assertIsNone(fact_span(document, self.fact()))

    def test_later_identical_quote_with_date_does_not_rescue_first_range(self):
        document = {"excerpts": [self.span(0, "Alpha North said: We agree."),
                                   self.span(100, "2026-10-15 Alpha North said: We agree.")]}
        self.assertIsNone(fact_span(document, self.fact(date_evidence="2026-10-15")))

    def test_gap_limit_is_left_to_evidence_layer(self):
        text = "Alpha North said:" + "." * 400 + "We agree."
        document = {"excerpts": [self.span(0, text)]}
        self.assertEqual(fact_span(document, self.fact()), self.span(0, text))

    def test_malformed_ranges_and_ambiguous_representations_are_rejected(self):
        invalid = [
            None, {}, {"text": None}, {"text": "x", "chunk_start": -1},
            {"text": "x", "chunk_start": True}, {"excerpts": []},
            {"excerpts": None}, {"excerpts": [None]},
            {"excerpts": [self.span(0, "x")], "text": "full text"},
            {"excerpts": [{"start": 0, "end": 2, "text": "x"}]},
            {"excerpts": [{"start": 0, "end": 0, "text": ""}]},
            {"excerpts": [{"start": True, "end": 2, "text": "x"}]},
            {"excerpts": [{"start": 0.0, "end": 1, "text": "x"}]},
            {"excerpts": [{"start": -1, "end": 0, "text": "x"}]},
            {"excerpts": [self.span(5, "abcd"), self.span(7, "xy")]},
            {"excerpts": [self.span(5, "abcd"), self.span(5, "abcd")]},
        ]
        for document in invalid:
            with self.subTest(document=document), self.assertRaises(ValueError):
                document_spans(document)

    def test_missing_or_invalid_fact_evidence_does_not_match(self):
        document = {"text": "Alpha North said: We agree."}
        changes = [{"quote": ""}, {"quote": " "}, {"quote": None},
                   {"attribution_evidence": None}, {"attribution_evidence": " "},
                   {"attribution_evidence": "Fabricated"}, {"date_evidence": 1},
                   {"date_evidence": " "}, {"quote": "We disagree."}]
        for change in changes:
            with self.subTest(change=change):
                self.assertIsNone(fact_span(document, self.fact(**change)))
        self.assertIsNone(fact_span(document, None))
        self.assertIsNone(quote_location(document, ""))
        self.assertIsNone(quote_location(document, "Missing"))


if __name__ == "__main__":
    unittest.main()
