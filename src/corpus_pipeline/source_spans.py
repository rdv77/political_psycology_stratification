"""Exact, separate source ranges for full-text and bounded documents.

Offsets are Python character offsets in the original document. This module
validates the declared ranges; only the excerpt producer has the original
full text with which to verify their provenance. Ranges are never joined.
"""
from __future__ import annotations

from collections.abc import Mapping


def _offset(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def document_spans(document: dict) -> list[dict]:
    """Return sorted, nonoverlapping ``{start, end, text}`` copies.

    Bounded documents require a nonempty excerpts list and omit ``text``.
    Legacy documents retain their text and optional chunk_start; chunk_end
    is not trusted for the range length. Empty legacy text has no ranges.
    Malformed representations raise ValueError rather than falling back to
    an accidentally retained full text.
    """
    if not isinstance(document, Mapping):
        raise ValueError("A source document must be an object")
    if "excerpts" not in document:
        text = document.get("text")
        if not isinstance(text, str):
            raise ValueError("Legacy document.text must be a string")
        start = _offset(document.get("chunk_start", 0), "chunk_start")
        return [{"start": start, "end": start + len(text), "text": text}] if text else []

    if "text" in document:
        raise ValueError("Bounded documents must omit document.text")
    excerpts = document["excerpts"]
    if not isinstance(excerpts, list) or not excerpts:
        raise ValueError("document.excerpts must be a nonempty list")
    spans = []
    for index, item in enumerate(excerpts):
        if not isinstance(item, Mapping):
            raise ValueError(f"excerpts[{index}] must be an object")
        start = _offset(item.get("start"), f"excerpts[{index}].start")
        end = _offset(item.get("end"), f"excerpts[{index}].end")
        text = item.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError(f"excerpts[{index}].text must be a nonempty string")
        if end <= start or end - start != len(text):
            raise ValueError(f"excerpts[{index}] range must match its text length")
        spans.append({"start": start, "end": end, "text": text})
    spans.sort(key=lambda span: span["start"])
    for previous, current in zip(spans, spans[1:]):
        if current["start"] < previous["end"]:
            raise ValueError("Source excerpts must not overlap")
    return spans


def document_texts(document: dict) -> list[str]:
    """Return separate passages for identity checks, without concatenation."""
    return [span["text"] for span in document_spans(document)]


def quote_location(document: dict, quote: str) -> int | None:
    """Return the first exact quote's absolute start, or None if not found."""
    spans = document_spans(document)
    if not isinstance(quote, str) or not quote.strip():
        return None
    for span in spans:
        offset = span["text"].find(quote)
        if offset >= 0:
            return span["start"] + offset
    return None


def fact_span(document: dict, fact: dict) -> dict | None:
    """Find the first quote range and require all evidence in that range.

    A later copy of the quote cannot rescue its first occurrence. Evidence
    in another range cannot validate it, even when the ranges are adjacent.
    Attribution gap, identity, date semantics and claim validity remain the
    caller's responsibility (in particular, evidence.py's locality check).
    """
    spans = document_spans(document)
    if not isinstance(fact, Mapping):
        return None
    quote = fact.get("quote")
    attribution = fact.get("attribution_evidence")
    date_evidence = fact.get("date_evidence")
    if not isinstance(quote, str) or not quote.strip():
        return None
    if not isinstance(attribution, str) or not attribution.strip():
        return None
    if date_evidence is not None and date_evidence != "":
        if not isinstance(date_evidence, str) or not date_evidence.strip():
            return None
    for span in spans:
        text = span["text"]
        if quote not in text:
            continue
        if attribution not in text or (date_evidence and date_evidence not in text):
            return None
        return span
    return None
