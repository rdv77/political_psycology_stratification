"""Deterministic capped adaptive context; lexical signals are not facts/identity.

Base mentions context is expanded for source-rare actors or long-form interview
candidates. Each returned passage is an exact source substring with global
character coordinates. No source, API, model result or approval is modified.
"""
from __future__ import annotations

import math
import re
import unicodedata

from .excerpts import EXCERPT_POLICY_VERSION, select_spans_with_metadata

CONTEXT_POLICY_VERSION = "adaptive-source-context-v1"
_INTERVIEW = re.compile(r"\b(?:interviews?|q\s*&\s*a|questions?[- ]and[- ]answers?|entretiens?|"
                        r"questions?[- ]r[eé]ponses?|ikiganiro|yaganiriye|mahojiano|maswali\s+na\s+majibu)\b", re.IGNORECASE)
_SPEAKER = re.compile(r"^[ \t]*([^\W\d_][^\r\n:?!]{0,70}):(?=[ \t])", re.MULTILINE | re.UNICODE)


def actor_mentions(text, objects):
    """Approved lexical actor matches only, not verified identity or prominence."""
    normalized = " ".join(unicodedata.normalize("NFC", text).casefold().split())
    found = []
    # Independent actor checks are deliberate: a longest full-name match must
    # not hide another actor's shared or nested alias at the same position.
    for obj in objects:
        for label in [obj.get("name", ""), *obj.get("aliases", [])]:
            label = " ".join(unicodedata.normalize("NFC", label).casefold().split())
            if label and re.search(r"(?<!\w)" + re.escape(label) + r"(?!\w)", normalized):
                found.append(obj["id"])
                break
    return sorted(set(found))


def _units(text, encoding):
    return len(text.encode("utf-8")) if encoding is None else len(encoding.encode(text, disallowed_special=()))


def _head(text, start, end, budget, encoding):
    if budget <= 0 or start >= end:
        return []
    selected = select_spans_with_metadata(text[start:end], [], "head", budget, encoding)
    return [dict(span, start=span["start"] + start, end=span["end"] + start) for span in selected["spans"]]


def _suffix_start(text, end, budget, encoding):
    # The head helper stops near its budget even for a very long suffix. This
    # binary search therefore never tokenizes the whole multi-megabyte source.
    low, high = 0, end
    while low < high:
        middle = (low + high) // 2
        selected = select_spans_with_metadata(text[middle:end], [], "head", max(1, budget), encoding)
        if not selected["metadata"]["truncated"]:
            high = middle
        else:
            low = middle + 1
    return low


def _merge(text, spans, encoding):
    bounds = []
    for span in sorted(spans, key=lambda s: (s["start"], s["end"])):
        if bounds and span["start"] <= bounds[-1][1]:
            bounds[-1] = (bounds[-1][0], max(bounds[-1][1], span["end"]))
        else:
            bounds.append((span["start"], span["end"]))
    return [{"start": a, "end": b, "text": text[a:b], "tokens": _units(text[a:b], encoding) if encoding is not None else None}
            for a, b in bounds]


def _longform(text, title, min_chars):
    targets = (len(text) // 3, len(text) * 2 // 3)
    nearest = [None, None]
    turns = 0
    speakers = set()
    for match in _SPEAKER.finditer(text):
        turns += 1
        if len(speakers) < 128:
            speakers.add(match.group(1).strip().casefold())
        for i, target in enumerate(targets):
            if nearest[i] is None or abs(match.start() - target) < abs(nearest[i] - target):
                nearest[i] = match.start()
    keyword = bool(_INTERVIEW.search(text))
    title_keyword = bool(_INTERVIEW.search(title))
    candidate = len(text) >= min_chars and (title_keyword or (keyword and turns >= 2) or (turns >= 4 and len(speakers) >= 2))
    return {"candidate": candidate, "interview_keyword_in_source": keyword,
            "interview_keyword_in_title": title_keyword, "speaker_turn_markers": turns,
            "distinct_speaker_markers_capped128": len(speakers),
            "middle_anchors": [point if point is not None else target for point, target in zip(nearest, targets)],
            "language_inferred": False}


def _rare_positions(text, objects, ids):
    found = {}
    for obj in objects:
        if obj["id"] not in ids:
            continue
        points = []
        for label in [obj.get("name", ""), *obj.get("aliases", [])]:
            for form in ("NFC", "NFD"):
                label_variant = unicodedata.normalize(form, label)
                if not label_variant.strip():
                    continue
                pattern = r"(?<!\w)" + r"\s+".join(re.escape(p) for p in label_variant.split()) + r"(?!\w)"
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    points.append(match.start())
        if points:
            found[obj["id"]] = min(points)
    return found


def select_document_context(document, objects, actor_document_counts, eligible_document_count, policy, encoding):
    """Return {spans, metadata}; encoding=None means explicit byte fallback.

    Policy keys: base_max_tokens (1500), expanded_max_tokens (6000),
    rare_actor_threshold (auto=max(5,floor(.001*N))), longform_min_chars (8000).
    Count rarity describes lexical coverage in eligible unique source documents,
    not an actor's importance, beliefs, or correctness of an ambiguous alias.
    """
    text = document.get("text", "")
    if not isinstance(text, str):
        raise TypeError("document.text must be a string")
    policy = policy or {}
    base_cap = policy.get("base_max_tokens", 1500)
    expanded_cap = policy.get("expanded_max_tokens", 6000)
    min_chars = policy.get("longform_min_chars", 8000)
    if (type(base_cap) is not int or type(expanded_cap) is not int
            or not 1 <= base_cap <= expanded_cap <= 6000
            or type(min_chars) is not int or min_chars < 1):
        raise ValueError("Require 1 <= base_max_tokens <= expanded_max_tokens <= 6000 and positive longform_min_chars")
    if type(eligible_document_count) is not int or eligible_document_count < 0:
        raise ValueError("eligible_document_count must be a nonnegative integer")
    threshold = policy.get("rare_actor_threshold")
    if threshold is None:
        threshold = max(5, math.floor(eligible_document_count * .001))
    if type(threshold) is not int or threshold < 0:
        raise ValueError("rare_actor_threshold must be a nonnegative integer")
    approved = {obj["id"] for obj in objects}
    for actor, count in actor_document_counts.items():
        if actor in approved and (type(count) is not int or count < 0 or count > eligible_document_count):
            raise ValueError("Actor counts must be nonnegative eligible-unique document counts")
    base = select_spans_with_metadata(text, objects, "mentions", base_cap, encoding)
    matched = set(actor_mentions(text, objects))
    rare = sorted(actor for actor in matched if actor in actor_document_counts
                  and eligible_document_count > 0 and actor_document_counts[actor] <= threshold)
    signals = _longform(text, str(document.get("title", "")), min_chars)
    reasons = (["rare_actor_source_coverage"] if rare else []) + (["longform_interview_candidate"] if signals["candidate"] else [])
    expand = base["metadata"]["truncated"] and bool(reasons) and expanded_cap > base_cap
    cap = expanded_cap if expand else base_cap
    spans = base["spans"]
    strategies = ["base_mentions"]
    if expand:
        complete = select_spans_with_metadata(text, [], "head", cap, encoding)
        if not complete["metadata"]["truncated"]:
            spans = complete["spans"]
            strategies = ["complete_source_within_expanded_cap"]
        else:
            # Four contextual passages: introduction, two middle/rare-actor
            # anchors, and ending. Anchor windows remain separate until overlap.
            middle = list(signals["middle_anchors"])
            positions = _rare_positions(text, objects, set(rare))
            preview_budget = max(1, cap // 4)
            head_preview = _head(text, 0, len(text), preview_budget, encoding)
            head_end = head_preview[-1]["end"] if head_preview else 0
            tail_start = _suffix_start(text, len(text), preview_budget, encoding)
            # Character-percent cutoffs can hide a rare name just before the
            # actual tail window. Use the real token-bounded head/tail ranges.
            inner_rare = sorted({pos for pos in positions.values() if head_end <= pos < tail_start})
            for i, point in enumerate(inner_rare[:2]):
                middle[i] = point
            anchors = [("head", 0), ("middle_or_rare", middle[0]), ("middle_or_rare", middle[1]), ("end", len(text))]
            spans = []
            for index, (kind, anchor) in enumerate(anchors):
                spent = sum(_units(span["text"], encoding) for span in spans)
                budget = max(0, (cap - spent) // (len(anchors) - index))
                if not budget:
                    break
                if kind == "end":
                    start = _suffix_start(text, len(text), budget, encoding)
                elif kind == "head":
                    start = 0
                else:
                    start = _suffix_start(text, anchor, max(1, budget // 3), encoding)
                proposed = _head(text, start, len(text), budget, encoding)
                merged = _merge(text, spans + proposed, encoding)
                if sum(_units(span["text"], encoding) for span in merged) <= cap:
                    spans = merged
            # Reuse a budget freed by overlapping passages, extending contiguous
            # source ranges; never fill a gap with manufactured joining text.
            for index in range(len(spans)):
                remaining = cap - sum(_units(span["text"], encoding) for span in spans)
                if remaining <= 1:
                    break
                old = spans[index]
                proposed = _head(text, old["start"], len(text), _units(old["text"], encoding) + remaining, encoding)
                merged = _merge(text, spans[:index] + spans[index + 1:] + proposed, encoding)
                if len(merged) <= 4 and sum(_units(span["text"], encoding) for span in merged) <= cap:
                    spans = merged
                    break  # Merging can change indices; one bounded fill pass.
            strategies = ["head", "middle_dialogue_or_rare_actor", "ending", "bounded_gap_fill"]
    selected_units = sum(_units(span["text"], encoding) for span in spans)
    if selected_units > cap or len(spans) > 4:
        raise AssertionError("Adaptive selection exceeded its document budget")
    covered = sorted({actor for span in spans for actor in actor_mentions(span["text"], objects)})
    metadata = dict(base["metadata"], context_policy_version=CONTEXT_POLICY_VERSION,
                    excerpt_policy_version=EXCERPT_POLICY_VERSION, adaptive=True,
                    base_max_tokens=base_cap, max_tokens=cap, effective_max_tokens=cap, expanded_max_tokens=expanded_cap,
                    effective_mode="adaptive_mentions" if expand else base["metadata"]["effective_mode"],
                    base_no_mentions_fallback=base["metadata"]["no_mentions_fallback"],
                    no_mentions_fallback=base["metadata"]["no_mentions_fallback"] and not expand,
                    expanded=bool(expand), reason=reasons if expand else ["source_fits_base" if not base["metadata"]["truncated"] else "base_context"],
                    selection_strategies=strategies, rare_actor_threshold=threshold, rare_actor_ids=rare,
                    missing_actor_count_ids=sorted(matched - actor_document_counts.keys()),
                    rarity_basis="lexical_source_document_coverage_not_identity_or_prominence",
                    longform_signals={key: value for key, value in signals.items() if key != "middle_anchors"},
                    matched_actor_ids=sorted(matched), covered_actor_ids=covered,
                    uncovered_rare_actor_ids=sorted(set(rare) - set(covered)), selected_budget_units=selected_units,
                    selected_tokens=selected_units if encoding is not None else None,
                    span_count=len(spans), truncated=not (len(spans) == 1 and spans[0]["start"] == 0 and spans[0]["end"] == len(text)))
    return {"spans": spans, "metadata": metadata}
