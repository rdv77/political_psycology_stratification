"""Exact source-coordinate excerpts with one global document budget.

Never joins separated passages. Tokenizer assets are read from local cache only;
without an encoder, UTF-8 bytes are a conservative o200k token-count upper bound,
not an exact token count. Selection is lexical retrieval, not identity proof.
"""
from __future__ import annotations

import base64
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import re
import tempfile
from types import FunctionType
import unicodedata


EXCERPT_POLICY_VERSION = "source-coordinate-document-budget-v2"
_ASSET_URL = "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
_ASSET_KEY = hashlib.sha1(_ASSET_URL.encode()).hexdigest()
_ASSET_SHA = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"
_AUTO_ENCODING = object()


def offline_encoding(cache_dir=None):
    """Load o200k_base ONLY from a verified local rank asset; never download.

    An explicit missing cache does not fall back to another directory. No
    environment/global tokenizer registry is modified; concurrent callers can
    use their own cached encoding. None means conservative byte budgeting.
    """
    if cache_dir is not None:
        paths = [Path(cache_dir)]
    else:
        paths = [Path(x) for x in (os.environ.get("TIKTOKEN_CACHE_DIR"), os.environ.get("DATA_GYM_CACHE_DIR")) if x]
        paths += [Path(__file__).resolve().parents[2] / ".corpus_runtime/tokenizer_cache",
                  Path(tempfile.gettempdir()) / "data-gym-cache"]
    return _offline_encoding(tuple(str(path.resolve()) for path in paths))


@lru_cache(maxsize=8)
def _offline_encoding(paths):
    try:
        import tiktoken
        from tiktoken_ext.openai_public import o200k_base
    except ImportError:
        return None
    for directory in paths:
        try:
            data = (Path(directory) / _ASSET_KEY).read_bytes()
        except OSError:
            continue
        if hashlib.sha256(data).hexdigest() != _ASSET_SHA:
            continue
        ranks = {base64.b64decode(token, validate=True): int(rank)
                 for token, rank in (line.split() for line in data.splitlines() if line)}
        def local_ranks(url, expected_hash=None):
            if url != _ASSET_URL or expected_hash != _ASSET_SHA:
                raise ValueError("Unexpected tokenizer asset; network loading is disabled")
            return ranks
        # Run the installed package's official pattern/special-token constructor
        # with an isolated, local-only loader. No monkeypatch or network fallback.
        constructor = FunctionType(o200k_base.__code__, dict(o200k_base.__globals__, load_tiktoken_bpe=local_ranks))
        return tiktoken.Encoding(**constructor())
    return None


load_local_encoding = offline_encoding


class _Meter:
    def __init__(self, encoding):
        self.encoding = encoding

    def count(self, text):
        if self.encoding is None:
            return len(text.encode("utf-8"))
        return len(self.encoding.encode(text, disallowed_special=()))

    def span(self, text, start, end):
        value = text[start:end]
        return {"start": start, "end": end, "text": value,
                "tokens": self.count(value) if self.encoding is not None else None}


def _fit_length(text, anchor, limit, budget, meter, *, left=False):
    """Find a safe length with exponential bounds, then bounded binary search.

    Head selection does not tokenize an entire multi-megabyte article just to
    keep a short prefix. Prefix token counts can be slightly nonmonotonic; the
    returned boundary is rechecked, not asserted to be the longest possible one.
    """
    if budget <= 0 or limit <= 0:
        return 0
    def cost(length):
        return meter.count(text[anchor - length:anchor] if left else text[anchor:anchor + length])
    low, high = 0, min(limit, max(1, budget * 4))
    while cost(high) <= budget:
        low = high
        if high == limit:
            return high
        high = min(limit, high * 2)
    while low + 1 < high:
        middle = (low + high) // 2
        if cost(middle) <= budget:
            low = middle
        else:
            high = middle
    return low


@lru_cache(maxsize=32)
def _compiled(labels):
    groups = {}
    alternatives = []
    for i, (label, actor_ids) in enumerate(labels):
        name = "a" + str(i)
        variants = sorted({unicodedata.normalize(form, label) for form in ("NFC", "NFD")}, key=lambda x: (-len(x), x))
        parts = [r"\s+".join(re.escape(part) for part in variant.split()) for variant in variants]
        alternatives.append("(?P<" + name + ">" + "|".join(parts) + ")")
        groups[name] = actor_ids
    if not alternatives:
        return None, groups
    return re.compile(r"(?<!\w)(?:" + "|".join(alternatives) + r")(?!\w)", re.IGNORECASE), groups


def _matcher(objects):
    labels = {}
    for obj in objects:
        actor_id = obj.get("id")
        if not isinstance(actor_id, str) or not actor_id:
            raise ValueError("Objects require nonempty actor IDs")
        for label in [obj.get("name", ""), *obj.get("aliases", [])]:
            if not isinstance(label, str):
                raise ValueError("Names and aliases must be strings")
            label = " ".join(label.split())
            if label:
                labels.setdefault(label, set()).add(actor_id)
    return _compiled(tuple((label, tuple(sorted(ids))) for label, ids in sorted(labels.items(), key=lambda x: (-len(x[0]), x[0]))))


def _merge(spans):
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def select_spans_with_metadata(text, objects, mode="full", max_tokens=500, encoding=_AUTO_ENCODING, *, max_windows=4):
    """Select exact source spans. Omitted encoding auto-loads local cache only.

    Explicit encoding=None forces conservative UTF-8 byte budgeting, even when
    another workspace has a tokenizer cache. This keeps a caller's recipe and
    selection policy consistent after its own store-specific cache lookup.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if mode not in {"full", "head", "mentions"}:
        raise ValueError("mode must be full, head or mentions")
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError("Positive integer max_tokens required")
    if type(max_windows) is not int or not 1 <= max_windows <= 4:
        raise ValueError("max_windows must be 1..4")
    encoding = offline_encoding() if encoding is _AUTO_ENCODING else encoding
    meter = _Meter(encoding)
    matched, covered = set(), set()
    bounds = []
    effective, fallback = mode, None
    if text and mode == "full":
        bounds = [(0, len(text))]
    elif text and mode == "head":
        end = _fit_length(text, 0, len(text), max_tokens, meter)
        if end:
            bounds = [(0, end)]
    elif text:
        pattern, groups = _matcher(objects)
        # Retain first/last matches in four document regions per actor. Memory
        # is bounded by roster size, even if Kagame appears a million times.
        retained = {}
        if pattern:
            for match in pattern.finditer(text):
                for actor_id in groups[match.lastgroup]:
                    matched.add(actor_id)
                    region = min(3, match.start() * 4 // len(text))
                    key = actor_id, region
                    point = (match.start(), match.end())
                    if key not in retained:
                        retained[key] = [point, point]
                    else:
                        retained[key][1] = point
        # A source already within the document cap needs no excerpting. The
        # bounded prefix search stops near the cap for huge articles instead of
        # encoding the entire source merely to discover that it is too long.
        whole_fits = _fit_length(text, 0, len(text), max_tokens, meter) == len(text)
        if whole_fits:
            bounds = [(0, len(text))]
            covered = set(matched)
            if not matched:
                effective, fallback = "head", "no_mentions"
        points = [] if whole_fits else sorted({point for pair in retained.values() for point in pair})
        point_actors = {}
        for (actor_id, _), pair in retained.items():
            for point in pair:
                point_actors.setdefault(point, set()).add(actor_id)
        target_windows = min(max_windows, max(1, max_tokens // 150))
        target_size = min(400, max(1, max_tokens // target_windows))
        attempted = set()
        while points and len(bounds) < max_windows and len(attempted) < max_windows * 8:
            spent = sum(meter.count(text[a:b]) for a, b in bounds)
            if spent >= max_tokens:
                break
            candidates = [point for point in points if point not in attempted
                          and not any(a <= point[0] and point[1] <= b for a, b in bounds)]
            if not candidates:
                break
            def priority(point):
                ids = point_actors[point]
                distance = min((max(a - point[1], point[0] - b, 0) for a, b in bounds), default=0)
                return (bool(ids - covered), len(ids - covered), distance, -point[0])
            start, end = max(candidates, key=priority)
            attempted.add((start, end))
            budget = min(target_size, max_tokens - spent)
            mention_cost = meter.count(text[start:end])
            if budget < mention_cost:
                continue
            left_budget = max(0, (budget - mention_cost) // 2)
            left = start - _fit_length(text, start, start, left_budget, meter, left=True)
            right = left + _fit_length(text, left, len(text) - left, budget, meter)
            if right < end:
                left = start
                right = start + _fit_length(text, start, len(text) - start, budget, meter)
            if right < end:
                continue
            proposal = _merge(bounds + [(left, right)])
            if sum(meter.count(text[a:b]) for a, b in proposal) > max_tokens:
                continue
            bounds = proposal
            covered = {actor_id for a, b in bounds for match in pattern.finditer(text[a:b])
                       for actor_id in groups[match.lastgroup]}
        if not bounds:
            effective = "head"
            fallback = "no_mentions" if not matched else "mentions_exceed_budget"
            end = _fit_length(text, 0, len(text), max_tokens, meter)
            if end:
                bounds = [(0, end)]
    spans = [meter.span(text, start, end) for start, end in bounds]
    units = sum(meter.count(span["text"]) for span in spans)
    return {"spans": spans, "metadata": {
        "policy_version": EXCERPT_POLICY_VERSION, "mode": mode, "effective_mode": effective,
        "tokenizer": getattr(encoding, "name", "provided_encoding") if encoding is not None else None,
        "token_count_exact": encoding is not None,
        "budget_unit": "tokens" if encoding is not None else "utf8_bytes_upper_bound",
        "max_tokens": max_tokens, "budget_applied": mode != "full", "selected_budget_units": units,
        "selected_tokens": units if encoding is not None else None,
        "no_mentions_fallback": fallback == "no_mentions", "fallback_reason": fallback,
        "matched_actor_ids": sorted(matched), "covered_actor_ids": sorted(covered),
        "truncated": bounds != ([(0, len(text))] if text else []), "span_count": len(spans)}}


def select_spans(text, objects, mode="full", max_tokens=500, encoding=_AUTO_ENCODING, *, max_windows=4):
    """Return source spans only; companion API exposes exact/fallback metadata."""
    return select_spans_with_metadata(text, objects, mode, max_tokens, encoding, max_windows=max_windows)["spans"]
