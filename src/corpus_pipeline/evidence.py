"""Conservative per-fact extraction normalization; no API or persistence.

Lexical locality and exact substrings are necessary checks, not proof that a
source assertion is true or that a nearby date describes the selected action.
The caller retains the original model response as immutable provenance.
"""
from __future__ import annotations

import copy
from datetime import date
import re
import unicodedata
from typing import Callable
from .source_spans import document_spans, fact_span


EVIDENCE_POLICY_VERSION = "local-attribution-date-review-v1"
MAX_ATTRIBUTION_GAP = 300
_BASES = frozenset({"direct_statement", "reported_statement", "reported_action",
                    "third_party_claim", "context", "uncertain"})
_OWN_KINDS = frozenset({"public_statement", "action"})
_EXPECTED_OWN_KIND = {"direct_statement": "public_statement",
                      "reported_statement": "public_statement", "reported_action": "action"}


def _lexical_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def _actor_mentioned(actor: dict, quote: str, attribution: str) -> bool:
    # Check the two fields separately: concatenating them must not manufacture
    # a full name out of unrelated fragments.
    passages = [_lexical_text(quote), _lexical_text(attribution)]
    for label in [actor.get("name", ""), *actor.get("aliases", [])]:
        label = _lexical_text(label)
        if label and any(re.search(r"(?<!\w)" + re.escape(label) + r"(?!\w)", p) for p in passages):
            return True
    return False


def _local_passages(text: str, quote: str, attribution: str) -> bool:
    # Match the first quote occurrence, as the persistence layer uses .index()
    # for its source offset. A later identical quote must not validate an
    # earlier unrelated occurrence.
    start = text.find(quote)
    end = start + len(quote)
    offset = text.find(attribution)
    while offset >= 0:
        other_end = offset + len(attribution)
        gap = max(start - other_end, offset - end, 0)
        if gap <= MAX_ATTRIBUTION_GAP:
            return True
        offset = text.find(attribution, offset + 1)
    return False


def _outside_period(value, context: dict) -> bool:
    try:
        event = date.fromisoformat(value)
        config = context.get("config", {})
        return bool((config.get("date_from") and event < date.fromisoformat(config["date_from"]))
                    or (config.get("date_to") and event > date.fromisoformat(config["date_to"])))
    except (TypeError, ValueError):
        return False  # The supplied validator handles malformed dates/config.


def _date_problem(fact: dict, text: str) -> str | None:
    evidence = fact.get("date_evidence")
    if not isinstance(evidence, str) or not evidence or evidence not in text:
        return "date_not_grounded"
    if evidence not in fact["quote"] and evidence not in fact["attribution_evidence"]:
        return "date_not_local"
    event = fact.get("event_date")
    if not isinstance(event, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", event):
        return None  # Do not repair an invalid date into an accepted unknown.
    year = event[:4]
    if not re.search(r"(?<!\d)" + re.escape(year) + r"(?!\d)", evidence):
        return "date_year_missing"
    numeric = re.findall(r"(?<!\d)(\d{4}-\d{2}-\d{2}|\d{1,2}\.\d{1,2}\.\d{4})(?!\d)", evidence)
    if numeric:
        candidates = set()
        for item in numeric:
            if "." in item:
                day, month, y = item.split(".")
                candidates.add(f"{y}-{int(month):02d}-{int(day):02d}")
            else:
                candidates.add(item)
        if event not in candidates:
            return "date_value_mismatch"
    return None


def normalize_extraction(data: dict, context: dict, validator: Callable) -> dict:
    """Validate each fact independently, retaining valid siblings with QA counts.

    Counts of downgrades/date clearing refer only to accepted facts. Issues use
    original zero-based fact indices, including facts subsequently skipped.
    comment/repost never becomes own speech/action, even if author metadata
    claims a verified actor. No personal identity is inferred from an account.
    """
    if not isinstance(data, dict) or not isinstance(data.get("facts"), list):
        raise ValueError("Extraction requires a facts list")
    document = context.get("document", {})
    passages = document_spans(document)
    actors = {actor["id"]: actor for actor in context.get("config", {}).get("objects", [])}
    source_type = str(document.get("source_type", "unknown")).strip().casefold()
    accepted, issues = [], []
    downgraded_count = cleared_count = 0
    for index, original in enumerate(data["facts"]):
        def issue(code: str) -> None:
            issues.append({"index": index, "code": code})

        if not isinstance(original, dict):
            issue("invalid_fact")
            continue
        fact = copy.deepcopy(original)
        basis = fact.get("attribution_basis")
        if not isinstance(basis, str) or basis not in _BASES:
            issue("invalid_attribution_basis")
            continue
        quote, attribution = fact.get("quote"), fact.get("attribution_evidence")
        if not isinstance(quote, str) or not quote.strip() or not any(quote in s["text"] for s in passages):
            issue("quote_not_exact")
            continue
        if not isinstance(attribution, str) or not attribution.strip() or not any(attribution in s["text"] for s in passages):
            issue("attribution_not_exact")
            continue
        # Invalid dates can be cleared below, but source attribution must already
        # fit the first original quote passage. Never bridge separate excerpts.
        passage = fact_span(document, dict(fact, date_evidence=None))
        if passage is None:
            issue("attribution_not_local")
            continue
        text = passage["text"]
        actor_id = fact.get("actor_id")
        actor = actors.get(actor_id, {}) if isinstance(actor_id, str) else {}
        if not _actor_mentioned(actor, quote, attribution):
            issue("actor_not_local")
            continue
        if not _local_passages(text, quote, attribution):
            issue("attribution_not_local")
            continue

        if not isinstance(fact.get("kind"), str):
            issue("invalid_kind")
            continue
        downgraded = False
        if fact.get("kind") in _OWN_KINDS:
            if source_type in {"comment", "repost"} or basis in {"third_party_claim", "uncertain"}:
                fact["kind"] = "attributed_image"
                fact["capacity"] = "third-party"
                if source_type in {"comment", "repost"} or basis == "third_party_claim":
                    fact["attribution_basis"] = "third_party_claim"
                issue("own_attribution_downgraded")
                downgraded = True
            elif basis == "context":
                fact["kind"] = "institutional_context"
                issue("context_attribution_downgraded")
                downgraded = True
        expected = _EXPECTED_OWN_KIND.get(fact["attribution_basis"])
        if expected is not None and fact.get("kind") != expected:
            issue("kind_basis_mismatch")
            continue

        cleared = False
        if "date_evidence" not in fact:
            issue("missing_date_evidence_field")
            continue
        if fact.get("event_date") is not None:
            try:
                event = fact["event_date"]
                if not isinstance(event, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", event):
                    raise ValueError("Expected an ISO date")
                date.fromisoformat(event)
            except (ValueError, TypeError):
                issue("invalid_event_date")
                continue
        if fact.get("event_date") is not None and not _outside_period(fact["event_date"], context):
            problem = _date_problem(fact, text)
            if problem:
                fact["event_date"] = fact["date_evidence"] = None
                issue(problem)
                cleared = True
        elif fact.get("event_date") is None and fact["date_evidence"] is not None:
            fact["date_evidence"] = None
            issue("orphan_date_evidence_cleared")
        try:
            validated = validator("extract", {"facts": [fact]}, context)
            accepted.append(copy.deepcopy(validated["facts"][0]))
        except (ValueError, KeyError, TypeError, IndexError):
            issue("validator_rejected")
            continue
        downgraded_count += downgraded
        cleared_count += cleared
    return {"facts": accepted, "quality_control": {
        "input_facts": len(data["facts"]), "skipped_facts": len(data["facts"]) - len(accepted),
        "downgraded_facts": downgraded_count, "cleared_dates": cleared_count, "issues": issues}}
