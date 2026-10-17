"""Offline screening annotations; no semantic verdicts or destructive cleaning.

Documents retain their text and publication provenance elsewhere. Known-name
matches are retrieval hints, not identity verification or political positions.
Only configured country/date rules exclude records; incomplete metadata stays
visible for review. Near-duplicate components never authorise deletion.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import date
import hashlib
import json
import re
import unicodedata

from .context_policy import actor_mentions


_STATUSES = ("ready_for_semantic_filter", "needs_semantic_review", "rule_excluded")
_LANGUAGES = {
    "rw": "rw", "kinyarwanda": "rw", "киньяруанда": "rw",
    "fr": "fr", "french": "fr", "французский": "fr",
    "en": "en", "english": "en", "английский": "en",
    "sw": "sw", "swahili": "sw", "kiswahili": "sw", "суахили": "sw",
}


def _lexical(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def _matched_actors(text: str, objects: list) -> tuple[list[str], bool]:
    labels = {}
    for actor in objects:
        if not isinstance(actor, Mapping) or not isinstance(actor.get("id"), str) or not actor["id"]:
            raise ValueError("Approved objects require nonempty string IDs")
        for label in [actor.get("name", ""), *actor.get("aliases", [])]:
            if not isinstance(label, str):
                raise ValueError("Approved names and aliases must be strings")
            if label := _lexical(label):
                labels.setdefault(label, set()).add(actor["id"])
    # Independent actor checks deliberately retain nested/shared aliases; a
    # longer full-name match must not hide another actor's shorter label.
    matched = actor_mentions(text, objects)
    normalized = _lexical(text)
    ambiguous = any(len(ids) > 1 and re.search(r"(?<!\w)" + re.escape(label) + r"(?!\w)", normalized)
                    for label, ids in labels.items())
    return matched, ambiguous


def _language(value) -> str | None:
    return _LANGUAGES.get(str(value or "").strip().casefold())


def annotate_document(document: Mapping, config: Mapping) -> dict:
    """Classify offline readiness without returning or changing source text.

    ``published_at`` is already normalised by the importer. Date parsing and
    configured country/date exclusions mirror CorpusPipeline._rule_exclusion.
    ``country_review_terms`` only flags an absent lexical country match for
    semantic review; it never creates an exclusion. Language hints are not
    verified language detection and never hard-exclude.
    """
    if not isinstance(document, Mapping) or not isinstance(config, Mapping):
        raise ValueError("document and config must be objects")
    text = document.get("text", "")
    if not isinstance(text, str):
        raise ValueError("The imported document text must be a string")
    matched, ambiguous = _matched_actors(text, config.get("objects", []))
    flags = []
    if not text.strip():
        flags.append("empty_text")
    if not matched:
        flags.append("no_known_actor_name")
    if ambiguous:
        flags.append("ambiguous_approved_alias")
    language = _language(document.get("language_hint"))
    allowed = {_language(value) or str(value).casefold() for value in config.get("languages", [])}
    if language is None:
        flags.append("unknown_language_hint")
    elif allowed and language not in allowed:
        flags.append("metadata_language_outside_allowed")
    try:
        dated = date.fromisoformat(str(document.get("published_at", ""))[:10]).isoformat()
    except ValueError:
        dated = None
        flags.append("unknown_publication_date")

    def missing_country(terms):
        return bool(terms) and not any(
            re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text, re.IGNORECASE) for term in terms)

    country_missing = missing_country(config.get("country_terms", []))
    country_review_missing = missing_country(config.get("country_review_terms", []))
    outside = dated is not None and bool(
        config.get("date_from") and dated < config["date_from"]
        or config.get("date_to") and dated > config["date_to"])
    if country_missing or country_review_missing:
        flags.append("country_not_matched")
        if matched:
            flags.append("missing_country_with_actor_match")
    if outside:
        flags.append("outside_period")
    if country_missing or outside:
        status, reason = "rule_excluded", "country_not_matched" if country_missing else "outside_period"
    elif flags:
        status, reason = "needs_semantic_review", flags[0]
    else:
        status, reason = "ready_for_semantic_filter", "offline_checks_passed"
    return {"status": status, "reason": reason, "matched_actor_ids": matched, "flags": flags}


def summarize_annotations(annotations: Iterable[Mapping]) -> dict:
    """Count screening results in one pass; actor counts are lexical documents."""
    statuses, reasons, flags, actors = Counter(), Counter(), Counter(), Counter()
    actor_statuses = {status: Counter() for status in _STATUSES}
    total = 0
    for annotation in annotations:
        status = annotation.get("status")
        if status not in _STATUSES:
            raise ValueError("Unknown screening status")
        total += 1
        statuses[status] += 1
        reasons[annotation.get("reason", "")] += 1
        flags.update(set(annotation.get("flags", [])))
        matched = set(annotation.get("matched_actor_ids", []))
        actors.update(matched)
        actor_statuses[status].update(matched)
    return {"documents": total, "status_counts": dict(sorted(statuses.items())),
            "reason_counts": dict(sorted(reasons.items())), "flag_counts": dict(sorted(flags.items())),
            "actor_document_counts": dict(sorted(actors.items())),
            "actor_document_counts_by_status": {status: dict(sorted(counts.items()))
                                                for status, counts in actor_statuses.items()}}


def candidate_families(edges: Iterable[Mapping]) -> list[dict]:
    """Group candidate edges for review, not equivalence or hard deletion.

    Transitive membership does not assert that all members have the same claim,
    speaker or event date. Distinct publications and source provenance remain.
    Duplicate/reversed edges count once; a member order never chooses a winner.
    """
    parents, unique_edges = {}, set()

    def find(member):
        parents.setdefault(member, member)
        while parents[member] != member:
            parents[member] = parents[parents[member]]
            member = parents[member]
        return member

    for edge in edges:
        left, right = edge.get("left_id"), edge.get("right_id")
        if any(not isinstance(member, str) or not member for member in (left, right)) or left == right:
            raise ValueError("A candidate edge needs two different nonempty string document IDs")
        unique_edges.add(tuple(sorted((left, right))))
        a, b = find(left), find(right)
        if a != b:
            parents[max(a, b)] = min(a, b)
    members = {}
    for member in parents:
        members.setdefault(find(member), []).append(member)
    pairs = Counter(find(left) for left, _ in unique_edges)
    result = []
    for root, group in members.items():
        group.sort()
        fingerprint = hashlib.sha256(json.dumps(group, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()[:16]
        result.append({"family_id": "near-" + fingerprint, "member_ids": group,
                       "pair_count": pairs[root], "status": "needs_duplicate_review",
                       "delete_automatically": False})
    return sorted(result, key=lambda family: (-len(family["member_ids"]), family["member_ids"]))
