"""Local deterministic sampling of a supplied, already eligible document frame.

The frame must contain exact-duplicate representatives, not all raw repost rows.
Names, dates and source/language labels remain noisy upstream observations. A
lexical actor ID is a retention signal, not proof of identity or endorsement.
No text, political position, length, API or random-number generator is used.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date
import hashlib
import json
import re
import unicodedata


SAMPLING_POLICY_VERSION = "certainty-stratified-half-v1"
_DIMENSIONS = ("month", "source_type", "language_hint", "status", "name_status")


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _sha(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _label(value, field):
    if value is None:
        return "unknown"
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string or null")
    return " ".join(unicodedata.normalize("NFC", value).casefold().split()) or "unknown"


def _month(value):
    if value is None or value == "":
        return "unknown", "missing"
    if not isinstance(value, str):
        raise ValueError("published_at must be an ISO date/timestamp string or null")
    value = value.strip()
    if not value:
        return "unknown", "missing"
    if re.fullmatch(r"\d{4}-\d{2}", value):
        try:
            date.fromisoformat(value + "-01")
            return value, "parsed"
        except ValueError:
            return "unknown", "unparseable"
    if re.match(r"^\d{4}-\d{2}-\d{2}(?:$|[T ])", value):
        try:
            date.fromisoformat(value[:10])
            return value[:7], "parsed"
        except ValueError:
            pass
    return "unknown", "unparseable"


def _validated_ids(values, label):
    if not isinstance(values, (list, tuple, set, frozenset)):
        raise ValueError(f"{label} must be a collection of nonempty string IDs")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{label} must contain only nonempty string IDs")
    return sorted(set(values))


def _allocate(strata, slots, minimum):
    """Hamilton by integer remainders, after any feasible mandatory minima."""
    counts = {key: len(values) for key, values in strata.items()}
    minima_applied = bool(minimum and counts and slots >= len(counts))
    lower = 1 if minima_applied else 0
    quotas = {key: lower for key in counts}
    capacities = {key: count - lower for key, count in counts.items()}
    remaining = slots - lower * len(counts)
    total_capacity = sum(capacities.values())
    if not total_capacity:
        return quotas, minima_applied
    remainders = {}
    for key, capacity in capacities.items():
        whole, remainder = divmod(remaining * capacity, total_capacity)
        quotas[key] += whole
        remainders[key] = remainder
    extra = slots - sum(quotas.values())
    # Lexical stratum ordering resolves equal Hamilton remainders independently
    # of input order; the seed controls document ordering within each stratum.
    for key in sorted(counts, key=lambda key: (-remainders[key], key))[:extra]:
        quotas[key] += 1
    return quotas, minima_applied


def sample_documents(records, *, approved_actor_ids, primary_actor_id="paul-kagame",
                     target_count=None, seed="stratified-half-v1",
                     min_one_per_stratum=True):
    """Return ``{'manifest': ..., 'audit': ...}`` without changing the input.

    ``records`` are slim dictionaries with id, published_at, source_type,
    language_hint, status and actor_ids. Missing labels/dates have explicit
    unknown buckets. Missing actor_ids means no approved lexical name match.
    Any non-approved actor ID is an error, never silently ignored.

    By default exactly half of an even-size frame is selected. An odd frame
    requires an explicit integer target_count, so rounding cannot be hidden.
    All documents mentioning an approved actor other than primary_actor_id are
    certainty units. If they exceed the target the function raises ValueError.

    Remaining units are stratified by month x source type x language hint x
    input status x named/unknown-name. Status labels are not interpreted or
    merged, so review labels stay separate. If feasible, first reserve one unit
    per remaining nonempty stratum; Hamilton then allocates the other slots
    proportionally to the remaining capacities (N_s - 1). Otherwise use ordinary
    Hamilton on N_s. Within strata order by SHA256 of JSON [seed, document_id].

    inclusion_probability is 1 for certainty units and quota/N_s for every
    other unit, selected or not. These are design probabilities under varying
    hash/seed ordering; with a fixed seed selection itself is deterministic.
    They do not imply a representative political population or reliable labels.
    """
    approved = _validated_ids(approved_actor_ids, "approved_actor_ids")
    if not isinstance(primary_actor_id, str) or primary_actor_id not in approved:
        raise ValueError("primary_actor_id must be an approved actor ID")
    if not isinstance(seed, str) or not seed:
        raise ValueError("seed must be a nonempty string")
    if not isinstance(min_one_per_stratum, bool):
        raise ValueError("min_one_per_stratum must be boolean")
    if isinstance(records, (str, bytes, dict)):
        raise ValueError("records must be an iterable of document dictionaries")
    approved_set = set(approved)
    frame = []
    seen = set()
    date_quality = Counter()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("each record must be a dictionary")
        identifier = record.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("each document id must be a nonempty string")
        if identifier in seen:
            raise ValueError(f"duplicate document id: {identifier}")
        seen.add(identifier)
        actors = _validated_ids(record.get("actor_ids", []), "actor_ids")
        unknown = set(actors) - approved_set
        if unknown:
            raise ValueError(f"document {identifier} has unapproved actor IDs: {sorted(unknown)}")
        month, quality = _month(record.get("published_at"))
        date_quality[quality] += 1
        stratum = {
            "month": month,
            "source_type": _label(record.get("source_type"), "source_type"),
            "language_hint": _label(record.get("language_hint"), "language_hint"),
            "status": _label(record.get("status"), "status"),
            "name_status": "matched" if actors else "unknown-name",
        }
        protected_actors = [actor for actor in actors if actor != primary_actor_id]
        frame.append({"id": identifier, "published_at": record.get("published_at"),
                      "actor_ids": actors, "stratum": stratum,
                      "protected_actor_ids": protected_actors})
    frame.sort(key=lambda record: record["id"])
    size = len(frame)
    if target_count is None:
        if size % 2:
            raise ValueError("an odd-size frame cannot be halved exactly; set target_count explicitly")
        target_count = size // 2
    if isinstance(target_count, bool) or not isinstance(target_count, int) or not 0 <= target_count <= size:
        raise ValueError("target_count must be an integer between zero and frame size")
    protected_count = sum(bool(record["protected_actor_ids"]) for record in frame)
    if protected_count > target_count:
        raise ValueError(f"certainty retention requires {protected_count} documents but target_count="
                         f"{target_count}; raise the target or explicitly revise the retention policy")

    strata = defaultdict(list)
    all_strata = defaultdict(list)
    for record in frame:
        key = tuple(record["stratum"][dimension] for dimension in _DIMENSIONS)
        all_strata[key].append(record)
        if not record["protected_actor_ids"]:
            strata[key].append(record)
    slots = target_count - protected_count
    quotas, minima_applied = _allocate(strata, slots, min_one_per_stratum)
    selected = {record["id"] for record in frame if record["protected_actor_ids"]}
    for key, members in strata.items():
        members.sort(key=lambda record: (_sha([seed, record["id"]]), record["id"]))
        selected.update(record["id"] for record in members[:quotas[key]])

    documents = []
    for record in frame:
        protected = bool(record["protected_actor_ids"])
        key = tuple(record["stratum"][dimension] for dimension in _DIMENSIONS)
        is_selected = record["id"] in selected
        reason = ("certainty_non_primary_actor" if protected else
                  "stratified_selected" if is_selected else "stratified_not_selected")
        documents.append({"id": record["id"], "selected": is_selected,
                          "protected": protected,
                          "protected_actor_ids": record["protected_actor_ids"],
                          "selection_reason": reason,
                          "inclusion_probability": 1.0 if protected else quotas[key] / len(strata[key]),
                          "stratum": dict(record["stratum"])})

    stratum_audit = []
    for key in sorted(all_strata):
        members = all_strata[key]
        certainty = sum(bool(record["protected_actor_ids"]) for record in members)
        residual = len(strata.get(key, []))
        quota = quotas.get(key, 0)
        stratum_audit.append({"stratum": dict(zip(_DIMENSIONS, key)),
                              "frame_count": len(members), "certainty_count": certainty,
                              "residual_count": residual, "residual_quota": quota,
                              "selected_count": certainty + quota,
                              "residual_inclusion_probability": quota / residual if residual else None})
    dimension_counts = {dimension: Counter() for dimension in _DIMENSIONS}
    dimension_selected = {dimension: Counter() for dimension in _DIMENSIONS}
    for record in frame:
        for dimension in _DIMENSIONS:
            label = record["stratum"][dimension]
            dimension_counts[dimension][label] += 1
            if record["id"] in selected:
                dimension_selected[dimension][label] += 1
    frame_fingerprint = _sha(frame)
    manifest = {
        "format_version": 1, "policy_version": SAMPLING_POLICY_VERSION,
        "seed": seed, "primary_actor_id": primary_actor_id,
        "approved_actor_ids": approved, "frame_fingerprint": frame_fingerprint,
        "target_count": target_count, "documents": documents,
    }
    audit = {
        "frame_count": size, "target_count": target_count,
        "selected_count": len(selected), "not_selected_count": size - len(selected),
        "certainty_count": protected_count, "stratified_count": slots,
        "residual_frame_count": size - protected_count,
        "exact_half": 2 * target_count == size,
        "stratum_dimensions": list(_DIMENSIONS),
        "residual_nonempty_strata": len(strata),
        "minimum_one_requested": min_one_per_stratum,
        "minimum_one_applied": minima_applied,
        "minimum_one_not_feasible": bool(min_one_per_stratum and strata and slots < len(strata)),
        "date_quality": dict(sorted(date_quality.items())),
        "strata": stratum_audit,
        "by_dimension": {
            dimension: [
                {"value": label,
                 "frame_count": dimension_counts[dimension][label],
                 "selected_count": dimension_selected[dimension][label]}
                for label in sorted(dimension_counts[dimension])]
            for dimension in _DIMENSIONS},
        "limitations": [
            "Input is an already eligible exact-unique frame; eligibility and duplicate equivalence are not inferred.",
            "Actor IDs and month/source/language/status labels are noisy upstream labels, not verified identities or language detection.",
            "Certainty retention and stratification do not establish political or population representativeness.",
            "A fixed seed gives a deterministic sample; inclusion probabilities describe the hash-order sampling design.",
            "A stratum can have zero inclusion probability when the residual budget cannot cover all strata.",
        ],
    }
    return {"manifest": manifest, "audit": audit}
