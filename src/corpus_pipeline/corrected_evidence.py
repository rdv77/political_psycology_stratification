"""Versioned offline evidence corrections, without changing extraction receipts.

Exact source adjudications may correct a fact. Lexical risk cues merely request
attention: a protocol word, old year or accusation elsewhere in a quotation is
not sufficient to decide its meaning. This module never generates ratings,
resolves relative dates from publication time, or merges presumed episodes.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from datetime import date
import re
import unicodedata

from .evidence import _actor_mentioned, _date_problem, _local_passages
from .date_precision import event_date_precision_problem as _precision_problem
from .scoring_preparation import digest, filter_eligible_fact
from .source_spans import document_spans


POLICY_VERSION = "source-corrections-no-api-v1"
_BASE_KIND = {"direct_statement": "public_statement", "reported_statement": "public_statement",
              "reported_action": "action", "third_party_claim": "attributed_image",
              "context": "institutional_context", "uncertain": "attributed_image"}
_EXACT = {"supported_exact", "corrected_exact"}
_DATE_STATUS = _EXACT | {"supported_interval", "publication_only", "unresolved", "outside_window"}
_ATTR_STATUS = {"supported", "corrected", "unresolved", "not_own", "homonym"}
_PROTOCOL = re.compile(r"\b(?:handshake|greeted|welcomed|welcome|airport|courtesy|ceremonial|"
                       r"photo|photograph|attended|participated|salutation|accueilli|aéroport|"
                       r"cérémonie|assiste|assister|participé|réception|yakiriye|bitabiriye|"
                       r"itabiriye|ikibuga cy['’]indege|yalishiriki|alishiriki)\b", re.I)
_PASSIVE = re.compile(r"\b(?:accused|alleged|allegations|convicted|sentenced|arrested|appointed|"
                      r"accusé|accusée|condamné|condamnée|arrêté|arrêtée|nommé|nommée|"
                      r"ashinjwa|yashinjwe|yatawe|yagizwe|anatuhumiwa|alihukumiwa)\b", re.I)
def _text(value):
    return " ".join(unicodedata.normalize("NFC", str(value or "")).split())


def _iso(value):
    if value is not None and (not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value)):
        raise ValueError("A correction date needs explicit ISO precision")
    if value is not None:
        date.fromisoformat(value)
    return value


def _hold(fact):
    fact["review_status"] = "unreviewed"
    fact["measurement_status"] = "requires_review"


def _context_checks(fact, actor, document, config):
    """Source-integrity decisions plus non-decisive semantic risk flags."""
    flags = []
    if document is None:
        flags.append("original_extraction_context_unavailable")
        _hold(fact)
    else:
        quote, attribution = fact.get("quote"), fact.get("attribution_evidence")
        spans = document_spans(document)
        candidates = [s["text"] for s in spans if isinstance(quote, str) and quote.strip()
                      and quote in s["text"]]
        if not candidates:
            flags.append("quote_not_in_original_extraction_passage")
            _hold(fact)
        elif not isinstance(attribution, str) or not attribution.strip() or not any(
                attribution in text and _local_passages(text, quote, attribution) for text in candidates):
            flags.append("local_attribution_not_source_grounded")
            _hold(fact)
        elif not _actor_mentioned(actor, quote, attribution):
            flags.append("approved_actor_not_in_local_attribution")
            _hold(fact)
        source_type = str(document.get("source_type", "unknown")).casefold().strip()
        if source_type in {"comment", "repost"} and fact.get("kind") in {"public_statement", "action"}:
            fact.update(kind="attributed_image", attribution_basis="third_party_claim",
                        is_own_position=False, capacity="third-party")
            flags.append("unverified_comment_or_repost_not_own_speech")
        if fact.get("event_date"):
            try:
                _iso(fact["event_date"])
                problem = (_date_problem(fact, next((t for t in candidates if attribution in t), ""))
                           or _precision_problem(fact))
            except (ValueError, TypeError):
                problem = "invalid_event_date"
            if problem:
                old_date = fact.get("event_date")
                fact.update(event_date=None, date_evidence=None, temporal_status="unresolved")
                flags.append(problem)
                # An unsupported historical label must not silently become an
                # eligible current observation merely by clearing its date.
                if (isinstance(old_date, str) and config.get("date_from") and old_date < config["date_from"]
                        or isinstance(old_date, str) and config.get("date_to") and old_date > config["date_to"]):
                    flags.append("unresolved_scope_after_date_clearing")
                    _hold(fact)
        elif fact.get("date_evidence") is not None:
            fact["date_evidence"] = None
            flags.append("orphan_event_date_evidence_cleared")

    basis, kind = fact.get("attribution_basis"), fact.get("kind")
    if basis in {"third_party_claim", "context", "uncertain"} and kind in {"action", "public_statement"}:
        fact["kind"] = _BASE_KIND[basis]
        fact["is_own_position"] = False
        flags.append("non_own_basis_downgraded")
    elif basis not in _BASE_KIND or kind != _BASE_KIND.get(basis):
        flags.append("inconsistent_attribution_kind_requires_review")
        _hold(fact)
    event = fact.get("event_date")
    if event and ((config.get("date_from") and event < config["date_from"])
                  or (config.get("date_to") and event > config["date_to"])):
        fact["temporal_status"] = "historical_outside_window"
        flags.append("explicit_event_date_outside_study_window")
    else:
        fact.setdefault("temporal_status", "explicit_event_date" if event else "unknown_event_time")

    passage = " ".join(str(fact.get(k) or "") for k in ("quote", "attribution_evidence"))
    if _PROTOCOL.search(passage):
        flags.append("protocol_or_participation_cue_review_only")
    if _PASSIVE.search(passage):
        flags.append("passive_agency_or_accusation_cue_review_only")
    # Deliberately no rule 'old year in quote => historical actor position':
    # valid current speeches often discuss 1994/1995 or an old court case.
    return flags


def _validate_annotation(annotation):
    if annotation.get("verdict") not in {"retain", "narrow", "hold", "exclude"}:
        raise ValueError("Unknown reviewed verdict")
    attribution, timing = annotation["reviewed_attribution"], annotation["reviewed_event_date"]
    if attribution.get("status") not in _ATTR_STATUS or attribution.get("basis") not in _BASE_KIND:
        raise ValueError("Unknown reviewed attribution")
    status = timing.get("status")
    if status not in _DATE_STATUS:
        raise ValueError("Unknown reviewed date precision")
    exact, start, end = (_iso(timing.get(k)) for k in ("date", "date_from", "date_to"))
    if status in _EXACT and (not exact or start or end):
        raise ValueError("An exact reviewed date requires one exact day")
    if status == "supported_interval" and (exact or not start or not end):
        raise ValueError("A reviewed interval requires both bounds")
    if status in {"publication_only", "unresolved"} and any((exact, start, end)):
        raise ValueError("An unresolved event cannot inherit a publication date")
    if bool(start) != bool(end) or start and start > end:
        raise ValueError("Invalid reviewed date interval")
    if status == "outside_window" and exact and (start or end):
        raise ValueError("An outside date cannot be both exact and an interval")
    ids = annotation.get("source_fact_ids")
    if not isinstance(ids, list) or not ids or len(ids) != len(set(ids)):
        raise ValueError("Reviewed source fact IDs must be unique and nonempty")


def _apply_review(fact, annotation, config):
    attribution, timing = annotation["reviewed_attribution"], annotation["reviewed_event_date"]
    fact.update(attribution_basis=attribution["basis"], kind=_BASE_KIND[attribution["basis"]],
                event_date=timing.get("date"), event_date_from=timing.get("date_from"),
                event_date_to=timing.get("date_to"), date_evidence=timing.get("evidence_quote"),
                temporal_status=timing["status"], review_status="reviewed",
                measurement_status="eligible_source_fragment",
                is_own_position=attribution["status"] in {"supported", "corrected"}
                and attribution["basis"] in {"direct_statement", "reported_statement", "reported_action"},
                correction_origin="exact_source_adjudication",
                reviewed_case_id=annotation["case_id"], reviewed_verdict=annotation["verdict"],
                reviewed_attribution=deepcopy(attribution), reviewed_event_date=deepcopy(timing))
    if annotation.get("scope_summary_ru"):
        fact["reviewed_scope_ru"] = annotation["scope_summary_ru"]
    if annotation["verdict"] == "exclude" or attribution["status"] in {"not_own", "homonym"}:
        fact.update(review_status="rejected", measurement_status="context_only")
    elif annotation["verdict"] == "hold" or attribution["status"] == "unresolved":
        _hold(fact)
    start = timing.get("date") or timing.get("date_from")
    end = timing.get("date") or timing.get("date_to")
    if timing["status"] == "outside_window" or (start and end and (
            config.get("date_from") and end < config["date_from"]
            or config.get("date_to") and start > config["date_to"])):
        fact.update(temporal_status="historical_outside_window", review_status="rejected")
    elif start and end and ((config.get("date_from") and start < config["date_from"])
                           or (config.get("date_to") and end > config["date_to"])):
        fact["temporal_status"] = "interval_crosses_study_boundary"
        _hold(fact)


def correct(snapshot, annotations, *, source_cases=None):
    """Return a complete derived fact version and a quote-free public summary.

    ``annotations`` is the frozen chat-review export (or its annotations list).
    Optional original ``source_cases`` enforces exact fact-content provenance.
    Absent facts, stale actor IDs and conflicting duplicate corrections fail
    closed. The caller may replace only ``facts`` in a *copy* of a snapshot.
    The original snapshot, queue, receipts and database are never modified.
    """
    config = snapshot["corpus"]["config"]
    actors = {obj["id"]: obj for obj in config["objects"]}
    if len(actors) != len(config["objects"]) or not actors:
        raise ValueError("Unique approved actors are required")
    originals = {}
    for fact in snapshot["facts"]:
        identifier = fact.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in originals:
            raise ValueError("Unique source fact IDs are required")
        originals[identifier] = fact
    rows = annotations.get("annotations", []) if isinstance(annotations, dict) else annotations
    if not isinstance(rows, list):
        raise ValueError("Corrections must be a list")
    cases = None if source_cases is None else {c["case_id"]: c for c in source_cases}
    if source_cases is not None and len(cases) != len(source_cases):
        raise ValueError("Duplicate source case")
    by_id, case_ids = {}, set()
    for annotation in rows:
        _validate_annotation(annotation)
        cid = annotation.get("case_id")
        if not isinstance(cid, str) or not cid or cid in case_ids:
            raise ValueError("Correction cases must be unique")
        case_ids.add(cid)
        case = cases.get(cid) if cases is not None else None
        if cases is not None and case is None:
            raise ValueError("Correction has no frozen source dossier")
        members = {f["id"]: f for f in case["member_facts"]} if case else {}
        for fid in annotation["source_fact_ids"]:
            original = originals.get(fid)
            if original is None or original.get("actor_id") != annotation.get("original_actor_id"):
                raise ValueError("Correction references a missing fact or a different actor")
            if fid in by_id:
                raise ValueError("A fact cannot silently receive two source adjudications")
            if case is not None and (case["actor_id"] != original["actor_id"] or fid not in members
                                     or digest(members[fid]) != digest(original)):
                raise ValueError("Original reviewed fact changed after dossier preparation")
            by_id[fid] = annotation

    documents = {}
    for job in snapshot.get("jobs", []):
        if job.get("stage") == "extract" and isinstance(job.get("context"), dict):
            document = job["context"].get("document")
            if isinstance(document, dict):
                documents[job["id"]] = document
    private_facts, audit = [], []
    flags_count, exact_verdicts = Counter(), Counter()
    actor_rows = {actor_id: {"actor_id": actor_id, "name": obj["name"], "input_facts": 0,
                            "eligible_before": 0, "eligible_after": 0, "exact_reviewed_facts": 0,
                            "general_risk_flagged_facts": 0} for actor_id, obj in actors.items()}
    for original in snapshot["facts"]:
        fact = deepcopy(original)
        actor_id = fact.get("actor_id")
        row = actor_rows.get(actor_id)
        if row is not None:
            row["input_facts"] += 1
            row["eligible_before"] += bool(filter_eligible_fact(original, config))
        document = documents.get(fact.get("extraction_job_id"))
        if document is not None and document.get("id") != fact.get("document_id"):
            raise ValueError("Extraction context belongs to a different source document")
        flags = _context_checks(fact, actors.get(actor_id, {}), document, config)
        if actor_id not in actors:
            flags.append("unapproved_actor")
            _hold(fact)
        annotation = by_id.get(fact["id"])
        if annotation is not None:
            _apply_review(fact, annotation, config)
            exact_verdicts[annotation["verdict"]] += 1
        else:
            fact["correction_origin"] = "general_source_checks_only"
        fact["correction_policy_version"] = POLICY_VERSION
        fact["correction_flags"] = sorted(set(flags))
        flags_count.update(fact["correction_flags"])
        if row is not None:
            row["eligible_after"] += bool(filter_eligible_fact(fact, config))
            row["exact_reviewed_facts"] += annotation is not None
            row["general_risk_flagged_facts"] += bool(flags)
        changed = [key for key in sorted(set(original) | set(fact))
                   if key not in {"correction_policy_version", "correction_origin", "correction_flags"}
                   and original.get(key) != fact.get(key)]
        audit.append({"fact_id": fact["id"], "source_fact_sha256": digest(original),
                      "derived_fact_sha256": digest(fact), "reviewed_case_id": fact.get("reviewed_case_id"),
                      "changed_fields": changed, "flags": fact["correction_flags"],
                      "eligible_before": filter_eligible_fact(original, config),
                      "eligible_after": filter_eligible_fact(fact, config),
                      "before": {key: deepcopy(original.get(key)) for key in changed},
                      "after": {key: deepcopy(fact.get(key)) for key in changed}})
        private_facts.append(fact)

    literal_groups = defaultdict(list)
    for fact in private_facts:
        if fact.get("actor_id") in actors and filter_eligible_fact(fact, config):
            # Review intervals are kept separate; no date is filled from source_date.
            key = (fact["actor_id"], _text(fact.get("quote")), fact.get("event_date"),
                   fact.get("event_date_from"), fact.get("event_date_to"))
            literal_groups[key].append(fact)
    for key, group in literal_groups.items():
        group_id = "literal-" + digest(key)[:24]
        for fact in group:
            fact["literal_material_group_id"] = group_id
            fact["episode_identity_verified"] = False
            fact["source_independence_verified"] = False
    # Hashes after all added fields allow the caller to verify the actual output.
    for fact, entry in zip(private_facts, audit):
        entry["derived_fact_sha256"] = digest(fact)
    summary = {"version": POLICY_VERSION, "method": "offline_source_checks_and_exact_chat_adjudications",
               "counts": {"all_extracted_facts_processed": len(private_facts),
                          "exact_reviewed_cases_applied": len(rows), "exact_reviewed_facts_applied": len(by_id),
                          "general_checks_only_facts": len(private_facts) - len(by_id),
                          "facts_with_changed_measurement_fields": sum(any(
                              key in originals[a["fact_id"]] and key in {
                                  "kind", "attribution_basis", "event_date", "date_evidence", "excluded",
                                  "review_status", "measurement_status", "is_own_position"}
                              for key in a["changed_fields"]) for a in audit),
                          "facts_with_added_or_changed_review_metadata": sum(bool(a["changed_fields"]) for a in audit),
                          "facts_with_general_risk_flags": sum(bool(a["flags"]) for a in audit),
                          "eligible_before": sum(a["eligible_before"] for a in audit),
                          "eligible_after": sum(a["eligible_after"] for a in audit),
                          "literal_material_groups": len(literal_groups),
                          "literal_repeat_groups": sum(len(g) > 1 for g in literal_groups.values()),
                          "literal_repeat_excess": sum(len(g) - 1 for g in literal_groups.values())},
               "rule_flags": dict(sorted(flags_count.items())), "exact_reviewed_verdicts": dict(sorted(exact_verdicts.items())),
               "actors": list(actor_rows.values()), "api_calls": 0, "new_api_cost_usd": 0,
               "ratings_generated": False, "original_extraction_modified": False,
               "database_modified": False, "queue_modified": False, "canonical_study_modified": False,
               "frequency_used_as_intensity": False, "publication_time_used_as_event_time": False,
               "raw_quotes_published": False,
               "limitations": ["General source checks are not a full semantic review of every publication.",
                               "Protocol, passive-agency and accusation cues are flags, never automatic meaning judgments.",
                               "Only explicitly adjudicated source fact IDs receive semantic/date corrections.",
                               "Unknown event times remain unknown; old years mentioned inside speeches do not date those speeches.",
                               "Literal material groups are not independent events and do not establish recurrence.",
                               "No lexical match, article frequency or missing value creates a numerical property score."]}
    metadata = {"policy_version": POLICY_VERSION, "source_facts_content_sha256": digest(snapshot["facts"]),
                "annotations_content_sha256": digest(rows), "derived_facts_content_sha256": digest(private_facts),
                "source_cases_content_sha256": digest(source_cases) if source_cases is not None else None}
    summary["provenance"] = {k: v for k, v in metadata.items() if k != "source_cases_content_sha256"}
    return {"private_facts": private_facts, "private_audit": audit, "public_summary": summary, "metadata": metadata}
