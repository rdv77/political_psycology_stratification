"""Recover reviewed individual-pole proposals without changing the canonical study.

No network access, no new numerical scoring and no automatic acceptance of a
previously rejected bipolar pair. A pole becomes observed only through a pinned
per-pole review entry. All other model values remain proposals, never zero-filled.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.semantic_service.providers import fact_within_period
from src.semantic_service.store import fingerprint

STUDY_ID = "rwanda-politicians-2021-2026"
RESEARCH = Path("research/rwanda_politicians_2021_2026")
POLES = ("positive", "negative")


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Input must stay within the study directory")
    return path


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 5


def inventory(root=ROOT):
    root = Path(root).resolve()
    directory = root / "studies" / STUDY_ID
    historical = directory / "attempts/paired_difference_2026-10-15"
    # A new run may replace the live matrix. Recovery is pinned to the historical
    # pair snapshot once it exists, never to concurrent partial unipolar output.
    pair_directory = historical if historical.exists() else directory
    study = load(pair_directory / "study.json")
    ratings = load(pair_directory / "ratings.json")
    index = load(directory / "evidence/model_response_index.json")
    objects = {obj["id"]: obj for obj in study["objects"]}
    scales = {scale["id"]: scale for scale in study["scales"]}
    inputs = {}
    docs = {}

    def document(relative):
        path = safe(directory, relative)
        if relative not in docs:
            docs[relative] = load(path)
            inputs[path.relative_to(root).as_posix()] = sha(path)
        return docs[relative]

    for path in (pair_directory / "study.json", pair_directory / "ratings.json", directory / "evidence/model_response_index.json"):
        inputs[path.relative_to(root).as_posix()] = sha(path)
    audits = []
    for path in sorted((root / RESEARCH).glob("AUDIT_BATCH_*.json")):
        inputs[path.relative_to(root).as_posix()] = sha(path)
        audits.extend(load(path)["decisions"])
    audit_lookup = {(x["object_id"], x["scale_id"]): x for x in audits}
    required = {s["id"] if isinstance(s, dict) else s
                for s in study.get("research_requirements", {}).get("required_scales", [])}
    reviewed_only = bool(study.get("research_requirements", {}).get("require_reviewed_thematic_evidence") or
                         study.get("source_supplement", {}).get("reviewed_thematic_only"))
    candidates = []

    def add(oid, row, evidence_file, origin, *, positions=None, basis=None, mapping_error=None):
        sid = row.get("scale_id")
        if oid not in objects or sid not in scales:
            raise ValueError("Unexpected object/scale identity")
        doc = document(evidence_file)
        facts = doc.get("facts", [])
        raw_indices = row.get("evidence_indices", [])
        indices = []
        errors = [mapping_error] if mapping_error else []
        acceptable_bases = {doc.get("basis"), fingerprint({"basis": doc.get("basis"), "facts": facts})}
        if basis not in acceptable_bases:
            errors.append("source_basis_mismatch")
        if not isinstance(raw_indices, list):
            errors.append("invalid_indices_container")
            raw_indices = []
        for value in raw_indices:
            if not isinstance(value, int) or isinstance(value, bool):
                errors.append("invalid_evidence_index")
                continue
            if positions is not None:
                if not 0 <= value < len(positions):
                    errors.append("invalid_local_thematic_index")
                    continue
                value = positions[value]
            if not 0 <= value < len(facts):
                errors.append("evidence_index_out_of_range")
                continue
            indices.append(value)
        indices = list(dict.fromkeys(indices))
        if not indices:
            errors.append("no_evidence_indices")
        for i in indices:
            fact = facts[i]
            parts = urlsplit(str(fact.get("source_url", "")))
            if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password:
                errors.append("invalid_source_url")
            if not fact_within_period(fact, study):
                errors.append("fact_outside_or_ambiguous_period")
            if sid in required and (sid not in fact.get("topic_ids", []) or
                                   fact.get("measurement_status") == "context_only" or
                                   (reviewed_only and fact.get("evidence_origin") != "assistant_verified_source")):
                errors.append("ineligible_required_topic_fact")
        for pole in POLES:
            value = row.get(pole)
            if not numeric(value):
                continue
            cid = fingerprint({"origin": origin, "object_id": oid, "scale_id": sid, "pole": pole})[:24]
            candidates.append({"candidate_id": cid, "object_id": oid, "scale_id": sid, "pole": pole,
                "value": value, "model_status": row.get("status"), "confidence": row.get("confidence"),
                "rationale": row.get("reason", ""), "evidence_file": evidence_file,
                "evidence_indices": indices, "raw_evidence_indices": raw_indices,
                "source_urls": list(dict.fromkeys(facts[i].get("source_url") for i in indices)),
                "source_basis": basis, "facts_sha256": fingerprint(facts),
                "validation_errors": sorted(set(errors)), "provenance": origin,
                "prior_pair_audit": audit_lookup.get((oid, sid))})

    # The canonical origin retains the exact score proposed before the pair audit.
    # An index repair is explicit and traceable; it does not change either number.
    for row in ratings:
        original = row.get("original_provider_rating", row)
        if not original.get("evidence_file"):
            continue
        source = dict(original)
        audit = row.get("source_audit", {})
        if audit.get("decision") == "repair_indices":
            source["evidence_indices"] = row["evidence_indices"]
        add(row["object_id"], source, original["evidence_file"],
            {"kind": "canonical_pre_audit_proposal", "file": (pair_directory / "ratings.json").relative_to(directory).as_posix(),
             "model": original.get("model"), "at": original.get("at"),
             "rating_pass": original.get("rating_pass"),
             "index_repair": audit if audit.get("decision") == "repair_indices" else None},
            basis=original.get("evidence_basis"))

    # Raw caches can contain a known pole discarded by the former pair contract.
    # Reconstruct ONLY the recorded initial/supplement snapshots and exact local
    # thematic slicing; never interpret a local index as an original global index.
    for entry in index["responses"]:
        response = document(entry["file"])
        task = response.get("task", "")
        if task != entry["task"] or response.get("cache_key") != entry.get("cache_key"):
            raise ValueError("Response index identity mismatch")
        match = None
        for oid in objects:
            for supplemental in (False, True):
                prefix = ("ratings-supplement-" if supplemental else "ratings-") + oid + "-"
                if task.startswith(prefix):
                    suffix = task[len(prefix):]
                    if re.fullmatch(r"(?:0|12)(?:-russia-[a-z-]+)?", suffix):
                        match = (oid, supplemental, suffix)
                        break
            if match:
                break
        if match is None:
            raise ValueError("Unrecognized cached task " + task)
        oid, supplemental, suffix = match
        initial = document("evidence/scoring_initial-" + oid + ".json")
        ef = ("evidence/dossier_supplement-" + oid + ".json") if supplemental else initial["evidence_file"]
        doc = document(ef)
        if supplemental:
            basis = doc.get("basis")
            error = None if doc.get("scales_basis") == fingerprint(study["scales"]) else "supplement_scale_basis_mismatch"
        else:
            basis = fingerprint({"basis": doc.get("basis"), "facts": doc.get("facts", [])})
            error = None if basis == initial.get("dossier_basis") else "initial_dossier_basis_mismatch"
        thematic = suffix.split("-", 1)[1] if "-" in suffix else None
        positions = None
        if thematic:
            positions = [i for i, f in enumerate(doc.get("facts", []))
                         if thematic in f.get("topic_ids", []) and fact_within_period(f, study)
                         and f.get("measurement_status") != "context_only"
                         and (not reviewed_only or f.get("evidence_origin") == "assistant_verified_source")]
        for row in response.get("data", {}).get("ratings", []):
            if thematic and row.get("scale_id") != thematic:
                raise ValueError("Thematic cache scale mismatch")
            add(oid, row, ef, {"kind": "raw_model_response", "file": entry["file"],
                "cache_key": entry["cache_key"], "task": task, "model": response.get("model"),
                "at": response.get("at"), "rating_pass": "supplement" if supplemental else "initial",
                "thematic_local_to_global": positions}, positions=positions, basis=basis, mapping_error=error)

    by_cell = defaultdict(list)
    for c in candidates:
        by_cell[(c["object_id"], c["scale_id"], c["pole"])].append(c)
    coverage = []
    for sid in scales:
        for pole in POLES:
            eligible = {oid for (oid, s, p), rows in by_cell.items() if s == sid and p == pole and
                        any(c["value"] > 0 and not c["validation_errors"] for c in rows)}
            coverage.append({"scale_id": sid, "pole": pole, "label": scales[sid][pole],
                             "candidate_objects": len(eligible), "n_objects": len(objects),
                             "candidate_threshold_26": len(eligible) >= 26})
    return {"version": 1, "study_id": STUDY_ID, "purpose": "Unaccepted source-linked historical model proposals; NOT a measurement matrix",
            "basis": {"objects": fingerprint(study["objects"]), "scales": fingerprint(study["scales"]),
                      "period": [study["date_from"], study["date_to"]]},
            "objects": list(objects.values()), "scales": list(scales.values()),
            "input_sha256": inputs, "candidates": candidates, "candidate_coverage": coverage,
            "evidence_snapshots": {key: {"object_id": d.get("object_id"), "basis": d.get("basis"),
                                  "facts": d["facts"], "retrieval_citations": d.get("retrieval_citations", []),
                                  "supplement_provenance": d.get("supplement_provenance"),
                                  "source_supplement_provenance": d.get("source_supplement_provenance")}
                                   for key, d in docs.items() if "facts" in d},
            "counts": {"raw_responses": len(index["responses"]), "candidate_proposals": len(candidates),
                       "candidate_nonzero": sum(c["value"] > 0 for c in candidates),
                       "candidate_zero": sum(c["value"] == 0 for c in candidates),
                       "distinct_numeric_poles": len(by_cell),
                       "distinct_nonzero_valid_poles": sum(any(c["value"] > 0 and not c["validation_errors"] for c in cs) for cs in by_cell.values())}}


def audit_fresh(root=ROOT, start=14, stop=28):
    """Create a read-only packet for semantic review, never edit measured values.

    Structural errors and semantic warnings are distinct. Repeated mentions of
    one event are not independent recurrence. Event-date checks are conservative
    warning generators; only a substantive reviewer can accept/reject a pole.
    """
    root = Path(root).resolve()
    directory = root / "studies" / STUDY_ID
    study = load(directory / "study.json")
    objects = study["objects"][start:stop]
    scales = {s["id"]: s for s in study["scales"]}
    inputs, cells, unavailable = {}, [], []
    checked, counts = [], Counter()
    for obj in objects:
        rating_file = directory / "evidence" / ("independent-ratings-" + obj["id"] + ".json")
        if not rating_file.exists():
            unavailable.append(obj["id"])
            continue
        rows = load(rating_file)
        inputs[rating_file.relative_to(root).as_posix()] = sha(rating_file)
        checked.append(obj["id"])
        for row in rows:
            if row.get("object_id") != obj["id"] or row.get("scale_id") not in scales:
                raise ValueError("Fresh score identity mismatch")
            scale = scales[row["scale_id"]]
            for pole, score in row.get("pole_scores", {}).items():
                if pole not in POLES:
                    raise ValueError("Unrecognized pole")
                counts["poles_total"] += 1
                if score.get("status") != "observed":
                    if score.get("value") is not None:
                        raise ValueError("Unknown pole must have null value")
                    counts["unknown"] += 1
                    continue
                counts["observed"] += 1
                errors, warnings = [], []
                value = score.get("value")
                if not numeric(value):
                    errors.append("out_of_range_or_nonfinite_value")
                relative = score.get("evidence_file", row.get("evidence_file"))
                if not relative:
                    errors.append("missing_evidence_file")
                    dossier = {"facts": []}
                else:
                    path = safe(directory, relative)
                    dossier = load(path)
                    inputs[path.relative_to(root).as_posix()] = sha(path)
                    if dossier.get("object_id") != obj["id"]:
                        errors.append("dossier_actor_mismatch")
                    if score.get("evidence_basis", row.get("evidence_basis")) != dossier.get("basis"):
                        errors.append("dossier_basis_mismatch")
                facts = dossier.get("facts", [])
                indices = score.get("evidence_indices", [])
                if not indices or len(indices) != len(set(indices)) or any(
                        isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(facts) for i in indices):
                    errors.append("missing_duplicate_or_invalid_evidence_index")
                    indices = [i for i in indices if isinstance(i, int) and not isinstance(i, bool) and 0 <= i < len(facts)]
                evidence = [{"index": i, **facts[i]} for i in indices]
                source_urls = list(dict.fromkeys(f["source_url"] for f in evidence if f.get("source_url")))
                if set(score.get("source_urls", [])) != set(source_urls):
                    errors.append("cited_urls_do_not_match_indexed_facts")
                for fact in evidence:
                    parts = urlsplit(str(fact.get("source_url", "")))
                    if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password:
                        errors.append("invalid_source_url")
                    if not fact_within_period(fact, study):
                        errors.append("fact_outside_or_ambiguous_period")
                    if fact.get("measurement_status") == "context_only":
                        warnings.append("context_only_fact_does_not_establish_this_property")
                    if scale.get("research_domain") == "russia_relations" and (
                            scale["id"] not in fact.get("topic_ids", []) or
                            fact.get("evidence_origin") != "assistant_verified_source"):
                        errors.append("unreviewed_or_offtopic_thematic_fact")
                dates = {m.group(0) for f in evidence for m in re.finditer(r"\b\d{4}-\d{2}-\d{2}\b", str(f.get("period", "")))}
                events = {f.get("event_id") for f in evidence if f.get("event_id")}
                if numeric(value) and value > 2 and (len(dates) < 2 or len(events) == 1):
                    warnings.append("value_above_2_requires_distinct_recurring_events")
                if value == 0:
                    warnings.append("zero_requires_substantive_explicit_absence_review")
                    if score.get("support_type") != "explicit_absence":
                        errors.append("zero_without_explicit_absence_support_type")
                elif score.get("support_type") != "positive_manifestation":
                    errors.append("nonzero_without_positive_manifestation_support_type")
                rationale = score.get("rationale", "")
                if re.search(r"нет (?:данных|сведений|упоминан|основан)|не найден|отсутстви[ея] (?:данных|сведений|упоминан)", rationale.lower()):
                    warnings.append("rationale_mentions_missing_evidence_check_pole_specificity")
                counts["structural_error_poles"] += bool(errors)
                counts["warning_poles"] += bool(warnings)
                cells.append({"object_id": obj["id"], "object_name": obj["name"], "scale_id": scale["id"],
                    "pole": pole, "label": scale[pole], "definition": scale["definition"], "value": value,
                    "rationale": rationale, "support_type": score.get("support_type"),
                    "evidence_file": relative, "evidence_basis": dossier.get("basis"),
                    "evidence": evidence, "distinct_dates": sorted(dates), "event_ids": sorted(events),
                    "validation_errors": sorted(set(errors)), "review_warnings": sorted(set(warnings)),
                    "provenance": score.get("provenance")})
    return {"version": 1, "purpose": "Read-only per-pole semantic review packet; no changes or new scores",
            "study_id": STUDY_ID, "object_range": [start, stop], "checked_objects": checked,
            "unavailable_objects": unavailable, "scales_basis": fingerprint(study["scales"]),
            "input_sha256": inputs, "counts": dict(counts), "cells": cells, "decisions": [],
            "decision_contract": {"allowed": ["reject", "repair_indices", "flag", "recode"],
                "required": ["object_id", "scale_id", "pole", "decision", "reason"],
                "rule": "Reject only the unsupported pole; preserve an unknown opposite. Index repairs require direct support in the same frozen dossier. For this fresh-score review only, recode to an approved anchor is permitted with explicit evidence and rationale; preserve the original model proposal. Historical recovery never invents a new number. No decision means reviewed and retained."}}


def recover(data, reviews):
    if reviews.get("status") != "completed_per_pole_semantic_review":
        raise ValueError("Historical recovery requires an explicitly completed per-pole semantic review")
    candidates = {c["candidate_id"]: c for c in data["candidates"]}
    decisions = {(d["object_id"], d["scale_id"], d["pole"]): d for d in reviews.get("decisions", [])}
    if len(decisions) != len(reviews.get("decisions", [])):
        raise ValueError("Duplicate per-pole review")
    if reviews.get("inventory_basis") != fingerprint({"basis": data["basis"], "input_sha256": data["input_sha256"]}):
        raise ValueError("Review must be pinned to the exact inventory/source snapshots")
    rows = []
    reasons = Counter()
    for obj in data["objects"]:
        for scale in data["scales"]:
            row = {"object_id": obj["id"], "scale_id": scale["id"], "pole_scores": {}}
            for pole in POLES:
                key = (obj["id"], scale["id"], pole)
                review = decisions.get(key)
                unknown = {"status": "unknown", "value": None, "evidence_indices": [], "source_urls": [],
                           "rationale": "Нет отдельно проверенного численного предложения для этого полюса",
                           "provenance": {"method": "historical_proposal_recovery", "review_status": "not_accepted"}}
                if review and review["decision"] == "retain":
                    c = candidates[review["candidate_id"]]
                    if (c["object_id"], c["scale_id"], c["pole"]) != key or c["validation_errors"]:
                        raise ValueError("Review selects an invalid or unrelated candidate")
                    indices = review.get("evidence_indices", c["evidence_indices"])
                    facts = data["evidence_snapshots"][c["evidence_file"]]["facts"]
                    if not indices or any(isinstance(i, bool) or not isinstance(i, int) or i not in c["evidence_indices"] for i in indices):
                        raise ValueError("Review indices must be a nonempty subset of the validated candidate evidence")
                    if c["value"] == 0 and not review.get("explicit_zero_evidence"):
                        raise ValueError("Zero needs explicit reviewed evidence of nonexpression")
                    row["pole_scores"][pole] = {"status": "observed", "value": c["value"],
                        "evidence_indices": indices, "source_urls": list(dict.fromkeys(facts[i]["source_url"] for i in indices)),
                        "rationale": review["reason"], "provenance": {"method": "historical_proposal_recovery",
                        "candidate_id": c["candidate_id"], "origin": c["provenance"], "original_rationale": c["rationale"],
                        "evidence_file": c["evidence_file"], "source_basis": c["source_basis"], "facts_sha256": c["facts_sha256"],
                        "prior_pair_audit": c.get("prior_pair_audit"), "reviewer": "assistant_excerpt_consistency_review_not_human_expert",
                        "warning": review.get("warning")}}
                    reasons["retained"] += 1
                else:
                    if review:
                        unknown["rationale"] = review["reason"]
                        unknown["provenance"]["review_status"] = review["decision"]
                        reasons[review["decision"]] += 1
                    else:
                        reasons["no_retained_proposal"] += 1
                    row["pole_scores"][pole] = unknown
            rows.append(row)
    coverage = []
    for scale in data["scales"]:
        for pole in POLES:
            count = sum(r["scale_id"] == scale["id"] and r["pole_scores"][pole]["status"] == "observed" for r in rows)
            coverage.append({"scale_id": scale["id"], "pole": pole, "label": scale[pole], "observed": count,
                             "unknown": len(data["objects"]) - count, "coverage": count / len(data["objects"]),
                             "threshold_26": count >= 26})
    return {"version": 1, "study_id": data["study_id"], "status": "proposal_not_applied",
            "measurement_model": "independent_poles", "basis": data["basis"],
            "input_sha256": data["input_sha256"], "inventory_basis": reviews["inventory_basis"],
            "ratings": rows, "coverage": coverage, "counts": dict(reasons),
            "limitations": ["Числа не назначались заново; сохранены только существовавшие предложения, прошедшие отдельную проверку полюса.",
                "Проверка относится к сохранённым выпискам; не является независимой человеческой экспертизой и не заменяет чтение всех первичных страниц.",
                "Unknown противоположного полюса сохраняется; разности и суммы полюсов не вычисляются.",
                "Низкая полнота и зависимость от доступности источников исключают автоматический вывод о политическом пространстве."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--recover-reviewed", action="store_true", help="Build a proposal only from a completed pinned per-pole review")
    parser.add_argument("--audit-fresh", action="store_true", help="Write a read-only packet for the current independent-pole scores")
    parser.add_argument("--start", type=int, default=14)
    parser.add_argument("--stop", type=int, default=28)
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()
    root = args.root.resolve()
    destination = root / RESEARCH
    if args.audit_fresh:
        packet = audit_fresh(root, args.start, args.stop)
        path = destination / ("unipolar_recovery_fresh_audit_batch_" + str(args.batch) + ".json")
        path.write_text(json.dumps(packet, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in packet.items() if k not in {"cells", "input_sha256", "decision_contract"}}, ensure_ascii=False, indent=2))
        return
    data = inventory(root)
    target = destination / "unipolar_recovery_inventory.json"
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    brief = {"counts": data["counts"], "candidate_coverage": data["candidate_coverage"],
             "inventory_basis": fingerprint({"basis": data["basis"], "input_sha256": data["input_sha256"]})}
    (destination / "unipolar_recovery_counts.json").write_text(json.dumps(brief, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.recover_reviewed and not args.inventory_only:
        reviews = load(destination / "unipolar_recovery_reviews.json")
        proposal = recover(data, reviews)
        (destination / "unipolar_recovery_proposal.json").write_text(json.dumps(proposal, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        brief["reviewed_counts"] = proposal["counts"]
        brief["reviewed_coverage"] = proposal["coverage"]
        (destination / "unipolar_recovery_counts.json").write_text(json.dumps(brief, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(brief, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
