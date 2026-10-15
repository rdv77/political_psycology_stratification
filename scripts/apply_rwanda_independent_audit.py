"""Apply complete per-property review; retain all original numerical proposals."""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.semantic_service.independent_poles import _write, apply_fact_review
from src.semantic_service.providers import fact_within_period
from src.semantic_service.store import StudyStore, fingerprint, now, ratings_basis

IDENTIFIER = "rwanda-politicians-2021-2026"


def apply_decisions(study, ratings, reports, load_facts):
    checked = [oid for report in reports for oid in report["checked_objects"]]
    if len(checked) != len(set(checked)) or set(checked) != {o["id"] for o in study["objects"]}:
        raise ValueError("Independent audit must cover each approved actor exactly once")
    observed = sum(p["status"] == "observed" for row in ratings for p in row["pole_scores"].values())
    if sum(r["checked_observed_properties"] for r in reports) != observed:
        raise ValueError("Independent audit count does not match the frozen pre-review matrix")
    revised = copy.deepcopy(ratings)
    lookup = {(row["object_id"], row["scale_id"], pole): (row, entry)
              for row in revised for pole, entry in row["pole_scores"].items()}
    seen, counts = set(), Counter()
    for report in reports:
        cohort = set(report["checked_objects"])
        expected = sum(p["status"] == "observed" for row in ratings if row["object_id"] in cohort
                       for p in row["pole_scores"].values())
        if report["checked_observed_properties"] != expected:
            raise ValueError("Audit cohort observed count does not match its input")
        for decision in report["decisions"]:
            key = tuple(decision[k] for k in ("object_id", "scale_id", "pole"))
            if key not in lookup or key in seen or key[0] not in cohort:
                raise ValueError("Unknown or duplicated per-property audit decision")
            seen.add(key)
            row, entry = lookup[key]
            if entry["status"] != "observed":
                raise ValueError("Audit cannot create an observation from an unknown property")
            action = decision["decision"]
            if action not in {"reject", "flag", "repair_indices", "recode"} or not decision.get("reason"):
                raise ValueError("Invalid per-property review action or empty reason")
            counts[action] += 1
            original = copy.deepcopy(entry)
            entry["original_provider_score"] = original
            entry["source_audit"] = {**decision, "reviewer": "assistant_excerpt_consistency_and_anchor_review_not_human_expert"}
            if action == "reject":
                entry.update(status="unknown", value=None, confidence=0, evidence_indices=[], source_urls=[], event_ids=[],
                             rationale="Проверка оснований: " + decision["reason"], support_type="unknown")
            elif action in {"repair_indices", "recode"}:
                facts = load_facts(row)
                indices = decision.get("evidence_indices", original["evidence_indices"])
                if not indices or len(indices) != len(set(indices)) or any(
                    not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < len(facts) or
                    not facts[i].get("source_url", "").startswith(("https://", "http://")) or
                    not fact_within_period(facts[i], study) for i in indices):
                    raise ValueError("Reviewed score needs valid dated source indices")
                value = decision.get("value", entry["value"])
                if action == "recode":
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 5:
                        raise ValueError("Recode must use a finite 0..5 value")
                    if value == 0 and decision.get("explicit_absence") is not True:
                        raise ValueError("Reviewed zero requires explicit absence evidence")
                    entry["value"] = value
                    entry["support_type"] = "explicit_absence" if value == 0 else "positive_manifestation"
                entry.update(evidence_indices=indices, source_urls=list(dict.fromkeys(facts[i]["source_url"] for i in indices)),
                             event_ids=list(dict.fromkeys(facts[i].get("event_id", "") for i in indices)),
                             rationale="Переразметка по фактам и якорям: " + decision["reason"])
    for row in revised:
        entries = list(row["pole_scores"].values())
        count = sum(p["status"] == "observed" for p in entries)
        row["status"] = "observed" if count == len(entries) else "partial" if count else "unknown"
        row["positive"] = row["pole_scores"]["positive"]["value"]
        row["negative"] = row["pole_scores"].get("negative", {}).get("value")
        row["difference"], row["mass"] = None, None
    return revised, {"checked_objects": len(checked), "checked_observed_properties": observed,
                      "counts": dict(counts), "remaining_observed_properties": sum(
                          p["status"] == "observed" for row in revised for p in row["pole_scores"].values())}


def main():
    store = StudyStore(ROOT / "studies")
    study = store.require_approved(IDENTIFIER)
    if study.get("measurement_model") != "independent_poles" or study.get("independent_source_audit", {}).get("applied"):
        raise ValueError("Independent matrix required; refuse repeated application")
    directory = store.directory(IDENTIFIER)
    ratings_path = directory / "ratings.json"
    ratings = json.loads(ratings_path.read_text(encoding="utf-8"))
    receipt = study.get("ratings_cache", {})
    if receipt.get("basis") != ratings_basis(study) or receipt.get("ratings_sha256") != fingerprint(ratings):
        raise ValueError("Review requires a complete current bound matrix")
    paths = [ROOT / "research/rwanda_politicians_2021_2026" / f"INDEPENDENT_AUDIT_BATCH_{i}.json" for i in (1, 2, 3)]
    reports = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    if any(report.get("status") in {"in_progress", "pending", "incomplete"} for report in reports):
        raise ValueError("All three semantic reviews must be complete before applying decisions")
    original_dossiers = {}
    for row in ratings:
        path = (directory / row["evidence_file"]).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("Evidence path escaped study")
        if row["object_id"] not in original_dossiers:
            original_dossiers[row["object_id"]] = json.loads(path.read_text(encoding="utf-8"))
    rules = {}
    for report in reports:
        for field in ("fact_exclusions", "fact_corrections", "event_groups"):
            for item in report.get(field, []):
                oid = item["object_id"]
                if oid not in report["checked_objects"]:
                    raise ValueError("Source correction belongs to another review cohort")
                facts = original_dossiers[oid]["facts"]
                for i in item["evidence_indices"]:
                    if not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < len(facts):
                        raise ValueError("Invalid source-correction index")
                    fact = facts[i]
                    signature = hashlib.sha256(fact["text"].encode("utf-8")).hexdigest()
                    key = (oid, fact["source_url"], signature)
                    rule = rules.setdefault(key, {"object_id": oid, "source_url": fact["source_url"], "text_sha256": signature,
                                                 "original_evidence_indices": [], "reasons": []})
                    if i not in rule["original_evidence_indices"]:
                        rule["original_evidence_indices"].append(i)
                    rule["reasons"].append(item["reason"])
                    if field == "fact_exclusions":
                        rule["excluded"] = True
                    elif field == "fact_corrections":
                        rule["corrected_period"] = item["corrected_period"]
                    else:
                        rule["canonical_event_id"] = item["canonical_event_id"]
    registry_path = ROOT / "research/rwanda_politicians_2021_2026/INDEPENDENT_FACT_REVIEW.json"
    _write(registry_path, {"at": now(), "facts": list(rules.values()),
                           "scope": "Known source/date errors and repeated-event equivalences; original source snapshots retained"})
    config = dict(study["independent_evidence"], fact_review_registry={"path": registry_path.relative_to(ROOT).as_posix(),
                  "sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest()})
    revised_study = dict(study, independent_evidence=config)
    corrected_dossiers = {obj["id"]: apply_fact_review(SimpleNamespace(root=ROOT), revised_study, obj, original_dossiers[obj["id"]])
                          for obj in study["objects"]}
    revised, summary = apply_decisions(study, ratings, reports, lambda row: corrected_dossiers[row["object_id"]]["facts"])
    for row in revised:
        facts = corrected_dossiers[row["object_id"]]["facts"]
        for entry in row["pole_scores"].values():
            if entry["status"] == "observed" and any(facts[i].get("measurement_status") == "excluded_known_source_issue" or
                not fact_within_period(facts[i], study) for i in entry["evidence_indices"]):
                raise ValueError(f"Accepted score still refers to a known invalid fact: {row['object_id']}/{row['scale_id']}")
    snapshot = directory / "attempts/independent_before_review/ratings.json"
    if snapshot.exists():
        raise ValueError("Pre-review snapshot already exists")
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(ratings_path.read_bytes())
    (snapshot.parent / "evidence").mkdir()
    for oid, dossier in original_dossiers.items():
        file = "independent-dossier-" + oid + ".json"
        (snapshot.parent / "evidence" / file).write_bytes((directory / "evidence" / file).read_bytes())
        _write(directory / "evidence" / file, corrected_dossiers[oid])
    for row in revised:
        basis = corrected_dossiers[row["object_id"]]["basis"]
        original_context = {"file": "attempts/independent_before_review/" + row["evidence_file"], "basis": row["evidence_basis"]}
        row.update(input_evidence_snapshot=original_context, evidence_basis=basis)
        for entry in row["pole_scores"].values():
            entry.update(input_evidence_snapshot=original_context, evidence_basis=basis)
            if entry["status"] == "observed":
                facts = corrected_dossiers[row["object_id"]]["facts"]
                entry["event_ids"] = list(dict.fromkeys(facts[i].get("event_id", "") for i in entry["evidence_indices"]))
    _write(ratings_path, revised)
    audit = {"applied": True, "at": now(), **summary,
             "scope": "Assistant review of saved excerpts and ordinal anchors; not independent human expert validation or full verification of all primary pages",
             "recode_policy": "New explicit assistant review score when a known property was overstated against preregistered event-frequency anchors; original proposal preserved",
             "reports": [{"path": p.relative_to(ROOT).as_posix(), "fingerprint": fingerprint(r)} for p, r in zip(paths, reports)],
             "fact_review_registry": config["fact_review_registry"], "fact_review_rule_count": len(rules),
             "before_fingerprint": fingerprint(ratings), "after_fingerprint": fingerprint(revised)}
    store.update(IDENTIFIER, independent_source_audit=audit, independent_evidence=config, status="ready", ratings_cache={
        "basis": ratings_basis(revised_study), "ratings_sha256": fingerprint(revised), "mode": "independent_poles_source_and_anchor_reviewed"})
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
