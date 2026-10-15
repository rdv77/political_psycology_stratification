"""Apply a preregistered coverage-only pass to unknown unipolar properties."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.semantic_service.independent_poles import _write, attach_reviewed_sources
from src.semantic_service.providers import fact_within_period
from src.semantic_service.store import StudyStore, fingerprint, now, ratings_basis

IDENTIFIER = "rwanda-politicians-2021-2026"
RESEARCH = ROOT / "research/rwanda_politicians_2021_2026"


def main():
    store = StudyStore(ROOT / "studies")
    study = store.require_approved(IDENTIFIER)
    if study.get("independent_completion", {}).get("applied"):
        raise ValueError("Completion was already applied; do not overwrite this version")
    directory = store.directory(IDENTIFIER)
    ratings_path = directory / "ratings.json"
    ratings = json.loads(ratings_path.read_text(encoding="utf-8"))
    plan = json.loads((RESEARCH / "INDEPENDENT_COMPLETION_PLAN.json").read_text(encoding="utf-8"))
    if fingerprint(ratings) != plan["input_ratings_fingerprint"] or study.get("ratings_cache", {}).get("basis") != ratings_basis(study):
        raise ValueError("Completion plan is stale or measurements were not bound")
    lookup = {(row["object_id"], row["scale_id"], pole): (row, entry) for row in ratings for pole, entry in row["pole_scores"].items()}
    target_pairs = {(t["scale_id"], t["pole"]) for t in plan["targets"]}
    expected = {key for key, (_, entry) in lookup.items() if key[1:] in target_pairs and entry["status"] == "unknown"}
    paths = [RESEARCH / f"INDEPENDENT_COMPLETION_BATCH_{i}.json" for i in (1, 2, 3)]
    reports = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    checked = [tuple(key) for report in reports for key in report["checked_properties"]]
    if len(checked) != len(set(checked)) or set(checked) != expected:
        raise ValueError("Completion must review every planned unknown exactly once")
    proposed, supplemental = [], {}
    for path, report in zip(paths, reports):
        for raw in report["observations"]:
            item = dict(raw, review_report=path.relative_to(ROOT).as_posix())
            key = tuple(item[k] for k in ("object_id", "scale_id", "pole"))
            if key not in expected or key in {tuple(x[k] for k in ("object_id", "scale_id", "pole")) for x in proposed}:
                raise ValueError("Completion cannot overwrite known or duplicate a proposed property")
            if isinstance(item["value"], bool) or not isinstance(item["value"], (int, float)) or not 0 < item["value"] <= 2 or not item.get("rationale"):
                raise ValueError("Completion uses directly supported limited expression1..2, never inferred zeros")
            proposed.append(item)
            for raw_fact in item.get("new_facts", []):
                fact = dict(raw_fact, actor_id=item["object_id"], eligibility="eligible_direct",
                            source_review_report=item["review_report"])
                fact["topic_ids"] = sorted(set(fact.get("topic_ids", []) + [item["scale_id"]]))
                if not fact_within_period(fact, study):
                    raise ValueError("New completion evidence must be dated inside the study period")
                signature = (fact["actor_id"], fact["source_url"], fact["text"])
                if signature in supplemental:
                    supplemental[signature]["topic_ids"] = sorted(set(supplemental[signature]["topic_ids"] + fact["topic_ids"]))
                else:
                    supplemental[signature] = fact
    source_file = RESEARCH / "INDEPENDENT_COMPLETION_SOURCES.json"
    _write(source_file, {"at": now(), "facts": list(supplemental.values()), "purpose": "New direct sources collected before factor fitting"})
    config = dict(study["independent_evidence"], completion_source_supplement={"path": source_file.relative_to(ROOT).as_posix(),
                  "sha256": hashlib.sha256(source_file.read_bytes()).hexdigest()})
    revised_study = dict(study, independent_evidence=config)
    old_dossiers = {obj["id"]: json.loads((directory / "evidence" / ("independent-dossier-" + obj["id"] + ".json")).read_text(encoding="utf-8")) for obj in study["objects"]}
    dossiers = {obj["id"]: attach_reviewed_sources(SimpleNamespace(root=ROOT), revised_study, obj, old_dossiers[obj["id"]], "completion_source_supplement") for obj in study["objects"]}
    revised = copy.deepcopy(ratings)
    revised_lookup = {(row["object_id"], row["scale_id"]): row for row in revised}
    for item in proposed:
        row = revised_lookup[(item["object_id"], item["scale_id"])]
        facts = dossiers[item["object_id"]]["facts"]
        indices = list(item.get("evidence_indices", []))
        for fact in item.get("new_facts", []):
            indices += [i for i, f in enumerate(facts) if (f["source_url"], f["text"]) == (fact["source_url"], fact["text"])]
        indices = list(dict.fromkeys(indices))
        if not indices or any(not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < len(facts) or
                              facts[i].get("measurement_status") == "excluded_known_source_issue" or
                              not fact_within_period(facts[i], study) for i in indices):
            raise ValueError("Completion proposal has invalid, excluded or undated support")
        previous = row["pole_scores"][item["pole"]]
        row["pole_scores"][item["pole"]] = {"status": "observed", "value": item["value"], "confidence": item.get("confidence", .6),
            "evidence_indices": indices, "source_urls": list(dict.fromkeys(facts[i]["source_url"] for i in indices)),
            "event_ids": list(dict.fromkeys(facts[i].get("event_id", "") for i in indices)), "support_type": "positive_manifestation",
            "rationale": item["rationale"], "evidence_file": row["evidence_file"],
            "provenance": {"method": "coverage_only_assistant_completion_before_factor_fit", "review_report": item["review_report"],
                           "review_report_fingerprint": fingerprint(next(r for p, r in zip(paths, reports) if p.relative_to(ROOT).as_posix() == item["review_report"]))},
            "previous_unknown_measurement": previous}
    snapshot = directory / "attempts/independent_before_completion"
    if snapshot.exists():
        raise ValueError("Pre-completion snapshot exists")
    (snapshot / "evidence").mkdir(parents=True)
    (snapshot / "ratings.json").write_bytes(ratings_path.read_bytes())
    (snapshot / "study.json").write_bytes((directory / "study.json").read_bytes())
    for oid, dossier in old_dossiers.items():
        name = "independent-dossier-" + oid + ".json"
        (snapshot / "evidence" / name).write_bytes((directory / "evidence" / name).read_bytes())
        _write(directory / "evidence" / name, dossiers[oid])
    for row in revised:
        old = revised_lookup[(row["object_id"], row["scale_id"])]
        context = {"file": "attempts/independent_before_completion/" + row["evidence_file"], "basis": row["evidence_basis"]}
        row["pre_completion_evidence_snapshot"] = context
        row["evidence_basis"] = dossiers[row["object_id"]]["basis"]
        for entry in row["pole_scores"].values():
            entry["evidence_basis"] = row["evidence_basis"]
        entries = row["pole_scores"]
        known = sum(e["status"] == "observed" for e in entries.values())
        row.update(positive=entries["positive"]["value"], negative=entries["negative"]["value"],
                   status="observed" if known == 2 else "partial" if known else "unknown", difference=None, mass=None)
    for old, new in zip(ratings, revised):
        for pole, entry in old["pole_scores"].items():
            if entry["status"] == "observed" and any(entry.get(k) != new["pole_scores"][pole].get(k) for k in ("status", "value", "evidence_indices", "rationale")):
                raise AssertionError("Completion changed an already known measurement")
    _write(ratings_path, revised)
    receipt = {"applied": True, "at": now(), "plan_fingerprint": fingerprint(plan), "checked_unknown_properties": len(checked),
               "new_observations": len(proposed), "unresolved_unknown_properties": len(expected) - len(proposed),
               "new_source_facts": len(supplemental), "before_fingerprint": fingerprint(ratings), "after_fingerprint": fingerprint(revised),
               "reports": [{"path": p.relative_to(ROOT).as_posix(), "fingerprint": fingerprint(r)} for p, r in zip(paths, reports)],
               "thresholds_changed": False, "factor_fit_used_to_select_measurements": False}
    store.update(IDENTIFIER, independent_evidence=config, independent_completion=receipt, status="ready", ratings_cache={
        "basis": ratings_basis(revised_study), "ratings_sha256": fingerprint(revised), "mode": "independent_poles_reviewed_and_completed"})
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
