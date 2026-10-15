"""Add reviewed evidence to predeclared unknown cells without imputing ratings."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.semantic_service.independent_poles import _write, apply_fact_review, attach_reviewed_sources
from src.semantic_service.providers import fact_within_period
from src.semantic_service.store import StudyStore, fingerprint, now, ratings_basis

IDENTIFIER = "rwanda-politicians-2021-2026"
RESEARCH = ROOT / "research/rwanda_politicians_2021_2026"
CONFIG_KEY = "coverage_expansion_source_supplement"


def collect_proposals(study, ratings, plan, reports):
    """Validate complete review scope before accepting any new measurements."""
    if fingerprint(ratings) != plan["input_ratings_fingerprint"]:
        raise ValueError("Expansion plan belongs to another ratings snapshot")
    if any(fingerprint(study[field]) != plan[f"input_{field}_fingerprint"] for field in ("objects", "scales")):
        raise ValueError("Approved objects or scales changed after the expansion plan")
    scope = plan.get("input_study_scope")
    if not isinstance(scope, dict) or not scope or any(study.get(key) != value for key, value in scope.items()):
        raise ValueError("Study scope, period or measurement model changed after the expansion plan")
    expected = {(t["object_id"], t["scale_id"], t["pole"]) for t in plan["checked_unknown_properties"]}
    lookup = {(r["object_id"], r["scale_id"], pole): p for r in ratings for pole, p in r["pole_scores"].items()}
    if any(key not in lookup or lookup[key]["status"] != "unknown" for key in expected):
        raise ValueError("Expansion scope includes an observed or nonexistent property")
    checked = [tuple(key) for report in reports for key in report["checked_properties"]]
    if len(checked) != len(set(checked)) or set(checked) != expected:
        raise ValueError("Review every planned unknown exactly once")
    proposals, signatures, facts = [], set(), {}
    for report in reports:
        if report.get("plan_fingerprint") != fingerprint(plan) or not isinstance(report.get("batch"), str) or not report["batch"].strip():
            raise ValueError("Source review is not bound to the frozen expansion plan")
        for raw in report.get("observations", []):
            item = copy.deepcopy(raw)
            key = tuple(item[k] for k in ("object_id", "scale_id", "pole"))
            if key not in expected or key in signatures:
                raise ValueError("An expansion cannot overwrite known or duplicate proposed ratings")
            value = item["value"]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 3 or not item.get("rationale"):
                raise ValueError("Require a justified positive manifestation1..3, never an evidence-free zero")
            signatures.add(key)
            for raw_fact in item.get("new_facts", []):
                if raw_fact.get("actor_id", item["object_id"]) != item["object_id"]:
                    raise ValueError("A source is attributed to a different actor")
                if raw_fact.get("measurement_status") in {"context_only", "excluded_known_source_issue"} or raw_fact.get("eligibility") == "context_only":
                    raise ValueError("A context-only or excluded fact cannot authorize an observed property")
                fact = dict(raw_fact, actor_id=item["object_id"], eligibility="eligible_direct")
                fact["topic_ids"] = sorted(set(fact.get("topic_ids", []) + [item["scale_id"]]))
                if not fact_within_period(fact, study) or any(not str(fact.get(k, "")).strip() for k in ("source_url", "text", "event_id", "capacity")):
                    raise ValueError("New facts must have dated, attributable, nonempty support")
                signature = (fact["actor_id"], fact["source_url"], fact["text"])
                if signature in facts:
                    facts[signature]["topic_ids"] = sorted(set(facts[signature]["topic_ids"] + fact["topic_ids"]))
                else:
                    facts[signature] = fact
            item["review_batch"] = report["batch"]
            proposals.append(item)
    return proposals, list(facts.values())


def supported_indices(proposal, facts, study):
    """Use the same source exclusions as ordinary scoring, before any writes."""
    indices = list(proposal.get("evidence_indices", []))
    for new in proposal.get("new_facts", []):
        indices.extend(i for i, fact in enumerate(facts)
                       if (fact.get("source_url"), fact.get("text")) == (new["source_url"], new["text"]))
    indices = list(dict.fromkeys(indices))
    exclusions = {(item["object_id"], item["source_url"], item["text_sha256"])
                  for item in study.get("independent_evidence", {}).get("additional_excluded_facts", [])}
    scale = next((item for item in study["scales"] if item["id"] == proposal["scale_id"]), {})
    thematic = scale.get("research_domain", (scale.get("metadata") or {}).get("research_domain")) == "russia_relations"
    topics = {proposal["scale_id"], proposal["scale_id"] + "__" + proposal["pole"]}
    if not indices:
        raise ValueError("Unsupported measurement has no source indices")
    for index in indices:
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(facts):
            raise ValueError("Unsupported source index")
        fact = facts[index]
        excluded_key = (proposal["object_id"], fact.get("source_url"),
                        hashlib.sha256(str(fact.get("text", "")).encode("utf-8")).hexdigest())
        if (not fact_within_period(fact, study) or
            fact.get("measurement_status") in {"context_only", "excluded_known_source_issue"} or
            fact.get("eligibility") == "context_only" or excluded_key in exclusions or
            fact.get("actor_id", fact.get("object_id", proposal["object_id"])) != proposal["object_id"] or
            (thematic and (fact.get("evidence_origin") != "assistant_verified_source" or
                          not topics.intersection(fact.get("topic_ids", []))))):
            raise ValueError("Unsupported, excluded, foreign, context-only or out-of-scope measurement")
    if proposal["value"] >= 3 and len({facts[i].get("event_id") for i in indices if facts[i].get("event_id")}) < 2:
        raise ValueError("A repeated emphasis needs distinct events, not repeated publications")
    return indices


def main():
    store = StudyStore(ROOT / "studies")
    study = store.require_approved(IDENTIFIER)
    if study.get("coverage_expansion", {}).get("applied"):
        raise ValueError("This expansion was already applied; preserve its history")
    directory = store.directory(IDENTIFIER)
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    ratings_path = directory / "ratings.json"
    ratings = read(ratings_path)
    if study.get("ratings_cache", {}).get("basis") != ratings_basis(study):
        raise ValueError("Current measurements are not bound to this study")
    plan = read(RESEARCH / "COVERAGE_EXPANSION_PLAN.json")
    paths = [RESEARCH / f"COVERAGE_EXPANSION_BATCH_{batch}.json" for batch in "ABC"]
    reports = [read(p) for p in paths]
    proposals, new_facts = collect_proposals(study, ratings, plan, reports)
    decision_path = RESEARCH / "COVERAGE_EXPANSION_FINAL_REVIEW.json"
    decision = read(decision_path)
    if decision["plan_fingerprint"] != fingerprint(plan) or decision["report_fingerprints"] != {p.name: fingerprint(r) for p, r in zip(paths, reports)}:
        raise ValueError("Final content review belongs to different reports or plan")
    proposed_keys = {tuple(p[k] for k in ("object_id", "scale_id", "pole")) for p in proposals}
    if {tuple(k) for k in decision["accepted_proposal_keys"]} != proposed_keys:
        raise ValueError("Each accepted proposal requires an explicit final content decision")
    flags = {tuple(f[k] for k in ("object_id", "scale_id", "pole")): {"decision": "flag", "reason": f["reason"]}
             for f in decision.get("context_flags", [])}
    for proposal in proposals:
        key = tuple(proposal[k] for k in ("object_id", "scale_id", "pole"))
        if key in flags:
            proposal["source_audit"] = flags[key]
    supplement = RESEARCH / "COVERAGE_EXPANSION_SOURCES.json"
    _write(supplement, {"at": now(), "plan_fingerprint": fingerprint(plan), "facts": new_facts,
                        "purpose": "Direct public evidence, separate from model-based imputation"})
    config = dict(study["independent_evidence"], **{CONFIG_KEY: {
        "path": supplement.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(supplement.read_bytes()).hexdigest()}})
    revised_study = dict(study, independent_evidence=config)
    provider = SimpleNamespace(root=ROOT)
    dossiers = {obj["id"]: apply_fact_review(provider, revised_study, obj,
        attach_reviewed_sources(provider, revised_study, obj,
            read(directory / "evidence" / f"independent-dossier-{obj['id']}.json"), CONFIG_KEY)) for obj in study["objects"]}
    revised = copy.deepcopy(ratings)
    lookup = {(r["object_id"], r["scale_id"]): r for r in revised}
    for proposal in proposals:
        row = lookup[proposal["object_id"], proposal["scale_id"]]
        facts = dossiers[proposal["object_id"]]["facts"]
        indices = supported_indices(proposal, facts, study)
        previous = row["pole_scores"][proposal["pole"]]
        entry = {"status": "observed", "value": proposal["value"], "confidence": proposal.get("confidence", .6),
                 "support_type": "positive_manifestation", "rationale": proposal["rationale"],
                 "evidence_indices": indices, "source_urls": list(dict.fromkeys(facts[i]["source_url"] for i in indices)),
                 "event_ids": list(dict.fromkeys(facts[i]["event_id"] for i in indices)),
                 "evidence_file": row["evidence_file"], "previous_unknown_measurement": previous,
                 "provenance": {"method": "assistant_reviewed_coverage_expansion", "review_batch": proposal["review_batch"],
                                "plan_fingerprint": fingerprint(plan), "final_content_review": decision_path.relative_to(ROOT).as_posix(),
                                "final_content_review_fingerprint": fingerprint(decision)}}
        if proposal.get("source_audit"):
            entry["source_audit"] = proposal["source_audit"]
        row["pole_scores"][proposal["pole"]] = entry
    for old, row in zip(ratings, revised):
        for pole, entry in old["pole_scores"].items():
            if entry["status"] == "observed" and entry != row["pole_scores"][pole]:
                raise AssertionError("An already observed property was changed")
    snapshot = directory / "attempts/independent_before_coverage_expansion"
    if snapshot.exists():
        raise ValueError("Do not overwrite the pre-expansion snapshot")
    snapshot.mkdir()
    for name in ("study.json", "ratings.json", "README.md"):
        (snapshot / name).write_bytes((directory / name).read_bytes())
    shutil.copytree(directory / "output", snapshot / "output")
    (snapshot / "evidence").mkdir()
    for obj in study["objects"]:
        filename = f"independent-dossier-{obj['id']}.json"
        (snapshot / "evidence" / filename).write_bytes((directory / "evidence" / filename).read_bytes())
        _write(directory / "evidence" / filename, dossiers[obj["id"]])
    for row in revised:
        row["pre_coverage_expansion_evidence_snapshot"] = {"file": "attempts/independent_before_coverage_expansion/" + row["evidence_file"],
                                                          "basis": row["evidence_basis"]}
        row["evidence_basis"] = dossiers[row["object_id"]]["basis"]
        for entry in row["pole_scores"].values():
            entry["evidence_basis"] = row["evidence_basis"]
        poles = row["pole_scores"]
        count = sum(entry["status"] == "observed" for entry in poles.values())
        row.update(positive=poles["positive"]["value"], negative=poles["negative"]["value"], difference=None, mass=None,
                   status="observed" if count == 2 else "partial" if count else "unknown")
    _write(ratings_path, revised)
    receipt = {"applied": True, "at": now(), "plan_fingerprint": fingerprint(plan),
               "checked_unknown_properties": len(plan["checked_unknown_properties"]), "new_observations": len(proposals),
               "new_source_facts": len(new_facts), "before_fingerprint": fingerprint(ratings), "after_fingerprint": fingerprint(revised),
               "reports": [{"path": p.relative_to(ROOT).as_posix(), "fingerprint": fingerprint(r)} for p, r in zip(paths, reports)],
               "final_content_review": {"path": decision_path.relative_to(ROOT).as_posix(), "fingerprint": fingerprint(decision)},
               "thresholds_changed": False, "imputed_values_added_to_ratings": False,
               "selection_note": "Coverage targets chosen from the preceding analysis; this is an explicitly adaptive exploratory followup"}
    store.update(IDENTIFIER, independent_evidence=config, coverage_expansion=receipt, status="ready", analysis=None, artifacts=[],
        ratings_cache={"basis": ratings_basis(revised_study), "ratings_sha256": fingerprint(revised), "mode": "independent_poles_coverage_expansion"})
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
