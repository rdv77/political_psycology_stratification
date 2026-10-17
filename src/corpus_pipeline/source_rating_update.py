"""Offline, source-reviewed episode aggregation for an approved dictionary.

This module never infers ratings from keywords, frequency, a person's office or
legacy scales. Its only numerical input is a separately validated native-chat
review. Exact reviewed event links remove repeated reporting before aggregation.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date
from statistics import median

from .scoring_preparation import digest


VERSION = "rwanda-reviewed-source-profiles-v1"
STRICT = "strict_dated"
LOOSE = "content_supported_unknown_time_sensitivity"
_KNOWN_DATE = {"supported_exact", "corrected_exact", "supported_interval"}
_ATTRIBUTED = {"supported", "corrected"}


def _date_interval(timing):
    status = timing.get("status")
    if status not in _KNOWN_DATE:
        if any(timing.get(k) for k in ("date", "date_from", "date_to")):
            raise ValueError("Unknown event time must not carry invented dates")
        return None
    exact = timing.get("date")
    start, end = ((exact, exact) if exact else (timing.get("date_from"), timing.get("date_to")))
    if not start or not end:
        raise ValueError("Known event time requires an explicit interval")
    date.fromisoformat(start)
    date.fromisoformat(end)
    if start > end:
        raise ValueError("Reversed event interval")
    return start, end


def _clean_actor(actor):
    return {k: actor[k] for k in ("id", "name", "name_ru", "name_original", "category", "role", "role_years", "inclusion_reason")
            if actor.get(k) is not None}


def _cases(review_bundles, dictionary):
    cases, ids = [], set()
    for review in review_bundles:
        if (review.get("measurement_dictionary_approved") is not True
                or review.get("dictionary_content_sha256") != digest(dictionary)
                or review.get("raw_quotes_published") is not False):
            raise ValueError("Use the validated safe review of this approved dictionary")
        for case in review["cases"]:
            if case["case_id"] in ids:
                raise ValueError("Duplicate source review case")
            ids.add(case["case_id"])
            cases.append(case)
    return cases


def _episode_groups(cases):
    """Union only explicit reviewed same-event links, never literal/topic matches."""
    by_id = {c["case_id"]: c for c in cases}
    parent = {key: key for key in by_id}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for case in cases:
        identity = case["event_identity"]
        related = identity.get("related_cases", [])
        if related and identity.get("status") != "same_event_supported":
            raise ValueError("Cross-case merging requires reviewed same-event support")
        for key in related:
            if key not in by_id or by_id[key]["actor_id"] != case["actor_id"]:
                raise ValueError("An episode link must refer to the same approved actor")
            parent[find(key)] = find(case["case_id"])
    groups = defaultdict(list)
    for case in cases:
        groups[find(case["case_id"])].append(case)
    return sorted((sorted(group, key=lambda c: c["case_id"]) for group in groups.values()),
                  key=lambda group: group[0]["case_id"])


def build_profiles(roster, dictionary, review_bundles, *, window_start="2021-10-15", window_end="2026-10-15"):
    """Return all roster cells, strict and unknown-time sensitivity separately.

    Multiple sources of one reviewed episode contribute one intensity. Conflicts
    inside an episode stay unknown. Distinct dated events receive equal weights;
    unresolved independence is held rather than silently treated as replication.
    The loose sensitivity can retain a common value invariant to duplicate
    weighting; differing undated-event values are held until event separation is
    reviewed. Held cases never become numerical personal evidence.
    """
    if dictionary.get("stage_status") != "approved" or dictionary.get("approved") is not True:
        raise ValueError("An approved measurement dictionary is required")
    start, end = date.fromisoformat(window_start), date.fromisoformat(window_end)
    if start > end:
        raise ValueError("Reversed study window")
    actors = [_clean_actor(actor) for actor in roster]
    actor_ids = [a["id"] for a in actors]
    scale_ids = [s["id"] for s in dictionary["scales"]]
    if len(set(actor_ids)) != len(actor_ids) or len(set(scale_ids)) != len(scale_ids):
        raise ValueError("Actor and property IDs must be unique")
    cases = _cases(review_bundles, dictionary)
    for case in cases:
        if case["actor_id"] not in actor_ids:
            raise ValueError("Source review actor is outside the approved roster")
        seen_scales = set()
        for rating in case["ratings"]:
            value, sid = rating["value"], rating["scale_id"]
            if sid not in scale_ids or sid in seen_scales:
                raise ValueError("Duplicate or unapproved reviewed property")
            seen_scales.add(sid)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 5:
                raise ValueError("Reviewed episode intensities must be integers from zero to five")
    episodes = []
    for index, group in enumerate(_episode_groups(cases), 1):
        admissible = [c for c in group if c.get("sample_kind") != "excluded_control"
                      and c["verdict"] in {"retain", "narrow"}
                      and c["attribution"]["status"] in _ATTRIBUTED
                      and c["event_date"]["status"] != "outside_window"
                      and c["event_identity"]["status"] not in {"unresolved", "distinct_events_in_literal_group"}]
        known_intervals = {_date_interval(c["event_date"]) for c in admissible}
        known_intervals.discard(None)
        date_conflict = len(known_intervals) > 1
        timing = next(iter(known_intervals)) if len(known_intervals) == 1 else None
        inside = bool(timing and timing[0] >= window_start and timing[1] <= window_end)
        rating_groups = defaultdict(list)
        for case in admissible:
            for rating in case["ratings"]:
                rating_groups[rating["scale_id"]].append((case["case_id"], rating))
        ratings = []
        for sid, evidence in sorted(rating_groups.items()):
            values = {r["value"] for _, r in evidence}
            ratings.append({"scale_id": sid, "value": next(iter(values)) if len(values) == 1 else None,
                            "status": "reviewed_episode_intensity" if len(values) == 1 else "within_episode_conflict",
                            "supporting_cases": [cid for cid, _ in evidence],
                            "reason_ru": list(dict.fromkeys(r["reason_ru"] for _, r in evidence)),
                            "case_rating_count": len(evidence)})
        episodes.append({"episode_id": f"E{index:03d}", "actor_id": group[0]["actor_id"],
                         "case_ids": [c["case_id"] for c in group],
                         "admissible_case_ids": [c["case_id"] for c in admissible],
                         "event_date_from": timing[0] if timing else None,
                         "event_date_to": timing[1] if timing else None,
                         "event_time_status": "conflicting_reviewed_dates" if date_conflict else
                             "known_inside_window" if inside else "known_not_fully_inside_window" if timing else "unverified",
                         "eligible_strict": bool(admissible and inside and not date_conflict),
                         "eligible_unknown_time_sensitivity": bool(admissible and not date_conflict and (inside or timing is None)),
                         "summaries_ru": list(dict.fromkeys(c["summary_ru"] for c in group)),
                         "capacities_ru": list(dict.fromkeys(c["capacity_review_ru"] for c in group)),
                         "ratings": ratings})

    scenarios = {}
    for scenario, eligibility in ((STRICT, "eligible_strict"), (LOOSE, "eligible_unknown_time_sensitivity")):
        candidates = defaultdict(list)
        for episode in episodes:
            if episode[eligibility]:
                for rating in episode["ratings"]:
                    if rating["value"] is not None:
                        candidates[episode["actor_id"], rating["scale_id"]].append((episode, rating))
        cells = []
        for actor in actors:
            for sid in scale_ids:
                evidence = candidates[actor["id"], sid]
                separation_verified = True
                for i, (left, _) in enumerate(evidence):
                    for right, _ in evidence[i + 1:]:
                        a, b = left["event_date_from"], left["event_date_to"]
                        c, d = right["event_date_from"], right["event_date_to"]
                        if not (a and b and c and d and (b < c or d < a)):
                            separation_verified = False
                distinct_values = {r["value"] for _, r in evidence}
                invariant = bool(evidence and len(distinct_values) == 1)
                observed = bool(evidence and (separation_verified or invariant))
                cells.append({"actor_id": actor["id"], "object_id": actor["id"], "scale_id": sid,
                              "status": "observed" if observed else "unknown",
                              "value": median(r["value"] for _, r in evidence) if observed else None,
                              "aggregation": "median_equal_verified_episodes" if observed and separation_verified else
                                  "common_value_invariant_to_unverified_event_multiplicity" if observed else None,
                              "eligible_episode_count": len(evidence),
                              "episode_separation_verified": separation_verified if evidence else None,
                              "distinct_event_count": len(evidence) if evidence and separation_verified else None,
                              "unknown_reason": None if observed else
                                  "unresolved_event_independence" if evidence else "no_admissible_reviewed_episode",
                              "episode_ids": [e["episode_id"] for e, _ in evidence],
                              "case_ids": sorted({cid for _, r in evidence for cid in r["supporting_cases"]}),
                              "is_stable_personal_trait_estimate": False})
        scenario_actors = []
        for actor in actors:
            known = [c for c in cells if c["actor_id"] == actor["id"] and c["status"] == "observed"]
            scenario_actors.append({**actor, "observed_properties": len(known), "unknown_properties": len(scale_ids) - len(known),
                                    "coverage": len(known) / len(scale_ids),
                                    "status": "source_episode_profile" if known else "insufficient_reviewed_evidence",
                                    "ratings": [{"scale_id": c["scale_id"], "value": c["value"], "case_ids": c["case_ids"],
                                                 "eligible_episode_count": c["eligible_episode_count"],
                                                 "episode_separation_verified": c["episode_separation_verified"]} for c in known]})
        observed_cells = [c for c in cells if c["status"] == "observed"]
        scenarios[scenario] = {"cells": cells, "actors": scenario_actors,
                              "counts": {"roster_actors": len(actors), "properties": len(scale_ids), "total_cells": len(cells),
                                         "observed_cells": len(observed_cells), "unknown_cells": len(cells) - len(observed_cells),
                                         "coverage": len(observed_cells) / len(cells) if cells else 0,
                                         "actors_with_observed_properties": sum(a["observed_properties"] > 0 for a in scenario_actors),
                                         "properties_with_observations": len({c["scale_id"] for c in observed_cells}),
                                         "numeric_case_ratings_before_episode_deduplication": sum(r["case_rating_count"] for e in episodes if e[eligibility] for r in e["ratings"]),
                                         "episode_property_values_after_deduplication": sum(r["value"] is not None for e in episodes if e[eligibility] for r in e["ratings"]),
                                         "cells_held_for_event_independence": sum(c["unknown_reason"] == "unresolved_event_independence" for c in cells)}}
    return {"version": VERSION, "dictionary_content_sha256": digest(dictionary),
            "approved_window": {"from": window_start, "to": window_end},
            "counts": {"approved_actors": len(actors), "approved_properties": len(scale_ids), "reviewed_cases": len(cases),
                       "reviewed_case_ratings": sum(len(c["ratings"]) for c in cases), "reviewed_episode_groups": len(episodes),
                       "same_event_groups_with_multiple_cases": sum(len(e["case_ids"]) > 1 for e in episodes)},
            "scales": [{k: s[k] for k in ("id", "name", "definition", "block", "anchor0", "anchor3", "anchor5", "coding_boundary") if k in s}
                       for s in dictionary["scales"]],
            "actors": scenarios[STRICT]["actors"], "cells": scenarios[STRICT]["cells"],
            "episodes": episodes, "scenarios": scenarios,
            "api_calls": 0, "new_api_cost_usd": 0, "canonical_study_modified": False,
            "old_ratings_transferred": False, "quotes_published": False,
            "limitations": ["Known values describe reviewed episodes, not stable five-year political beliefs.",
                            "All 42 approved people are retained as cards; card presence does not imply a measured profile.",
                            "Unreviewed facts and unmentioned properties never generate numerical scores or zeros.",
                            "Frequency and duplicate reporting do not increase intensity.",
                            "The unknown-time sensitivity is descriptive and cannot verify an episode's inclusion in the research period.",
                            "The source review is not an independent human panel or external truth verification.",
                            "Institutional and private capacities are reported and must not be generalized to Rwandan state policy."]}


def factor_admission(study, cells):
    """Run the existing, unchanged service gate before any imputation or fit."""
    from src.semantic_service.analysis import InsufficientMeasurementError, _prepare

    ratings = [{"object_id": c["actor_id"], "scale_id": c["scale_id"], "status": c["status"], "value": c["value"]} for c in cells]
    try:
        _, _, _, _, diagnostics = _prepare(study, ratings)
    except InsufficientMeasurementError as error:
        return {"status": "not_admissible", "reason_ru": str(error), "diagnostics": error.diagnostics,
                "imputation_performed": False, "factor_models_estimated": False,
                "solutions": [{"q": q, "status": "not_estimated_insufficient_measurement", "variance_total": None} for q in (2, 3, 4)]}
    p = diagnostics["retained_scales"]
    return {"status": "admissible_for_parameter_screen", "diagnostics": diagnostics,
            "imputation_performed": False, "factor_models_estimated": False,
            "solutions": [{"q": q, "status": "requires_model_fit" if p * (p + 1) / 2 - (p * q + p - q * (q - 1) / 2) >= 0
                           else "fails_necessary_parameter_count_screen", "variance_total": None} for q in (2, 3, 4)]}


def attach_factor_comparison(result, roster, dictionary, historical):
    """Attach safe coverage audits and explicitly non-equivalent old solutions."""
    study = {"objects": [_clean_actor(a) for a in roster], "scales": dictionary["scales"], "measurement_model": "independent_poles"}
    for scenario in result["scenarios"].values():
        scenario["factor_admission"] = factor_admission(study, scenario["cells"])
    result["historical_comparison"] = {
        "direct_loading_or_cluster_comparison_valid": False,
        "reason_ru": "Изменились словарь, основания оценивания и состав измеренной матрицы; численные различия нельзя приписать изменению политических взглядов.",
        "historical": {"input_actors": historical["diagnostics"]["input_objects"],
                       "input_variables": historical["diagnostics"]["input_scales"],
                       "known_cells": historical["diagnostics"]["observed_input_ratings"],
                       "input_coverage": historical["diagnostics"]["input_coverage"],
                       "retained_actors": historical["n_objects"], "retained_variables": historical["n_scales"],
                       "retained_coverage": historical["diagnostics"]["retained_coverage"],
                       "models": [{"q": model["q"], "variance_total": model["variance_total"]} for model in historical["models"]]},
        "new": {key: {"counts": scenario["counts"], "status": scenario["factor_admission"]["status"],
                      "solutions": scenario["factor_admission"]["solutions"]} for key, scenario in result["scenarios"].items()},
        "russia_factor_status": "not_testable_without_admissible_measurement_matrix",
        "russia_factor_variance_threshold": .02,
        "clusters_reestimated": False,
        "previous_cluster_claims_reconfirmed": False}
    return result
