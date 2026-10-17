"""Conditional, poststratified cost scenarios from a document-level pilot.

No API, model fitting, token pricing, persistence or probability intervals. Costs
must already include *all* chunks belonging to the document. Targeted extras are
reported as expenditure, never silently used as the random-sample baseline.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math


_BOOLEANS = ("random_sample", "relevance_complete", "relevant", "review",
             "extraction_complete", "extraction_random")
_COSTS = ("relevance_cost_usd", "relevance_uncached_cost_usd",
          "extraction_cost_usd", "extraction_uncached_cost_usd")


def _nonnegative(value, label, *, integer=False):
    if (type(value) not in ((int,) if integer else (int, float))
            or not math.isfinite(value) or value < 0):
        raise ValueError(label + " must be a finite nonnegative " + ("integer" if integer else "number"))
    return value


def _mean(rows, key):
    values = [row[key] for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def _sum_known(rows, key):
    return {"known_sum_usd": sum(row[key] for row in rows if row.get(key) is not None),
            "n_known": sum(row.get(key) is not None for row in rows),
            "n_missing": sum(row.get(key) is None for row in rows)}


def _scale(value, multiplier):
    return value * multiplier if value is not None else None


def _add(a, b):
    return a + b if a is not None and b is not None else None


def estimate_costs(population: list[dict], observations: list[dict], *,
                   target_count=32000, raw_count: int,
                   descriptor_cost_usd=0, descriptor_input_facts=0) -> dict:
    """Estimate one-pass classification/extraction, with explicit missing strata.

    ``population`` is the complete eligible, unique pilot sampling frame. Each
    observation is one whole document, not a chunk. Missing boolean fields are
    False; absent monetary/fact fields are unknown, not zero. Complete random
    documents estimate classification and relevance. Extraction uses only
    completed, randomly chosen *positive* documents from that random sample.

    Observed relevant fraction is the central scenario; relevant OR review is
    an ambiguity upper scenario. This is not a sampling confidence interval.
    Incomplete random classifications are excluded, with an explicit warning.
    A global extraction fallback is weighted by estimated positive population
    sizes of strata with usable extraction observations, not their sample sizes.
    """
    _nonnegative(raw_count, "raw_count", integer=True)
    _nonnegative(target_count, "target_count", integer=True)
    _nonnegative(descriptor_cost_usd, "descriptor_cost_usd")
    _nonnegative(descriptor_input_facts, "descriptor_input_facts", integer=True)
    if raw_count == 0 or target_count == 0 or not isinstance(population, list) or not population:
        raise ValueError("A nonempty population and positive raw/target counts are required")
    if not isinstance(observations, list) or len(population) > raw_count:
        raise ValueError("Observations must be a list and eligible unique count cannot exceed raw_count")
    frame = {}
    for row in population:
        if (not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"].strip()
                or not isinstance(row.get("stratum"), str) or not row["stratum"].strip()):
            raise ValueError("Population needs nonempty string id and stratum")
        if row["id"] in frame:
            raise ValueError("Duplicate population document ID")
        frame[row["id"]] = row["stratum"]
    seen, records = set(), []
    for original in observations:
        if not isinstance(original, dict) or original.get("id") not in frame:
            raise ValueError("Observation ID must belong to the population")
        if original["id"] in seen or original.get("stratum") != frame[original["id"]]:
            raise ValueError("Duplicate observation or inconsistent stratum")
        seen.add(original["id"])
        row = dict(original)
        for key in _BOOLEANS:
            if key in row and type(row[key]) is not bool:
                raise ValueError(key + " must be boolean")
            row.setdefault(key, False)
        for key in (*_COSTS, "facts"):
            if row.get(key) is not None:
                _nonnegative(row[key], key, integer=key == "facts")
            else:
                row[key] = None
        records.append(row)

    flags = []

    def flag(code, message, stratum=None):
        item = {"code": code, "message": message}
        if stratum is not None:
            item["stratum"] = stratum
        flags.append(item)

    groups = defaultdict(list)
    for row in records:
        groups[row["stratum"]].append(row)
    counts = Counter(frame.values())
    strata = []
    for name, size in sorted(counts.items()):
        selected = [r for r in groups[name] if r["random_sample"]]
        complete = [r for r in selected if r["relevance_complete"]]
        positive = [r for r in complete if r["relevant"]]
        extraction = [r for r in positive if r["extraction_complete"] and r["extraction_random"]]
        if len(complete) < len(selected):
            flag("incomplete_random_classification", "Incomplete documents excluded; completed-sample transfer is an assumption.", name)
        if not complete:
            flag("uncovered_classification_stratum", "No complete random classification; its rate and classification cost remain unknown.", name)
        rates = {"observed_relevant": len(positive) / len(complete) if complete else None,
                 "relevant_or_review": sum(r["relevant"] or r["review"] for r in complete) / len(complete) if complete else None,
                 "review": sum(r["review"] for r in complete) / len(complete) if complete else None}
        classification = {"baseline_uncached_usd": _mean(complete, "relevance_uncached_cost_usd"),
                          "observed_cache_usd": _mean(complete, "relevance_cost_usd")}
        conditional = {"baseline_uncached_usd": _mean(extraction, "extraction_uncached_cost_usd"),
                       "observed_cache_usd": _mean(extraction, "extraction_cost_usd"),
                       "facts": _mean(extraction, "facts")}
        classification_n = {"baseline_uncached_usd": sum(r["relevance_uncached_cost_usd"] is not None for r in complete),
                            "observed_cache_usd": sum(r["relevance_cost_usd"] is not None for r in complete)}
        extraction_n = {"baseline_uncached_usd": sum(r["extraction_uncached_cost_usd"] is not None for r in extraction),
                        "observed_cache_usd": sum(r["extraction_cost_usd"] is not None for r in extraction),
                        "facts": sum(r["facts"] is not None for r in extraction)}
        for key, n in classification_n.items():
            if complete and n < len(complete):
                flag("missing_classification_cost", "Missing " + key + "; only reported costs enter its mean, never zero. Missing-cost transfer is an assumption.", name)
        if any(n < len(extraction) for n in extraction_n.values()):
            flag("missing_extraction_field", "Extraction costs/yield use their own reported denominators; missing values are not zero.", name)
        strata.append({"stratum": name, "population_count": size, "population_weight": size / len(frame),
                       "random_selected": len(selected), "classification_complete": len(complete),
                       "design_weight": size / len(selected) if selected else None,
                       "complete_case_expansion_weight": size / len(complete) if complete else None,
                       "relevant_complete": len(positive), "extraction_random_complete": len(extraction),
                       "targeted_observations": sum(not r["random_sample"] for r in groups[name]),
                       "relevance_rates": rates, "classification_mean_per_document": classification,
                       "classification_cost_n": classification_n, "extraction_field_n": extraction_n,
                       "extraction_conditional_local": conditional})

    # The two-phase conditional mean respects population prevalence, rather
    # than allowing oversampled small strata to dominate a global fallback.
    global_conditional = {}
    for key in ("baseline_uncached_usd", "observed_cache_usd", "facts"):
        available = [(s["population_count"] * s["relevance_rates"]["observed_relevant"],
                      s["extraction_conditional_local"][key]) for s in strata
                     if s["relevance_rates"]["observed_relevant"] is not None
                     and s["extraction_conditional_local"][key] is not None]
        weight = sum(w for w, value in available)
        global_conditional[key] = sum(w * value for w, value in available) / weight if weight > 0 else None
    for s in strata:
        effective, sources = {}, {}
        for key, local in s["extraction_conditional_local"].items():
            effective[key] = local if local is not None else global_conditional[key]
            sources[key] = "stratum_random" if local is not None else "global_random_conditional" if effective[key] is not None else "unavailable"
        s["extraction_conditional_effective"] = effective
        s["extraction_conditional_source"] = sources
        may_be_positive = s["relevance_rates"]["relevant_or_review"]
        if may_be_positive is None or may_be_positive > 0:
            if "global_random_conditional" in sources.values():
                flag("global_extraction_fallback", "Missing within-stratum extraction uses the population-weighted conditional random mean; exchangeability is assumed.", s["stratum"])
            if "unavailable" in sources.values():
                flag("uncovered_extraction_stratum", "No eligible random extraction baseline for at least one cost/yield field; forecasts remain unknown.", s["stratum"])
    if not any(s["extraction_random_complete"] for s in strata):
        flag("no_random_extraction", "No completed eligible random extraction sample; targeted extras cannot supply the baseline.")

    def weighted_classification(key):
        values = [(s["population_weight"], s["classification_mean_per_document"][key]) for s in strata]
        return sum(w * value for w, value in values) if all(value is not None for w, value in values) else None

    def weighted_extraction(key, rate_key):
        total = 0.
        for s in strata:
            rate = s["relevance_rates"][rate_key]
            conditional = s["extraction_conditional_effective"][key]
            if rate is None or (rate > 0 and conditional is None):
                return None
            total += s["population_weight"] * rate * (conditional if conditional is not None else 0.)
        return total

    per_eligible = {"classification": {}, "extraction": {}, "core": {}}
    for key in ("baseline_uncached_usd", "observed_cache_usd"):
        c = weighted_classification(key)
        e = weighted_extraction(key, "observed_relevant")
        upper = weighted_extraction(key, "relevant_or_review")
        upper_key = "review_upper_uncached_usd" if key == "baseline_uncached_usd" else "review_upper_observed_cache_usd"
        per_eligible["classification"].update({key: c, upper_key: c})
        per_eligible["extraction"].update({key: e, upper_key: upper})
        per_eligible["core"].update({key: _add(c, e), upper_key: _add(c, upper)})
    fact_yield = {"observed_relevant": weighted_extraction("facts", "observed_relevant"),
                  "relevant_or_review": weighted_extraction("facts", "relevant_or_review")}
    overall_rates = {}
    for key in ("observed_relevant", "relevant_or_review", "review"):
        overall_rates[key] = (sum(s["population_weight"] * s["relevance_rates"][key] for s in strata)
                              if all(s["relevance_rates"][key] is not None for s in strata) else None)

    def costs_at(count):
        return {stage: {key: _scale(value, count) for key, value in costs.items()} for stage, costs in per_eligible.items()}

    descriptor_unit = descriptor_cost_usd / descriptor_input_facts if descriptor_input_facts else None
    if descriptor_unit is None:
        flag("descriptor_unit_cost_unavailable", "No positive descriptor input-fact denominator; descriptor forecast is unknown.")
    scenarios = []
    fractions = (("observed_eligible_fraction", len(frame) / raw_count),
                 ("assumed_eligible_fraction_0.75", .75), ("assumed_eligible_fraction_1.0", 1.))
    for name, fraction in fractions:
        count = target_count * fraction
        yields = {key: _scale(value, count) for key, value in fact_yield.items()}
        scenarios.append({"name": name, "eligible_fraction": fraction, "eligible_count": count,
                          "fraction_basis": "measured_pilot_ratio_transferred" if name == "observed_eligible_fraction" else "assumption",
                          "stages": costs_at(count), "per_1000_raw": costs_at(1000 * fraction),
                          "estimated_fact_yield": yields,
                          "descriptor_forecast_usd": {key: value * descriptor_unit if value is not None and descriptor_unit is not None else None
                                                      for key, value in yields.items()}})
    return {
        "methodology": {"unit": "whole_unique_document_including_all_chunks", "sampling_weights": "N_h/n_h",
                        "central_relevance": "observed_relevant", "ambiguity_upper": "relevant_or_review",
                        "interval_kind": "scenario_only_not_probability_or_confidence_interval",
                        "incomplete_classification": "complete_case_with_explicit_nonresponse_warning",
                        "global_extraction_weight": "estimated_positive_population_in_covered_strata",
                        "cache_assumption": "baseline assumes no guaranteed cache; observed-cache transfer is a separate scenario",
                        "scope": "one_pass_classification_and_extraction; descriptor estimate separate; scoring, review, retries and revisions excluded"},
        "population": {"raw_count": raw_count, "eligible_unique_count": len(frame),
                       "observed_eligible_fraction": len(frame) / raw_count, "target_raw_count": target_count},
        "measured": {"observations": len(records), "random_documents": sum(r["random_sample"] for r in records),
                     "targeted_documents": sum(not r["random_sample"] for r in records),
                     "classification_complete_random": sum(s["classification_complete"] for s in strata),
                     "relevant_complete_random": sum(s["relevant_complete"] for s in strata),
                     "review_complete_random": sum(r["random_sample"] and r["relevance_complete"] and r["review"] for r in records),
                     "extraction_complete_random_positive": sum(s["extraction_random_complete"] for s in strata),
                     "reported_costs": {key: _sum_known(records, key) for key in _COSTS},
                     "descriptor_cost_usd": descriptor_cost_usd, "descriptor_input_facts": descriptor_input_facts,
                     "note": "Reported expenditure includes targeted and incomplete records; it is not the forecast and excludes any unreported/uncertain charge."},
        "strata": strata,
        "forecasts": {"primary_scenario": "observed_eligible_fraction", "per_1000_eligible": costs_at(1000),
                      "relevance_rates": overall_rates,
                      "scenarios": scenarios, "global_extraction_conditional": global_conditional},
        "descriptor_forecast": {"observed_usd_per_input_fact": descriptor_unit,
                                "assumption": "Linear fact packaging transfer; group overhead, repeated synthesis and output growth may invalidate it.",
                                "included_in_core": False},
        "flags": flags,
    }
