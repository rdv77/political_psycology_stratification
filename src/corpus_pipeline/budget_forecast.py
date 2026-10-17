"""Empirical, offline cost scenarios for actual planned one-document payloads.

Known tokenized fields are summed from the planned wire format. Provider prompt
rendering, new classification outcomes, output sizes and fact yield remain
calibration/scenario assumptions. No file access, network, queue or key lookup.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import json
import math
import statistics

from .models import DEFAULT_PRICING, _eligible_fact, estimate_cost, make_task
from .service import MAX_OUTPUT, compact_objects
from .source_spans import document_texts

_KINDS = {"article", "post", "comment", "repost", "unknown"}
_STATS = ("mean", "median", "p90")


def _dump(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _quantile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] * (hi - position) + values[hi] * (position - lo) if hi != lo else values[lo]


def _stats(values):
    return {"n": len(values), "mean": statistics.mean(values), "median": statistics.median(values),
            "p90": _quantile(values, .9), "min": min(values), "max": max(values)} if values else None


def _kind(value):
    return value if value in _KINDS else "unknown"


def _bucket(length):
    return "0..500" if length <= 500 else "501..1500" if length <= 1500 else "1501..6000" if length <= 6000 else ">6000"


class _Counter:
    def __init__(self, encoding):
        self.encoding, self.static = encoding, {}

    def count(self, text):
        return len(self.encoding.encode_ordinary(text)) if hasattr(self.encoding, "encode_ordinary") else len(self.encoding.encode(text, disallowed_special=()))

    def fixed(self, text):
        if text not in self.static:
            self.static[text] = self.count(text)
        return self.static[text]

    def parts(self, payload):
        schema = json.dumps(payload["text"]["format"]["schema"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return {"user": self.count(payload["input"][0]["content"][0]["text"]),
                "instructions": self.fixed(payload["instructions"]), "schema": self.fixed(schema)}

    def body(self, document):
        return sum(self.count(text) for text in document_texts(document))


def _pilot(pilot, counter):
    jobs = [j for j in pilot["jobs"] if j.get("active_recipe") and j["stage"] in {"relevance", "extract", "describe"}
            and j["status"] in {"done", "review"} and j.get("input_tokens", 0) > 0]
    residuals, groups, kinds, descriptors = defaultdict(list), {}, {}, []
    for j in jobs:
        stage, model = j["stage"], j["payload"]["model"]
        part = counter.parts(j["payload"])
        residuals[(stage, model)].append(j["input_tokens"] - sum(part.values()))
        if stage == "describe":
            if j["status"] == "done" and j["context"].get("facts"):
                empty = copy.deepcopy(j["context"])
                empty["facts"] = []
                payload = make_task("describe", empty, model=model, max_output_tokens=j["payload"]["max_output_tokens"])
                dependent = sum(part.values()) - sum(counter.parts(payload).values())
                descriptors.append({"model": model, "facts": len(j["context"]["facts"]), "dependent": dependent,
                                    "fixed": j["input_tokens"] - dependent, "output": j["output_tokens"]})
            continue
        doc = j["context"]["document"]
        key = stage, model, doc["id"]
        if key not in groups:
            groups[key] = {"stage": stage, "model": model, "id": doc["id"], "body_tokens": 0,
                           "output_tokens": 0, "eligible_facts": 0, "relevant": False, "review": False,
                           "representative": j["context"]}
        row = groups[key]
        row["body_tokens"] += counter.body(doc)
        row["output_tokens"] += j["output_tokens"]
        row["relevant"] |= stage == "relevance" and j["status"] == "done" and (j.get("result") or {}).get("status") == "relevant"
        row["review"] |= j["status"] == "review" or (j.get("result") or {}).get("status") == "review"
        kind = _kind(doc.get("source_type"))
        if kind != "unknown":
            kinds[doc["id"]] = kind
    ext_jobs = {j["id"]: (j["payload"]["model"], j["context"]["document"]["id"]) for j in jobs if j["stage"] == "extract" and j["status"] == "done"}
    config = pilot["corpus"]["config"]
    eligible_total = 0
    for fact in pilot.get("facts", []):
        key = ext_jobs.get(fact.get("extraction_job_id"))
        if key and _eligible_fact(fact, config):
            groups[("extract", *key)]["eligible_facts"] += 1
            eligible_total += 1
    records = list(groups.values())
    for row in records:
        row["source_type"] = kinds.get(row["id"], "unknown")
        row["bucket"] = _bucket(row["body_tokens"])
    rel = [r for r in records if r["stage"] == "relevance"]
    relevant = sum(r["relevant"] for r in rel)
    rates = {"classification_documents": len(rel), "observed_relevant_documents": relevant,
             "observed_relevant_fraction": relevant / len(rel) if rel else None,
             "relevant_or_review_fraction": sum(r["relevant"] or r["review"] for r in rel) / len(rel) if rel else None,
             "eligible_discovery_facts": eligible_total}
    return records, residuals, descriptors, rates


def _pools(records):
    grouped = defaultdict(list)
    for row in records:
        prefix = row["stage"], row["model"]
        grouped[(*prefix, "kind_length", row["source_type"], row["bucket"])].append(row)
        grouped[(*prefix, "length", row["bucket"])].append(row)
        grouped[(*prefix, "stage")].append(row)
    return grouped


def _choose(pools, stage, model, kind, bucket):
    for key in ((stage, model, "kind_length", kind, bucket), (stage, model, "length", bucket), (stage, model, "stage")):
        values = pools.get(key, [])
        if values and (len(values) >= 5 or key[2] == "stage"):
            return values, key[2]
    return [], "unavailable"


def _task_config(config):
    defaults = {"scope": "", "aspect": "public_positions", "objects": [], "languages": [],
                "date_from": "", "date_to": "", "country_terms": []}
    return {key: config.get(key, default) for key, default in defaults.items()}


def _cost(model, inputs, outputs, pricing):
    if inputs is None or outputs is None:
        return None
    # Fractional counts represent a scenario expectation, not a literal API
    # receipt. estimate_cost validates the explicit rates and integer counts.
    unit_in = estimate_cost(model, 1_000_000, 0, pricing=pricing)
    unit_out = estimate_cost(model, 0, 1_000_000, pricing=pricing)
    return inputs * unit_in / 1_000_000 + outputs * unit_out / 1_000_000


def forecast(plans, pilot_results, config, encoding, progress=None):
    """Aggregate actual planned payloads, without new relevance or scoring.

    plans is {variant:[{document, actor_ids, source_type, token_budget?},...]};
    a list is accepted as one ``planned`` variant. Private metadata is ignored.
    Output quantiles are per-document empirical proxies, not total-cost CIs.
    """
    if encoding is None:
        raise ValueError("An explicit verified local encoding is required")
    if isinstance(plans, list):
        plans = {"planned": plans}
    if not isinstance(plans, dict) or not plans:
        raise ValueError("Nonempty variant mapping required")
    counter = _Counter(encoding)
    records, residuals, descriptors, rates = _pilot(pilot_results, counter)
    pools = _pools(records)
    pricing = config.get("pricing", DEFAULT_PRICING)
    models = {stage: config["routes"][stage] for stage in ("relevance", "extract", "describe")}
    for model in models.values():
        estimate_cost(model, 1, 1, pricing=pricing)
    calibration = {stage: _stats(residuals.get((stage, model), [])) for stage, model in models.items()}
    descriptor_rows = [d for d in descriptors if d["model"] == models["describe"]]
    fact_inputs = sum(d["facts"] for d in descriptor_rows)
    descriptor = {"model": models["describe"], "pilot_calls": len(descriptor_rows), "pilot_fact_occurrences": fact_inputs,
                  "input_data_tokens_per_fact": sum(d["dependent"] for d in descriptor_rows) / fact_inputs if fact_inputs else None,
                  "fixed_input_tokens_per_group": statistics.mean(d["fixed"] for d in descriptor_rows) if descriptor_rows else None,
                  "outputs_per_group": _stats([d["output"] for d in descriptor_rows]), "assumed_group_size": 60}
    summaries, flag_counts = {}, Counter()
    pool_summary = []
    for key, rows in sorted(pools.items()):
        pool_summary.append({"stage": key[0], "model": key[1], "level": key[2],
                             "stratum": list(key[3:]), "documents": len(rows),
                             "output_tokens": _stats([r["output_tokens"] for r in rows]),
                             "eligible_facts": _stats([r["eligible_facts"] for r in rows]) if key[0] == "extract" else None,
                             "body_tokens_min": min(r["body_tokens"] for r in rows), "body_tokens_max": max(r["body_tokens"] for r in rows)})
    for variant, documents in plans.items():
        if not isinstance(variant, str) or not isinstance(documents, list):
            raise ValueError("Variant names and document-plan lists required")
        inputs = {s: {"requests": len(documents), "known_components": {"user": 0, "instructions": 0, "schema": 0}}
                  for s in ("relevance", "extract")}
        outputs = {s: {key: 0. for key in _STATS} for s in inputs}
        lookups, source_kinds, caps = {s: Counter() for s in inputs}, Counter(), Counter()
        eligible_facts_all = 0.
        missing_outputs, missing_yield = set(), False
        task_config = _task_config(config)
        all_actor_ids = {o["id"] for o in task_config["objects"]}
        for index, plan in enumerate(documents, 1):
            document = plan["document"]
            body = counter.body(document)
            kind = _kind(plan.get("source_type", document.get("source_type")))
            source_kinds[kind] += 1
            cap = plan.get("token_budget", document.get("text_selection", {}).get("max_tokens", "unspecified"))
            caps[str(cap)] += 1
            local = plan.get("actor_ids", [])
            if not isinstance(local, list) or not set(local) <= all_actor_ids:
                raise ValueError("Local actor candidates must belong to the approved roster")
            actors = compact_objects(task_config["objects"], document) if config.get("text_selection", {}).get("compact_roster") else task_config["objects"]
            allowed_local = [x for x in local if x in {a["id"] for a in actors}]
            if len(allowed_local) != len(local):
                flag_counts["candidate_outside_retained_roster"] += 1
            if not allowed_local:
                flag_counts["no_local_actor_for_hypothetical_extract"] += 1
            per_task_config = dict(task_config, objects=actors)
            for stage in ("relevance", "extract"):
                selected, level = _choose(pools, stage, models[stage], kind, _bucket(body))
                lookups[stage][level] += 1
                context = {"document": document, "config": per_task_config}
                if stage == "extract":
                    context["document"] = dict(document, **plan.get("source_context", {
                        "source_type": kind, "source_platform": "unknown", "author_is_verified_actor": False}))
                    # Only a token-length template, never a classification of
                    # this document. No exemplar prose is returned or sent.
                    if selected:
                        mean_output = statistics.mean(r["output_tokens"] for r in selected)
                        representative = min(selected, key=lambda r: abs(r["output_tokens"] - mean_output))["representative"]
                        relevance = copy.deepcopy(representative.get("relevance", {}))
                        relevance["actor_ids"] = allowed_local
                        context["relevance"] = relevance
                        if representative.get("relevance_job_id"):
                            context["relevance_job_id"] = representative["relevance_job_id"]
                    else:
                        context["relevance"] = {"status": "relevant", "actor_ids": allowed_local,
                                                "language": "unknown", "reason": "Hypothetical cost template; not a classification."}
                payload = make_task(stage, context, model=models[stage], max_output_tokens=MAX_OUTPUT[stage])
                part = counter.parts(payload)
                for key, value in part.items():
                    inputs[stage]["known_components"][key] += value
                if not selected:
                    missing_outputs.add(stage)
                    if stage == "extract":
                        missing_yield = True
                    continue
                if body > max(r["body_tokens"] for r in selected):
                    flag_counts["planned_body_above_selected_pilot_max_" + stage] += 1
                stat = _stats([r["output_tokens"] for r in selected])
                for key in _STATS:
                    outputs[stage][key] += min(stat[key], MAX_OUTPUT[stage])
                if stage == "extract":
                    eligible_facts_all += statistics.mean(r["eligible_facts"] for r in selected)
            if progress is not None and (index % 1000 == 0 or index == len(documents)):
                progress({"variant": variant, "processed": index, "planned_documents": len(documents)})
        for stage, values in inputs.items():
            known = sum(values["known_components"].values())
            values["known_tokenized_sum"] = known
            values["model"] = models[stage]
            values["calibration_residual"] = calibration[stage]
            values["projected_api_input_tokens"] = {
                key: known + len(documents) * calibration[stage][key] if calibration[stage] is not None else None
                for key in ("mean", "median", "min", "max")}
            if any(v is not None and v < 0 for v in values["projected_api_input_tokens"].values()):
                raise ValueError("Incompatible negative calibrated token projection")
            if stage in missing_outputs:
                outputs[stage] = {key: None for key in _STATS}
        scenarios = []
        fractions = [("assumed_lower_50pct", .5), ("transferred_pilot_observed_positive", rates["observed_relevant_fraction"]), ("assumed_all_pass", 1.)]
        for name, fraction in fractions:
            stages = {}
            for stage in inputs:
                weight = 1. if stage == "relevance" else fraction
                api_input = inputs[stage]["projected_api_input_tokens"]["mean"]
                stage_input = api_input * weight if api_input is not None and weight is not None else None
                stage_outputs = {k: outputs[stage][k] * weight if outputs[stage][k] is not None and weight is not None else None for k in _STATS}
                stages[stage] = {"model": models[stage], "expected_requests": len(documents) * weight if weight is not None else None,
                                 "estimated_input_tokens": stage_input, "estimated_output_tokens": stage_outputs,
                                 "cost_usd": {k: _cost(models[stage], stage_input, stage_outputs[k], pricing) for k in _STATS}}
            expected_facts = eligible_facts_all * fraction if not missing_yield and fraction is not None else None
            groups = math.ceil(expected_facts / 60) if expected_facts is not None else None
            desc_input = (expected_facts * descriptor["input_data_tokens_per_fact"] + groups * descriptor["fixed_input_tokens_per_group"]
                          if expected_facts is not None and descriptor["input_data_tokens_per_fact"] is not None else None)
            desc_outputs = {k: groups * descriptor["outputs_per_group"][k] if groups is not None and descriptor["outputs_per_group"] is not None else None for k in _STATS}
            if groups == 0:
                desc_input, desc_outputs = 0., {k: 0. for k in _STATS}
            desc_costs = {k: _cost(models["describe"], desc_input, desc_outputs[k], pricing) for k in _STATS}
            core = {k: stages["relevance"]["cost_usd"][k] + stages["extract"]["cost_usd"][k]
                    if stages["relevance"]["cost_usd"][k] is not None and stages["extract"]["cost_usd"][k] is not None else None for k in _STATS}
            scenarios.append({"name": name, "pass_fraction": fraction, "stages": stages,
                              "describe": {"expected_eligible_facts": expected_facts, "assumed_groups_of_60": groups,
                                           "estimated_input_tokens": desc_input, "estimated_output_tokens": desc_outputs, "cost_usd": desc_costs},
                              "core_cost_usd": core, "with_describe_cost_usd": {k: core[k] + desc_costs[k] if core[k] is not None and desc_costs[k] is not None else None for k in _STATS}})
        summaries[variant] = {"planned_documents": len(documents), "source_kinds": dict(source_kinds), "token_budget_counts": dict(caps),
                              "known_inputs": inputs, "output_proxy_tokens_if_all_pass": outputs,
                              "output_pool_fallback_counts": {s: dict(v) for s, v in lookups.items()},
                              "pass_scenarios": scenarios}
    return {"schema_version": "empirical-planned-budget-v1", "paid_calls": 0,
            "encoding": getattr(encoding, "name", "provided"), "pilot_pass_calibration": rates,
            "input_residual_calibration": calibration, "output_calibration_pools": pool_summary,
            "describe_calibration": descriptor, "variants": summaries,
            "flags": [{"code": key, "count": value} for key, value in sorted(flag_counts.items())],
            "methodology": {"known_input": "Exact local tokenization of current instructions, serialized USER, canonical output schema; actual provider rendering remains calibrated",
                "input_proxy": "known input sum + per-request mean pilot residual for the SAME stage/model",
                "outputs": "Per-document sum of old chunk outputs; mean/median/p90 proxies stratified by source kind and supplied body-token bucket (pilot: sum of original chunks; plan: selected bounded passages), min5 support then length/stage fallback; capped by the request output limit",
                "p90_warning": "Sum of per-document p90 proxies is not a p90 total or confidence bound",
                "pass_assumption": "50%, old observed-positive doc fraction, and100%; common fraction applied uniformly to planned extraction costs, not actual new classifications",
                "extract_context": "Local candidate IDs plus one representative pilot relevance metadata template, only for token-cost construction; not a model answer",
                "describe": "Conditional eligible fact yield transferred by sourcekind/length; ceil(expectedfacts/60), calibrated fixed/group+data/fact; only two pilot descriptor calls may exist",
                "cache": "Standard uncached rates, no guaranteed future cache saving",
                "representativeness": "Pilot259-record three-day/topic composition is not representative of the full source; sparse strata and longer/adaptive contexts can change outputs and fact yield",
                "scope": "One relevance/extraction/descriptor pass only; final scoring, human audit, synthesis, reruns and retries excluded"}}
