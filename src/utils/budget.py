from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

from src.utils.costs import estimate_cost_usd
from src.utils.logging import get_logger


logger = get_logger("utils.budget")


def _get(cfg: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _count_lines(path: str) -> int:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def _load_json(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _default_assumptions() -> Dict[str, Any]:
    return {
        "objects": {"in": 500, "out": 600},
        "parties": {"in": 1200, "out": 2000},
        "traits_extract": {"in": 900, "out": 400},
        "traits_dedupe": {"in": 2000, "out": 600},
        # scoring uses dynamic formula below
        "factor_describe": {"in": 400, "out": 350},
        "factor_groups_report": {"in": 1200, "out": 1200},
        "cluster_thematic_report": {"in": 3000, "out": 2000},
        "estimate_n_factors": 6,
    }


def estimate_project_budget(config: Dict[str, Any]) -> Dict[str, Any]:
    data_dir = _get(config, "io.data_dir", "data")

    # Resolve counts from existing artifacts or fallbacks to limits
    countries_file = os.path.join(data_dir, "countries.json")
    countries = _load_json(countries_file)
    if isinstance(countries, list) and countries:
        countries_count = len(countries)
    else:
        user_c = _get(config, "user_input.countries", []) or []
        max_c = int(_get(config, "limits.max_countries", 20))
        if user_c:
            countries_count = min(len(user_c), max_c)
        else:
            countries_count = max_c

    parties_file = os.path.join(data_dir, "parties.jsonl")
    parties_count = _count_lines(parties_file)
    if parties_count <= 0:
        max_ppc = int(_get(config, "limits.max_parties_per_country", 8))
        parties_count = countries_count * max_ppc

    traits_file = os.path.join(data_dir, "traits.json")
    traits = _load_json(traits_file)
    if isinstance(traits, list) and traits:
        traits_count = len(traits)
    else:
        traits_count = int(_get(config, "limits.max_traits", 120))

    factors_meta_file = os.path.join(data_dir, "factors_meta.json")
    meta = _load_json(factors_meta_file)
    if isinstance(meta, dict) and "n_factors" in meta:
        n_factors = int(meta["n_factors"])
    else:
        n_factors = int(_get(config, "budget.assumptions.estimate_n_factors", _default_assumptions()["estimate_n_factors"]))

    # Assumptions (overridable via config.budget.assumptions)
    asm = _default_assumptions()
    cfg_asm = _get(config, "budget.assumptions", {}) or {}
    for k, v in cfg_asm.items():
        if isinstance(v, dict) and k in asm:
            asm[k].update(v)
        else:
            asm[k] = v

    # Models
    pplx_model = _get(config, "providers.perplexity.model", "sonar-pro")
    oai_score_model = _get(config, "providers.openai.model_scoring", "gpt-4.1")
    oai_dedupe_model = _get(config, "providers.openai.model_dedupe", "gpt-4.1")
    oai_desc_model = _get(config, "providers.openai.model_describe", "o3")
    oai_search_model = _get(config, "providers.openai.model_search", "gpt-5-search-api")

    steps: List[Dict[str, Any]] = []

    # 1) Objects (countries) with selectable provider
    obj_in, obj_out = int(asm["objects"]["in"]), int(asm["objects"]["out"]) 
    collect_provider = _get(config, "collect.provider", "perplexity")
    if collect_provider == "openai_search":
        steps.append(_step_cost("objects", "openai", oai_search_model, calls=1, in_per_call=obj_in, out_per_call=obj_out))
    else:
        steps.append(_step_cost("objects", "perplexity", pplx_model, calls=1, in_per_call=obj_in, out_per_call=obj_out))

    # 2) Parties per country (selectable provider)
    prt_in, prt_out = int(asm["parties"]["in"]), int(asm["parties"]["out"]) 
    if collect_provider == "openai_search":
        steps.append(_step_cost("parties", "openai", oai_search_model, calls=countries_count, in_per_call=prt_in, out_per_call=prt_out))
    else:
        steps.append(_step_cost("parties", "perplexity", pplx_model, calls=countries_count, in_per_call=prt_in, out_per_call=prt_out))

    # 4) Traits per party (OpenAI, use dedupe model for traits extraction prompts)
    tr_in, tr_out = int(asm["traits_extract"]["in"]), int(asm["traits_extract"]["out"]) 
    steps.append(_step_cost("traits_extract", "openai", oai_dedupe_model, calls=parties_count, in_per_call=tr_in, out_per_call=tr_out))

    # 5) Traits dedupe (OpenAI)
    td_in, td_out = int(asm["traits_dedupe"]["in"]), int(asm["traits_dedupe"]["out"]) 
    steps.append(_step_cost("traits_dedupe", "openai", oai_dedupe_model, calls=1, in_per_call=td_in, out_per_call=td_out))

    # 6) Scoring per party with chunking
    chunk = int(_get(config, "scoring.max_traits_per_call", 10))
    num_chunks = max(1, (traits_count + max(1, chunk) - 1) // max(1, chunk))
    sc_in = min(2000, 500 + min(traits_count, chunk) * 6)
    sc_out = min(2000, min(traits_count, chunk) * 5)
    steps.append(_step_cost("scoring", "openai", oai_score_model, calls=parties_count * num_chunks, in_per_call=sc_in, out_per_call=sc_out))

    # 9) Factor describe per factor (OpenAI)
    fd_in, fd_out = int(asm["factor_describe"]["in"]), int(asm["factor_describe"]["out"]) 
    steps.append(_step_cost("factor_describe", "openai", oai_desc_model, calls=n_factors, in_per_call=fd_in, out_per_call=fd_out))

    # 10) Factor groups report per factor (Perplexity)
    fr_in, fr_out = int(asm["factor_groups_report"]["in"]), int(asm["factor_groups_report"]["out"]) 
    steps.append(_step_cost("factor_groups_report", "perplexity", pplx_model, calls=n_factors, in_per_call=fr_in, out_per_call=fr_out))

    # 13) Cluster thematic report (Perplexity)
    ct_in, ct_out = int(asm["cluster_thematic_report"]["in"]), int(asm["cluster_thematic_report"]["out"]) 
    steps.append(_step_cost("cluster_thematic_report", "perplexity", pplx_model, calls=1, in_per_call=ct_in, out_per_call=ct_out))

    total = sum(s["cost_usd"] for s in steps)
    total_tokens_in = sum(s["tokens_in_total"] for s in steps)
    total_tokens_out = sum(s["tokens_out_total"] for s in steps)
    return {
        "counts": {
            "countries": countries_count,
            "parties": parties_count,
            "traits": traits_count,
            "factors": n_factors,
        },
        "steps": steps,
        "total_cost_usd": round(total, 4),
        "total_tokens_in": int(total_tokens_in),
        "total_tokens_out": int(total_tokens_out),
    }


def _step_cost(name: str, provider: str, model: str, calls: int, in_per_call: int, out_per_call: int) -> Dict[str, Any]:
    cost_one = estimate_cost_usd(provider, model, in_per_call, out_per_call)
    subtotal = cost_one * calls
    return {
        "name": name,
        "provider": provider,
        "model": model,
        "calls": int(calls),
        "tokens_in_per_call": int(in_per_call),
        "tokens_out_per_call": int(out_per_call),
        "tokens_in_total": int(in_per_call) * int(calls),
        "tokens_out_total": int(out_per_call) * int(calls),
        "cost_usd": round(subtotal, 4),
    }


