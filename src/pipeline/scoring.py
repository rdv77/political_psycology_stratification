from __future__ import annotations

import asyncio
import json
import math
import os
from typing import Dict, List, Optional

from src.clients.openai_client import OpenAIClient
from src.utils.costs import estimate_cost_usd, clamp_tokens
from src.utils.logging import get_logger
from src.utils.concurrency import bounded_gather
from src.utils.json_utils import extract_json_object


logger = get_logger("pipeline.scoring")


def _estimate_budget(config: dict, parties_count: int, traits_count: int) -> float:
    model = config["providers"]["openai"]["model_scoring"]
    chunk = int(config.get("scoring", {}).get("max_traits_per_call", 10))
    num_chunks = max(1, math.ceil(traits_count / max(1, chunk)))
    # Per-call token estimates scale with chunk size
    approx_in_per_call = min(2000, 500 + min(traits_count, chunk) * 6)
    approx_out_per_call = min(2000, min(traits_count, chunk) * 5)
    per_call_cost = estimate_cost_usd("openai", model, approx_in_per_call, approx_out_per_call)
    total_calls = parties_count * num_chunks
    return per_call_cost * total_calls


async def score_parties(config: dict, openai: OpenAIClient) -> None:
    data_dir = config["io"]["data_dir"]
    parties_path = os.path.join(data_dir, "parties.jsonl")
    parties_map_path = os.path.join(data_dir, "parties_map.json")
    traits_path = os.path.join(data_dir, "traits.json")
    out_path = os.path.join(data_dir, "scores.parquet")

    with open(traits_path, "r", encoding="utf-8") as f:
        traits: List[str] = json.load(f)

    # Build iterable of (party_name, country)
    items: List[tuple[str, str]] = []
    if os.path.exists(parties_map_path):
        try:
            pmap = json.load(open(parties_map_path, "r", encoding="utf-8"))
            if isinstance(pmap, dict):
                items = [(str(k), str(v or "")) for k, v in pmap.items()]
        except Exception:
            items = []
    if not items:
        # fallback legacy path
        parties: List[Dict] = [json.loads(line) for line in open(parties_path, "r", encoding="utf-8")]
        for party in parties:
            name = party.get("name_original") or party.get("name") or party.get("title")
            country = party.get("country", "")
            if name:
                items.append((str(name), str(country)))
    parties_count = len(items)
    traits_count = len(traits)

    est_cost = _estimate_budget(config, parties_count, traits_count)
    logger.info("Estimated scoring cost: ~$%.2f (parties=%d, traits=%d)", est_cost, parties_count, traits_count)

    if est_cost > float(config["limits"]["max_cost_usd"]):
        logger.error("Budget exceeded: need ~$%.2f > max_cost_usd=%.2f", est_cost, float(config["limits"]["max_cost_usd"]))
        raise RuntimeError("Budget gate: increase max_cost_usd or reduce dataset")

    prompt_template = open(config["prompts"]["scoring"], "r", encoding="utf-8").read()

    import pandas as pd

    max_per_call = int(config.get("scoring", {}).get("max_traits_per_call", 10))
    max_concurrent = int(config.get("limits", {}).get("max_concurrent_requests", 4))

    async def _score_one_party(party_name: str, country: str, ordinal: int) -> Dict[str, Optional[float]]:
        merged_scores: Dict[str, Optional[float]] = {}
        for start_idx in range(0, len(traits), max_per_call):
            chunk_traits = traits[start_idx : start_idx + max_per_call]
            prompt = (
                prompt_template
                .replace("{party_name}", party_name)
                .replace("{country}", country)
                .replace("{traits_json}", json.dumps(chunk_traits, ensure_ascii=False))
            )
            max_tokens = clamp_tokens(int(config["limits"]["max_tokens_per_call"]), 1500)
            messages = [{"role": "user", "content": prompt}]
            try:
                data = await openai.chat(messages, model=config["providers"]["openai"]["model_scoring"], max_tokens=max_tokens)
                content = data["choices"][0]["message"]["content"]
            except Exception as e:
                logger.warning("LLM call failed for %s chunk %d-%d: %s", party_name, start_idx, start_idx+len(chunk_traits), e)
                content = "{}"

            # Robust JSON extraction: try dict, else try list aligned to chunk order
            scores_chunk: Dict[str, Optional[float]] = {}
            obj = extract_json_object(content)
            if obj is None:
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, list) and len(parsed) == len(chunk_traits):
                        for t, v in zip(chunk_traits, parsed):
                            scores_chunk[t] = float(v) if isinstance(v, (int, float)) else None
                    elif isinstance(parsed, dict):
                        obj = parsed  # non-fenced dict
                except Exception:
                    pass

            if obj is not None:
                for t in chunk_traits:
                    val = obj.get(t, None)
                    if isinstance(val, (int, float)):
                        scores_chunk[t] = float(val)
                    else:
                        try:
                            scores_chunk[t] = float(val) if val is not None else None
                        except Exception:
                            scores_chunk[t] = None

            # Merge clamped values, keep None for missing
            for t in chunk_traits:
                val = scores_chunk.get(t, None)
                if isinstance(val, (int, float)):
                    merged_scores[t] = max(0.0, min(10.0, float(val)))
                else:
                    merged_scores.setdefault(t, None)

        logger.info(
            "Scored %s (%d/%d) traits=%d via chunks=%d",
            party_name,
            ordinal,
            parties_count,
            len(traits),
            math.ceil(len(traits) / max_per_call),
        )
        return {"party": party_name, **merged_scores}

    # Run parties in parallel with bounded concurrency
    tasks = [
        _score_one_party(party_name, country, idx)
        for idx, (party_name, country) in enumerate(items, start=1)
    ]
    rows = await bounded_gather(tasks, max_concurrent=max_concurrent)

    df = pd.DataFrame(rows)
    os.makedirs(data_dir, exist_ok=True)
    try:
        df.to_parquet(out_path, index=False)
    except ImportError as e:
        # Provide a friendly fallback and guidance if parquet engine is missing
        csv_path = os.path.join(data_dir, "scores.csv")
        logger.warning(
            "Parquet engine not installed (%s). Saved CSV fallback to %s. Install 'pyarrow' for parquet support.",
            e,
            csv_path,
        )
        df.to_csv(csv_path, index=False, encoding="utf-8")
        return
    logger.info("Saved scores to %s | shape=%s", out_path, df.shape)


