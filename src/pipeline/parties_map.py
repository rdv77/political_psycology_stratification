from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List

from src.clients.perplexity_client import PerplexityClient
from src.clients.openai_client import OpenAIClient
from src.utils.json_utils import extract_json_object
from src.utils.logging import get_logger


logger = get_logger("pipeline.parties_map")


def _array_of_strings_schema() -> dict:
    return {
        "type": "json_schema",
        "json_schema": {"schema": {"type": "array", "items": {"type": "string"}}},
    }


def _sanitize_name(name: str) -> str:
    return str(name).strip().strip('"').strip("' ")


async def collect_parties_map(
    config: dict,
    pplx: PerplexityClient,
    oai: OpenAIClient,
    countries: Iterable[str],
) -> Dict[str, str]:
    provider = config.get("collect", {}).get("provider", "openai_search")
    fallback = config.get("collect", {}).get("fallback_provider", "perplexity")

    prompts_path = config["prompts"]["parties"]
    prompt_template = open(prompts_path, "r", encoding="utf-8").read()

    parties_map: Dict[str, str] = {}

    async def _ask_perplexity(country: str) -> List[str] | None:
        prompt = prompt_template.replace("{country}", country)
        messages = [{"role": "user", "content": prompt}]
        text = await pplx.chat(messages, response_format=_array_of_strings_schema())
        obj = extract_json_object(text)
        if isinstance(obj, list):
            return [n for n in (_sanitize_name(x) for x in obj) if n]
        return None

    async def _ask_openai(country: str) -> List[str] | None:
        prompt = prompt_template.replace("{country}", country)
        messages = [{"role": "user", "content": prompt}]
        data = await oai.chat(messages, model=config["providers"]["openai"]["model_search"])
        text = data["choices"][0]["message"]["content"] or ""
        obj = extract_json_object(text)
        if isinstance(obj, list):
            return [n for n in (_sanitize_name(x) for x in obj) if n]
        try:
            arr = json.loads(text)
            if isinstance(arr, list):
                return [n for n in (_sanitize_name(x) for x in arr) if n]
        except Exception:
            return None

    for country in countries:
        names: List[str] | None = None
        for prov in (provider, fallback):
            if prov == "perplexity":
                names = await _ask_perplexity(country)
            else:
                names = await _ask_openai(country)
            if names:
                break
        if not names:
            logger.warning("No parties for %s", country)
            continue
        added = 0
        for name in names:
            if name not in parties_map:
                parties_map[name] = country
                added += 1
        logger.info("Added %d parties for %s (total=%d)", added, country, len(parties_map))

    # Persist
    data_dir = config["io"]["data_dir"]
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, "parties_map.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(parties_map, f, ensure_ascii=False, indent=2)
    logger.info("Saved parties map to %s | size=%d", out_path, len(parties_map))
    return parties_map


