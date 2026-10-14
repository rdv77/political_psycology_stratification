from __future__ import annotations

import json
import os
from typing import Dict, Iterable

from src.clients.perplexity_client import PerplexityClient
from src.clients.openai_client import OpenAIClient
from src.io_schemas.models import Party
from src.utils.logging import get_logger
from src.utils.json_utils import extract_json_object


logger = get_logger("pipeline.parties")


async def collect_parties_for_countries(config: dict, pplx: PerplexityClient, oai: OpenAIClient, countries: Iterable[str]) -> None:
    prompts_path = config["prompts"]["parties"]
    with open(prompts_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    data_dir = config["io"]["data_dir"]
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, "parties.jsonl")

    max_per_country = int(config["limits"]["max_parties_per_country"])

    def _party_schema():
        # Prefer a JSON array of party objects with required name_original
        return {
            "type": "json_schema",
            "json_schema": {
                "schema": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 10,
                    "items": {
                        "type": "object",
                        "properties": {
                            "name_original": {"type": "string"},
                            "country": {"type": "string"},
                            "ideology": {"type": "string"},
                            "leaders": {"type": "array", "items": {"type": "string"}},
                            "summary": {"type": "string"},
                        },
                        "required": ["name_original"],
                        "additionalProperties": False,
                    },
                }
            },
        }

    async def _ask_perplexity(country: str) -> Dict[str, Dict] | list | None:
        prompt = prompt_template.replace("{country}", country) + "\nВерни строго JSON-массив объектов, без пояснений."
        messages = [{"role": "user", "content": prompt}]
        text = await pplx.chat(messages, response_format=_party_schema())
        obj = extract_json_object(text)
        return obj if isinstance(obj, (dict, list)) else None

    async def _ask_openai(country: str) -> Dict[str, Dict] | list | None:
        prompt = prompt_template.replace("{country}", country) + "\nВерни строго JSON-массив объектов, без пояснений."
        messages = [{"role": "user", "content": prompt}]
        data = await oai.chat(messages, model=config["providers"]["openai"].get("model_search", config["providers"]["openai"]["model_describe"]))
        text = data["choices"][0]["message"]["content"]
        obj = extract_json_object(text)
        return obj if isinstance(obj, (dict, list)) else None

    primary = config.get("collect", {}).get("provider", "perplexity")
    fallback = config.get("collect", {}).get("fallback_provider", "openai_search")

    def _iter_parties_from_obj(obj: Dict[str, Dict] | list, default_country: str):
        if isinstance(obj, dict):
            reserved = {"country", "ideology", "leaders", "summary"}
            for key, value in obj.items():
                if key in reserved:
                    # skip mis-shaped dicts like {country: ..., ideology: ...}
                    continue
                # key treated as name if value is mapping/string
                if isinstance(value, dict):
                    yield Party(
                        name_original=str(key),
                        country=str(value.get("country", default_country)),
                        ideology=(value.get("ideology") if isinstance(value.get("ideology"), str) else None),
                        leaders=[str(x) for x in (value.get("leaders") or []) if isinstance(x, str)],
                        summary=(value.get("summary") if isinstance(value.get("summary"), str) else None),
                    )
                elif isinstance(value, str):
                    # value is summary, key is the name
                    yield Party(
                        name_original=str(key),
                        country=default_country,
                        ideology=None,
                        leaders=[],
                        summary=value,
                    )
                else:
                    # fallback minimal
                    yield Party(name_original=str(key), country=default_country)
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, str):
                    yield Party(name_original=item, country=default_country)
                elif isinstance(item, dict):
                    # try fields: name/name_original + meta
                    name = item.get("name_original") or item.get("name") or item.get("title")
                    # sometimes models put the name into summary instead of name_original
                    if not isinstance(name, str):
                        if isinstance(item.get("summary"), str) and len(item["summary"]) <= 120:
                            name = item["summary"]
                            item = {**item, "summary": None}
                    if isinstance(name, str):
                        yield Party(
                            name_original=name,
                            country=default_country,
                            ideology=(item.get("ideology") if isinstance(item.get("ideology"), str) else None),
                            leaders=[str(x) for x in (item.get("leaders") or []) if isinstance(x, str)],
                            summary=(item.get("summary") if isinstance(item.get("summary"), str) else None),
                        )
        else:
            return

    with open(out_path, "w", encoding="utf-8") as out_f:
        for country in countries:
            data_any: Dict[str, Dict] | list | None = None
            for provider in (primary, fallback):
                if provider == "perplexity":
                    data_any = await _ask_perplexity(country)
                elif provider == "openai_search":
                    data_any = await _ask_openai(country)
                if data_any is not None:
                    size = len(data_any) if hasattr(data_any, "__len__") else 0
                    logger.info("Parties for %s collected via %s: %d", country, provider, size)
                    break
                else:
                    logger.warning("Failed parties for %s via %s; trying fallback...", country, provider)
            if data_any is None:
                logger.error("Skip country %s: unable to parse valid JSON from providers", country)
                continue

            count = 0
            for party in _iter_parties_from_obj(data_any, country):
                if count >= max_per_country:
                    break
                out_f.write(json.dumps(party.model_dump(), ensure_ascii=False) + "\n")
                count += 1
            logger.info("Collected %d parties for %s", count, country)


