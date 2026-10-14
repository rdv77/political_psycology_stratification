from __future__ import annotations

import ast
import json
import os
from typing import List

from src.clients.perplexity_client import PerplexityClient
from src.clients.openai_client import OpenAIClient
from src.utils.json_utils import extract_json_object
from src.utils.logging import get_logger


logger = get_logger("pipeline.objects")


async def collect_countries(config: dict, pplx: PerplexityClient, oai: OpenAIClient) -> List[str]:
    prompts_path = config["prompts"]["objects"]
    with open(prompts_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    primary = config.get("collect", {}).get("provider", "perplexity")
    fallback = config.get("collect", {}).get("fallback_provider", "openai_search")

    def _schema_array_of_strings():
        return {"type": "json_schema", "json_schema": {"schema": {"type": "array", "items": {"type": "string"}}}}

    async def _ask_perplexity() -> List[str] | None:
        messages = [{"role": "user", "content": system_prompt + "\nВерни строго JSON-массив строк."}]
        text = await pplx.chat(messages, response_format=_schema_array_of_strings())
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [str(x).strip() for x in data if str(x).strip()]
        except Exception:
            pass
        obj = extract_json_object(text)
        if obj and isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
        return None

    async def _ask_openai() -> List[str] | None:
        prompt = system_prompt + "\nВерни строго JSON-массив строк без пояснений."
        messages = [{"role": "user", "content": prompt}]
        data = await oai.chat(messages, model=config["providers"]["openai"].get("model_search", config["providers"]["openai"]["model_describe"]))
        text = data["choices"][0]["message"]["content"]
        try:
            arr = json.loads(text)
            if isinstance(arr, list):
                return [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            pass
        obj = extract_json_object(text)
        if obj and isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
        return None

    result = None
    for provider in (primary, fallback):
        if provider == "perplexity":
            result = await _ask_perplexity()
        elif provider == "openai_search":
            result = await _ask_openai()
        if result:
            logger.info("Countries collected via %s: %d", provider, len(result))
            break
        else:
            logger.warning("Countries collection failed on provider=%s, trying fallback...", provider)
    if not result:
        raise RuntimeError("Failed to collect countries from all providers")

    max_c = int(config["limits"]["max_countries"])
    result = result[:max_c]
    logger.info("Collected countries: %d (limit=%d)", len(result), max_c)

    data_dir = config["io"]["data_dir"]
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, "countries.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.debug("Saved countries to %s", out_path)
    return result


