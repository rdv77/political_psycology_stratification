from __future__ import annotations

import ast
import json
import os
from typing import Dict, Iterable, List

from src.clients.openai_client import OpenAIClient
from src.utils.logging import get_logger


logger = get_logger("pipeline.traits")


async def extract_traits_for_parties(config: dict, openai: OpenAIClient) -> Dict[str, List[str]]:
    data_dir = config["io"]["data_dir"]
    parties_path = os.path.join(data_dir, "parties.jsonl")
    parties_map_path = os.path.join(data_dir, "parties_map.json")
    out_path = os.path.join(data_dir, "party_traits.jsonl")

    prompt_path = config["prompts"]["traits"]
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    traits_map: Dict[str, List[str]] = {}
    total = 0
    # Prefer simplified parties_map.json if present
    items: List[tuple[str, str]] = []
    if os.path.exists(parties_map_path):
        try:
            m = json.load(open(parties_map_path, "r", encoding="utf-8"))
            if isinstance(m, dict):
                items = list(m.items())
        except Exception:
            items = []
    if not items:
        with open(parties_path, "r", encoding="utf-8") as f_in:
            for line in f_in:
                party = json.loads(line)
                party_name = party.get("name_original") or party.get("name") or party.get("title")
                country = party.get("country", "")
                items.append((party_name, country))

    with open(out_path, "w", encoding="utf-8") as f_out:
        for party_name, country in items:
            prompt = prompt_template.replace("{party_name}", party_name).replace("{country}", country)
            messages = [{"role": "user", "content": prompt}]
            # Prefer research model if specified; fall back to dedupe model
            model_research = config["providers"]["openai"].get("model_research")
            try:
                if model_research:
                    data = await openai.chat(messages, model=model_research)
                else:
                    data = await openai.chat(messages, model=config["providers"]["openai"]["model_dedupe"])  # small model ok
            except Exception:
                data = await openai.chat(messages, model=config["providers"]["openai"]["model_dedupe"])  # fallback
            text = data["choices"][0]["message"]["content"]
            traits = ast.literal_eval(text)
            traits_map[party_name] = [str(x).strip() for x in traits]
            f_out.write(json.dumps({"party": party_name, "traits": traits_map[party_name]}, ensure_ascii=False) + "\n")
            total += 1
    logger.info("Extracted traits for %d parties", total)
    # Also save as a single mapping for convenience
    map_path = os.path.join(data_dir, "party_traits_map.json")
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(traits_map, f, ensure_ascii=False, indent=2)
    return traits_map


