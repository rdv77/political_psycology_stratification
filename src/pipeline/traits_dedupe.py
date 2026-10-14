from __future__ import annotations

import json
import os
from typing import Dict, List

from src.clients.openai_client import OpenAIClient
from src.utils.logging import get_logger
from src.utils.json_utils import extract_json_object
from src.utils.text import canonicalize_trait


logger = get_logger("pipeline.traits_dedupe")


async def dedupe_traits(config: dict, openai: OpenAIClient) -> List[str]:
    data_dir = config["io"]["data_dir"]
    in_path = os.path.join(data_dir, "party_traits.jsonl")
    out_path = os.path.join(data_dir, "traits.json")

    traits: List[str] = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            traits.extend(row.get("traits", []))

    traits_blob = "\n".join(f"- {t}" for t in traits)
    prompt_path = config["prompts"]["dedupe"]
    with open(prompt_path, "r", encoding="utf-8") as f:
        tmpl = f.read()
    max_traits = int(config["limits"]["max_traits"])
    prompt = tmpl.replace("{max_traits}", str(max_traits)).replace("{traits_blob}", traits_blob)

    messages = [{"role": "user", "content": prompt}]
    data = await openai.chat(messages, model=config["providers"]["openai"]["model_dedupe"])
    text = data["choices"][0]["message"]["content"] or ""
    obj = extract_json_object(text)
    if isinstance(obj, list):
        initial = [str(x).strip() for x in obj if str(x).strip()]
    else:
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                initial = [str(x).strip() for x in parsed if str(x).strip()]
            else:
                initial = []
        except Exception:
            logger.warning("Dedupe traits: failed to parse JSON, falling back to heuristic unique list")
            seen = set()
            initial = []
            for t in traits:
                k = str(t).strip().casefold()
                if k and k not in seen:
                    seen.add(k)
                    initial.append(str(t).strip())

    # Post-process: canonicalization + fuzzy merge by Jaccard of stem sets
    sim_threshold = float((config.get("dedupe", {}) or {}).get("similarity_threshold", 0.7))
    def _jaccard(a: str, b: str) -> float:
        sa = set(canonicalize_trait(a).split())
        sb = set(canonicalize_trait(b).split())
        if not sa or not sb:
            return 0.0
        inter = len(sa & sb)
        union = len(sa | sb)
        return inter / union if union else 0.0

    merged: List[str] = []
    canon_seen: set = set()
    for t in initial:
        c = canonicalize_trait(t)
        if not c:
            continue
        # Direct canonical duplicate
        if c in canon_seen:
            continue
        # Near-duplicate check against kept items
        is_dup = False
        for kept in merged:
            sim = _jaccard(t, kept)
            if sim >= sim_threshold:  # collapse very similar forms
                is_dup = True
                break
        if not is_dup:
            canon_seen.add(c)
            merged.append(t)

    uniq = merged[:max_traits]

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(uniq, f, ensure_ascii=False, indent=2)
    logger.info("Dedupe traits: %d -> %d (post-processed)", len(traits), len(uniq))
    return uniq


