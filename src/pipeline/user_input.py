from __future__ import annotations

from typing import Iterable, List

from src.utils.logging import get_logger
import json
import os


logger = get_logger("pipeline.user_input")


def merge_user_countries(config: dict, countries: Iterable[str]) -> List[str]:
    if not config.get("user_input", {}).get("enable_input_objects", False):
        return list(countries)
    extra = config["user_input"].get("countries", []) or []
    merged = list(dict.fromkeys(list(countries) + [str(x) for x in extra]))
    logger.info("Merged user countries: +%d (total=%d)", len(extra), len(merged))
    return merged


def merge_user_parties(config: dict) -> None:
    """Append user-specified parties to parties.jsonl if missing.
    Expected config.user_input.parties: list of {name_original: str, country?: str}
    """
    ui = config.get("user_input", {})
    if not ui.get("enable_input_objects", False):
        return
    add = ui.get("parties", []) or []
    if not add:
        return
    data_dir = config["io"]["data_dir"]
    # Prefer new simplified map file
    map_path = os.path.join(data_dir, "parties_map.json")
    if os.path.exists(map_path):
        try:
            cur = json.load(open(map_path, "r", encoding="utf-8"))
            if not isinstance(cur, dict):
                cur = {}
        except Exception:
            cur = {}
        before = len(cur)
        if isinstance(add, dict):
            for name, country in add.items():
                name_s = str(name).strip()
                if name_s and name_s not in cur:
                    cur[name_s] = str(country or "")
        else:
            for p in add:
                if isinstance(p, str):
                    name = p.strip()
                    if name and name not in cur:
                        cur[name] = ""
                elif isinstance(p, dict):
                    name = str(p.get("name_original", "")).strip() or str(p.get("name", "")).strip()
                    if name and name not in cur:
                        cur[name] = str(p.get("country", ""))
        json.dump(cur, open(map_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        logger.info("Merged user parties into parties_map.json: +%d", len(cur) - before)
        return

    # Legacy: append to parties.jsonl if map not present
    path = os.path.join(data_dir, "parties.jsonl")
    os.makedirs(data_dir, exist_ok=True)
    existing = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    existing.add(str(obj.get("name_original", "")).strip())
                except Exception:
                    continue
    new = []
    for p in add:
        if isinstance(p, str):
            name = p.strip()
            if name and name not in existing:
                new.append({"name_original": name, "country": ""})
        elif isinstance(p, dict):
            name = str(p.get("name_original", "")).strip()
            if name and name not in existing:
                new.append({"name_original": name, "country": str(p.get("country", ""))})
    if not new:
        return
    with open(path, "a", encoding="utf-8") as f:
        for obj in new:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    logger.info("Merged user parties: +%d", len(new))


