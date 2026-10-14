from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional


def _strip_code_fences(text: str) -> str:
    # Extract content inside the first triple backtick block if present
    fence = re.search(r"```(json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if fence:
        return fence.group(2).strip()
    return text


def _try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
        return None
    except Exception:
        return None


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Try to robustly extract a JSON object from free-form LLM text.

    Strategy:
    1) Strip markdown code fences.
    2) Direct json.loads.
    3) Scan for the largest balanced-brace substring that parses as JSON.
    """
    cleaned = _strip_code_fences(text).strip()

    obj = _try_parse_json(cleaned)
    if obj is not None:
        return obj

    s = cleaned
    opens = [i for i, ch in enumerate(s) if ch == '{']
    for start in opens:
        depth = 0
        for end in range(start, len(s)):
            if s[end] == '{':
                depth += 1
            elif s[end] == '}':
                depth -= 1
                if depth == 0:
                    cand = s[start : end + 1]
                    obj = _try_parse_json(cand)
                    if obj is not None:
                        return obj
    return None


