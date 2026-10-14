from __future__ import annotations

import hashlib
import json
from typing import Any, Dict

import httpx

from src.utils.cache import FileCache
from src.utils.logging import get_logger


OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"


class OpenAIClient:
    def __init__(self, api_key: str, default_model: str, timeout_s: int = 60, cache_dir: str = "cache") -> None:
        self.api_key = api_key
        self.default_model = default_model
        self.timeout_s = timeout_s
        self.logger = get_logger("openai")
        self.cache = FileCache(cache_dir)

    def _cache_key(self, messages: Any, model: str, **kwargs: Any) -> str:
        raw = json.dumps({"messages": messages, "model": model, **kwargs}, ensure_ascii=False, sort_keys=True)
        return f"oai:{hashlib.sha256(raw.encode('utf-8')).hexdigest()}"

    async def chat(self, messages: list[dict], model: str | None = None, temperature: float = 0.2, max_tokens: int = 800) -> dict:
        use_model = model or self.default_model
        key = self._cache_key(messages, model=use_model, temperature=temperature, max_tokens=max_tokens)
        cached = self.cache.get(key)
        if cached:
            self.logger.debug("OpenAI cache hit")
            return cached

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": use_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        self.logger.debug("OpenAI request | model=%s | max_tokens=%s", use_model, max_tokens)
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            resp = await client.post(OPENAI_API_URL, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            self.cache.set(key, data)
            return data


