from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional

import httpx

from src.utils.cache import FileCache
from src.utils.logging import get_logger


PPLX_API_URL = "https://api.perplexity.ai/chat/completions"


class PerplexityClient:
    def __init__(self, api_key: str, model: str, timeout_s: int = 60, cache_dir: str = "cache") -> None:
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        self.logger = get_logger("perplexity")
        self.cache = FileCache(cache_dir)

    def _cache_key(self, messages: Any, **kwargs: Any) -> str:
        raw = json.dumps({"messages": messages, "model": self.model, **kwargs}, ensure_ascii=False, sort_keys=True)
        return f"pplx:{hashlib.sha256(raw.encode('utf-8')).hexdigest()}"

    async def chat(self, messages: list[dict], temperature: float = 0.2, max_tokens: int = 800, response_format: dict | None = None) -> str:
        key = self._cache_key(messages, temperature=temperature, max_tokens=max_tokens, response_format=response_format)
        cached = self.cache.get(key)
        if cached:
            self.logger.debug("Perplexity cache hit")
            return cached["text"]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        self.logger.debug("Perplexity request | model=%s | max_tokens=%s", self.model, max_tokens)
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            resp = await client.post(PPLX_API_URL, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            self.cache.set(key, {"text": text, "raw": data})
            return text


