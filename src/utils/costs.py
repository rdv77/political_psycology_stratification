from __future__ import annotations

from typing import Dict, Tuple


# Pricing tables derived from the user's specification.
# OpenAI: USD per 1M tokens (input, cached_input, output)
OPENAI_PRICE_MAP: Dict[str, Tuple[float, float, float]] = {
    # model: (input_per_1k, cached_input_per_1k, output_per_1k)
    "gpt-5": (1.25, 0.125, 10.00),
    "gpt-5-mini": (0.25, 0.025, 2.00),
    "gpt-5-nano": (0.05, 0.005, 0.40),
    "gpt-5-chat-latest": (1.25, 0.125, 10.00),
    "gpt-5-codex": (1.25, 0.125, 10.00),
    "gpt-5-pro": (15.00, 15.00, 120.00),  # no cached listed; use input as fallback
    "gpt-4.1": (2.00, 0.50, 8.00),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "gpt-4.1-nano": (0.10, 0.025, 0.40),
    "gpt-4o": (2.50, 1.25, 10.00),
    "gpt-4o-2024-05-13": (5.00, 5.00, 15.00),  # no cached listed; use input as fallback
    "gpt-4o-mini": (0.15, 0.075, 0.60),
    "gpt-realtime": (4.00, 0.40, 16.00),
    "gpt-realtime-mini": (0.60, 0.06, 2.40),
    "gpt-4o-realtime-preview": (5.00, 2.50, 20.00),
    "gpt-4o-mini-realtime-preview": (0.60, 0.30, 2.40),
    "gpt-audio": (2.50, 2.50, 10.00),  # output given; cached not listed -> input fallback
    "gpt-audio-mini": (0.60, 0.60, 2.40),
    "gpt-4o-audio-preview": (2.50, 2.50, 10.00),
    "gpt-4o-mini-audio-preview": (0.15, 0.15, 0.60),
    "o1": (15.00, 7.50, 60.00),
    "o1-pro": (150.00, 150.00, 600.00),  # cached not listed -> input fallback
    "o3-pro": (20.00, 20.00, 80.00),
    "o3": (2.00, 0.50, 8.00),
    "o3-deep-research": (10.00, 2.50, 40.00),
    "o4-mini": (1.10, 0.275, 4.40),
    "o4-mini-deep-research": (2.00, 0.50, 8.00),
    "o3-mini": (1.10, 0.55, 4.40),
    "o1-mini": (1.10, 0.55, 4.40),
    "codex-mini-latest": (1.50, 0.375, 6.00),
    "gpt-5-search-api": (1.25, 0.125, 10.00),
    "gpt-4o-mini-search-preview": (0.15, 0.15, 0.60),
    "gpt-4o-search-preview": (2.50, 2.50, 10.00),
    "computer-use-preview": (3.00, 3.00, 12.00),
    # Alias seen in config
    "3o": (2.00, 0.50, 8.00),  # map to o3
}


# Perplexity: given in USD per 1M tokens; convert to per 1K inside estimator
PERPLEXITY_PRICE_MAP_PER_1M: Dict[str, Tuple[float, float]] = {
    # model: (input_per_1M, output_per_1M)
    "sonar": (1.0, 1.0),
    "sonar-pro": (3.0, 15.0),
    "sonar-reasoning": (1.0, 5.0),
    "sonar-reasoning-pro": (2.0, 8.0),
    "sonar-deep-research": (2.0, 8.0),  # ignoring citation/search/reasoning addons here
}


def _lookup_openai(model: str) -> Tuple[float, float, float] | None:
    if model in OPENAI_PRICE_MAP:
        return OPENAI_PRICE_MAP[model]
    # Try family fallback by prefix
    for key in OPENAI_PRICE_MAP.keys():
        if model.startswith(key):
            return OPENAI_PRICE_MAP[key]
    return None


def estimate_cost_usd(
    provider: str,
    model: str,
    tokens_in: int,
    tokens_out: int,
    cached_tokens_in: int = 0,
) -> float:
    """Estimate USD cost for a single call.

    - provider: "openai" | "perplexity"
    - model: model identifier as used in config
    - tokens_in/out: non-cached input and output tokens
    - cached_tokens_in: input tokens charged at cached-input rate (OpenAI only)
    """
    if provider == "openai":
        price = _lookup_openai(model)
        if not price:
            return 0.0
        in_cost_1m, cached_in_cost_1m, out_cost_1m = price
        cost = (tokens_in / 1_000_000.0) * in_cost_1m
        cost += (cached_tokens_in / 1_000_000.0) * (cached_in_cost_1m or in_cost_1m)
        cost += (tokens_out / 1_000_000.0) * out_cost_1m
        return float(cost)

    if provider == "perplexity":
        # Normalize model id (lowercase)
        m = model.lower()
        if m not in PERPLEXITY_PRICE_MAP_PER_1M:
            # try to strip extra suffixes
            for key in PERPLEXITY_PRICE_MAP_PER_1M.keys():
                if m.startswith(key):
                    m = key
                    break
        price_1m = PERPLEXITY_PRICE_MAP_PER_1M.get(m)
        if not price_1m:
            return 0.0
        in_per_1m, out_per_1m = price_1m
        return (tokens_in / 1_000_000.0) * in_per_1m + (tokens_out / 1_000_000.0) * out_per_1m

    return 0.0


def clamp_tokens(max_tokens_per_call: int, requested: int) -> int:
    return max(1, min(int(max_tokens_per_call), int(requested)))


