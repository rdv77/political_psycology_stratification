from __future__ import annotations

import json
import os
from typing import List

import numpy as np
import pandas as pd

from src.clients.openai_client import OpenAIClient
from src.utils.logging import get_logger


logger = get_logger("analysis.describe_factors")


async def describe_factors(config: dict, openai: OpenAIClient, top_k: int = 8) -> None:
    data_dir = config["io"]["data_dir"]
    loadings = np.load(os.path.join(data_dir, "factor_loadings.npy"))  # traits x factors
    traits: List[str] = json.load(open(os.path.join(data_dir, "traits.json"), "r", encoding="utf-8"))
    meta = json.load(open(os.path.join(data_dir, "factors_meta.json"), "r", encoding="utf-8"))
    n_factors = meta["n_factors"]
    explained = meta.get("explained_variance", [])

    tmpl_path = config["prompts"]["factor_describe"]
    tmpl = open(tmpl_path, "r", encoding="utf-8").read()

    reports_dir = config["io"]["reports_dir"]
    os.makedirs(reports_dir, exist_ok=True)
    out_path = os.path.join(reports_dir, "factors.md")

    parts = []
    for i in range(n_factors):
        col = loadings[:, i]
        order_pos = np.argsort(-col)[:top_k]
        order_neg = np.argsort(col)[:top_k]
        top_pos = [traits[idx] for idx in order_pos]
        top_neg = [traits[idx] for idx in order_neg]
        var_share = 0.0
        try:
            var_share = float(explained[i]) if i < len(explained) else 0.0
        except Exception:
            var_share = 0.0
        prompt = (
            tmpl.replace("{top_pos}", ", ".join(top_pos)).replace("{top_neg}", ", ".join(top_neg))
        )
        messages = [{"role": "user", "content": prompt}]
        data = await openai.chat(messages, model=config["providers"]["openai"]["model_describe"])
        desc = data["choices"][0]["message"]["content"]
        parts.append(
            "\n".join(
                [
                    f"## Фактор F{i+1} (доля дисперсии: {var_share:.2%})",
                    f"Топ признаки (положительные нагрузки): {', '.join(top_pos)}",
                    f"Топ признаки (отрицательные нагрузки): {', '.join(top_neg)}",
                    "",
                    desc,
                    "",
                ]
            )
        )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(parts))
    logger.info("Saved factor descriptions to %s", out_path)


