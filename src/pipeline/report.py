from __future__ import annotations

import json
import os
from typing import Dict, List

import pandas as pd

from src.clients.perplexity_client import PerplexityClient
from src.utils.logging import get_logger


logger = get_logger("pipeline.report")


async def factor_top_parties_report(config: dict, pplx: PerplexityClient, top_n: int = 10) -> None:
    """Cluster report: sizes, percentages, nearest parties, and web-backed description.

    Reuses function name for CLI compatibility.
    """
    data_dir = config["io"]["data_dir"]
    clusters_path = os.path.join(data_dir, "clusters.parquet")
    parties_map_path = os.path.join(data_dir, "parties_map.json")

    df = pd.read_parquet(clusters_path)  # index: party, includes cluster_id and dist_to_center
    total = float(len(df)) if len(df) > 0 else 1.0

    # Party -> country map for context
    pmap: Dict[str, str] = {}
    if os.path.exists(parties_map_path):
        try:
            pmap = json.load(open(parties_map_path, "r", encoding="utf-8"))
        except Exception:
            pmap = {}

    reports_dir = config["io"]["reports_dir"]
    os.makedirs(reports_dir, exist_ok=True)
    out_path = os.path.join(reports_dir, "clusters.md")

    parts: List[str] = []
    for cid, group in df.groupby("cluster_id"):
        n = int(len(group))
        pct = (n / total) if total else 0.0
        nearest = group.sort_values("dist_to_center").head(top_n)
        items = [f"- {idx} ({pmap.get(idx, '?')})" for idx in nearest.index]

        header = f"## Кластер {cid} — {n} партий ({pct:.1%})\n\n" + "\n".join(items) + "\n\n"
        prompt = "\n".join([
            "Ты исследователь партийных систем с веб-доступом.",
            f"Сформулируй характеристику кластера на основе ближайших к центроиду партий.",
            f"Кластер {cid}:",
            *items,
            "Опиши, что их объединяет, с акцентом на общие темы и идеологию. Приведи ссылки (URL).",
        ])
        messages = [{"role": "user", "content": prompt}]
        desc = await pplx.chat(messages)
        parts.append(header + desc + "\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(parts))
    logger.info("Saved clusters report to %s", out_path)


