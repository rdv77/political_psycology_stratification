from __future__ import annotations

import json
import os
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.logging import get_logger


logger = get_logger("analysis.scatter_factors")


def _available_factor_pairs(factor_cols: List[str]) -> List[Tuple[str, str]]:
    # Desired sequence: (F1,F2), (F2,F3), ..., (F8,F9)
    pairs: List[Tuple[str, str]] = []
    # Extract factor indices as ints from labels like "F3"
    nums = sorted([int(c[1:]) for c in factor_cols if c.startswith("F")])
    for a, b in zip(nums, nums[1:]):
        if a >= 1 and b == a + 1 and a <= 8:
            pa, pb = f"F{a}", f"F{b}"
            if pa in factor_cols and pb in factor_cols:
                pairs.append((pa, pb))
    return pairs


def scatter_factor_pairs(config: dict) -> None:
    data_dir = config["io"]["data_dir"]
    plots_dir = config["io"]["plots_dir"]
    os.makedirs(plots_dir, exist_ok=True)

    # Load factor scores and party-country mapping
    fpath = os.path.join(data_dir, "factors.parquet")
    df = pd.read_parquet(fpath)  # index: party
    factor_cols = [c for c in df.columns if c.startswith("F")]

    # Join country info from parties_map.json if present
    pmap_path = os.path.join(data_dir, "parties_map.json")
    countries_map = {}
    if os.path.exists(pmap_path):
        try:
            countries_map = json.load(open(pmap_path, "r", encoding="utf-8"))
        except Exception:
            countries_map = {}
    df["country"] = [countries_map.get(p, "?") for p in df.index]

    # Injected Russian parties highlighting
    injected = set()
    try:
        injected = set((config.get("user_input", {}) or {}).get("parties", {}).keys())
    except Exception:
        injected = set()
    inj_ru_mask = df.index.isin([p for p in injected if countries_map.get(p, "") == "Россия"])
    inj_all_mask = df.index.isin(injected)

    # Build color map per country
    countries = pd.Categorical(df["country"]).categories.tolist()
    # Use tab20 palette cycling as needed
    palette = plt.cm.get_cmap("tab20", max(1, len(countries)))
    country_to_color = {c: palette(i % palette.N) for i, c in enumerate(countries)}

    for x_col, y_col in _available_factor_pairs(factor_cols):
        plt.figure(figsize=(9, 7))
        # Plot all by country color
        for c in countries:
            mask = df["country"] == c
            if mask.any():
                plt.scatter(df.loc[mask, x_col], df.loc[mask, y_col], s=28, alpha=0.85, color=country_to_color[c], label=c)

        # Overlay injected Russian parties
        if inj_ru_mask.any():
            plt.scatter(
                df.loc[inj_ru_mask, x_col],
                df.loc[inj_ru_mask, y_col],
                s=90,
                marker="D",
                c="#d62728",
                edgecolors="black",
                linewidths=0.6,
                label="Injected RU parties",
                zorder=5,
            )

            # Add labels for all injected parties (independent of country)
            try:
                for name, x, y in zip(df.index[inj_all_mask], df.loc[inj_all_mask, x_col], df.loc[inj_all_mask, y_col]):
                    plt.annotate(
                        str(name),
                        xy=(float(x), float(y)),
                        xytext=(6, 6),
                        textcoords="offset points",
                        fontsize=8,
                        color="#d62728",
                        ha="left",
                        va="bottom",
                        zorder=6,
                    )
            except Exception:
                pass

        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f"Factor scatter: {x_col} vs {y_col}")
        plt.legend(loc="best", fontsize=8, ncol=1)
        plt.tight_layout()
        out_path = os.path.join(plots_dir, f"factors_{x_col}_{y_col}.png")
        plt.savefig(out_path, dpi=150)
        logger.info("Saved factor scatter to %s", out_path)


