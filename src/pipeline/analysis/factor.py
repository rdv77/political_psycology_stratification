from __future__ import annotations

import json
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer

from src.utils.logging import get_logger


logger = get_logger("analysis.factor")


def _run_fa_once(df: pd.DataFrame, n_factors: int, rotation: str | None) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    fa = FactorAnalyzer(n_factors=n_factors, rotation=None if rotation == "none" else rotation)
    fa.fit(df)
    loadings = fa.loadings_  # shape: traits x factors
    var, prop_var, cum_var = fa.get_factor_variance()  # arrays of length factors
    scores = fa.transform(df)  # factor scores, shape: samples x factors
    scores_df = pd.DataFrame(scores, index=df.index, columns=[f"F{i+1}" for i in range(n_factors)])
    return loadings, prop_var, scores_df


def run_factor_analysis(config: dict) -> None:
    data_dir = config["io"]["data_dir"]
    scores_path = os.path.join(data_dir, "scores.parquet")
    traits_path = os.path.join(data_dir, "traits.json")
    out_path = os.path.join(data_dir, "factors.parquet")

    df = pd.read_parquet(scores_path)
    df = df.set_index("party")
    with open(traits_path, "r", encoding="utf-8") as f:
        traits: List[str] = json.load(f)

    # Keep only traits present in scores; log missing
    present_traits = [t for t in traits if t in df.columns]
    missing_traits = [t for t in traits if t not in df.columns]
    if missing_traits:
        logger.warning("FA: %d traits missing from scores; they will be skipped", len(missing_traits))
    df = df[present_traits]

    # Ensure numeric and handle NaNs: drop all-NaN columns, fill remaining NaNs with column means, drop rows still NaN
    df = df.apply(pd.to_numeric, errors="coerce")
    all_nan_before = int(df.isna().all(axis=0).sum())
    if all_nan_before:
        logger.warning("FA: dropping %d traits with all NaN values", all_nan_before)
    df = df.dropna(axis=1, how="all")
    # Drop zero-variance columns (cannot contribute to FA)
    std = df.std(axis=0, skipna=True)
    zero_var = std[std <= 0].index.tolist()
    if zero_var:
        logger.warning("FA: dropping %d traits with zero variance", len(zero_var))
        df = df.drop(columns=zero_var)
    # Impute remaining NaNs with column means
    col_means = df.mean(axis=0, skipna=True)
    df = df.fillna(col_means)
    # Drop any rows still containing NaN (e.g., if a column was entirely NaN and then dropped)
    rows_before = len(df)
    df = df.dropna(axis=0, how="any")
    dropped_rows = rows_before - len(df)
    if dropped_rows:
        logger.warning("FA: dropped %d parties with incomplete data after imputation", dropped_rows)

    min_last = float(config["analysis"]["factor"]["min_last_factor_variance"])  # e.g., 0.03
    rotation = config["analysis"]["factor"].get("rotation", "varimax")
    max_factors_cfg = int(config["analysis"]["factor"].get("max_factors", 25))
    # Clamp max_factors to available features
    max_factors = max(1, min(max_factors_cfg, max(1, df.shape[1])))

    n = 1
    chosen = 1
    chosen_load = None
    chosen_prop = None
    chosen_scores = None
    while n <= max_factors:
        loadings, prop_var, scores_df = _run_fa_once(df, n, rotation)
        last_var = float(prop_var[-1])
        logger.info("FA factors=%d | last_var=%.4f", n, last_var)
        chosen, chosen_load, chosen_prop, chosen_scores = n, loadings, prop_var, scores_df
        if last_var < min_last:
            break
        n += 1

    # Save factor scores per party
    chosen_scores.to_parquet(out_path, index=True)
    logger.info("Saved factor scores to %s | shape=%s", out_path, chosen_scores.shape)

    # Save loadings and variance separately for later steps
    meta = {
        "traits": traits,
        "explained_variance": chosen_prop.tolist(),
        "n_factors": chosen,
    }
    with open(os.path.join(data_dir, "factors_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    np.save(os.path.join(data_dir, "factor_loadings.npy"), chosen_load)


