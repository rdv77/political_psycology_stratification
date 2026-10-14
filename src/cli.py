from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Any, Dict

import yaml

from src.clients.openai_client import OpenAIClient
from src.clients.perplexity_client import PerplexityClient
from src.pipeline.objects import collect_countries
from src.pipeline.parties import collect_parties_for_countries
from src.pipeline.parties_map import collect_parties_map
from src.pipeline.user_input import merge_user_countries, merge_user_parties
from src.pipeline.traits import extract_traits_for_parties
from src.pipeline.traits_dedupe import dedupe_traits
from src.pipeline.scoring import score_parties
from src.pipeline.analysis.factor import run_factor_analysis
from src.pipeline.analysis.describe_factors import describe_factors
from src.pipeline.analysis.cluster import cluster_factors
from src.pipeline.analysis.tsne import tsne_plot
from src.pipeline.analysis.scatter_factors import scatter_factor_pairs
from src.pipeline.report import factor_top_parties_report
from src.utils.budget import estimate_project_budget
from src.utils.logging import configure_logging, get_logger
from dotenv import load_dotenv


logger = get_logger("cli")


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Expand env vars in providers keys
    for prov in ("perplexity", "openai"):
        if prov in cfg.get("providers", {}):
            for k, v in list(cfg["providers"][prov].items()):
                if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
                    env_key = v[2:-1]
                    cfg["providers"][prov][k] = os.environ.get(env_key, "")
    return cfg


async def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Parties Pipeline")
    parser.add_argument("run", nargs="?")
    parser.add_argument("--stage", default="all", help="Stage to run: all|collect|traits|dedupe|score|fa|describe|cluster|tsne|report|budget")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    # Load environment from .env if present
    load_dotenv(dotenv_path=".env", override=False)

    config = load_config(args.config)
    configure_logging(config, quiet_flag=args.quiet)
    logger.info("Starting stage=%s dry_run=%s", args.stage, args.dry_run)

    # Budget stage is read-only, does not require approval
    if args.stage != "budget" and config.get("governance", {}).get("require_user_approval", True) and not args.apply:
        logger.warning("Approval required: re-run with --apply to execute")
        if not args.dry_run:
            return

    # Full dry-run: print budget and exit without API calls
    if args.dry_run and args.stage != "budget":
        est = estimate_project_budget(config)
        logger.info("DRY-RUN budget counts: countries=%s parties=%s traits=%s factors=%s",
                    est["counts"]["countries"], est["counts"]["parties"], est["counts"]["traits"], est["counts"]["factors"])
        for s in est["steps"]:
            logger.info(
                "%-24s | %s/%s | calls=%-5d | in=%-5d (total=%-7d) | out=%-5d (total=%-7d) | $%.4f",
                s["name"], s["provider"], s["model"], s["calls"],
                s["tokens_in_per_call"], s["tokens_in_total"],
                s["tokens_out_per_call"], s["tokens_out_total"],
                s["cost_usd"],
            )
        logger.warning(
            "DRY-RUN Estimated TOTAL: $%.4f (limit=$%.2f) | tokens: in=%d out=%d",
            est["total_cost_usd"], float(config["limits"]["max_cost_usd"]),
            est.get("total_tokens_in", 0), est.get("total_tokens_out", 0),
        )
        return

    # Sample mode: reduce limits for fast exploratory runs
    if args.sample:
        config["limits"]["max_countries"] = min(3, int(config["limits"]["max_countries"]))
        config["limits"]["max_parties_per_country"] = min(3, int(config["limits"]["max_parties_per_country"]))
        config["limits"]["max_traits"] = min(40, int(config["limits"]["max_traits"]))
        logger.warning("Sample mode active: limits reduced to countries=%s parties_per_country=%s traits=%s",
                       config["limits"]["max_countries"],
                       config["limits"]["max_parties_per_country"],
                       config["limits"]["max_traits"])

    pplx = PerplexityClient(
        api_key=config["providers"]["perplexity"]["api_key"],
        model=config["providers"]["perplexity"]["model"],
        timeout_s=int(config["resilience"]["request_timeout_s"]),
        cache_dir=config["io"]["cache_dir"],
    )
    openai = OpenAIClient(
        api_key=config["providers"]["openai"]["api_key"],
        default_model=config["providers"]["openai"]["model_scoring"],
        timeout_s=int(config["resilience"]["request_timeout_s"]),
        cache_dir=config["io"]["cache_dir"],
    )

    features = config.get("features", {})

    # Budget estimation (no API calls)
    if args.stage == "budget":
        est = estimate_project_budget(config)
        logger.info("Budget counts: countries=%s parties=%s traits=%s factors=%s",
                    est["counts"]["countries"], est["counts"]["parties"], est["counts"]["traits"], est["counts"]["factors"])
        for s in est["steps"]:
            logger.info(
                "%-24s | %s/%s | calls=%-5d | in=%-5d (total=%-7d) | out=%-5d (total=%-7d) | $%.4f",
                s["name"], s["provider"], s["model"], s["calls"],
                s["tokens_in_per_call"], s["tokens_in_total"],
                s["tokens_out_per_call"], s["tokens_out_total"],
                s["cost_usd"],
            )
        logger.warning(
            "Estimated TOTAL: $%.4f (limit=$%.2f) | tokens: in=%d out=%d",
            est["total_cost_usd"], float(config["limits"]["max_cost_usd"]),
            est.get("total_tokens_in", 0), est.get("total_tokens_out", 0),
        )
        return

    # DISPATCH
    if args.stage in ("all", "collect") and features.get("collect_countries", True):
        countries = await collect_countries(config, pplx, openai)
        countries = merge_user_countries(config, countries)
        if features.get("collect_parties", True):
            await collect_parties_map(config, pplx, openai, countries)
            merge_user_parties(config)

    if args.stage in ("all", "traits") and features.get("extract_traits", True):
        await extract_traits_for_parties(config, openai)

    if args.stage in ("all", "dedupe") and features.get("dedupe_traits", True):
        await dedupe_traits(config, openai)

    if args.stage in ("all", "score") and features.get("scoring", True):
        await score_parties(config, openai)

    if args.stage in ("all", "fa") and features.get("factor_analysis", True):
        run_factor_analysis(config)

    if args.stage in ("all", "describe") and features.get("describe_factors", True):
        await describe_factors(config, openai)

    if args.stage in ("all", "cluster") and features.get("cluster", True):
        cluster_factors(config)

    if args.stage in ("all", "tsne") and features.get("tsne", True):
        tsne_plot(config)
        # Also produce factor-pair scatter plots
        scatter_factor_pairs(config)

    if args.stage in ("all", "report") and features.get("reports", True):
        await factor_top_parties_report(config, pplx)


if __name__ == "__main__":
    asyncio.run(main())


