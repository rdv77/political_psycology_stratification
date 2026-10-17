"""Reproducible small-budget pilot; raw inputs/results stay in ignored runtime.

All API phases require --execute. The corpus-wide cap includes earlier runs and
uncertain reservations. No study, scale approval or research result is changed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.corpus_pipeline.service import CorpusPipeline, candidates, digest
from src.corpus_pipeline.models import DEFAULT_PRICING

SEED = 20261017
CHALLENGE_ROWS = [105, 111, 220, 222, 200, 199, 206, 147]


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def stratum(document):
    hint = document.get("language_hint", "").casefold()
    language = "fr" if hint in {"fr", "french", "французский"} else "en" if hint in {"en", "english", "английский"} else "other"
    length = len(document["text"])
    return language + "/" + ("short" if length <= 1000 else "medium" if length <= 5000 else "long")


def stratified(documents, count, seed=SEED):
    groups = defaultdict(list)
    for document in documents:
        groups[stratum(document)].append(document)
    rng = random.Random(seed)
    for key in sorted(groups):
        groups[key].sort(key=lambda x: x["id"])
        rng.shuffle(groups[key])
    target = min(count, len(documents))
    quota = {key: min(3, len(group)) for key, group in groups.items()}
    if sum(quota.values()) > target:
        quota = {key: 0 for key in groups}
    while sum(quota.values()) < target:
        eligible = [key for key in sorted(groups) if quota[key] < len(groups[key])]
        key = max(eligible, key=lambda key: len(groups[key]) / (quota[key] + 1))
        quota[key] += 1
    selected = [document for key in sorted(groups) for document in groups[key][:quota[key]]]
    return selected, {key: {"population": len(groups[key]), "selected": quota[key]} for key in sorted(groups)}


def setup(pipeline, source, sample_size, budget):
    study_path = ROOT / "studies/rwanda-politicians-2021-2026/study.json"
    study = read(study_path)
    dictionary = read(ROOT / "research/rwanda_politicians_2021_2026/corpus_search/actor_aliases.json")
    extras = read(ROOT / "research/rwanda_politicians_2021_2026/corpus_search/verified_language_aliases.json")
    by_id = {obj["id"]: obj for obj in dictionary["actors"]}
    objects = []
    for obj in study["objects"]:
        actor = by_id[obj["id"]]
        aliases = [x["text"] for x in actor["aliases"] if not re.search(r"[\u0400-\u052f]", x["text"])]
        aliases += [x["text"] for x in extras["aliases"] if x["actor_id"] == obj["id"]]
        objects.append({"id": obj["id"], "name": actor["canonical_latin"], "aliases": sorted(set(aliases))})
    config = {"scope": "Публичные политические позиции и действия утверждённых 42 политиков Руанды. Технический пилот на отдельной выгрузке; различай собственные высказывания, действия, характеристики другими и институциональный контекст.",
              "aspect": "public_positions", "objects": objects, "languages": ["rw", "en", "fr", "sw"],
              "date_from": study["date_from"], "date_to": "2026-10-17",
              "country_terms": ["Rwanda", "#Rwanda", "u Rwanda", "uRwanda"], "budget_limit_usd": budget}
    corpus = pipeline.import_csv(source, title="Руанда — технический пилот 15–17 октября 2026", config=config)
    identifier = corpus["id"]
    directory = pipeline.runtime / "pilots" / identifier
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = read(manifest_path)
        saved = manifest["budget_cap_usd"]
        if abs(saved - budget) > 1e-12 or corpus["config"].get("budget_limit_usd") != saved:
            raise ValueError("Лимит сохранённого пилота не совпадает с запрошенным: "
                             f"manifest={saved}, config={corpus['config'].get('budget_limit_usd')}, requested={budget}. "
                             "Запуск отменён без API; бюджет нельзя молча переопределить.")
        return manifest
    with pipeline.connection() as db:
        documents = [json.loads(row[0]) for row in db.execute("SELECT data FROM documents WHERE corpus_id=? AND status='unique' ORDER BY source_row", (identifier,))]
    eligible = [doc for doc in documents if pipeline._rule_exclusion(doc, corpus["config"]) is None]
    selected, strata = stratified(eligible, sample_size)
    challenge = [doc for doc in eligible if doc["source_row"] in CHALLENGE_ROWS and doc["id"] not in {x["id"] for x in selected}]
    manifest = {"corpus_id": identifier, "seed": SEED, "budget_cap_usd": budget,
                "source_sha256": corpus["source_sha256"], "parent_study_sha256": digest(study),
                "canonical_research_window": [study["date_from"], study["date_to"]],
                "technical_pilot_window": [config["date_from"], config["date_to"]],
                "population_documents": len(documents), "eligible_documents": len(eligible),
                "excluded_by_rules": len(documents) - len(eligible), "strata": strata,
                "random_document_ids": [x["id"] for x in selected],
                "challenge_document_ids": [x["id"] for x in challenge],
                "selection": [{"id": doc["id"], "source_row": doc["source_row"], "stratum": stratum(doc),
                               "chars": len(doc["text"]), "url": doc["url"], "source_name": doc["source_name"],
                               "kind": "random" if doc["id"] in {x["id"] for x in selected} else "challenge"}
                              for doc in selected + challenge],
                "notes": ["Стратификация использует метаданные языка, которые проверяются отдельно.",
                          "Целевые сложные случаи не участвуют в основном взвешенном прогнозе.",
                          "Пилот не изменяет шкалы, оценки, окно или отчёты прежнего исследования."]}
    write(manifest_path, manifest)
    return manifest


def private_results(pipeline, manifest):
    corpus = pipeline.get_corpus(manifest["corpus_id"])
    with pipeline.connection() as db:
        rows = db.execute("SELECT * FROM jobs WHERE corpus_id=? ORDER BY created_at,id", (corpus["id"],)).fetchall()
    jobs = []
    for row in rows:
        value = dict(row)
        value["active_recipe"] = value["recipe"] == pipeline._recipe(value["stage"], corpus["config"])
        for key in ("context", "payload", "result"):
            value[key] = json.loads(value[key]) if value[key] else None
        jobs.append(value)
    output = {"manifest": manifest, "corpus": corpus, "jobs": jobs,
              "facts": pipeline._facts(corpus["id"], corpus["config"])}
    write(pipeline.runtime / "pilots" / corpus["id"] / "results_private.json", output)
    return output


def relevant_documents(results):
    ids = set()
    for job in results["jobs"]:
        if job["stage"] == "relevance" and job.get("active_recipe") and job["status"] == "done" and job["result"]["status"] == "relevant":
            ids.add(job["context"]["document"]["id"])
    return ids


def phase(pipeline, manifest, stage, extract_size, execute, maximum, selection="full"):
    identifier = manifest["corpus_id"]
    results = private_results(pipeline, manifest)
    ids = manifest["random_document_ids"] + manifest["challenge_document_ids"]
    if selection == "audit":
        if stage != "relevance":
            raise ValueError("Контрольная сравнительная выборка применяется только к relevance")
        review = read(pipeline.runtime / "pilots" / identifier / "independent_review.json")
        ids = [item["document_id"] for item in review]
    if stage == "extract":
        if results["corpus"]["jobs"].get("pending"):
            raise ValueError("Сначала завершите подготовленную очередь relevance")
        manifest_path = pipeline.runtime / "pilots" / identifier / "manifest.json"
        if "extraction_document_ids" not in manifest:
            relevant = relevant_documents(results)
            with pipeline.connection() as db:
                documents = [json.loads(row[0]) for row in db.execute("SELECT data FROM documents WHERE corpus_id=? AND status='unique'", (identifier,))]
            core = [doc for doc in documents if doc["id"] in relevant and doc["id"] in manifest["random_document_ids"]]
            selected, strata = stratified(core, extract_size, SEED + 1)
            challenge = [doc for doc in documents if doc["id"] in relevant and doc["source_row"] in {105, 111, 220, 222, 200}]
            ids = list(dict.fromkeys([x["id"] for x in selected + challenge]))
            manifest.update(extraction_document_ids=ids, extraction_random_ids=[x["id"] for x in selected], extraction_strata=strata)
            write(manifest_path, manifest)
        ids = manifest["extraction_document_ids"]
    kwargs = {"document_ids": ids} if stage in {"relevance", "extract"} else {}
    if stage == "describe" and manifest.get("discovery_fact_ids"):
        kwargs["fact_ids"] = manifest["discovery_fact_ids"]
    prepared = pipeline.prepare(identifier, stage=stage, limit=5000, **kwargs)
    print(json.dumps({"phase": stage, "prepared": prepared}, ensure_ascii=False), flush=True)
    if not execute:
        return private_results(pipeline, manifest)
    attempts = 0
    while attempts < maximum:
        current = pipeline.get_corpus(identifier)
        usage = current["usage"]
        remaining = manifest["budget_cap_usd"] - usage["estimated_usd"] - usage["uncertain_reserved_usd"]
        if remaining <= 0 or not current["jobs"].get("pending"):
            break
        run = pipeline.run(identifier, max_jobs=min(5, maximum - attempts), budget_usd=remaining)
        attempts += run["attempted"]
        write(pipeline.runtime / "pilots" / identifier / (run["run_id"] + ".json"), run)
        print(json.dumps({"phase": stage, "run": run, "usage": pipeline.get_corpus(identifier)["usage"]}, ensure_ascii=False), flush=True)
        private_results(pipeline, manifest)
        if run["failed"] or run["uncertain"] or run["stopped_reason"] not in {"job_limit", "queue_empty"}:
            break
    return private_results(pipeline, manifest)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--phase", choices=["setup", "relevance", "extract", "describe", "summary"], default="setup")
    parser.add_argument("--sample-size", type=int, default=48)
    parser.add_argument("--extract-size", type=int, default=12)
    parser.add_argument("--budget-usd", type=float, default=2)
    parser.add_argument("--max-jobs", type=int, default=200)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--relevance-model", help="Explicit comparison model; earlier paid receipts are preserved")
    parser.add_argument("--selection", choices=["full", "audit"], default="full")
    args = parser.parse_args()
    pipeline = CorpusPipeline(ROOT)
    manifest = setup(pipeline, args.source, args.sample_size, args.budget_usd)
    if args.relevance_model:
        current = pipeline.get_corpus(manifest["corpus_id"])["config"]
        if current["routes"]["relevance"] != args.relevance_model:
            routes = dict(current["routes"], relevance=args.relevance_model)
            pricing = dict(current["pricing"])
            if args.relevance_model in DEFAULT_PRICING:
                pricing[args.relevance_model] = DEFAULT_PRICING[args.relevance_model]
            pipeline.configure(manifest["corpus_id"], {"routes": routes, "pricing": pricing})
    if args.phase in {"setup", "summary"}:
        results = private_results(pipeline, manifest)
    else:
        results = phase(pipeline, manifest, args.phase, args.extract_size, args.execute, args.max_jobs, args.selection)
    output = {"corpus_id": manifest["corpus_id"], "counts": results["corpus"]["counts"],
              "strata": manifest["strata"], "random_documents": len(manifest["random_document_ids"]),
              "challenge_documents": len(manifest["challenge_document_ids"]),
              "jobs": results["corpus"]["jobs"], "usage": results["corpus"]["usage"],
              "private_results": str(pipeline.runtime / "pilots" / manifest["corpus_id"] / "results_private.json")}
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
