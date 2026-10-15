"""Versioned Rwanda unipolar follow-up: prepare, retrieve, score, then preview.

Each phase is resumable and retains raw responses. No full report, clustering
or presentation is generated before the user's factor-solution selection.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.semantic_service.independent_poles import (
    _write, apply_fact_review, assemble_dossier, attach_reviewed_sources, enrich_independent_dossier, score_independent_dossier,
)
from src.semantic_service.providers import ModelProvider
from src.semantic_service.store import StudyStore, fingerprint, now, ratings_basis

IDENTIFIER = "rwanda-politicians-2021-2026"


def prepare(store):
    study = store.require_approved(IDENTIFIER)
    if study.get("measurement_model") == "independent_poles":
        return study
    directory = store.directory(IDENTIFIER)
    snapshot = directory / "attempts/paired_difference_2026-10-15"
    if snapshot.exists():
        raise ValueError("Snapshot exists; refuse to overwrite historical measurements")
    snapshot.mkdir(parents=True)
    for name in ("study.json", "ratings.json", "README.md"):
        (snapshot / name).write_bytes((directory / name).read_bytes())
    shutil.copytree(directory / "output", snapshot / "output")
    requirements = copy.deepcopy(study["research_requirements"])
    requirements["instructions"] += [
        "С 15.10.2026 пользователь утвердил независимые однополюсные переменные. Каждый из двух прежних полюсов 0..5 анализируется отдельно; неизвестный другой полюс не аннулирует измерение.",
        "Публичное служебное действие измеряет конкретное действие; дипломатический контакт может характеризовать взаимодействие, но не личную симпатию или поддержку внешнеполитических действий России.",
    ]
    protocol = {"version": 2, "approved": True, "at": now(), "actor": "human_user_chat_2026-10-15_independent_poles",
        "authorization": "Давай анализировать полюса как отдельные переменные. В принципе, методология должна допускать оценки по одной шкале.",
        "model": "independent_poles", "input_scale_definitions": 24, "planned_variables": 48,
        "approved_scale_fingerprint": fingerprint(study["scales"]),
        "range": [0, 5], "unknown": "NaN; no inferred zero and no antipode subtraction",
        "anchors": {"0": "Explicitly evidenced absence, never lack of mentions", "1": "Single weak/narrow manifestation",
                    "2": "Clear limited manifestation", "3": "Several consistent episodes",
                    "4": "Strong repeated emphasis", "5": "Defining persistent trait supported by independent episodes"},
        "factor_polarity": "Inferred from signed loadings; no automatic antonyms or forced bipolar factors",
        "previous_snapshot": snapshot.relative_to(directory).as_posix(),
        "source_limitations": "Exploratory LLM measurement of public evidence, not independent human ratings or private beliefs"}
    study = store.update(IDENTIFIER, measurement_model="independent_poles", measurement_protocol=protocol,
        independent_evidence={"version": 1, "retrieval_rounds": 1, "thematic_review": "individually_source_reviewed",
                              "scope": "public_statements_and_actions"},
        research_requirements=requirements, status="ready", analysis=None, artifacts=[], error=None,
        ratings_cache=None)
    _write(ROOT / "research/rwanda_politicians_2021_2026/INDEPENDENT_POLES_PROTOCOL.json", protocol)
    return study


def run_phase(store, provider, study, phase, workers):
    directory = store.directory(study["id"])
    results, errors = {}, []
    def one(obj):
        dossier = assemble_dossier(provider, study, obj)
        dossier = enrich_independent_dossier(provider, study, obj, dossier)
        if phase == "retrieve":
            return {"facts": len(dossier["facts"]), "added": dossier.get("retrieval", {}).get("added_fact_count", 0)}
        dossier = attach_reviewed_sources(provider, study, obj, dossier)
        dossier = attach_reviewed_sources(provider, study, obj, dossier, "completion_source_supplement")
        dossier = apply_fact_review(provider, study, obj, dossier)
        rows = score_independent_dossier(provider, study, obj, dossier)
        _write(directory / "evidence" / ("independent-ratings-" + obj["id"] + ".json"), rows)
        return rows
    store.update(study["id"], status="running", error=None)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, obj): obj for obj in study["objects"]}
        for index, future in enumerate(as_completed(futures), 1):
            obj = futures[future]
            try:
                result = future.result()
                results[obj["id"]] = result
                detail = result if phase == "retrieve" else {"observed_properties": sum(p["status"] == "observed" for row in result for p in row["pole_scores"].values())}
                print(json.dumps({"phase": phase, "completed": index, "total": len(futures), "object": obj["id"], **detail}, ensure_ascii=False), flush=True)
            except Exception as exc:
                errors.append({"object_id": obj["id"], "error": str(exc)[:1000]})
                print(json.dumps({"phase": phase, "completed": index, "object": obj["id"], "error": str(exc)[:1000]}, ensure_ascii=False), flush=True)
            store.log(study["id"], f"Независимые признаки: {phase}, объектов обработано {index}/{len(futures)}", "sources" if phase == "retrieve" else "ratings", index, len(futures))
    _write(directory / "evidence" / ("independent-" + phase + "-receipt.json"),
           {"at": now(), "objects": len(futures), "completed": len(results), "errors": errors,
            "measurement_model": "independent_poles", "protocol": study["measurement_protocol"]})
    if errors:
        store.update(study["id"], status="interrupted", error=f"Не завершён этап {phase}: {len(errors)} объектов; кэш сохранён")
        raise ValueError(f"Phase {phase} incomplete; see receipt, no partial matrix substituted")
    if phase == "score":
        ratings = [row for obj in study["objects"] for row in results[obj["id"]]]
        _write(directory / "ratings.json", ratings)
        current = store.get(study["id"])
        store.update(study["id"], status="ready", ratings_cache={"basis": ratings_basis(current),
                     "ratings_sha256": fingerprint(ratings), "mode": "independent_poles_evidence_bound"})
    else:
        store.update(study["id"], status="ready")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["prepare", "retrieve", "score", "preview"])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    store = StudyStore(ROOT / "studies")
    study = prepare(store)
    if args.phase == "prepare":
        print(json.dumps(study["measurement_protocol"], ensure_ascii=False, indent=2))
        return
    provider = ModelProvider(ROOT, store)
    if args.phase in {"retrieve", "score"}:
        if args.phase == "score":
            supplement = ROOT / "research/rwanda_politicians_2021_2026/russia_independent_poles_supplement.json"
            if not supplement.is_file():
                raise ValueError("Complete and freeze the per-actor Russia source review before scoring")
            config = dict(study["independent_evidence"], reviewed_russia_supplement={
                "path": supplement.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(supplement.read_bytes()).hexdigest()})
            study = store.update(IDENTIFIER, independent_evidence=config)
        run_phase(store, provider, study, args.phase, args.workers)
    else:
        from src.semantic_service.factor_preview import preview_factors
        ratings = json.loads((store.directory(IDENTIFIER) / "ratings.json").read_text(encoding="utf-8"))
        receipt = study.get("ratings_cache") or {}
        if receipt.get("basis") != ratings_basis(study) or receipt.get("ratings_sha256") != fingerprint(ratings):
            raise ValueError("Complete current independent scoring before preview; stale or unbound ratings are forbidden")
        if len(ratings) != len(study["objects"]) * len(study["scales"]) or any(row.get("measurement_model") != "independent_poles" for row in ratings):
            raise ValueError("Preview requires a complete independent-poles matrix, not historical paired ratings")
        result = preview_factors(study, ratings, store.directory(IDENTIFIER) / "output")
        status = "needs_measurement" if result.get("measurement_status") == "insufficient_data" else "await_factor_selection"
        artifacts = [{"label": "Полное сравнение факторов и диагностика", "format": "md", "url": f"/api/studies/{IDENTIFIER}/files/factor_preview.md"},
                     {"label": "Нагрузки всех решений", "format": "json", "url": f"/api/studies/{IDENTIFIER}/files/factor_preview.json"}]
        artifacts += [{"label": c["title"], "format": "png", "url": f"/api/studies/{IDENTIFIER}/files/{c['file']}"} for c in result.get("charts", [])]
        store.update(IDENTIFIER, status=status, analysis=result, artifacts=artifacts, error=None)
        print(json.dumps({"status": status, "objects": result["n_objects"], "variables": result["n_scales"],
                          "models": [{"q": m["q"], "variance_total": m["variance_total"]} for m in result.get("models", [])]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
