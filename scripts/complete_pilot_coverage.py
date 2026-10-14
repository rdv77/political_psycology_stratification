"""One final, bounded source-and-rating pass for excluded pilot objects.

Dry-run is the default. This script never approves data, changes the dictionary,
or runs analysis/export. Example: python -m scripts.complete_pilot_coverage --apply
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import shutil
import uuid
from pathlib import Path

from src.semantic_service.providers import ModelProvider, RATING, SOURCE, TEXT, grounded_facts, object_schema
from src.semantic_service.store import StudyStore, fingerprint, now


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY = "russian-politicians-2016-2026"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def merge_facts(*groups: list[dict]) -> list[dict]:
    result, seen = [], set()
    for facts in groups:
        for fact in facts:
            key = (" ".join(str(fact.get("text", "")).split()).casefold(),
                   str(fact.get("source_url", "")).rstrip("/"))
            if key not in seen:
                seen.add(key)
                result.append(fact)
    return result


def input_basis(study: dict) -> str:
    return fingerprint({"version": 1, "objects": study["objects"], "scales": study["scales"],
                        "scope": study["scope"], "aspect": study["aspect"],
                        "period": [study["date_from"], study["date_to"]]})


def require_idle_approved(store: StudyStore, study_id: str, expected_basis: str | None = None) -> dict:
    study = store.require_approved(study_id)
    if study["status"] != "complete" and study.get("progress", {}).get("stage") != "coverage_completion":
        raise ValueError("Сначала завершите текущий анализ; допускается только complete или этап coverage_completion")
    if expected_basis is not None and input_basis(study) != expected_basis:
        raise ValueError("Утверждённые входные данные изменились во время добора")
    if study.get("budget_limit_usd", 20) > 20:
        raise ValueError("Для ограниченного добора бюджет исследования должен быть не выше $20")
    return study


def source_material(directory: Path, obj: dict, scales_basis: str) -> tuple[list[dict], list[str], list[dict]]:
    """Load only the original and first supplement; the final pass never feeds itself."""
    material, citations = [], []
    for name in (obj["id"] + ".json", "dossier_supplement-" + obj["id"] + ".json"):
        path = directory / "evidence" / name
        if not path.exists():
            continue
        document = read_json(path)
        if name.startswith("dossier_supplement-") and document.get("scales_basis") != scales_basis:
            raise ValueError("Первый добор досье относится к другому словарю шкал: " + obj["id"])
        urls = document.get("retrieval_citations", []) + document.get("supplement_provenance", {}).get("retrieval_citations", [])
        citations.extend(url for url in urls if isinstance(url, str) and url.startswith(("http://", "https://")))
        material.append({"file": "evidence/" + name, "basis": document.get("basis"),
                         "facts": document.get("facts", []), "sources": document.get("sources", []),
                         "retrieval_citations": urls})
    citations = list(dict.fromkeys(citations))
    facts = merge_facts(*(grounded_facts(item, citations) for item in material))
    return material, citations, facts


def complete_object(provider: ModelProvider, study: dict, directory: Path, obj: dict,
                    targets: list[dict], retained_ids: list[str], study_basis: str,
                    previous: dict | None) -> dict:
    material, base_citations, base_facts = source_material(directory, obj, fingerprint(study["scales"]))
    basis = fingerprint({"version": 1, "approved_basis": study_basis,
                         "retained_scale_ids": retained_ids, "object_id": obj["id"],
                         "base_material": material, "search_model": provider.search_model,
                         "rating_model": provider.model})
    if previous and previous.get("basis") == basis:
        return dict(previous, reused=True)
    record = {"object_id": obj["id"], "basis": basis, "approved_basis": study_basis,
              "objects_basis": fingerprint(study["objects"]), "scales_basis": fingerprint(study["scales"]),
              "target_scale_ids": [scale["id"] for scale in targets],
              "retained_scale_ids": retained_ids, "started_at": now(), "updates": [],
              "maximum_retrieval_rounds": 1, "status": "started"}
    evidence_file = "evidence/coverage_completion-" + obj["id"] + ".json"
    destination = directory / evidence_file
    try:
        if destination.exists() and read_json(destination).get("basis") == basis:
            enriched = read_json(destination)
        else:
            prompt = (f"Последний целевой добор исследовательского досье: {obj['name']}. "
                      f"Область {study['scope']}; аспект {study['aspect']}; период {study['date_from']}—{study['date_to']}. "
                      "Найди только реальные публичные высказывания, программы, документы и наблюдаемые действия, "
                      "относящиеся к указанным недостаточно покрытым шкалам. По возможности охвати оба свойства "
                      "каждой шкалы, включая противоречащие примеры. Приведи даты в исследуемом периоде. "
                      "Предпочитай прямые заявления и первичные документы, подтверждай независимыми источниками. "
                      "Каждому факту нужна реальная ссылка из citations поиска. Не выводи свойство из молчания, "
                      "должности или общей репутации. Не фабрикуй сведения ради полноты: неустранимые пробелы "
                      "честно опиши в limitations. Не ставь оценки; не повторяй исходные факты. Пересказывай кратко. "
                      f"Целевые шкалы: {json.dumps(targets, ensure_ascii=False)}\n"
                      f"Имеющиеся факты: {json.dumps(base_facts, ensure_ascii=False)}")
            fact = object_schema({"text": TEXT, "source_url": TEXT, "period": TEXT})
            response = provider.request(study["id"], "coverage-completion-search-" + obj["id"], prompt,
                                        object_schema({"facts": {"type": "array", "items": fact},
                                                       "sources": {"type": "array", "items": SOURCE}, "limitations": TEXT}),
                                        search=True, max_tokens=8000)
            new_citations = response.get("citations", [])
            accepted = grounded_facts(response["data"], new_citations)
            facts = merge_facts(base_facts, accepted)
            additions = facts[len(base_facts):]
            citations = list(dict.fromkeys(base_citations + [url for url in new_citations
                if isinstance(url, str) and url.startswith(("http://", "https://"))]))
            allowed = {url.rstrip("/") for url in citations}
            sources, seen = [], set()
            for item in material + [response["data"]]:
                for source in item.get("sources", []):
                    url = str(source.get("url", "")).rstrip("/")
                    if url in allowed and url not in seen:
                        sources.append(source)
                        seen.add(url)
            enriched = {"object_id": obj["id"], "basis": basis, "approved_basis": study_basis,
                        "objects_basis": record["objects_basis"], "scales_basis": record["scales_basis"],
                        "facts": facts, "sources": sources, "retrieval_citations": citations,
                        "new_retrieval_citations": new_citations, "completion_facts": additions,
                        "base_evidence_files": [item["file"] for item in material],
                        "provenance": {"model": response["model"], "at": response["at"],
                                       "cache_key": response.get("cache_key"),
                                       "base_material_basis": fingerprint(material),
                                       "target_scale_ids": record["target_scale_ids"],
                                       "retained_scale_ids": retained_ids, "maximum_retrieval_rounds": 1,
                                       "accepted_fact_count": len(accepted), "added_fact_count": len(additions),
                                       "limitations": response["data"].get("limitations", "")}}
            atomic_json(destination, enriched)
        facts = enriched["facts"]
        record["evidence_file"] = evidence_file
        record["accepted_fact_count"] = enriched.get("provenance", {}).get("accepted_fact_count", 0)
        record["added_fact_count"] = len(enriched.get("completion_facts", []))
        if not record["added_fact_count"]:
            # A changed scoring prompt is not new evidence; leave those cells unknown.
            record.update(status="completed_no_new_evidence", finished_at=now(), observed_updates=0)
            return record
        for start in range(0, len(targets), 12):
            scales = targets[start:start+12]
            prompt = (f"Последняя разведочная оценка недостающих ячеек публичного образа {obj['name']}. "
                      f"Область {study['scope']}; аспект {study['aspect']}; период {study['date_from']}—{study['date_to']}. "
                      "Для каждой шкалы отдельно оцени выраженность positive и negative от 0 до 5. "
                      "Оба свойства могут быть выражены. 0 означает подтверждённую невыраженность, "
                      "5 — определяющую черту образа. Используй только предоставленные факты. "
                      "Не выводи полюс из молчания или должности и не подменяй неизвестное нулём. "
                      "Если фактов недостаточно, status=unknown, positive=null и negative=null. "
                      "Для observed нужны evidence_indices — индексы поддерживающих фактов с нуля, "
                      "reason — краткое обоснование без психодиагноза, confidence 0..1 — достаточность материалов. "
                      "Для изменения неизвестной оценки нужно содержательное основание хотя бы в одном новом факте; "
                      "ссылка на новый факт должна действительно поддерживать оцениваемое свойство, а не служить формальностью. "
                      f"Индексы новых фактов: {list(range(len(base_facts), len(facts)))}. "
                      f"Шкалы: {json.dumps(scales, ensure_ascii=False)}\nФакты: {json.dumps(facts, ensure_ascii=False)}")
            try:
                response = provider.request(study["id"], f"coverage-completion-ratings-{obj['id']}-{start}", prompt,
                                            object_schema({"ratings": {"type": "array", "items": RATING}}), max_tokens=6500)
                lookup = {row["scale_id"]: row for row in response["data"]["ratings"]}
                for scale in scales:
                    row = lookup.get(scale["id"], {})
                    indices = [index for index in row.get("evidence_indices", [])
                               if isinstance(index, int) and not isinstance(index, bool) and 0 <= index < len(facts)]
                    positive, negative = row.get("positive"), row.get("negative")
                    supported_new = any(index >= len(base_facts) for index in indices)
                    valid = row.get("status") == "observed" and supported_new and all(
                        isinstance(value, (int, float)) and not isinstance(value, bool)
                        and math.isfinite(value) and 0 <= value <= 5 for value in (positive, negative))
                    record["updates"].append({"object_id": obj["id"], "scale_id": scale["id"],
                        "positive": positive if valid else None, "negative": negative if valid else None,
                        "difference": positive-negative if valid else None, "mass": positive+negative if valid else None,
                        "status": "observed" if valid else "unknown", "confidence": row.get("confidence", 0),
                        "reason": "Новое основание для изменения оценки не найдено" if row.get("status") == "observed" and not supported_new
                                  else row.get("reason", "Модель не вернула оценку"), "evidence_indices": indices,
                        "sources": list(dict.fromkeys(facts[index]["source_url"] for index in indices)),
                        "evidence_file": evidence_file, "base_evidence_files": enriched["base_evidence_files"],
                        "evidence_basis": basis, "rating_pass": "coverage_completion",
                        "model": response["model"], "at": response["at"],
                        "measurement_note": "Последний ограниченный LLM-добор по найденным источникам; не независимая экспертная верификация"})
            except Exception as exc:
                record.setdefault("rating_errors", []).append({"scale_ids": [scale["id"] for scale in scales],
                                                                "message": str(exc)[:1000]})
        record["status"] = "completed_with_errors" if record.get("rating_errors") else "completed"
    except Exception as exc:
        record.update(status="failed", error=str(exc)[:1000])
    record.update(finished_at=now(), observed_updates=sum(row["status"] == "observed" for row in record["updates"]))
    return record


def complete_coverage(root: Path, study_id: str = DEFAULT_STUDY, apply: bool = False,
                      workers: int = 4, provider: ModelProvider | None = None) -> dict:
    root = Path(root)
    store = StudyStore(root / "studies")
    study = require_idle_approved(store, study_id)
    study_basis = input_basis(study)
    directory = store.directory(study_id)
    ratings_file, metadata_file = directory / "ratings.json", directory / "output" / "metadata.json"
    ratings, metadata = read_json(ratings_file), read_json(metadata_file)
    ratings_basis = fingerprint(ratings)
    excluded = {obj["id"] for obj in metadata.get("diagnostics", {}).get("excluded_objects", metadata.get("excluded_objects", []))}
    retained_ids = [row["id"] for row in metadata.get("loading_table", [])]
    if not retained_ids or len(set(retained_ids)) != len(retained_ids):
        raise ValueError("В metadata.loading_table нет однозначного набора retained шкал")
    if not set(retained_ids).issubset({scale["id"] for scale in study["scales"]}):
        raise ValueError("Metadata относится к другому словарю шкал")
    lookup = {(row["object_id"], row["scale_id"]): row for row in ratings}
    if len(lookup) != len(ratings):
        raise ValueError("В ratings обнаружены повторные ячейки")
    targets = [(obj, [scale for scale in study["scales"] if scale["id"] in retained_ids
                     and lookup.get((obj["id"], scale["id"]), {}).get("status") == "unknown"])
               for obj in study["objects"] if obj["id"] in excluded]
    targets = [(obj, scales) for obj, scales in targets if scales]
    summary = {"study_id": study_id, "apply": apply, "approved_basis": study_basis,
               "target_objects": len(targets), "target_cells": sum(len(scales) for _, scales in targets),
               "retained_scale_count": len(retained_ids), "workers": min(4, max(1, workers)),
               "observed_before": sum(row.get("status") == "observed" for row in ratings)}
    if not apply:
        return dict(summary, targets=[{"object_id": obj["id"], "scale_ids": [s["id"] for s in scales]} for obj, scales in targets])
    lock_file = directory / ".coverage_completion.lock"
    try:
        descriptor = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ValueError("Другой процесс ограниченного добора уже запущен") from exc
    os.close(descriptor)
    try:
        history = directory / "history" / "pre-coverage-completion"
        history.mkdir(parents=True, exist_ok=True)
        for source in (ratings_file, metadata_file):
            destination = history / source.name
            if not destination.exists():
                shutil.copy2(source, destination)
        manifest = history / "manifest.json"
        if not manifest.exists():
            atomic_json(manifest, {"approved_basis": study_basis, "ratings_basis": ratings_basis, "at": now()})
        ledger_file = directory / "evidence" / "coverage_completion_ledger.json"
        ledger = read_json(ledger_file) if ledger_file.exists() else {"version": 1, "objects": {}}
        ledger.update(approved_basis=study_basis, retained_scale_ids=retained_ids)
        provider = provider or ModelProvider(root, store)
        with concurrent.futures.ThreadPoolExecutor(max_workers=summary["workers"]) as pool:
            futures = {}
            for obj, scales in targets:
                require_idle_approved(store, study_id, study_basis)
                future = pool.submit(complete_object, provider, study, directory, obj, scales,
                                     retained_ids, study_basis, ledger["objects"].get(obj["id"]))
                futures[future] = obj
            for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
                result = future.result()
                ledger["objects"][result["object_id"]] = result
                ledger["updated_at"] = now()
                atomic_json(ledger_file, ledger)
                store.log(study_id, f"Последний добор: {completed} из {len(targets)}; {futures[future]['name']}: "
                          f"оценок с новым основанием {result['observed_updates']}",
                          "coverage_completion", completed, len(targets))
                print(json.dumps({"object_id": result["object_id"], "status": result["status"],
                                  "observed_updates": result["observed_updates"], "reused": result.get("reused", False)}, ensure_ascii=False), flush=True)
        require_idle_approved(store, study_id, study_basis)
        if fingerprint(read_json(ratings_file)) != ratings_basis:
            raise ValueError("Ratings изменились параллельно; добор сохранён в ledger, перезапись отменена")
        target_keys = {(obj["id"], scale["id"]) for obj, scales in targets for scale in scales}
        updates = {(row["object_id"], row["scale_id"]): row
                   for obj, _ in targets for row in ledger["objects"][obj["id"]].get("updates", [])
                   if (row["object_id"], row["scale_id"]) in target_keys}
        final = [updates.get((row["object_id"], row["scale_id"]), row)
                 if row.get("status") == "unknown" else row for row in ratings]
        atomic_json(ratings_file, final)
        summary.update(observed_after=sum(row.get("status") == "observed" for row in final),
                       newly_observed=sum(row.get("status") == "observed" for row in final)-summary["observed_before"],
                       remaining_target_unknown=sum(row.get("status") == "unknown" and
                           (row["object_id"], row["scale_id"]) in target_keys for row in final),
                       ledger="evidence/coverage_completion_ledger.json")
        ledger["last_summary"] = summary
        atomic_json(ledger_file, ledger)
        return summary
    finally:
        lock_file.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--study", default=DEFAULT_STUDY)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--apply", action="store_true", help="Разрешить реальные запросы и обновить только целевые unknown ratings")
    args = parser.parse_args()
    print(json.dumps(complete_coverage(args.root, args.study, args.apply, args.workers), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
