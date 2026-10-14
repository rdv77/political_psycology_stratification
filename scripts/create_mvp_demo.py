"""Offline UI/export example from the old party matrix, never a new study.

Run from the repository root with the existing research environment:
    python -m scripts.create_mvp_demo
    python -m scripts.create_mvp_demo --rebuild

Only studies/demo-legacy-parties is created or, with --rebuild, replaced.
No provider, API, PDF, Word, or presentation exporter is called.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.semantic_service.analysis import analyze_study
from src.semantic_service.reports import heading, html_document, markdown, paragraph, report_blocks
from src.semantic_service.store import StudyStore, fingerprint, validate_objects, validate_scales


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="Replace only the previously marked legacy example")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    studies = (root / "studies").resolve()
    identifier = "demo-legacy-parties"
    store = StudyStore(studies)
    destination = store.directory(identifier).resolve()
    # Recursive replacement is constrained to this literal, explicitly named fixture.
    if destination.parent != studies or destination.name != "demo-legacy-parties":
        raise ValueError("Unsafe example destination")
    if destination.exists():
        existing = store.get(identifier)
        if not args.rebuild:
            print("Example already exists; use --rebuild to replace this fixture only.")
            print(str(destination / "output" / "report_content.json"))
            return
        if existing.get("example") is not True or existing.get("measurement_mode") != "legacy_import_example":
            raise ValueError("Refusing to replace an unmarked study with the example identifier")
        print("Rebuilding only: " + str(destination), flush=True)
        shutil.rmtree(destination)

    inputs = {
        "scores": root / "data/scores.parquet",
        "traits": root / "data/traits.json",
        "countries": root / "data/parties_map.json",
    }
    frame = pd.read_parquet(inputs["scores"])
    trait_names = json.loads(inputs["traits"].read_text(encoding="utf-8"))[:40]
    countries = json.loads(inputs["countries"].read_text(encoding="utf-8"))
    if "party" not in frame.columns or len(frame) != 95 or len(trait_names) != 40:
        raise ValueError("The legacy example requires 95 parties and the first 40 saved traits")
    absent = [name for name in trait_names if name not in frame.columns]
    if absent:
        raise ValueError("Saved trait names do not match the legacy score columns")
    matrix = frame[trait_names].to_numpy(dtype=float)
    known = matrix[np.isfinite(matrix)]
    if np.isinf(matrix).any() or (known.size and (known.min() < 0 or known.max() > 10)):
        raise ValueError("Legacy observed values must be finite and between 0 and 10")

    note = (
        "ТЕХНИЧЕСКИЙ ПРИМЕР НА ПРЕЖНИХ ДАННЫХ. Это повторный анализ ранее сохранённых "
        "модельных оценок 95 партий по первым 40 признакам исходного словаря. Это не новое "
        "исследование и не результат исследования 70 российских политиков. Новые источники "
        "не собирались, LLM не вызывалась, старый словарь не проходил новую нормализацию. "
        "Оценка x по прежней шкале 0–10 перенесена как value = x − 5. Два полюса независимо "
        "не оценивались; их сумма и смешанность по этим данным не определяются. "
        "Ноль в примере означает прежнюю оценку 5, а не доказанную невыраженность обоих свойств. "
        "Отметки утверждения принадлежат импортированному примеру, а не пользователю."
    )
    source_metadata = {
        "kind": "legacy_import_example", "note": note,
        "files": {key: str(file.relative_to(root)).replace("\\", "/") for key, file in inputs.items()},
        "sha256": {key: hashlib.sha256(file.read_bytes()).hexdigest() for key, file in inputs.items()},
        "original_objects": len(frame), "original_trait_count": len(frame.columns) - 1,
        "selected_traits": 40, "trait_selection": "first 40 entries of the saved traits.json, not a validated short form",
        "conversion": "value = original_0_10 - 5; missing values remain unknown; mass is unavailable",
        "historical_period": "not fixed in the saved matrix; no new 2016-2026 claim",
        "fresh_model_calls": 0,
    }
    objects = validate_objects([
        {"id": f"party-{i + 1:03d}", "name": str(name), "category": "Политическая партия",
         "country": countries.get(str(name), "Не указана"),
         "role": "Партия из сохранённой международной выборки; страна: " + countries.get(str(name), "Не указана"),
         "inclusion_reason": "Импортирована из прежней матрицы для технического примера интерфейса и экспорта; новая выборка не формировалась",
         "sources": [], "source_files": list(source_metadata["files"].values())}
        for i, name in enumerate(frame["party"])
    ])
    scales = validate_scales([
        {"id": f"scale-{i + 1:03d}", "name": name, "positive": name,
         "negative": "Слабая выраженность: " + name,
         "definition": "Учебный перенос старой шкалы 0..10 через x−5; не заново оценённая пара антонимов",
         "aliases": [], "legacy_source_trait": name,
         "anchors": {"-5": "Прежняя оценка 0", "0": "Прежняя оценка 5; не равнозначна отсутствию обоих полюсов", "5": "Прежняя оценка 10"}}
        for i, name in enumerate(trait_names)
    ])
    study = store.create({
        "title": "Пример: ранее собранные партии",
        "scope": "Международное пространство политических партий из прежней выборки",
        "aspect": "Старые модельные оценки партий", "target_count": 95,
        "query": "Технический импорт прежней матрицы для демонстрации UI, анализа и экспорта; новых объектов не искать",
        # Store dates are a technical import date, explicitly not a historical
        # observation period; the display label below takes precedence in UI/report.
        "focus": "Яблоко", "date_from": "2026-10-15", "date_to": "2026-10-15",
    }, identifier=identifier)
    study.update(objects=objects, scales=scales, example=True,
                 period_label="Исторический период в исходной матрице не зафиксирован",
                 date_semantics="Technical example-import date only; not an observation period",
                 example_note=note, measurement_mode="legacy_import_example",
                 selection_method="Прежняя целевая международная выборка 95 партий; первые 40 сохранённых шкал взяты только для технической демонстрации, а не как валидированная короткая форма.",
                 selection_limitations=[note], legacy_import=source_metadata,
                 imported_ratings_basis={"objects": fingerprint(objects), "scales": fingerprint(scales)})
    store.save(study)
    # These are explicitly marked fixture receipts, never approvals on the pilot.
    store.approve(identifier, "objects", actor="imported_example_fixture")
    study = store.approve(identifier, "scales", actor="imported_example_fixture")
    ratings = []
    for i, obj in enumerate(objects):
        for j, scale in enumerate(scales):
            value = matrix[i, j]
            observed = bool(np.isfinite(value))
            ratings.append({
                "object_id": obj["id"], "scale_id": scale["id"],
                "status": "observed" if observed else "unknown",
                "value": float(value - 5) if observed else None,
                "original_value": float(value) if observed else None,
                "reason": "Перенос сохранённой модельной оценки 0..10 по формуле x−5; не новая оценка и не независимое оценивание полюсов",
                "sources": [], "source_file": "data/scores.parquet",
                "measurement_mode": "legacy_import_example",
            })
    (destination / "ratings.json").write_text(json.dumps(ratings, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    store.update(identifier, imported_ratings_count=len(ratings), status="running")
    store.log(identifier, "Пример: офлайн-анализ прежних оценок 95 партий; новых LLM-запросов нет", "analysis", 0, 1)
    output = destination / "output"
    print("Analyzing the marked legacy example: 95 parties x 40 saved traits; no external calls.", flush=True)
    try:
        analysis = analyze_study(study, ratings, output)
        analysis["example"] = True
        analysis["legacy_import"] = source_metadata
        analysis["warnings"].insert(0, note)
        analysis["diagnostics"]["measurement"] = source_metadata["conversion"]
        (output / "metadata.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        study = store.update(identifier, analysis=analysis)
        blocks = report_blocks(study, analysis, ratings)
        blocks.insert(1, heading("Технический пример: ограничения исходных данных", 1))
        blocks.insert(2, paragraph(note))
        for block in blocks:
            if block["type"] != "paragraph":
                continue
            text = block["text"]
            if text.startswith("Область:"):
                block["text"] = f"Область: {study['scope']}. Аспект: {study['aspect']}. Исторический период в сохранённой матрице не зафиксирован. Пример не относится к новой выборке российских политиков 2016–2026 годов."
            elif text.startswith("Проанализировано"):
                block["text"] = text.replace("Результат представляет разведочную модель публичных образов в выбранном корпусе источников.", "Это техническая демонстрация нового анализа прежних оценок, а не заново выполненное исследование или валидация исходного измерения.")
            elif text.startswith("После утверждения объектов собраны"):
                block["text"] = "Досье и источники в этом примере не пересобирались. Импортированы 95 строк старой матрицы и первые 40 признаков сохранённого словаря. Близкие формулировки и разные уровни описания могли сохраниться; короткая форма не валидирована. Утверждения помечены actor=imported_example_fixture и не являются решениями пользователя."
            elif text.startswith("Каждый полюс шкалы оценён"):
                block["text"] = "Импортирована прежняя оценка x из диапазона 0–10; для нового численного конвейера использовано value=x−5. Независимые оценки двух полюсов отсутствуют, сумма полюсов неизвестна. Пропуск сохранён как unknown. Ноль после сдвига означает старую оценку 5, а не невыраженность обоих свойств."
            elif text.startswith("Наблюдаемых оценок:"):
                observed = sum(rating["status"] == "observed" for rating in ratings)
                block["text"] = f"Сохранённых наблюдаемых оценок: {observed} из {len(ratings)}. Режим legacy_import_example. Эти числа получены прежним модельным оценщиком; новых запросов к моделям и новых человеческих оценок нет. Исходные источники не восстановлены и не проверены."
            elif text.startswith("Слабая заполненность участка этой карты"):
                block["text"] = "Пустой участок этой международной карты относится только к прежней выборке и первым 40 признакам. Он не доказывает отсутствие соответствующих идей во всём публичном пространстве, полноту сравнения стран или наличие спроса на свободную нишу."
            elif text.startswith("Сохранены версии списков"):
                block["text"] = "Сохранены контрольные суммы исходных файлов, состав импортированного примера, преобразованные поэлементные оценки, параметры анализа и отметки imported_example_fixture. Новых досье, независимых оценок полюсов или пользовательских утверждений для этого примера нет."
        (output / "report.md").write_text(markdown(blocks), encoding="utf-8")
        (output / "report.html").write_text(html_document(blocks), encoding="utf-8")
        artifacts = [{"label": label, "format": extension, "url": f"/api/studies/{identifier}/files/{name}"}
                     for name, label, extension in (("report.html", "Открыть пример отчёта", "html"),
                                                     ("report.md", "Пример отчёта Markdown", "md"),
                                                     ("metadata.json", "Параметры примера", "json"))]
        study = store.update(identifier, status="complete", analysis=analysis, artifacts=artifacts, error=None)
        store.log(identifier, "Офлайн-пример готов: MD/HTML и данные экспорта; PDF и презентация ещё не создавались", "complete", 1, 1)
        study = store.get(identifier)
        payload = {"study": study, "analysis": analysis, "blocks": blocks}
        content = output / "report_content.json"
        content.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps({"study_id": identifier, "report_content": str(content),
                          "objects": len(objects), "scales": len(scales),
                          "factors": analysis["factor_count"], "clusters": analysis["cluster_count"],
                          "fresh_model_calls": 0}, ensure_ascii=True), flush=True)
    except Exception as exc:
        store.update(identifier, status="error", error=str(exc))
        raise


if __name__ == "__main__":
    main()
