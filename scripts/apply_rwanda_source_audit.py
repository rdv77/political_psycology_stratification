"""Apply documented source-consistency decisions, without inventing new scores."""
from __future__ import annotations

from collections import Counter
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.semantic_service.factor_preview import preview_factors
from src.semantic_service.store import StudyStore, fingerprint, now, ratings_basis

IDENTIFIER = "rwanda-politicians-2021-2026"
RESEARCH = ROOT / "research/rwanda_politicians_2021_2026"


def main():
    store = StudyStore(ROOT / "studies")
    study = store.require_approved(IDENTIFIER)
    if study.get("source_audit", {}).get("applied"):
        raise ValueError("Аудит уже применён; повторная перезапись запрещена")
    directory = store.directory(IDENTIFIER)
    ratings_file = directory / "ratings.json"
    original = json.loads(ratings_file.read_text(encoding="utf-8"))
    ratings = copy.deepcopy(original)
    lookup = {(row["object_id"], row["scale_id"]): row for row in ratings}
    if len(lookup) != len(study["objects"]) * len(study["scales"]):
        raise ValueError("Сначала необходимо завершить оценивание всей матрицы")
    audit_files = sorted(RESEARCH.glob("AUDIT_BATCH_*.json"))
    reports = [json.loads(file.read_text(encoding="utf-8")) for file in audit_files]
    checked = [identity for report in reports for identity in report["checked_objects"]]
    if len(checked) != len(set(checked)) or set(checked) != {obj["id"] for obj in study["objects"]}:
        raise ValueError("Аудит должен охватывать ровно все утверждённые объекты")
    observed_before = sum(row["status"] == "observed" for row in original)
    if sum(report["checked_observed"] for report in reports) != observed_before:
        raise ValueError("Количество проверенных ячеек не совпадает с исходной матрицей")
    attempts = directory / "attempts/pre_source_audit"
    attempts.mkdir(parents=True, exist_ok=True)
    (attempts / "ratings.json").write_bytes(ratings_file.read_bytes())
    for file in (directory / "output").glob("factor_preview.*"):
        (attempts / file.name).write_bytes(file.read_bytes())
    seen, counts = set(), Counter()
    for file, report in zip(audit_files, reports):
        for decision in report["decisions"]:
            key = (decision["object_id"], decision["scale_id"])
            row = lookup[key]
            if key in seen or row["status"] != "observed":
                raise ValueError("Повторное решение аудита или ссылка на ненаблюдаемую оценку")
            seen.add(key)
            action = decision["decision"]
            if action not in {"reject", "repair_indices", "flag"}:
                raise ValueError("Неизвестное решение аудита")
            counts[action] += 1
            row["original_provider_rating"] = copy.deepcopy(row)
            row["source_audit"] = dict(decision, audit_file=file.relative_to(ROOT).as_posix(),
                                       reviewer="assistant_consistency_review_not_independent_human_expert")
            if action == "reject":
                row.update(status="unknown", positive=None, negative=None, difference=None, mass=None, confidence=0,
                           reason="Аудит оснований: " + decision["reason"])
                if "value" in row:
                    row["value"] = None
            elif action == "repair_indices":
                evidence_file = (directory / row["evidence_file"]).resolve()
                if not evidence_file.is_relative_to(directory.resolve()):
                    raise ValueError("Досье должно находиться внутри исследования")
                evidence = json.loads(evidence_file.read_text(encoding="utf-8"))["facts"]
                indices = decision["evidence_indices"]
                if not indices or any(not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < len(evidence) for i in indices):
                    raise ValueError("Недопустимый исправленный индекс")
                row["evidence_indices"] = indices
                row["sources"] = list(dict.fromkeys(evidence[i]["source_url"] for i in indices))
    if any(row["status"] == "observed" and
           (row["positive"], row["negative"]) != (old["positive"], old["negative"])
           for row, old in zip(ratings, original)):
        raise AssertionError("Аудит не должен назначать новые численные оценки")
    pending = ratings_file.with_suffix(".json.tmp")
    pending.write_text(json.dumps(ratings, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(ratings_file)
    audit = {"applied": True, "at": now(), "checked_objects": len(checked), "checked_observed": observed_before,
             "counts": dict(counts), "remaining_observed": sum(row["status"] == "observed" for row in ratings),
             "files": [file.relative_to(ROOT).as_posix() for file in audit_files],
             "scope": "Семантическая согласованность с извлечёнными выписками; не независимая человеческая экспертиза и не полная проверка каждой исходной страницы",
             "before_fingerprint": fingerprint(original), "after_fingerprint": fingerprint(ratings)}
    store.update(IDENTIFIER, source_audit=audit, status="running", analysis=None, artifacts=[],
                 ratings_cache={"basis": ratings_basis(study), "ratings_sha256": fingerprint(ratings),
                                "mode": "source_consistency_audited_exploratory"})
    study = store.get(IDENTIFIER)
    output = directory / "output"
    result = preview_factors(study, ratings, output)
    result["measurement_context"]["source_audit"] = audit
    result["russia_review"]["evidence_audit"] = "assistant_excerpt_consistency_review_not_human_validation"
    before = Counter(row["scale_id"] for row in original if row["status"] == "observed")
    after = Counter(row["scale_id"] for row in ratings if row["status"] == "observed")
    scales = study["scales"]
    positions = np.arange(len(scales))
    figure, axes = plt.subplots(figsize=(13, 13))
    axes.barh(positions - .18, [100 * before[s["id"]] / len(study["objects"]) for s in scales], height=.35,
              color="#a8b8bb", label="Предложения модели до проверки")
    axes.barh(positions + .18, [100 * after[s["id"]] / len(study["objects"]) for s in scales], height=.35,
              color="#226d66", label="После исключения явных ошибок")
    axes.set_yticks(positions, [s["name"] for s in scales], fontsize=9)
    axes.invert_yaxis()
    axes.axvline(100 * 26 / 42, color="#a33f39", linestyle="--", label="Порог первого шага: 26 из 42")
    axes.set_xlim(0, 100)
    axes.set_xlabel("Доля политиков с известной разностью двух полюсов, %")
    axes.set_title("Руанда: полнота измерений по 24 шкалам\nЭто покрытие данных, а не доля объяснённой дисперсии")
    axes.legend(loc="lower right", fontsize=9)
    figure.tight_layout()
    (output / "assets").mkdir(exist_ok=True)
    chart_file = "assets/coverage_before_after_audit.png"
    figure.savefig(output / chart_file, dpi=150, bbox_inches="tight")
    plt.close(figure)
    result["charts"].append({"q": None, "file": chart_file, "title": "Покрытие данных до и после проверки"})
    (output / "factor_preview.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    with (output / "factor_preview.md").open("a", encoding="utf-8") as stream:
        stream.write("\n## Проверка оснований оценок\n\n" + audit["scope"] + ".\n\n" +
                     f"Проверено {observed_before} предложенных пар; отклонено {counts['reject']}, исправлены ссылки у {counts['repair_indices']}, " +
                     f"сохранены с предупреждениями {counts['flag']}. Новые числа не назначались.\n\n" +
                     f"![Покрытие данных]({chart_file})\n")
    artifacts = [{"label": "Короткий аудит измерений", "format": "md", "url": f"/api/studies/{IDENTIFIER}/files/factor_preview.md"},
                 {"label": "Матрица и диагностика", "format": "json", "url": f"/api/studies/{IDENTIFIER}/files/factor_preview.json"},
                 {"label": "Диаграмма покрытия", "format": "png", "url": f"/api/studies/{IDENTIFIER}/files/{chart_file}"}]
    status = "needs_measurement" if result.get("measurement_status") == "insufficient_data" else "await_factor_selection"
    store.update(IDENTIFIER, analysis=result, artifacts=artifacts, status=status, error=None)
    store.log(IDENTIFIER, "Короткий аудит готов; данных для обоснованного выбора факторов недостаточно" if status == "needs_measurement" else "Сравнение факторов готово",
              "measurement_review" if status == "needs_measurement" else "factor_selection", 1, 1)
    print(json.dumps({"audit": audit, "status": status, "retained_objects": result["n_objects"],
                      "retained_scales": result["n_scales"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
