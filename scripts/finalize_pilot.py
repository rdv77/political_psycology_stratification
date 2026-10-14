"""Recompute and export the approved pilot from its saved, audited ratings.

Default: inspect readiness only. --apply runs the numerical analysis, cached
interpretation requests and document export, without rating the objects again.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.semantic_service.store import StudyStore, fingerprint
from src.semantic_service.providers import ModelProvider


ROOT = Path(__file__).resolve().parents[1]
IDENTIFIER = "russian-politicians-2016-2026"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    store = StudyStore(ROOT / "studies")
    study = store.require_approved(IDENTIFIER)
    directory = store.directory(IDENTIFIER)
    ratings = json.loads((directory / "ratings.json").read_text(encoding="utf-8"))
    # Every original score batch must be tied to the currently approved scales.
    # This pilot-only command cannot repurpose an old matrix after scale editing.
    for obj in study["objects"]:
        receipt = json.loads((directory / "evidence" / ("scoring_initial-" + obj["id"] + ".json")).read_text(encoding="utf-8"))
        if receipt["scales_basis"] != fingerprint(study["scales"]):
            raise ValueError("Исходные оценки относятся к другой версии шкал")
        dossier = json.loads((directory / "evidence" / (obj["id"] + ".json")).read_text(encoding="utf-8"))
        expected = fingerprint({"version": 2, "object": obj, "period": [study["date_from"], study["date_to"]], "aspect": study["aspect"]})
        if dossier.get("basis") != expected or receipt["dossier_basis"] != fingerprint({"basis": dossier["basis"], "facts": dossier["facts"]}):
            raise ValueError("Исходные оценки относятся к другой версии объекта или досье")
    print(json.dumps({"study": IDENTIFIER, "ratings": len(ratings), "objects": len(study["objects"]),
                      "observed": sum(row["status"] == "observed" for row in ratings), "apply": args.apply}))
    if not args.apply:
        return
    from src.semantic_service.server import Application
    # Only runtime configuration is needed; do not instantiate Application while
    # the explicit research worker owns this study's progress state.
    Application.configure_export_runtime()
    from src.semantic_service.analysis import analyze_study
    from src.semantic_service.reports import export_report
    provider = ModelProvider(ROOT, store)
    output = directory / "output"
    store.update(IDENTIFIER, status="running", error=None)
    try:
        store.log(IDENTIFIER, "Пересчёт по сохранённым оценкам после проверки покрытия", "analysis", 0, 1)
        analysis = analyze_study(study, ratings, output)
        store.log(IDENTIFIER, "Содержательное сравнение группировок", "cluster_review", 0, 1)
        choice = provider.choose_clusters(store.get(IDENTIFIER), analysis)
        if choice["recommended_k"] != analysis["cluster_count"]:
            analysis = analyze_study(study, ratings, output, factor_count=analysis["factor_count"], cluster_count=choice["recommended_k"])
        analysis["cluster_content_review"] = choice
        analysis["cluster_selection_reason"] = choice["reason"]
        analysis["diagnostics"]["cluster_selection"].update(selection_authority="semantic_and_numerical_review", content_recommended_k=choice["recommended_k"])
        if not choice["discrete_structure_supported"]:
            analysis["warnings"].append("Содержательное сравнение не подтверждает естественные дискретные классы: группы — описательная типология непрерывного пространства.")
        store.log(IDENTIFIER, "Интерпретация осей и профилей групп", "interpretation", 0, 1)
        analysis = provider.interpret(store.get(IDENTIFIER), analysis)
        (output / "metadata.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        store.update(IDENTIFIER, analysis=analysis)
        store.log(IDENTIFIER, "Подготовка окончательных файлов отчёта и слайдов", "export", 0, 1)
        artifacts = export_report(store.get(IDENTIFIER), analysis, ratings, output, ROOT)
        store.update(IDENTIFIER, status="complete", artifacts=artifacts, error=None)
        store.log(IDENTIFIER, "Исследование завершено: отчёт и слайды готовы", "complete", 1, 1)
        print(json.dumps({key: analysis[key] for key in ("n_objects", "n_scales", "factor_count", "cluster_count", "variance_total")}))
    except Exception as exc:
        store.update(IDENTIFIER, status="error", error=str(exc)[:1800])
        raise


if __name__ == "__main__":
    main()
