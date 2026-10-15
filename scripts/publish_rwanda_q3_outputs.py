"""Register the selected q3/k4 files in the local service and download bundle."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.semantic_service.store import StudyStore, fingerprint, now, ratings_basis
from src.semantic_service.reports import write_report_bundle

ID = "rwanda-politicians-2021-2026"
RESEARCH = ROOT / "research/rwanda_politicians_2021_2026"
read = lambda path: json.loads(Path(path).read_text(encoding="utf-8"))
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    store = StudyStore(ROOT / "studies")
    study = store.require_approved(ID)
    output = store.directory(ID) / "output"
    source_ratings_sha = sha(output.parent / "ratings.json")
    model = read(output / "selected_q3/metadata.json")
    if model["factor_count"] != 3 or model["cluster_count"] != 4 or model["ratings_sha256"] != source_ratings_sha:
        raise ValueError("The selected model does not match approved q3 or current ratings")
    if study["ratings_cache"]["basis"] != ratings_basis(study):
        raise ValueError("The measurement cache became stale")
    spec = read(output / "presentation_content.json")
    if spec["catalogue_count"] != 42 or len(spec["slides"]) != 50:
        raise ValueError("The reviewed catalogue or deck content changed")
    for relative, expected in spec.get("source_sha256", {}).items():
        if sha(output / relative) != expected:
            raise ValueError(f"Slide coordinates belong to a different source: {relative}")
    for relative, expected in read(output / "selected_q3/delivery_inputs.json")["source_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"A reviewed narrative or measurement input changed: {relative}")
    files = [("report.md", "Политологический отчёт Markdown"), ("report.pdf", "Политологический отчёт PDF"),
             ("slides.pptx", "Презентация 50 слайдов"), ("report.docx", "Политологический отчёт для Google Docs"),
             ("report.html", "Политологический отчёт в браузере"), ("report_bundle.zip", "Все форматы и материалы"),
             ("technical/report.pdf", "Технический отчёт PDF"), ("technical/report.md", "Технический отчёт Markdown"),
             ("technical/report.docx", "Технический отчёт DOCX"), ("technical/report.html", "Технический отчёт в браузере"),
             ("cluster_candidates_q3.md", "Сравнение числа групп и устойчивость"),
             ("selected_q3/scores_and_clusters.csv", "Координаты и составы групп"),
             ("coverage_by_actor.md", "Покрытие всех 42 участников"),
             ("short_report.md", "Предварительное сравнение факторных решений")]
    for name, _ in files:
        if name != "report_bundle.zip" and not (output / name).is_file():
            raise FileNotFoundError(name)
    labels = {m["id"]: c["id"] for c in model["clusters"] for m in c["members"]}
    if len(labels) != 37 or len(model["objects"]) != 37:
        raise ValueError("All 37 admitted people need exactly one group")
    review = read(RESEARCH / "CLUSTER_Q3_POLITICAL_REVIEW.json")
    model["cluster_selection_reason"] = review["selection_reason"]
    model["object_profiles"] = [{**o, "cluster_id": labels[o["id"]], "factor_scores": o["raw_scores"],
                                "observed_scales": o["observed_properties"], "missing_scales": 8-o["observed_properties"]}
                               for o in model["objects"]]
    model["charts"] = read(output / "assets/q3_political/manifest.json")["charts"]
    model["russia_review"] = read(output / "factor_preview.json")["russia_review"]
    model["warnings"] = ["Четыре группы — рабочая типология публичных профилей, не установленные политические коалиции.",
                         "Источники и недостающие сведения влияют на оси и группы; проверки приведены в техническом отчёте."]
    selection = {"approved": True, "factor_count": 3, "actor": "human_user_chat_2026_10_16", "at": now(),
                 "user_instruction": "Давай выберем решение с 3 факторами и подготовим кластеризацию, расширенный отчёт и презентацию.",
                 "numerical_preview_sha256": sha(output / "factor_preview.json"), "ratings_sha256": source_ratings_sha}
    artifacts = [{"label": label, "format": Path(name).suffix[1:], "url": f"/api/studies/{ID}/files/{name}"} for name, label in files]
    study = store.update(ID, delivery_mode="full_report", status="complete", analysis=model, artifacts=artifacts,
                         factor_selection=selection, report_profile="rwanda_q3_political_and_separate_technical",
                         presentation_config={"mode": "rwanda_q3_political", "factor_count": 3, "cluster_count": 4,
                                              "slide_count": len(spec["slides"]), "catalogue_count": 42},
                         progress={"stage": "complete", "message": "Политологический и технический отчёты, презентация и справочник готовы", "current": 1, "total": 1}, error=None)
    if study["ratings_cache"]["basis"] != ratings_basis(study) or sha(output.parent / "ratings.json") != source_ratings_sha:
        raise RuntimeError("Publication modified the measurements")
    payload = read(output / "report_content.json")
    payload["study"] = study
    (output / "report_content.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report_bundle(output)
    with zipfile.ZipFile(output / "report_bundle.zip", "a", compression=zipfile.ZIP_DEFLATED) as bundle:
        for folder in ("technical", "selected_q3", "cluster_candidates_q3"):
            for file in sorted((output / folder).rglob("*")):
                relative = file.relative_to(output)
                if file.is_file() and not any(part.startswith(".") for part in relative.parts):
                    bundle.write(file, relative.as_posix())
        for name in ("cluster_candidates_q3.json", "cluster_candidates_q3.md", "factor_preview.json", "factor_preview.md",
                     "imputation_review_2026-10-15.json", "imputation_review_2026-10-15.md", "coverage_by_actor.md"):
            bundle.write(output / name, name)
        for name in ("REPORT_Q3_NARRATIVE.json", "ACTOR_PROFILES.json", "POLITICAL_SYSTEM_CONTEXT.json", "POLITICAL_CONTEXT_RUSSIA.json",
                     "CLUSTER_Q3_POLITICAL_REVIEW.json", "TECHNICAL_Q3_CONTENT.json"):
            bundle.write(RESEARCH / name, "context/" + name)
    with zipfile.ZipFile(output / "report_bundle.zip") as bundle:
        required = {"report.md", "report.pdf", "report.docx", "slides.pptx", "technical/report.md", "technical/report.pdf", "technical/report.docx",
                    "selected_q3/factor_parameters_q3.json", "selected_q3/scores_and_clusters.csv"}
        if not required.issubset(bundle.namelist()) or bundle.testzip() is not None:
            raise RuntimeError("The report bundle is incomplete or corrupted")
    print(json.dumps({"status": study["status"], "factor_count": 3, "cluster_count": 4,
                      "artifacts": len(artifacts), "ratings_unchanged": True,
                      "bundle_bytes": (output / "report_bundle.zip").stat().st_size}))

if __name__ == "__main__":
    main()
