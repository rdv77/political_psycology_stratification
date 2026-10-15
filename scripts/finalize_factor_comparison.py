"""Publish the reviewed q2/q3/q4 comparison from frozen, locally saved fits.

No search, scoring or language-model API is called. Numerical fitting is a
separate step (scripts.compare_factor_models). Default mode checks readiness;
--apply integrates reviewed interpretations and exports the report and slides.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src.semantic_service.store import StudyStore, fingerprint, ratings_basis


ROOT = Path(__file__).resolve().parents[1]
IDENTIFIER = "russian-politicians-2016-2026"
REVIEW = ROOT / "research/russian_politicians_2016_2026/FACTOR_COMPARISON_INTERPRETATION.json"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def partition_overlap(reference_path: Path, target_path: Path) -> dict:
    """Describe intersections even when the number of groups differs."""
    reference = pd.read_csv(reference_path).set_index("object_id")["cluster_id"]
    target = pd.read_csv(target_path).set_index("object_id")["cluster_id"]
    if not reference.index.is_unique or not target.index.is_unique or set(reference.index) != set(target.index):
        raise ValueError("Comparisons must use exactly the same unique objects")
    target = target.reindex(reference.index)
    overlap = pd.crosstab(reference.rename("reference"), target.rename("target"))
    return {"k_reference": int(reference.nunique()), "k_target": int(target.nunique()),
            "ari": float(adjusted_rand_score(reference, target)),
            "nmi": float(normalized_mutual_info_score(reference, target, average_method="arithmetic")),
            "overlap": {"rows": overlap.index.tolist(), "columns": overlap.columns.tolist(),
                        "values": overlap.to_numpy().tolist()},
            "scope": "Raw cluster labels are model-local. Intersections describe regrouping; no cross-k moved-count is inferred."}


def reviewed_metadata(metadata: dict, review: dict, cluster_field="clusters") -> dict:
    result = deepcopy(metadata)
    for field, descriptions in (("factors", review["factors"]), ("clusters", review[cluster_field])):
        mapping = {item["id"]: item for item in descriptions}
        if len(mapping) != len(descriptions) or set(mapping) != {item["id"] for item in result[field]}:
            raise ValueError(f"Review does not match all numerical {field}")
        for item in result[field]:
            text = mapping[item["id"]]
            if not text.get("label") or not text.get("interpretation"):
                raise ValueError("A reviewed label and interpretation are required")
            if field == "clusters" and "count" in text and text["count"] != item["count"]:
                raise ValueError("The reviewed cluster size disagrees with its actual members")
            item.setdefault("numeric_label", item.get("label") or item.get("name"))
            item.update(label=text["label"], name=text["label"], interpretation=text["interpretation"])
    result["interpretation_provenance"] = {
        "method": "assistant_review_of_computed_loadings_and_cluster_profiles",
        "source": str(REVIEW.relative_to(ROOT)).replace("\\", "/"),
        "review_sha256": hashlib.sha256(REVIEW.read_bytes()).hexdigest(),
        "scope": "Narrative review by the assistant; not an independent human validation or a new measurement"}
    return result


def build_comparison(output: Path, review: dict) -> tuple[dict, dict]:
    comparison = read_json(output / "models/comparison.json")
    primary = read_json(output / "models/q2/metadata.json")
    if primary["factor_count"] != 2 or primary["cluster_count"] != 3 or "model_comparison" in primary:
        raise ValueError("The original q2/k3 snapshot must remain a plain baseline")
    reviews = {int(item["q"]): item for item in review["models"]}
    if set(reviews) != {3, 4}:
        raise ValueError("Both alternative dimensionalities must be reviewed")
    comparison.update(executive_summary=review["executive_summary"], recommendation=review["recommendation"],
                      interpretation_provenance=review["provenance"], selected_k=[])
    preferred_pair = next((item for item in comparison.get("selected_k_comparisons", [])
                           if item["reference_q"] == 3 and item["target_q"] == 4 and item["k"] == 4), None)
    if preferred_pair is not None:
        comparison["preferred_k4_comparison"] = deepcopy(preferred_pair)
        if preferred_pair.get("chart"):
            comparison["charts"].append(deepcopy(preferred_pair["chart"]))
    for chart in comparison.get("charts", []):
        chart["file"] = "models/" + chart["file"]
    for group in ("factor_comparisons", "fixed_k3"):
        for item in comparison[group]:
            item["artifact_files"] = {key: "models/" + value for key, value in item.get("artifact_files", {}).items()}
    previous_variance = None
    for model in comparison["models"]:
        q = int(model["q"])
        folder = output / "models" / f"q{q}"
        metadata = read_json(folder / "metadata.json")
        if q > 2:
            metadata = reviewed_metadata(metadata, reviews[q])
            write_json(folder / "metadata.json", metadata)
        model.update(path=f"models/q{q}", factors=metadata["factors"], loading_table=metadata["loading_table"],
                     clusters=metadata["clusters"],
                     dimension_fit=next(item for item in metadata["dimension_candidates"] if item["q"] == q),
                     gain_percentage_points=None if previous_variance is None else 100 * (model["variance_total"] - previous_variance),
                     narrative=reviews.get(q, {}).get("narrative", {}),
                     candidate_review=reviews.get(q, {}).get("candidate_review", []))
        previous_variance = model["variance_total"]
        # Full semantic candidate profiles remain in each original metadata file.
        model["cluster_candidates"] = [{key: value for key, value in item.items() if key != "semantic_profiles"}
                                       for item in model["cluster_candidates"]]
        model["artifacts"] = {key: f"models/q{q}/" + value for key, value in model["artifacts"].items()}
        if q == 2:
            continue
        preferred = int(reviews[q]["preferred_k"])
        selected_folder = folder if preferred == 3 else folder / f"selected_k{preferred}"
        selected = metadata if preferred == 3 else reviewed_metadata(
            read_json(selected_folder / "metadata.json"), reviews[q], "preferred_clusters")
        if selected["factor_count"] != q or selected["cluster_count"] != preferred:
            raise ValueError("The selected model files disagree with the reviewed q/k")
        selected["cluster_selection_reason"] = reviews[q]["selection_reason"]
        selected["diagnostics"]["cluster_selection"].update(
            selection_authority="assistant_review_of_all_computed_candidate_profiles",
            content_recommended_k=preferred)
        write_json(selected_folder / "metadata.json", selected)
        quality = next(item for item in selected["cluster_candidates"] if item["k"] == preferred)
        comparison["selected_k"].append({
            "q": q, "k": preferred, "path": selected_folder.relative_to(output).as_posix(),
            "selection_reason": reviews[q]["selection_reason"],
            "selection_authority": "assistant_review_of_all_computed_candidate_profiles",
            "clusters": selected["clusters"], "quality": {"silhouette": quality["silhouette"], **quality["stability"]},
            "relative_to_q2": partition_overlap(output / "models/q2/cluster_labels.csv", selected_folder / "cluster_labels.csv"),
            "relative_to_fixed_k3": partition_overlap(folder / "cluster_labels.csv", selected_folder / "cluster_labels.csv")})
    # Candidate details are kept in model-specific metadata; compact comparison
    # avoids multiplying large, repeated semantic profile tables in every export.
    for candidate_group in comparison["numerical_free_k"]:
        candidate_group["candidates"] = [{key: value for key, value in item.items() if key != "semantic_profiles"}
                                         for item in candidate_group["candidates"]]
    write_json(output / "models/comparison_interpreted.json", comparison)
    primary["model_comparison"] = comparison
    return primary, comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    store = StudyStore(ROOT / "studies")
    study = store.require_approved(IDENTIFIER)
    directory = store.directory(IDENTIFIER)
    output = directory / "output"
    ratings_path = directory / "ratings.json"
    ratings = read_json(ratings_path)
    cache = study.get("ratings_cache", {})
    if cache.get("basis") != ratings_basis(study) or cache.get("ratings_sha256") != fingerprint(ratings):
        raise ValueError("The saved rating receipt does not match the approved inputs")
    comparison = read_json(output / "models/comparison.json")
    if hashlib.sha256(ratings_path.read_bytes()).hexdigest() != comparison["scope"]["ratings_sha256"]:
        raise ValueError("Ratings changed since fitting the comparison")
    if hashlib.sha256((output / "matrix.csv").read_bytes()).hexdigest() != comparison["scope"]["matrix_sha256"]:
        raise ValueError("The primary analysis rectangle changed")
    review = read_json(REVIEW)
    print(json.dumps({"ready": True, "apply": args.apply, "models": [2, 3, 4], "ratings": len(ratings),
                      "reviewed_alternatives": [item["q"] for item in review["models"]]}, ensure_ascii=False), flush=True)
    if not args.apply:
        return
    from src.semantic_service.server import Application
    from src.semantic_service.reports import export_report
    Application.configure_export_runtime()
    analysis, comparison = build_comparison(output, review)
    write_json(output / "metadata.json", analysis)
    store.update(IDENTIFIER, analysis=analysis, status="running", error=None)
    store.log(IDENTIFIER, "Сравнение q=2/3/4 по сохранённым оценкам, подготовка отчёта и слайдов", "export", 0, 1)
    artifacts = export_report(store.get(IDENTIFIER), analysis, ratings, output, ROOT)
    store.update(IDENTIFIER, analysis=analysis, status="complete", artifacts=artifacts, error=None)
    store.log(IDENTIFIER, "Сравнение факторов и групп завершено; отчёт и слайды обновлены", "complete", 1, 1)
    print(json.dumps({"complete": True, "preferred_k": {item["q"]: item["k"] for item in comparison["selected_k"]}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
