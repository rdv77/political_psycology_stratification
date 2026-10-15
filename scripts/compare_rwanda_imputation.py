"""Offline CLI; only writes separate imputation-review JSON/Markdown outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.semantic_service.imputation_review import review_imputation  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_dir", nargs="?", type=Path,
                        default=ROOT / "studies/rwanda-politicians-2021-2026")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output-stem", default="imputation_review_2026-10-15")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--mask-fraction", type=float, default=.12)
    parser.add_argument("--posterior-draws", type=int, default=10)
    parser.add_argument("--q", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--seed", type=int, default=20261015)
    args = parser.parse_args()
    study_dir = args.study_dir.resolve()
    paths = [study_dir / "study.json", study_dir / "ratings.json"]
    content = [path.read_bytes() for path in paths]
    hashes = {str(path): hashlib.sha256(data).hexdigest() for path, data in zip(paths, content)}
    study, ratings = [json.loads(data.decode("utf-8-sig")) for data in content]
    if isinstance(ratings, dict):
        ratings = ratings.get("ratings", ratings.get("rows"))
    output_dir = (args.output_dir or study_dir / "output").resolve()
    result = review_imputation(study, ratings, output_dir, repeats=args.repeats,
                               mask_fraction=args.mask_fraction, posterior_draws=args.posterior_draws,
                               q_values=tuple(args.q), seed=args.seed, output_stem=args.output_stem)
    result["file_input_receipt"] = {"sha256": hashes, "unchanged_during_review": all(
        hashlib.sha256(path.read_bytes()).hexdigest() == hashes[str(path)] for path in paths)}
    preview_path = study_dir / "output/factor_preview.json"
    if result["measurement_status"] == "admitted_real_observations" and preview_path.exists():
        preview = json.loads(preview_path.read_text(encoding="utf-8"))
        same_ids = (preview.get("diagnostics", {}).get("training_object_order") == result["object_ids"] and
                    preview.get("diagnostics", {}).get("trait_order") == result["variable_ids"])
        checks = []
        if same_ids:
            for model in result["factor_sensitivity"]["models"]:
                saved = next((x for x in preview.get("models", []) if x["q"] == model["q"]), None)
                if saved is None:
                    continue
                candidate = model["deterministic"][0]
                q = model["q"]
                a = np.array([[x["loadings"][f"F{j+1}"] for j in range(q)] for x in saved["loading_table"]])
                b = np.array([[x["loadings"][f"F{j+1}"] for j in range(q)] for x in candidate["aligned_loadings"]])
                checks.append({"q": q, "max_absolute_loading_difference": float(np.max(np.abs(a-b))),
                               "variance_total_difference": float(candidate["variance_total"] - saved["variance_total"])})
        result["existing_preview_comparison"] = {
            "path": str(preview_path), "sha256": hashlib.sha256(preview_path.read_bytes()).hexdigest(),
            "same_objects_and_variables": same_ids, "checks": checks,
            "note": "A later collection may change the admitted matrix. Old previews are not overwritten or silently treated as the new baseline."}
    output_path = output_dir / (args.output_stem + ".json")
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"measurement_status": result["measurement_status"],
                      "n_objects": result.get("n_objects"), "n_variables": result.get("n_variables"),
                      "validation": result.get("validation", {}).get("summary") if result.get("validation") else None,
                      "source_files_unchanged": result["file_input_receipt"]["unchanged_during_review"],
                      "output": str(output_path)}, ensure_ascii=False))
    if not result["file_input_receipt"]["unchanged_during_review"]:
        raise RuntimeError("Input files changed concurrently; this receipt describes the initial snapshot. Rerun after collection is stable.")


if __name__ == "__main__":
    main()
