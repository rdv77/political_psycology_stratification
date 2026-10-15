"""Frozen-input, offline Ward candidate comparison. No canonical files changed."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.semantic_service.analysis import SEED
from src.semantic_service.ward_review import render_ward_review, review_ward_q3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=ROOT / "studies/rwanda-politicians-2021-2026")
    parser.add_argument("--frozen-inputs", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--subsample-repeats", type=int, default=100)
    parser.add_argument("--imputation-draws", type=int, default=10)
    parser.add_argument("--k", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    output = (args.output_dir or args.study_dir / "output").resolve()
    names = ("source_study.json", "source_ratings.json", "source_factor_preview.json")
    paths = ([args.frozen_inputs / name for name in names] if args.frozen_inputs else
             [args.study_dir / "study.json", args.study_dir / "ratings.json", args.study_dir / "output/factor_preview.json"])
    blobs = [path.read_bytes() for path in paths]
    snapshots = output / "cluster_candidates_q3"
    snapshots.mkdir(parents=True, exist_ok=True)
    for name, blob in zip(names, blobs):
        target = snapshots / name
        if target.exists() and target.read_bytes() != blob:
            raise ValueError("Frozen source bytes differ. Reproduce with --frozen-inputs or choose another --output-dir.")
        target.write_bytes(blob)
    study, ratings, preview = [json.loads(blob.decode("utf-8-sig")) for blob in blobs]
    result = review_ward_q3(study, ratings, preview, output, k_values=tuple(args.k),
                            subsample_repeats=args.subsample_repeats, imputation_draws=args.imputation_draws, seed=args.seed)
    result["source_file_sha256"] = {str(path.resolve()): hashlib.sha256(blob).hexdigest() for path, blob in zip(paths, blobs)}
    result["reproduction"] = {"script": "scripts/compare_rwanda_ward_q3.py",
        "frozen_input_sha256": {name: hashlib.sha256(blob).hexdigest() for name, blob in zip(names, blobs)},
        "command": 'python -B -X utf8 scripts/compare_rwanda_ward_q3.py --frozen-inputs "studies/rwanda-politicians-2021-2026/output/cluster_candidates_q3" --output-dir ".service_runtime/reproduce_ward_q3"',
        "settings": {"k": args.k, "subsample_repeats": args.subsample_repeats, "imputation_draws": args.imputation_draws, "seed": args.seed},
        "no_APIs": True, "source_data_modified_by_this_script": False}
    result["source_files_changed_concurrently"] = [str(path.resolve()) for path, blob in zip(paths, blobs) if path.read_bytes() != blob]
    result["artifacts"] += [f"cluster_candidates_q3/{name}" for name in names]
    (output / "cluster_candidates_q3.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "cluster_candidates_q3.md").write_text(render_ward_review(result), encoding="utf-8")
    print(json.dumps({"output": str(output), "n_objects": result["n_objects"], "n_variables": result["n_variables"],
        "preview_loading_max_difference": result["factor_model"]["loadings_max_abs_difference_from_preview"],
        "score_sd": result["score_scaling"]["standard_deviation_ddof0"], "selected_k": None,
        "candidates": [{"k": c["k"], "sizes": c["sizes"], "silhouette": c["silhouette"],
            "subsample_ARI_median": c["stability"]["object_subsampling"]["median_adjusted_rand"],
            "imputation_ARI_median": c["stability"]["multiple_imputation"]["median_adjusted_rand"],
            "scaling_ARI": c["stability"]["empirical_axis_standardization"]["adjusted_rand"]} for c in result["candidates"]]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
