"""Rebuild source-reviewed 42-person V2 profiles without API calls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.corpus_pipeline.source_rating_update import attach_factor_comparison, build_profiles


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dictionary", type=Path, default=ROOT / "research/rwanda_politicians_2021_2026/source_scales_v2/approved_scales.json")
    parser.add_argument("--study", type=Path, default=ROOT / "studies/rwanda-politicians-2021-2026/study.json")
    parser.add_argument("--review", type=Path, default=ROOT / "reports/rwanda_chat_combined_review_2026_10_18.json")
    parser.add_argument("--extension-review", type=Path, action="append", default=[])
    parser.add_argument("--historical", type=Path, default=ROOT / "studies/rwanda-politicians-2021-2026/output/factor_preview.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/rwanda_source_reanalysis_2026_10_18.json")
    args = parser.parse_args(argv)
    dictionary, study, historical = read(args.dictionary), read(args.study), read(args.historical)
    bundles = [read(path) for path in [args.review, *args.extension_review]]
    result = build_profiles(study["objects"], dictionary, bundles)
    attach_factor_comparison(result, study["objects"], dictionary, historical)
    result["provenance"] = {
        "approved_dictionary_file_sha256": hashlib.sha256(args.dictionary.read_bytes()).hexdigest(),
        "validated_review_files_sha256": [hashlib.sha256(path.read_bytes()).hexdigest()
                                         for path in [args.review, *args.extension_review]],
        "historical_factor_preview_file_sha256": hashlib.sha256(args.historical.read_bytes()).hexdigest(),
        "historical_study_file_sha256": hashlib.sha256(args.study.read_bytes()).hexdigest()}
    corrections = ROOT / "reports/rwanda_corrected_evidence_2026_10_18.json"
    if corrections.exists():
        result["source_corrections"] = read(corrections)
        result["provenance"]["source_corrections_file_sha256"] = hashlib.sha256(corrections.read_bytes()).hexdigest()
    result["measurement_scope"] = "Only separately reviewed source episodes create scores; the general pass covers all facts but does not semantically rate all unreviewed material."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "counts": result["counts"], "scenarios": {
        key: {"counts": value["counts"], "factor_admission": value["factor_admission"]["status"]}
        for key, value in result["scenarios"].items()}, "api_calls": 0, "new_api_cost_usd": 0}, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
