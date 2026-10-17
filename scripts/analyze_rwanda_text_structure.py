"""Build a separate, exploratory LSA from saved corrected extraction results.

Does not call APIs, create V2 intensity scores, change FA gates or update the
original corpus/study. Source-bearing feature matrices stay in ignored runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.corpus_pipeline.scoring_preparation import digest
from src.corpus_pipeline.text_structure import analyze


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(root=ROOT, *, publish=False):
    root = Path(root)
    source = root / ".corpus_runtime/corrected_evidence_2026_10_18/facts_private.json"
    facts = read(source)
    manifest = read(source.parent / "manifest_private.json")
    if digest(facts) != manifest["derived_facts_content_sha256"]:
        raise ValueError("Corrected evidence differs from frozen correction manifest")
    study_path = root / "studies/rwanda-politicians-2021-2026/study.json"
    study = read(study_path)
    aliases_path = root / "research/rwanda_politicians_2021_2026/corpus_search/actor_aliases.json"
    result = analyze(facts, study["objects"], aliases=read(aliases_path), config={"aspect": "public_positions"})
    result["public"]["provenance"] = {
        "corrected_facts_file_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "correction_policy_version": manifest["policy_version"],
        "roster_study_file_sha256": hashlib.sha256(study_path.read_bytes()).hexdigest(),
        "actor_aliases_file_sha256": hashlib.sha256(aliases_path.read_bytes()).hexdigest(),
        "source_publication_period": {"from": "2025-10-01", "to": "2026-10-17"},
        "unknown_event_time_included_as_text_only": True,
        "canonical_study_modified": False, "corpus_database_modified": False,
    }
    private = root / ".corpus_runtime/text_structure_2026_10_18"
    write(private / "analysis_private.json", result["private"])
    write(private / "analysis_public.json", result["public"])
    if publish:
        write(root / "reports/rwanda_text_structure_2026_10_18.json", result["public"])
    return {"counts": result["public"]["input_counts"],
            "solutions": result["public"]["baseline"]["solutions"],
            "api_calls": 0, "new_api_cost_usd": 0, "published": publish,
            "is_factor_analysis_of_approved_scales": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    result = run(args.root, publish=args.publish)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
