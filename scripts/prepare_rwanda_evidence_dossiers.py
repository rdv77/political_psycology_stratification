"""No-API retrieval index over the complete frozen corrected Rwanda facts."""
from __future__ import annotations

import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.corpus_pipeline.evidence_dossiers import build_dossiers
from src.corpus_pipeline.scoring_preparation import digest, filter_eligible_fact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-per-cell", type=int, default=4)
    parser.add_argument("--output", type=Path, default=ROOT / ".corpus_runtime/evidence_dossiers_2026_10_18")
    parser.add_argument("--publish", action="store_true", help="Publish only quote-free aggregate retrieval availability")
    args = parser.parse_args()
    base = ROOT / ".corpus_runtime/corrected_evidence_2026_10_18"
    facts = json.loads((base / "facts_private.json").read_text(encoding="utf-8"))
    manifest = json.loads((base / "manifest_private.json").read_text(encoding="utf-8"))
    if digest(facts) != manifest["derived_facts_content_sha256"]:
        raise ValueError("Frozen corrected facts changed")
    dictionary = json.loads((ROOT / "research/rwanda_politicians_2021_2026/source_scales_v2/approved_scales.json").read_text(encoding="utf-8"))
    database = (ROOT / ".corpus_runtime/corpus.sqlite3").resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(database, uri=True)) as connection:
        row = connection.execute("SELECT config FROM corpora WHERE id=?", (dictionary["corpus_id"],)).fetchone()
    if row is None:
        raise ValueError("Source corpus not found")
    config = json.loads(row[0])
    result = build_dossiers(facts, dictionary, config["objects"], config, max_per_cell=args.max_per_cell)
    study = json.loads((ROOT / "studies/rwanda-politicians-2021-2026/study.json").read_text(encoding="utf-8"))
    study_config = dict(config, date_from=study["date_from"], date_to=study["date_to"])
    result["public_summary"]["scope_comparison"] = {
        "corpus_retrieval_window": {"date_from": config["date_from"], "date_to": config["date_to"]},
        "approved_study_window": {"date_from": study["date_from"], "date_to": study["date_to"]},
        "eligible_to_review_in_corpus_configuration": sum(filter_eligible_fact(f, config) for f in facts),
        "eligible_to_review_in_study_configuration": sum(filter_eligible_fact(f, study_config) for f in facts),
        "note": "The corpus retrieval configuration ends two days later. Eligibility is readiness for review, not verified event timing or a measured value."
    }
    result["config_sha256"] = digest(config)
    result["private_config"] = config
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, content in (("dossiers_private.json", result), ("public_summary.json", result["public_summary"])):
        (args.output / filename).write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.publish:
        (ROOT / "reports/rwanda_evidence_dossiers_2026_10_18.json").write_text(
            json.dumps(result["public_summary"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["public_summary"]["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
