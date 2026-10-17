"""Validate chat reviews and optionally publish paraphrases, never raw evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.corpus_pipeline.chat_review import apply_adjudications, compile_review, render
from src.corpus_pipeline.scoring_preparation import digest, packed

PUBLIC_JSON = "reports/rwanda_chat_semantic_review_2026_10_17.json"
PUBLIC_MD = "research/rwanda_politicians_2021_2026/source_scales_v2/CHAT_REVIEW_2026-10-17.md"


def run(root, directory, dictionary_path, *, publish=False):
    root, directory, dictionary_path = (Path(p).resolve() for p in (root, directory, dictionary_path))
    if not directory.is_relative_to(root / ".corpus_runtime"):
        raise ValueError("Private reviews must remain in ignored runtime")
    manifest = json.loads((directory / "manifest_private.json").read_text(encoding="utf-8-sig"))
    dictionary_raw = dictionary_path.read_bytes()
    dictionary = json.loads(dictionary_raw.decode("utf-8-sig"))
    cases = [json.loads((directory / "cases" / (row["case_id"] + ".json")).read_text(encoding="utf-8-sig"))
             for row in manifest["cases"]]
    for row, case in zip(manifest["cases"], cases):
        if digest(case) != row["sha256"]:
            raise ValueError("Source dossier changed after review preparation")
    if digest(dictionary) != manifest["dictionary_content_sha256"]:
        raise ValueError("Approved dictionary differs from dossier preparation")
    reviews = [json.loads((directory / name).read_text(encoding="utf-8-sig"))
               for name in ("reviews_a.json", "reviews_b.json", "reviews_c.json", "reviews_controls.json")]
    originals_sha = digest(reviews)
    amendment_path = directory / "adjudications_private.json"
    amendments = json.loads(amendment_path.read_text(encoding="utf-8-sig"))["amendments"] if amendment_path.exists() else []
    reviews, changes = apply_adjudications(reviews, amendments)
    result = compile_review(cases, dictionary, reviews)
    public = result["public"]
    public["calibration_changes"] = changes
    public["provenance"] = {"dictionary_file_sha256": hashlib.sha256(dictionary_raw).hexdigest(),
                            "source_package_file_sha256": manifest["source_package_file_sha256"],
                            "case_manifest_file_sha256": hashlib.sha256((directory / "manifest_private.json").read_bytes()).hexdigest(),
                            "original_review_submissions_sha256": originals_sha, "calibration_sha256": digest(amendments)}
    public_raw = (packed(public) + "\n").encode("utf-8")
    markdown_raw = render(public).encode("utf-8")
    validation_raw = (packed({"validation": result["private_validation"],
                              "public_sha256": hashlib.sha256(public_raw).hexdigest(),
                              "api_calls": 0}) + "\n").encode("utf-8")
    by_review = {r["case_id"]: r for file in reviews for r in file["cases"]}
    annotations = [{"case_id": c["case_id"], "source_fact_ids": [f["id"] for f in c["member_facts"]],
                    "original_actor_id": c["actor_id"], "reviewed_attribution": by_review[c["case_id"]]["attribution"],
                    "reviewed_event_date": by_review[c["case_id"]]["event_date"],
                    "verdict": by_review[c["case_id"]]["verdict"], "original_source_modified": False,
                    "applied_to_live_corpus": False, "canonical_study_modified": False} for c in cases]
    for name, raw in (("validated_public_review.json", public_raw),
                      ("validated_public_review.md", markdown_raw), ("validation_private.json", validation_raw),
                      ("reviewed_fact_annotations_private.json", (packed({"annotations": annotations, "api_calls": 0}) + "\n").encode("utf-8"))):
        (directory / name).write_bytes(raw)
    if publish:
        for relative, raw in ((PUBLIC_JSON, public_raw), (PUBLIC_MD, markdown_raw)):
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
    return {"counts": public["counts"], "selected_verdicts": public["selected_verdicts"],
            "control_verdicts": public["control_verdicts"], "published": publish,
            "api_calls": 0, "new_api_cost_usd": 0, "canonical_study_modified": False}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--directory", type=Path, default=ROOT / ".corpus_runtime/semantic_review_2026_10_17")
    parser.add_argument("--dictionary", type=Path, default=ROOT / "research/rwanda_politicians_2021_2026/source_scales_v2/approved_scales.json")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.root, args.directory, args.dictionary, publish=args.publish), ensure_ascii=False, indent=2))
        return 0
    except (ValueError, KeyError, TypeError, OSError):
        # Raw exceptions can contain source quotes. Details belong in private QA.
        print(json.dumps({"error": "Review validation failed; inspect local reviews. No API was called.", "api_calls": 0}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
