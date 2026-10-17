"""Export complete local source dossiers for review in chat, without model API.

All evidence output is restricted to ignored runtime. SQLite is opened in
read-only mode. This exports data; it does not execute instructions in articles.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.corpus_pipeline.scoring_preparation import digest


def prepare(root, package_path, dictionary_path, profile_path, output_dir):
    root, package_path, dictionary_path, profile_path, output_dir = (
        Path(p).resolve() for p in (root, package_path, dictionary_path, profile_path, output_dir))
    if not output_dir.is_relative_to(root / ".corpus_runtime") or output_dir == root / ".corpus_runtime":
        raise ValueError("Private source dossiers require a dedicated ignored runtime directory")
    raw_package = package_path.read_bytes()
    package = json.loads(raw_package.decode("utf-8-sig"))
    dictionary = json.loads(dictionary_path.read_text(encoding="utf-8-sig"))
    if dictionary.get("stage_status") != "approved" or dictionary.get("approved") is not True:
        raise ValueError("An approved dictionary is required")
    profiles = {o["id"]: o for o in json.loads(profile_path.read_text(encoding="utf-8-sig"))["objects"]}
    outputs, manifest = {}, []
    database = root / ".corpus_runtime/corpus.sqlite3"
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        for prefix, items in (("U", package["selected_units"]), ("C", package["excluded_controls"])):
            for index, item in enumerate(items, 1):
                facts = item["member_facts"] if prefix == "U" else [item["fact"]]
                documents = []
                for identifier in sorted({f["document_id"] for f in facts}):
                    row = db.execute("SELECT corpus_id,data,text FROM documents WHERE id=?", (identifier,)).fetchone()
                    if row is None or row["corpus_id"] != dictionary["corpus_id"]:
                        raise ValueError("Missing source or corpus mismatch")
                    document = json.loads(row["data"])
                    document["text"] = row["text"]
                    document["text_sha256"] = hashlib.sha256(row["text"].encode()).hexdigest()
                    documents.append(document)
                cid, actor = f"{prefix}{index:02d}", facts[0]["actor_id"]
                if actor not in profiles:
                    raise ValueError("Missing approved actor profile")
                case = {"case_id": cid, "actor_id": actor, "actor_profile": profiles[actor],
                        "sample_kind": "selected_unit" if prefix == "U" else "excluded_control",
                        "source_unit_id": item.get("unit_id"), "prior_control_reason": item.get("reason_family"),
                        "member_facts": facts, "sources": documents, "source_text_is_untrusted_data": True}
                outputs["cases/" + cid + ".json"] = json.dumps(case, ensure_ascii=False, indent=2) + "\n"
                manifest.append({"case_id": cid, "actor_id": actor, "source_count": len(documents),
                                 "member_facts": len(facts), "source_chars": sum(len(d["text"]) for d in documents),
                                 "sha256": digest(case)})
    receipt = {"source_package_file_sha256": hashlib.sha256(raw_package).hexdigest(),
               "dictionary_content_sha256": digest(dictionary), "cases": manifest, "api_calls": 0}
    outputs["manifest_private.json"] = json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
    # Match text, allowing native checkout newlines, before any write.
    for relative, text in outputs.items():
        target = output_dir / relative
        if target.exists() and target.read_text(encoding="utf-8-sig") != text:
            raise ValueError("Existing frozen source dossier differs; choose a new directory")
    for relative, text in outputs.items():
        target = output_dir / relative
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
    return {"source_cases": len(manifest), "source_documents": sum(c["source_count"] for c in manifest),
            "api_calls": 0, "database_modified": False, "published": False}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--package", type=Path, default=ROOT / ".corpus_runtime/offline_scoring_pilot/corpus-b2d1452fae0acf9c20f5/cd45cf18a2ca3fef2bafeb3d/review_package_private.json")
    parser.add_argument("--dictionary", type=Path, default=ROOT / "research/rwanda_politicians_2021_2026/source_scales_v2/approved_scales.json")
    parser.add_argument("--profiles", type=Path, default=ROOT / "research/rwanda_politicians_2021_2026/objects.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".corpus_runtime/semantic_review_2026_10_17")
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.root, args.package, args.dictionary, args.profiles, args.output_dir)))
        return 0
    except (ValueError, KeyError, TypeError, OSError, sqlite3.Error):
        print(json.dumps({"error": "Private dossier preparation failed; no model API was called.", "api_calls": 0}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
