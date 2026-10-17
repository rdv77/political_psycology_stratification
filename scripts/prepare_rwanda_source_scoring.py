"""Prepare private source units and optional public readiness entirely offline.

Reads an immutable extraction snapshot and a candidate dictionary. No database,
queue, corpus configuration, API key, client or paid-run manifest is opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.corpus_pipeline.excerpts import offline_encoding
from src.corpus_pipeline.scoring_preparation import packed, prepare, render_readiness


CORPUS_ID = "corpus-b2d1452fae0acf9c20f5"
PUBLIC_JSON = "reports/rwanda_source_scoring_preparation_2026_10_17.json"
PUBLIC_MD = "research/rwanda_politicians_2021_2026/SOURCE_SCORING_PREPARATION_2026-10-17.md"


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _inside(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _write_once(path, raw):
    """Identical reruns are harmless; changed content needs a new directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError("Existing preparation differs; use a new private output directory")
        return
    with path.open("xb") as stream:
        stream.write(raw)


def prepare_files(root, snapshot_path, dictionary_path, output_dir=None, *, publish=False, encoding=None):
    root = Path(root).resolve()
    snapshot_path, dictionary_path = Path(snapshot_path).resolve(), Path(dictionary_path).resolve()
    snapshot_raw, dictionary_raw = snapshot_path.read_bytes(), dictionary_path.read_bytes()
    snapshot, dictionary = json.loads(snapshot_raw.decode("utf-8-sig")), json.loads(dictionary_raw.decode("utf-8-sig"))
    result = prepare(snapshot, dictionary, encoding=encoding)
    public = result["public_readiness"]
    public["provenance"] = dict(public["provenance"], source_snapshot_file_sha256=_sha(snapshot_raw),
                                dictionary_file_sha256=_sha(dictionary_raw))
    identifier = snapshot["corpus"].get("id")
    if not isinstance(identifier, str) or not identifier or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in identifier):
        raise ValueError("Corpus ID must be a safe path component")
    if output_dir is None:
        output_dir = root / ".corpus_runtime" / "scoring_preparation" / identifier / (
            result["metadata"]["facts_content_sha256"][:12] + "-" + result["metadata"]["dictionary_content_sha256"][:12])
    directory = Path(output_dir).resolve()
    if not any(_inside(directory, root / name) for name in (".corpus_runtime", ".service_runtime")):
        raise ValueError("Private evidence export must stay inside an ignored local runtime directory")
    if directory in {root / ".corpus_runtime", root / ".service_runtime"}:
        raise ValueError("Use a dedicated private preparation subdirectory")

    unit_raw = "".join(packed(unit) + "\n" for unit in result["private_units"]).encode("utf-8")
    public_raw = (packed(public) + "\n").encode("utf-8")
    markdown_raw = render_readiness(public).encode("utf-8")
    receipt = {**result["metadata"],
               "source_snapshot_file_sha256": _sha(snapshot_raw), "dictionary_file_sha256": _sha(dictionary_raw),
               "private_source_snapshot": str(snapshot_path), "private_dictionary_source": str(dictionary_path),
               "private_artifacts": {"evidence_units_private.jsonl": {"sha256": _sha(unit_raw), "bytes": len(unit_raw),
                                                                         "rows": len(result["private_units"])},
                                     "readiness_for_review.json": {"sha256": _sha(public_raw), "bytes": len(public_raw)},
                                     "readiness_for_review.md": {"sha256": _sha(markdown_raw), "bytes": len(markdown_raw)}}}
    # No side effects have occurred before all validation and serialization.
    outputs = {"evidence_units_private.jsonl": unit_raw, "readiness_for_review.json": public_raw,
               "readiness_for_review.md": markdown_raw,
               "preparation_private.json": (packed(receipt) + "\n").encode("utf-8")}
    for name, raw in outputs.items():
        destination = directory / name
        if destination.exists() and destination.read_bytes() != raw:
            raise ValueError("Existing preparation differs; use a new private output directory")
    for name, raw in outputs.items():
        _write_once(directory / name, raw)
    if publish:
        # Only allowlisted aggregate data crosses into version-controlled paths.
        for relative, raw in ((PUBLIC_JSON, public_raw), (PUBLIC_MD, markdown_raw)):
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw)
    return {"directory": str(directory), "counts": public["counts"],
            "source_snapshot_file_sha256": _sha(snapshot_raw), "dictionary_file_sha256": _sha(dictionary_raw),
            "published": bool(publish), "public_json": PUBLIC_JSON if publish else None,
            "public_markdown": PUBLIC_MD if publish else None,
            "api_calls": 0, "new_cost_usd": 0, "scores_generated": False,
            "queue_modified": False, "corpus_config_modified": False,
            "canonical_study_modified": False, "source_receipts_modified": False}


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--publish", action="store_true", help="Publish only the safe aggregate JSON and Markdown")
    parser.add_argument("--without-tokenizer", action="store_true", help="Report bytes and unknown token counts")
    args = parser.parse_args(argv)
    try:
        encoding = None if args.without_tokenizer else offline_encoding(args.root / ".corpus_runtime" / "tokenizer_cache")
        result = prepare_files(args.root, args.snapshot, args.dictionary, args.output_dir,
                               publish=args.publish, encoding=encoding)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, KeyError, TypeError, OSError):
        # Do not echo potentially private JSON, source text or exception values.
        print(json.dumps({"error": "Offline preparation rejected invalid input or output; no API was called.", "api_calls": 0}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
