"""Freeze a source review sample offline; contains no API client or paid runner."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.corpus_pipeline.offline_scoring_pilot import build, render
from src.corpus_pipeline.scoring_preparation import packed, prepare


PUBLIC_JSON = "reports/rwanda_offline_scoring_pilot_2026_10_17.json"
PUBLIC_MD = "research/rwanda_politicians_2021_2026/source_scales_v2/OFFLINE_PILOT_2026-10-17.md"


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _inside(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def prepare_files(root, snapshot_path, dictionary_path, output_dir=None, *, publish=False):
    root = Path(root).resolve()
    snapshot_path, dictionary_path = Path(snapshot_path).resolve(), Path(dictionary_path).resolve()
    snapshot_raw, dictionary_raw = snapshot_path.read_bytes(), dictionary_path.read_bytes()
    snapshot = json.loads(snapshot_raw.decode("utf-8-sig"))
    dictionary = json.loads(dictionary_raw.decode("utf-8-sig"))
    prepared = prepare(snapshot, dictionary, encoding=None)
    result = build(snapshot, dictionary, prepared)
    private, public = result["private_package"], result["public_summary"]
    public["provenance"].update(source_snapshot_file_sha256=_sha(snapshot_raw), dictionary_file_sha256=_sha(dictionary_raw))
    raw = (packed(private) + "\n").encode("utf-8")
    public_raw = (packed(public) + "\n").encode("utf-8")
    markdown_raw = render(public).encode("utf-8")
    identifier = snapshot["corpus"].get("id")
    if not isinstance(identifier, str) or not identifier or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in identifier):
        raise ValueError("Invalid corpus identifier")
    if output_dir is None:
        output_dir = root / ".corpus_runtime" / "offline_scoring_pilot" / identifier / public["provenance"]["private_review_package_sha256"][:24]
    directory = Path(output_dir).resolve()
    if not any(_inside(directory, root / name) for name in (".corpus_runtime", ".service_runtime")) or directory in {root / ".corpus_runtime", root / ".service_runtime"}:
        raise ValueError("Private review material requires a dedicated ignored runtime directory")
    receipt = {"policy_version": public["policy_version"], "source_snapshot_file_sha256": _sha(snapshot_raw),
               "dictionary_file_sha256": _sha(dictionary_raw), "private_review_package_file_sha256": _sha(raw),
               "private_review_package_content_sha256": public["provenance"]["private_review_package_sha256"],
               "public_summary_file_sha256": _sha(public_raw), "public_markdown_file_sha256": _sha(markdown_raw),
               "execution_allowed": False, "api_calls": 0, "new_cost_usd": 0}
    outputs = {"review_package_private.json": raw, "review_summary.json": public_raw,
               "review_summary.md": markdown_raw, "package_receipt_private.json": (packed(receipt) + "\n").encode("utf-8")}
    # Check every existing target before writing any file. A changed package
    # needs another directory; an identical rerun never rewrites the package.
    for name, content in outputs.items():
        destination = directory / name
        if destination.exists() and destination.read_bytes() != content:
            raise ValueError("Existing frozen package differs; use a new directory")
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in outputs.items():
        destination = directory / name
        if not destination.exists():
            with destination.open("xb") as stream:
                stream.write(content)
    if publish:
        for relative, content in ((PUBLIC_JSON, public_raw), (PUBLIC_MD, markdown_raw)):
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
    return {"directory": str(directory), "counts": public["counts"], "published": bool(publish),
            "execution_allowed": False, "api_calls": 0, "new_cost_usd": 0,
            "private_review_package_sha256": public["provenance"]["private_review_package_sha256"],
            "queue_modified": False, "corpus_config_modified": False, "scores_generated": False}


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--publish", action="store_true", help="Publish only safe aggregate JSON and Markdown")
    args = parser.parse_args(argv)
    try:
        result = prepare_files(args.root, args.snapshot, args.dictionary, args.output_dir, publish=args.publish)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, KeyError, TypeError, OSError):
        print(json.dumps({"error": "Offline review package rejected invalid input or output; no API was called.",
                          "execution_allowed": False, "api_calls": 0}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
