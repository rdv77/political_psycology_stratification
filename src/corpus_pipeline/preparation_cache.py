"""Verified local checkpoints for completed offline cleaning, never API work.

The caller binds both the source CSV and every screening option (including
policy versions) into the supplied digests. Source text stays in private JSONL;
the small atomic checkpoint contains only hashes, sizes and aggregate counts.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile


VERSION = "offline-cleaning-checkpoint-v1"
CHECKPOINT = "cleaning_checkpoint.json"
_FILES = ("cleaned_unique.jsonl", "cleaning_index.jsonl")
_FIELDS = ("id", "source_row", "title", "text", "url", "published_at", "source_name", "author", "language_hint")
_ANNOTATION = {"status", "reason", "matched_actor_ids", "flags"}
_STATUSES = {"ready_for_semantic_filter", "needs_semantic_review", "rule_excluded"}


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _annotation(value):
    if not isinstance(value, dict) or set(value) != _ANNOTATION or value["status"] not in _STATUSES:
        raise ValueError("Invalid cleaning annotation")
    if not isinstance(value["reason"], str):
        raise ValueError("Invalid cleaning reason")
    for key in ("matched_actor_ids", "flags"):
        if (not isinstance(value[key], list) or any(not isinstance(v, str) or not v for v in value[key])
                or len(value[key]) != len(set(value[key]))):
            raise ValueError("Invalid cleaning labels")
    return value


def _identity(identifier, row):
    if not isinstance(identifier, str) or not identifier or type(row) is not int or row < 2:
        raise ValueError("Invalid original document identity")


def _read_jsonl(path):
    digest, size, rows = hashlib.sha256(), 0, []
    with path.open("rb") as stream:
        for raw in stream:
            digest.update(raw)
            size += len(raw)
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError("Checkpoint JSONL rows must be objects")
            rows.append(value)
    return rows, {"sha256": digest.hexdigest(), "bytes": size, "rows": len(rows)}


def _read_completed(directory):
    if any((directory / (name + ".tmp")).exists() for name in _FILES):
        raise ValueError("Cleaning generation is not complete")
    cleaned, cleaned_info = _read_jsonl(directory / _FILES[0])
    index, index_info = _read_jsonl(directory / _FILES[1])
    indexed, source_rows, annotations, statuses, kept = {}, set(), [], Counter(), {}
    for row in index:
        identifier, source_row, status = row.get("document_id"), row.get("source_row"), row.get("status")
        _identity(identifier, source_row)
        if identifier in indexed or source_row in source_rows:
            raise ValueError("Repeated document in cleaning index")
        indexed[identifier] = row
        source_rows.add(source_row)
        if status in _STATUSES:
            annotation = _annotation({key: value for key, value in row.items()
                                      if key not in {"document_id", "source_row"}})
            annotations.append(annotation)
            statuses["unique"] += 1
            if status != "rule_excluded":
                kept[identifier] = (source_row, annotation)
        elif status in {"exact_duplicate", "empty"}:
            if set(row) != {"document_id", "source_row", "status", "duplicate_of"}:
                raise ValueError("Invalid imported-status index row")
            statuses[status] += 1
        else:
            raise ValueError("Unknown cleaning index status")
    for row in index:
        if row["status"] == "exact_duplicate":
            representative = indexed.get(row["duplicate_of"])
            if representative is None or representative["status"] not in _STATUSES:
                raise ValueError("Duplicate has no unique representative")
        elif row["status"] == "empty" and row["duplicate_of"] is not None:
            raise ValueError("Empty document cannot have a duplicate representative")
    frames, seen = [], set()
    for row in cleaned:
        if set(row) != set(_FIELDS) | {"cleaning", "source_context"}:
            raise ValueError("Unexpected fields in cleaned document")
        identifier, source_row = row["id"], row["source_row"]
        _identity(identifier, source_row)
        annotation = _annotation(row["cleaning"])
        if identifier in seen or kept.get(identifier) != (source_row, annotation):
            raise ValueError("Cleaned document disagrees with cleaning index")
        if any(not isinstance(row[field], str) for field in _FIELDS if field != "source_row"):
            raise ValueError("Invalid imported document fields")
        context = row["source_context"]
        if (not isinstance(context, dict)
                or set(context) != {"source_type", "source_platform", "author_is_verified_actor"}
                or not isinstance(context["source_type"], str)
                or not isinstance(context["source_platform"], str)
                or type(context["author_is_verified_actor"]) is not bool):
            raise ValueError("Invalid source context")
        document = {field: row[field] for field in _FIELDS}
        document["_source_context"] = context
        frames.append((document, annotation))
        seen.add(identifier)
    if seen != set(kept):
        raise ValueError("Cleaning index and cleaned frame counts disagree")
    info = {"files": {_FILES[0]: cleaned_info, _FILES[1]: index_info},
            "counts": {"raw_records": len(index), "annotated_unique": len(annotations),
                       "frames": len(frames), "import_statuses": dict(sorted(statuses.items()))}}
    return frames, annotations, statuses, info


def create_checkpoint(directory, source_sha256, config_hash):
    """Certify two completed local JSONL files; return the atomic manifest.

    Inconsistent rows or unfinished files raise ValueError; no checkpoint is
    written. This function does not query SQLite or change either JSONL file.
    """
    if not _digest(source_sha256) or not _digest(config_hash):
        raise ValueError("Source and configuration must have SHA256 digests")
    directory = Path(directory)
    *_, info = _read_completed(directory)
    manifest = {"version": VERSION, "source_sha256": source_sha256,
                "config_hash": config_hash, **info}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory,
                                         prefix=CHECKPOINT + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, directory / CHECKPOINT)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return manifest


def load_checkpoint(directory, source_sha256, config_hash):
    """Return (frames, all unique annotations, import-status Counter), or None.

    Missing/stale/corrupted checkpoints are cache misses. Hashes are recomputed
    while reading the exact rows returned; counts and cross-file IDs must agree.
    ``cleaning`` never becomes part of a restored document's model payload.
    """
    try:
        if not _digest(source_sha256) or not _digest(config_hash):
            return None
        directory = Path(directory)
        manifest = json.loads((directory / CHECKPOINT).read_text(encoding="utf-8"))
        if (not isinstance(manifest, dict) or manifest.get("version") != VERSION
                or manifest.get("source_sha256") != source_sha256 or manifest.get("config_hash") != config_hash):
            return None
        frames, annotations, statuses, info = _read_completed(directory)
        if manifest.get("files") != info["files"] or manifest.get("counts") != info["counts"]:
            return None
        return frames, annotations, statuses
    except (OSError, ValueError, TypeError, KeyError, UnicodeError):
        return None
