"""Attach a pinned, separately checked source layer without rewriting retrieval."""
from __future__ import annotations

import copy
import hashlib
import json
import re
import uuid
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit

from .store import fingerprint


def _iso_date(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("Source supplement requires an exact ISO date")
    return date.fromisoformat(value)


def attach_source_supplement(study, obj, dossier, root, evidence_dir) -> dict:
    config = study.get("source_supplement")
    if not config:
        return dossier
    root = Path(root).resolve()
    relative = Path(config["path"])
    source_file = (root / relative).resolve()
    if relative.is_absolute() or not source_file.is_relative_to(root):
        raise ValueError("Source supplement path must stay inside the repository")
    payload = source_file.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != config.get("sha256"):
        raise ValueError("Source supplement SHA256 does not match the pinned file")
    supplement = json.loads(payload.decode("utf-8-sig"))
    rows = supplement.get("objects", [])
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Invalid source supplement objects")
    matches = [row for row in rows if row.get("object_id") == obj["id"]]
    if not matches:
        return dossier
    if len(matches) != 1 or dossier.get("object_id") != obj["id"]:
        raise ValueError("Source supplement object identity mismatch")
    identifier = obj["id"]
    if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", identifier):
        raise ValueError("Invalid source supplement object identifier")
    directory = Path(evidence_dir).resolve()
    if not directory.is_relative_to(root):
        raise ValueError("Reviewed evidence must stay inside the repository")
    row = matches[0]
    sources = row.get("sources", [])
    if not isinstance(sources, list) or any(not isinstance(s, dict) for s in sources):
        raise ValueError("Invalid source supplement sources")
    source_lookup = {s.get("url"): s for s in sources}
    allowed_topics = {s["id"] if isinstance(s, dict) else s
                      for s in study.get("research_requirements", {}).get("required_scales", [])}
    first, last = _iso_date(study["date_from"]), _iso_date(study["date_to"])
    facts, pending = [], []
    candidates = row.get("facts", [])
    if not isinstance(candidates, list):
        raise ValueError("Invalid source supplement facts")
    for raw in candidates:
        if not isinstance(raw, dict):
            raise ValueError("Invalid source supplement fact")
        fact = copy.deepcopy(raw)
        url = fact.get("source_url")
        parts = urlsplit(url) if isinstance(url, str) else None
        if (url not in source_lookup or parts is None or parts.scheme not in {"http", "https"}
                or not parts.hostname or parts.username or parts.password):
            raise ValueError("Supplement fact source_url must match its object's sources")
        topics = fact.get("topic_ids")
        if not isinstance(topics, list) or any(not isinstance(t, str) or t not in allowed_topics for t in topics):
            raise ValueError("Supplement topic_ids must belong to research requirements")
        if not first <= _iso_date(fact.get("period")) <= last:
            raise ValueError("Supplement fact is outside the study period")
        if any(not isinstance(fact.get(k), str) or not fact[k].strip() for k in ("text", "event_id", "capacity")):
            raise ValueError("Supplement fact needs text, event_id and capacity")
        if fact.get("channel") not in {"official_act", "personal_public_rhetoric", "public_statement", "attributed_report"}:
            raise ValueError("Unsupported supplement evidence channel")
        verification = str(source_lookup[url].get("verification", "")).lower()
        if "index" in verification and not any(s in verification for s in ("full_page_read", "full page read", "full relevant post text read", "full statement read")):
            pending.append(dict(fact, status="pending", pending_reason="Index-only source; full document not verified"))
            continue
        if fact["channel"] == "personal_public_rhetoric":
            fact["original_channel"] = fact["channel"]
            fact["channel"] = "public_statement"
            fact["attribution_note"] = "Именное публичное высказывание в указанной роли; не свидетельство частного убеждения."
        fact["evidence_origin"] = "assistant_verified_source"
        fact["source_provenance"] = copy.deepcopy(source_lookup[url])
        facts.append(fact)
    raw_facts_hash = fingerprint(dossier.get("facts", []))
    basis = fingerprint({"raw_basis": dossier.get("basis"), "raw_facts_sha256": raw_facts_hash,
                         "supplement_sha256": digest})
    destination = directory / ("reviewed-" + identifier + ".json")
    if destination.exists():
        saved = json.loads(destination.read_text(encoding="utf-8"))
        if saved.get("basis") == basis and saved.get("object_id") == identifier:
            return saved
    reviewed = copy.deepcopy(dossier)
    known = {(f.get("event_id"), f.get("text"), f.get("source_url")) for f in reviewed.get("facts", [])}
    additions = [f for f in facts if (f["event_id"], f["text"], f["source_url"]) not in known]
    reviewed["facts"] = reviewed.get("facts", []) + additions
    known_urls = {s.get("url") for s in reviewed.get("sources", [])}
    accepted_urls = {f["source_url"] for f in facts}
    reviewed["sources"] = reviewed.get("sources", []) + [copy.deepcopy(s) for s in sources
                           if s.get("url") in accepted_urls and s.get("url") not in known_urls]
    reviewed.update(basis=basis, evidence_file="evidence/" + destination.name,
                    original_dossier_file=dossier.get("evidence_file", "evidence/" + identifier + ".json"),
                    original_dossier_basis=dossier.get("basis"), original_dossier_model=dossier.get("model"),
                    model="mixed_source_provenance", source_supplement_facts=additions,
                    source_supplement_provenance={"path": relative.as_posix(), "sha256": digest,
                        "raw_facts_sha256": raw_facts_hash, "evidence_origin": "assistant_verified_source",
                        "sources": copy.deepcopy(sources), "pending_facts": pending,
                        "added_fact_count": len(additions)})
    directory.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(reviewed, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return reviewed
