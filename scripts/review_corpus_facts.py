"""Offline, conservative annotations for planned and third-party action facts.

Default is a private proposal only. --apply appends audit annotations through
CorpusPipeline.review_facts; model receipts and raw facts are never rewritten.
Lexical matches are review gates, not complete verification of source truth.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.corpus_pipeline.service import CorpusPipeline, digest

REVIEWER = "local_rule_v2"
POLICY = "draft_temporal_thirdparty_actions_v2"
_PLANNED = re.compile(
    r"\b(?:scheduled\s+to|plans?\s+to|planning\s+to|will\s+(?:travel|visit)|is\s+due\s+to|"
    r"se\s+rendra|pr[eé]voit(?:\s+de)?)\b", re.IGNORECASE)
_ANNOUNCED = re.compile(
    r"\b(?:announced|announces|announcing|has\s+announced|a\s+annonc[eé]|annonce|"
    r"annonc[eé]|a\s+annonc[eé]\s+son)\b", re.IGNORECASE)
_PASSIVE_CLAIM = re.compile(
    r"\b(?:(?:was|were|is|are|has\s+been|have\s+been|had\s+been)\s+"
    r"(?:(?:publicly|formally|allegedly|reportedly|also|previously)\s+){0,3}"
    r"(?:accused|alleged|named|identified|sanctioned|convicted)|"
    r"(?:experts|investigators|prosecutors|authorities)\s+(?:have\s+)?(?:named|identified|accused)|"
    r"(?:a\s+[eé]t[eé]|est|[eé]tait)\s+(?:accus[eé]|condamn[eé]|sanctionn[eé])"
    r")\b", re.IGNORECASE)


def build_proposals(facts, jobs):
    """Pure, non-upgrading rules over current merged facts and source contexts."""
    jobs_by_id = {job["id"]: job for job in jobs}
    decisions, codes, source_types = [], Counter(), Counter()
    action_count = 0
    for fact in facts:
        if fact.get("kind") != "action":
            continue
        action_count += 1
        job = jobs_by_id.get(fact.get("extraction_job_id"), {})
        context = job.get("context") or {}
        if isinstance(context, str):
            context = json.loads(context)
        document = context.get("document", {})
        source_type = str(document.get("source_type", "unknown")).strip().casefold()
        source_types[source_type] += 1
        claim = str(fact.get("claim", ""))
        annotation = None
        if _PLANNED.search(claim) and not _ANNOUNCED.search(claim):
            code = "planned_not_completed_action"
            annotation = {"kind": "institutional_context", "attribution_basis": "context",
                          "measurement_status": "context_only", "temporal_status": "planned",
                          "review_note": "Запись сообщает запланированное действие, а не его совершение. "
                                         "Явное уже состоявшееся объявление самим политиком в формулировке действия не выделено; "
                                         "не использовать план как совершённое собственное действие."}
        elif source_type != "article":
            code = "unverified_third_party_action"
            annotation = {"kind": "attributed_image", "attribution_basis": "third_party_claim",
                          "review_note": "Действие передано постом/комментарием/репостом либо источником неустановленного типа. "
                                         "В этой записи первичное подтверждение действия не установлено; сохраняется как стороннее "
                                         "приписывание для анализа образа, без повышения до собственной позиции."}
        elif _PASSIVE_CLAIM.search(claim):
            code = "accusation_or_external_decision_not_own_action"
            annotation = {"kind": "attributed_image", "attribution_basis": "third_party_claim",
                          "review_note": "Политик выступает объектом обвинения, экспертного приписывания, санкции или судебного решения. "
                                         "Такая запись сама по себе не подтверждает собственного политического действия; "
                                         "сохранить стороннее приписывание и его источник."}
        if annotation:
            decisions.append({"fact_id": fact["id"], "annotation": annotation})
            codes[code] += 1
    return {"policy": POLICY, "reviewer": REVIEWER, "input_facts_sha256": digest(facts),
            "checked_active_facts": len(facts), "checked_action_facts": action_count,
            "annotations_proposed": len(decisions), "rule_counts": dict(sorted(codes.items())),
            "action_source_types": dict(sorted(source_types.items())), "decisions": decisions,
            "limitations": ["Lexical review only: unchanged facts have not been established as true.",
                            "English/French claim patterns are incomplete; no document language is excluded.",
                            "An announcement may be an accomplished communicative action while the announced journey remains planned.",
                            "Third-party images remain usable only with their attribution; they are not own positions.",
                            "No new score, date, private motive or event is inferred; raw facts and model receipts remain intact."]}


def review(root, corpus_id, *, apply=False):
    root = Path(root).resolve()
    if not re.fullmatch(r"corpus-[A-Za-z0-9_-]+", corpus_id):
        raise ValueError("Invalid corpus ID")
    if not (root / ".corpus_runtime/corpus.sqlite3").is_file():
        raise FileNotFoundError("Corpus database does not exist")
    pipeline = CorpusPipeline(root)
    corpus = pipeline._row(corpus_id)
    facts = pipeline._facts(corpus_id, corpus["config"])
    with pipeline.connection() as db:
        jobs = [dict(row) for row in db.execute("SELECT id,context FROM jobs WHERE corpus_id=?", (corpus_id,))]
    proposal = build_proposals(facts, jobs)
    proposal.update(corpus_id=corpus_id, source_sha256=corpus["source_sha256"],
                    apply_requested=bool(apply), api_calls=0, model_receipts_modified=False)
    if apply:
        proposal["application"] = pipeline.review_facts(corpus_id, proposal["decisions"], reviewer=REVIEWER, policy=POLICY)
    destination = root / ".corpus_runtime/pilots" / corpus_id / "full-v2/fact_review_proposals.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(proposal, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return {"proposal_path": str(destination), "checked_active_facts": proposal["checked_active_facts"],
            "checked_action_facts": proposal["checked_action_facts"], "annotations_proposed": proposal["annotations_proposed"],
            "rule_counts": proposal["rule_counts"], "apply_requested": bool(apply),
            "application": proposal.get("application"), "api_calls": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--apply", action="store_true", help="Append annotations locally; never calls an API")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(review(args.root, args.corpus_id, apply=args.apply), ensure_ascii=False, allow_nan=False))
    except (ValueError, KeyError, FileNotFoundError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
