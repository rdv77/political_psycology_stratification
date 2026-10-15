"""Apply a reviewed, still-unapproved dictionary while preserving its auto draft."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_rwanda_scales import write_proposal
from src.semantic_service.store import BUSY, StudyStore, fingerprint, now, validate_required_scales, validate_scales

IDENTIFIER = "rwanda-politicians-2021-2026"
RESEARCH = ROOT / "research/rwanda_politicians_2021_2026"


def main():
    review_file = RESEARCH / "scales_reviewed_proposal.json"
    proposal = json.loads(review_file.read_text(encoding="utf-8-sig"))
    scales = validate_scales(proposal["scales"] if isinstance(proposal, dict) else proposal)
    store = StudyStore(ROOT / "studies")
    study = store.get(IDENTIFIER)
    if study["status"] in BUSY or not store.is_approved(study, "objects"):
        raise ValueError("Список должен быть утверждён, а другой этап завершён")
    validate_required_scales(dict(study, scales=scales))
    if store.is_approved(study, "scales"):
        if fingerprint(study["scales"]) != fingerprint(scales):
            raise ValueError("Нельзя перезаписывать утверждённые шкалы новым предложением")
        print("Reviewed dictionary already approved; unchanged")
        return
    draft_file = RESEARCH / "scales_automatic_draft.json"
    if not draft_file.exists():
        draft_file.write_text(json.dumps({"status": "superseded_automatic_draft_do_not_rate",
                                          "scales": study["scales"]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    provenance = study.get("scale_provenance", {})
    provenance.update(review_status="manual_reviewed_unapproved", manual_review={
        "actor": "research_assistant_source_review_not_independent_human_expert", "at": now(),
        "proposal_file": review_file.relative_to(ROOT).as_posix(), "proposal_fingerprint": fingerprint(scales),
        "review_note": "research/rwanda_politicians_2021_2026/SCALE_REVIEW.md",
        "automatic_draft_file": draft_file.relative_to(ROOT).as_posix(),
        "notes": ["Отраслевые темы и справки о должностях заменены переносимыми политическими контрастами по материалам объектов.",
                  "Отсутствие темы и молчание исключены как противоположные полюса. Недостаточное покрытие нельзя исправлять догадкой.",
                  "Сохранены точные определения шести обязательных тем России, без повышающего веса и фиктивного числа подтверждений.",
                  "Это проект словаря до утверждения; наличие шкалы не является наблюдаемой оценкой каждого объекта."]})
    provenance["normalization_notes"] = provenance["manual_review"]["notes"]
    store.edit(IDENTIFIER, "scales", scales)
    store.update(IDENTIFIER, scale_provenance=provenance)
    write_proposal(scales, provenance)
    print(json.dumps({"scales": len(scales), "status": store.get(IDENTIFIER)["status"],
                      "approved": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
