"""Import the Rwanda protocol as an unapproved draft; never invent a human receipt."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.semantic_service.store import BUSY, StudyStore, fingerprint, validate_objects

STUDY_ID = "rwanda-politicians-2021-2026"
RESEARCH = ROOT / "research/rwanda_politicians_2021_2026"


def main():
    roster = json.loads((RESEARCH / "objects.json").read_text(encoding="utf-8-sig"))
    topic = json.loads((RESEARCH / "russia_scales_proposal.json").read_text(encoding="utf-8-sig"))
    spec = dict(roster["study"])
    spec["title"] = "Политическое пространство Руанды, 2021–2026"
    if topic["period"] != {key: spec[key] for key in ("date_from", "date_to")}:
        raise ValueError("Периоды списка и тематического протокола должны совпадать")
    store = StudyStore(ROOT / "studies")
    objects = validate_objects(roster["objects"])
    supplement = RESEARCH / "russia_evidence_seed.json"
    sources = {"path": supplement.relative_to(ROOT).as_posix(),
               "sha256": hashlib.sha256(supplement.read_bytes()).hexdigest(),
               "reviewed_thematic_only": True,
               "verification_note": "Тематические оценки допускаются по отдельно просмотренным источникам ассистента; это не независимая человеческая экспертиза"}
    if (store.directory(STUDY_ID) / "study.json").exists():
        study = store.get(STUDY_ID)
        if fingerprint(study["objects"]) != fingerprint(objects):
            raise ValueError("Исследование уже существует с другим составом; автоматическая перезапись запрещена")
        if not study.get("source_supplement"):
            if study["status"] in BUSY or store.is_approved(study, "scales") or (store.directory(STUDY_ID) / "ratings.json").exists():
                raise ValueError("Нельзя менять источник измерения во время работы или после утверждения шкал")
            study = store.update(STUDY_ID, source_supplement=sources)
        elif study["source_supplement"] != sources:
            raise ValueError("Проверенный слой источников изменился; нужна явная новая версия протокола")
        print(json.dumps({"id": STUDY_ID, "status": study["status"], "action": "unchanged"}))
        return
    study = store.create(spec, identifier=STUDY_ID)
    study.update(objects=objects, status="await_objects_approval", delivery_mode="factor_preview",
                 source_supplement=sources,
                 selection_method=roster.get("selection_method", "Целевая выборка политических ролей"),
                 selection_limitations=roster.get("limitations", []),
                 seed_provenance="research/rwanda_politicians_2021_2026/objects.json",
                 research_requirements={
                     "domain": "russia_relations", "required_scales": topic["scales"],
                     "loading_threshold": .35, "minimum_explained_share": .02, "minimum_indicators": 3,
                     "instructions": [
                         "В досье каждого политика проверить все шесть аспектов отношения к России; пробелы явно отмечать.",
                         "Персонифицированные датированные позиции и действия, а не перенос позиции государства, партии или ведомства.",
                         "Различать официальное действие и частное убеждение; не выводить симпатию из должности или встречи.",
                         "Сохранить event_id, даты и противоречащие свидетельства. Несколько пересказов не являются разными событиями.",
                         "Не дополнять численные оценки ради российского фактора. Unknown остаётся пропуском.",
                         "Шесть тематических шкал входят в единый общий словарь без повышающего веса и дублирования синонимами.",
                         "Сначала короткое сравнение всех допустимых q=2..8; расширенный отчёт и презентация только после выбора решения."]},
                 progress={"stage": "objects", "message": "Проверьте и утвердите список объектов",
                           "current": len(objects), "total": len(objects)})
    store.save(study)
    print(json.dumps({"id": STUDY_ID, "objects": len(objects), "approved": False,
                      "delivery_mode": "factor_preview"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
