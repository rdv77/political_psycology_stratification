"""Retrieve dossiers and propose the Rwanda dictionary after actual roster approval.

This stage never approves scales, rates objects, fits factors or exports slides.
Unchanged requests resume from the existing provider cache.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.semantic_service.providers import ModelProvider
from src.semantic_service.store import BUSY, StudyStore

IDENTIFIER = "rwanda-politicians-2021-2026"
DESTINATION = ROOT / "research/rwanda_politicians_2021_2026"


def write_proposal(scales, provenance):
    (DESTINATION / "scales_for_approval.json").write_text(
        json.dumps(scales, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Руанда: шкалы для утверждения", "",
             "Период: 15.10.2021–15.10.2026. Список из 42 человек утверждён; **этот словарь пока не утверждён**.", "",
             "Измеряем публичные политические акценты и стратегии. Личные убеждения и характер не диагностируются. "
             "Каждый полюс оценивается отдельно от 0 до 5; в факторный анализ входит разность −5…+5. "
             "Сохраняется также сумма полюсов: нулевая разность может означать выраженность обоих свойств. "
             "Неизвестное остаётся пропуском; должность, молчание и позиция страны не определяют ответ.", "",
             "Шесть аспектов отношения к России заданы исследовательским вопросом. Их включение в словарь "
             "не означает, что по ним уже достаточно данных или существует общий фактор. У каждой шкалы одинаковый "
             "вес после стандартизации; синонимы не дублируют столбцы.", "",
             "## Шкалы", "", "| № | Шкала | Полюс + | Полюс − |", "|---:|---|---|---|"]
    for i, scale in enumerate(scales, 1):
        lines.append(f"| {i} | {scale['name']} | {scale['positive']} | {scale['negative']} |")
    lines += ["", "## Определения", ""]
    for i, scale in enumerate(scales, 1):
        origin = "Обязательная тема отношения к России" if scale.get("research_domain") == "russia_relations" else "Общий словарь публичных характеристик"
        lines += [f"### {i}. {scale['name']}", "", origin + ". " + scale["definition"], ""]
    lines += ["## Проверка словаря", ""] + ["- " + note for note in provenance.get("normalization_notes", [])]
    lines += ["", "После утверждения: оценивание по досье → проверка покрытия → сравнение допустимых q=2…8 → "
              "короткий отчёт с долями дисперсии и проверкой российской темы. Расширенный отчёт и слайды — после выбора решения.", ""]
    (DESTINATION / "scales_for_approval.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    store = StudyStore(ROOT / "studies")
    study = store.get(IDENTIFIER)
    if not store.is_approved(study, "objects"):
        raise ValueError("Сначала необходимо фактическое утверждение списка пользователем")
    if store.is_approved(study, "scales") or study["status"] in BUSY:
        raise ValueError("Утверждённые шкалы или выполняемый этап нельзя перезаписывать")
    provider = ModelProvider(ROOT, store)
    store.update(IDENTIFIER, status="generating_scales", error=None)
    try:
        scales, provenance = provider.generate_scales(study, progress=lambda message, current, total:
            store.log(IDENTIFIER, message, "sources", current, total))
        store.update(IDENTIFIER, scales=scales, scale_provenance=provenance,
                     status="await_scales_approval", analysis=None, artifacts=[])
        write_proposal(scales, provenance)
        store.log(IDENTIFIER, "Досье и общий словарь готовы; требуется согласование шкал", "scales", len(scales), len(scales))
        print(json.dumps({"status": "await_scales_approval", "scales": len(scales),
                          "dossiers": provenance["source_object_count"],
                          "usage": store.get(IDENTIFIER)["usage"]}, ensure_ascii=False))
    except Exception as exc:
        store.update(IDENTIFIER, status="error", error=str(exc)[:1800])
        raise


if __name__ == "__main__":
    main()
