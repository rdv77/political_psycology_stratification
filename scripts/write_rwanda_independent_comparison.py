"""Reader-facing first-stage comparison of independent-property FA solutions."""
from __future__ import annotations

import json
import hashlib
import copy
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn

from src.semantic_service.analysis import SEED, _prepare
from src.semantic_service.independent_poles import _write
from src.semantic_service.store import StudyStore, fingerprint, ratings_basis

IDENTIFIER = "rwanda-politicians-2021-2026"
RESEARCH = ROOT / "research/rwanda_politicians_2021_2026"


def percent(value):
    return f"{100*value:.2f}%".replace(".", ",")


def main():
    store = StudyStore(ROOT / "studies")
    study = store.require_approved(IDENTIFIER)
    if study.get("coverage_expansion", {}).get("applied"):
        raise ValueError("Use the coverage-expansion report writer for the new data; preserve the initial comparison")
    directory = store.directory(IDENTIFIER)
    output = directory / "output"
    result = json.loads((output / "factor_preview.json").read_text(encoding="utf-8"))
    json.loads((output / "factor_sensitivity.json").read_text(encoding="utf-8"))
    ratings = json.loads((directory / "ratings.json").read_text(encoding="utf-8"))
    if study["measurement_model"] != "independent_poles" or study["ratings_cache"]["ratings_sha256"] != fingerprint(ratings):
        raise ValueError("Current bound independent-pole measurements required")
    interpretation_file = RESEARCH / "INDEPENDENT_FACTOR_INTERPRETATIONS.json"
    interpretations = json.loads(interpretation_file.read_text(encoding="utf-8")) if interpretation_file.exists() else {}
    preview_sha = hashlib.sha256((output / "factor_preview.json").read_bytes()).hexdigest()
    if interpretations and interpretations.get("factor_preview_sha256") != preview_sha:
        raise ValueError("Semantic review belongs to another numerical preview; review it before delivery")
    audit = study["independent_source_audit"]
    completion = study.get("independent_completion", {})
    observed = sum(e["status"] == "observed" for r in ratings for e in r["pole_scores"].values())
    cells = 42 * 48
    models = result.get("models", [])
    lines = ["# Руанда: короткое сравнение факторных решений", "",
        "**42 политика и публичных руководителя · 15.10.2021–15.10.2026 · независимые признаки 0–5.**", "",
        "**Статус: предварительные факторные гипотезы.** При исключении всех оценок с контекстными оговорками остаётся 23 участника и 5 признаков — "
        "меньше принятого минимума для FA. Ни одно решение пока не подтверждено на этом строгом корпусе. "
        "[Проверка чувствительности](factor_sensitivity.md).", "",
        "## Что изменилось в измерении", "",
        "Из 24 утверждённых пар получены 48 самостоятельных признаков. Известный полюс сохраняется при неизвестном противоположном; "
        "разность и дополнение до фиксированной суммы не вычисляются. Однополюсная шкала также допустима. "
        "Ноль требует прямого основания, а отсутствие сведений остаётся пропуском.", "",
        "Дополнительный поиск по всем 42 участникам дал 308 новых выписок; до отдельного проверенного тематического дополнения "
        "было 905 кандидатных выписок. Это не 905 независимых подтверждений: среди них есть повторные описания событий и ошибки дат. "
        f"Из 774 первоначальных численных предложений отклонены {audit['counts']['reject']}, "
        f"у {audit['counts']['repair_indices']} уточнены ссылки, {audit['counts']['recode']} переоценены по якорям выраженности. "
        "Исходные ответы сохранены. Содержательная проверка относится к сохранённым выпискам с выборочным чтением первичных страниц; "
        "она не заменяет независимую человеческую экспертизу всего корпуса.", ""]
    if completion:
        lines += [f"До обучения факторов отдельно проверены все {completion['checked_unknown_properties']} пропусков в шести общих признаках "
                  "с покрытием от 50% до 60%. Признаки выбраны по полноте измерения, без просмотра факторных результатов. "
                  f"Добавлено {completion['new_observations']} прямых ограниченных проявлений; "
                  "прежние известные оценки и пороги допуска не менялись.", ""]
    lines += [f"Сейчас известны **{observed} из {cells} оценок ({percent(observed/cells)})**. "
              "Это разведочные LLM-оценки публичных заявлений и действий, а не измерение частных убеждений или взглядов населения.", ""]
    if models:
        lines += [f"В общую факторную матрицу вошли **{result['n_objects']} участников и {result['n_scales']} признаков**, "
                  f"покрытие — {percent(result['diagnostics']['retained_coverage'])}. "
                  "Все решения ниже рассчитаны по одной этой матрице. Их доли относятся к её переменным, а не ко всем 48 плановым признакам.", "",
                  "**Состав существенно ограничивает вывод.** В матрицу не прошли Поль Кагаме, Оливье Ндухунгирехе, Жювеналь Маризамунда и Луиза Мушикивабо. "
                  "Из восьми оппозиционных и независимых участников вошли только трое. Преобладают руководители и управленцы; "
                  "здесь прежде всего описываются доступные образцы публичной управленческой коммуникации, а не всё политическое сознание страны.", "",
                  "Оставлены восемь признаков: изменение курса; продолжение курса; поддержка курса; административное исполнение; "
                  "ценности, память и символы; доступность базовых благ; раскрытие проблем; упреждение рисков.", "",
                  "## Сравнение решений", "",
                  "| Факторов | Общая объяснённая дисперсия | Что добавляется |",
                  "|---|---:|---|"]
        additions = {2: "Поддержка курса и обновление как две связки", 3: "Ценностная преемственность в сопоставлении с профилактикой", 4: "Разделение профилактики и ценностной преемственности"}
        for model in models:
            lines.append(f"| {model['q']} | {percent(model['variance_total'])} | {additions.get(model['q'], 'См. полную диагностику')} |")
        lines += ["", "Дисперсия рассчитана по нагрузкам FA после varimax относительно общей дисперсии восьми стандартизованных переменных. "
                  "35 пропусков в 224 ячейках заполнены средним соответствующего признака перед стандартизацией. Это допущение модели, а не восстановленные факты. "
                  "При изменении числа факторов вся модель переоценивается; "
                  "разность общих процентов не является долей одного нового фактора.", ""]
        for model in models:
            if model["q"] not in {2, 3, 4}:
                continue
            key = str(model["q"])
            description = interpretations.get("solutions", {}).get(key, {})
            lines += [f"### {model['q']} фактора — {percent(model['variance_total'])}", ""]
            for factor in model["factors"]:
                manual = next((f for f in description.get("factors", []) if f["id"] == factor["id"]), {})
                name = manual.get("name", factor["name"])
                meaning = manual.get("meaning", factor["interpretation"])
                lines += [f"**{factor['id']}. {name} — {percent(factor['variance_share'])}.** {meaning}", ""]
            if description.get("difference"):
                lines += [description["difference"], ""]
        excluded = result["dimension_selection"].get("parameter_count_constraint", {}).get("excluded_candidates", [])
        if excluded:
            lines += ["По ограничению числа параметров не обучались: " + ", ".join(f"{r['q']} факторов (df={r['nominal_df']})" for r in excluded) +
                      ". Неотрицательный df — необходимое условие, а не гарантия надёжности модели.", ""]
        figure, axis = plt.subplots(figsize=(7, 3.8))
        axis.bar([str(m["q"]) for m in models], [100*m["variance_total"] for m in models], color="#23766c")
        axis.set(xlabel="Число факторов", ylabel="Общая дисперсия переменных, %", title="Руанда: сравнение размерностей на одной матрице")
        for i, model in enumerate(models):
            axis.text(i, 100*model["variance_total"]+.6, f"{100*model['variance_total']:.1f}%", ha="center", fontsize=9)
        figure.tight_layout()
        (output / "assets").mkdir(exist_ok=True)
        figure.savefig(output / "assets/independent_factor_comparison.png", dpi=160, bbox_inches="tight")
        plt.close(figure)
        lines += ["![Общая дисперсия решений](assets/independent_factor_comparison.png)", ""]
        lines += ["[Нагрузки 2 факторов](assets/factor_preview_q2.png) · [Нагрузки 3 факторов](assets/factor_preview_q3.png) · "
                  "[Нагрузки 4 факторов](assets/factor_preview_q4.png). "
                  "На этих диаграммах строки — признаки, столбцы — факторы, цвет и число показывают направление и силу связи. "
                  "Весь знак столбца можно поменять без изменения модели; отрицательная нагрузка сама по себе не означает отрицательную оценку политика.", "",
                  "## Насколько это устойчиво", "",
                  "При 30 повторных выборках участников с возвращением наиболее воспроизводима связка поддержки курса и базовых благ: "
                  "медианная схожесть нагрузок с исходными — 0,963 / 0,947 / 0,950 в решениях с 2 / 3 / 4 факторами. "
                  "Ось обновления в двухфакторном варианте слабее (0,776); новая ось ценностной преемственности в четырёхфакторном — 0,785. "
                  "Остальные оси также заметно меняются. Это условная проверка уже выбранной матрицы, не доказательство качества источников или охвата страны.", "",
                  "Без всех 41 оценок с оговорками остаётся **23 × 5**, без всех 20 новых тематических дополнений — **18 × 4**. "
                  "Обе матрицы не проходят прежние пороги, FA для них не обучалась. Оговорка не равна доказанной ошибке, "
                  "но результат зависит от допущения об использовании таких наблюдений. "
                  "[Подробная проверка и все повторы](factor_sensitivity.md).", ""]
    else:
        lines += ["**Матрица всё ещё не прошла требования допуска. Факторы и доли дисперсии не назначались.** "
                  "Подробные причины и покрытие каждого признака сохранены в диагностике.", ""]
    theme = result["russia_review"]
    lines += ["## Проверка отношения к России", "",
              "Проверены шесть тематических контрастов как 12 самостоятельных признаков: дипломатия, оборона, экономика и технологии, "
              "образование и культура, оценки конкретных внешних действий России, санкции и изоляция. Два признака одной пары считаются одной семантической семьёй.", ""]
    if theme["status"] == "insufficient_measurement":
        lines += ["**Проверить долю отдельного российского фактора ≥2% пока нельзя.** "
                  "Пригодные прямые основания не распределены достаточно широко по участникам; российский блок не прошёл допуск. "
                  "Неизвестным позициям не присвоены нули, а российскому фактору не приписан нулевой процент.", ""]
    else:
        lines += [f"Статус численного просмотра: `{theme['status']}`; сохранены все кандидаты каждого допустимого решения. "
                  "Доли считаются относительно общей дисперсии всех оставленных переменных, а не только российского блока.", ""]
        for candidate in theme["candidates"]:
            if candidate["mandatory_report"]:
                lines += [f"- Решение {candidate['q']}, {candidate['factor_id']}: {percent(candidate['variance_share'])}, "
                          f"семей индикаторов {candidate['distinct_family_count']}. Это численный кандидат, требующий содержательной проверки."]
        lines += [""]
    lines += ["В частности, проверены соглашение Ндухунгирехе о визовом режиме и последующие двусторонние заявления, "
              "оборонное взаимодействие Маризамунды, медицинское сотрудничество Нсанзиманы и критика российских действий Мушикивабо в роли главы OIF. "
              "Всего приняты 7 оценок отдельных российских свойств у четырёх человек; одно событие может относиться к нескольким темам и не становится независимыми событиями. "
              "Их нельзя переносить на остальных участников или смешивать с личной симпатией. "
              "[Карта источников и границы вывода](../../../research/rwanda_politicians_2021_2026/RUSSIA_INDEPENDENT_POLES_COVERAGE.md).", "",
              "## Как читать результат и что выбрать", "",
              interpretations.get("recommendation", "Численную рекомендацию следует сопоставить с содержанием нагрузок и ограничениями измерений; окончательное решение выбирает пользователь."), "",
              "Фактор с сильными нагрузками одного знака допускает однополюсный смысл. Разные знаки у разных содержательных групп "
              "могут дать два интерпретируемых полюса. Меньшая выраженность свойства не означает выраженность антонима. "
              "«Луч» здесь — содержательная метафора: центрированные FA-координаты математически допускают оба знака.", "",
              "Доступность источников и публичные служебные роли неравномерны. Неизмеренные и исключённые участники не становятся "
              "нейтральными, а полученная матрица не исчерпывает политическое пространство страны. Расширенный отчёт, кластеризация "
              "и презентация ожидают выбора факторного решения.", "",
              "[Полные нагрузки, исключения и диагностика](factor_preview.md) · [JSON всех решений](factor_preview.json) · "
              "[Методика независимых признаков](../../../docs/INDEPENDENT_POLES.md).", ""]
    text = "\n".join(lines)
    (output / "short_report.md").write_text(text, encoding="utf-8")
    mirrored = text.replace("(assets/", f"(../../studies/{IDENTIFIER}/output/assets/")
    for stem in ("factor_preview", "factor_sensitivity"):
        mirrored = mirrored.replace(f"({stem}.", f"(../../studies/{IDENTIFIER}/output/{stem}.")
    mirrored = mirrored.replace("(../../../", "(../../")
    (RESEARCH / "SHORT_REPORT.md").write_text(mirrored, encoding="utf-8")
    metadata = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
                "scikit_learn": sklearn.__version__, "seed": SEED, "ratings_fingerprint": fingerprint(ratings),
                "ratings_basis": ratings_basis(study), "measurement_model": study["measurement_model"],
                "source_review": audit, "completion": completion, "methodological_selection": "Fixed coverage thresholds; held-out LL+parsimony is exploratory"}
    metadata["completion_followup_review"] = study.get("independent_completion_review")
    metadata["sensitivity"] = {"file": "factor_sensitivity.json", "conditional_on_selected_matrix": True,
                               "strict_measurement_corpus_passes": False,
                               "input_hash_timing": "The full-study SHA in the sensitivity audit identifies the computational input before delivery artifacts were added. ratings_basis binds measurement inputs independently of output links and timestamps."}
    metadata["files_sha256"] = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in (directory / "ratings.json", output / "factor_preview.json", output / "factor_sensitivity.json", interpretation_file)}
    _write(output / "reproducibility.json", metadata)
    if models:
        frame, _, _, scale_map, _ = _prepare(study, ratings)
        _write(output / "factor_matrix.json", {"objects": frame.index.tolist(), "variables": [scale_map[sid] for sid in frame.columns],
               "values": [[None if pd.isna(x) else float(x) for x in row] for row in frame.to_numpy()]})
    extras = [{"label": "Короткое сравнение и интерпретация", "format": "md", "url": f"/api/studies/{IDENTIFIER}/files/short_report.md"},
              {"label": "Проверка чувствительности факторов", "format": "md", "url": f"/api/studies/{IDENTIFIER}/files/factor_sensitivity.md"}]
    artifacts = extras + [a for a in study.get("artifacts", []) if a["url"] not in {e["url"] for e in extras}
                          and not a["url"].endswith("independent_factor_comparison.png")]
    if models:
        artifacts.append({"label": "Дисперсия решений", "format": "png", "url": f"/api/studies/{IDENTIFIER}/files/assets/independent_factor_comparison.png"})
    display_analysis = copy.deepcopy(result)
    for model in display_analysis.get("models", []):
        reviewed = interpretations.get("solutions", {}).get(str(model["q"]), {})
        for factor in model["factors"]:
            manual = next((f for f in reviewed.get("factors", []) if f["id"] == factor["id"]), None)
            if manual:
                factor.update(automatic_name=factor["name"], name=manual["name"], label=manual["name"],
                              interpretation=manual["meaning"])
    display_analysis["semantic_review"] = {"file": interpretation_file.relative_to(ROOT).as_posix(),
        "status": "provisional_assistant_interpretation", "numerical_preview_sha256": preview_sha}
    display_analysis["limitations"] = [
        "Предварительные гипотезы: без всех оценок с контекстными оговорками остаётся 23 × 5, меньше прежнего минимума для FA. Устойчивость на строгом корпусе не подтверждена.",
        "В матрицу 28 × 8 не прошли Кагаме и ряд других значимых фигур; из 8 оппозиционных и независимых участников вошли 3. Эти данные не описывают всё политическое сознание Руанды.",
        *display_analysis.get("limitations", [])]
    store.update(IDENTIFIER, artifacts=artifacts, analysis=display_analysis)
    print(json.dumps({"words": len(text.split()), "known_properties": observed, "models": [m["q"] for m in models]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
