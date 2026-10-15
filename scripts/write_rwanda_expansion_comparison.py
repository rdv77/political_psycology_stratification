"""Deliver the evidence-expansion and imputation comparison, without clusters."""
from __future__ import annotations

import copy
import hashlib
import json
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
import sklearn

from src.semantic_service.analysis import _analysis_ratings, _prepare, SEED
from src.semantic_service.independent_poles import _write
from src.semantic_service.store import StudyStore, fingerprint, ratings_basis

IDENTIFIER = "rwanda-politicians-2021-2026"
RESEARCH = ROOT / "research/rwanda_politicians_2021_2026"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value):
    return f"{100*value:.2f}%".replace(".", ",")


def known(rows):
    return sum(e["status"] == "observed" for row in rows for e in row["pole_scores"].values())


def main():
    store = StudyStore(ROOT / "studies")
    study = store.require_approved(IDENTIFIER)
    directory = store.directory(IDENTIFIER)
    output = directory / "output"
    snapshot = directory / "attempts/independent_before_coverage_expansion"
    rows, previous = read(directory / "ratings.json"), read(snapshot / "ratings.json")
    preview, before = read(output / "factor_preview.json"), read(snapshot / "output/factor_preview.json")
    imputation = read(output / "imputation_review_2026-10-15.json")
    strict_comparison = read(output / "strict_without_flags/COMPARISON.json")
    semantic_path = RESEARCH / "COVERAGE_EXPANSION_INTERPRETATIONS.json"
    semantics = read(semantic_path)
    preview_sha = hashlib.sha256((output / "factor_preview.json").read_bytes()).hexdigest()
    if study["ratings_cache"]["basis"] != ratings_basis(study) or study["ratings_cache"]["ratings_sha256"] != fingerprint(rows):
        raise ValueError("Unbound or stale current measurements")
    if semantics["factor_preview_sha256"] != preview_sha:
        raise ValueError("Semantic interpretation belongs to a different numerical solution")
    actual_hash = hashlib.sha256((directory / "ratings.json").read_bytes()).hexdigest()
    if actual_hash not in imputation["file_input_receipt"]["sha256"].values():
        raise ValueError("Recompute imputation review on current ratings first")
    if actual_hash not in strict_comparison["source_file_sha256"].values() or preview_sha not in strict_comparison["source_file_sha256"].values():
        raise ValueError("Strict comparison belongs to different ratings")
    frame, _, objects, scale_map, diagnostics = _prepare(study, rows)
    ids = frame.columns.tolist()
    current = {(r["object_id"], r["scale_id"]): r for r in _analysis_ratings(study, rows)}
    old = {(r["object_id"], r["scale_id"]): r for r in _analysis_ratings(study, previous)}
    original_excluded = read(RESEARCH / "COVERAGE_EXPANSION_PLAN.json")["excluded_object_ids"]
    actor_coverage = []
    for obj in study["objects"]:
        oid = obj["id"]
        count = sum(current[oid, sid]["status"] == "observed" for sid in ids)
        old_count = sum(old[oid, sid]["status"] == "observed" for sid in ids)
        actor_coverage.append({"id": oid, "name": obj["name"], "known_common_variables_before": old_count,
                               "known_common_variables_now": count, "common_variables_total": len(ids),
                               "included_in_FA": oid in frame.index,
                               "known_all_properties_now": sum(e["status"] == "observed" for r in rows if r["object_id"] == oid for e in r["pole_scores"].values())})
    _write(output / "coverage_by_actor.json", actor_coverage)
    table = ["# Покрытие всех 42 утверждённых участников", "",
             "Числа относятся к реальным наблюдениям. Модельные заполнения не увеличивают покрытие. Прежние пороги допуска сохранены.", "",
             "| Участник | Общие признаки: было | Стало | В основной FA | Все известные свойства из 48 |",
             "|---|---:|---:|---|---:|"]
    table += [f"| {a['name']} | {a['known_common_variables_before']}/{len(ids)} | {a['known_common_variables_now']}/{len(ids)} | {'да' if a['included_in_FA'] else 'нет'} | {a['known_all_properties_now']} |" for a in actor_coverage]
    (output / "coverage_by_actor.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    retained = set(frame.index)
    newly = [a for a in actor_coverage if a["id"] in original_excluded and a["included_in_FA"]]
    excluded = [a for a in actor_coverage if not a["included_in_FA"]]
    themed = {s["id"] for s in study["scales"] if s.get("research_domain") == "russia_relations"}
    russia_known = sum(e["status"] == "observed" for r in rows if r["scale_id"] in themed for e in r["pole_scores"].values())
    russia_actors = {r["object_id"] for r in rows if r["scale_id"] in themed and any(e["status"] == "observed" for e in r["pole_scores"].values())}
    names = {o["id"]: o["name"] for o in study["objects"]}
    scales = {s["id"]: s for s in study["scales"]}
    thematic_lines = ["# Россия: непосредственно обоснованные публичные проявления", "",
        f"Известны {russia_known} из 504 свойств у {len(russia_actors)} участников. Остальные значения неизвестны. "
        "Это индивидуальные порядковые оценки отдельных проявлений, а не итоговый рейтинг отношения к России и не факторные баллы.", "",
        "| Участник | Измеренное свойство | Оценка 0–5 | Ограничение и основание |", "|---|---|---:|---|"]
    for row in rows:
        if row["scale_id"] not in themed:
            continue
        for pole, cell in row["pole_scores"].items():
            if cell["status"] != "observed":
                continue
            sources = " ".join(f"[источник{i+1}]({url})" for i, url in enumerate(cell["source_urls"]))
            explanation = cell["rationale"].replace("|", "/").replace("\n", " ")
            thematic_lines.append(f"| {names[row['object_id']]} | {scales[row['scale_id']][pole]} | {cell['value']} | {explanation} {sources} |")
    thematic_lines += ["", "Дипломатическое или отраслевое сотрудничество может сочетаться с критикой конкретного действия России. "
        "Один документ по нескольким темам остаётся одним событием. Роль OIF отличается от роли правительства Руанды. "
        "Молчание не даёт нуля, а прошлое обучение в России не является современной политической позицией.", ""]
    (output / "russia_profile_evidence.md").write_text("\n".join(thematic_lines), encoding="utf-8")
    receipt = study["coverage_expansion"]
    lines = ["# Руанда: расширение охвата и сравнение способов заполнения пропусков", "",
             "**42 утверждённых участника · 15.10.2021–15.10.2026 · независимые свойства 0–5.**", "",
             f"После адресного дополнения источников в основную факторную матрицу вошли **{len(frame)} из 42 участников и {len(ids)} признаков** "
             f"(раньше {before['n_objects']} × {before['n_scales']}). Покрытие матрицы — **{pct(diagnostics['retained_coverage'])}**. "
             "Все исходные пропуски и прежние наблюдения сохранены отдельно от аналитического заполнения.", "",
             "## Что удалось дополнить", "",
             f"Проверены все {receipt['checked_unknown_properties']} запланированных неизвестных свойства: прежние восемь общих признаков у 14 исключённых участников "
             "и российский блок у всех 42. Выбор направлений обусловлен предыдущим покрытием и поручением пользователя: это адаптивное разведочное продолжение. "
             f"Число добавленных оценок — **{receipt['new_observations']}**. Всего известно **{known(rows)}/2016 ({pct(known(rows)/2016)})**, "
             f"до этого — {known(previous)}/2016. Уже известные значения не переоценивались, пороги допуска не снижались.", "",
             "У Кагаме теперь измерены все восемь общих признаков. Поиск ограничен сохранёнными запросами и источниками; "
             "оставленные пропуски не доказывают, что соответствующих высказываний вообще не существует.", "",
             "Впервые прошли допуск: " + (", ".join(a["name"] for a in newly) or "новых участников нет") + ".", "",
             "Остались за пределами основной FA: " + (", ".join(a["name"] for a in excluded) or "никто") + ". "
             "Исключение означает недостаток сопоставимых измерений, а не нейтральность или отсутствие позиции. "
             "[Таблица всех 42 участников](coverage_by_actor.md).", ""]
    selected = [a for a in actor_coverage if a["id"] in original_excluded]
    figure, axis = plt.subplots(figsize=(10, 6.5))
    y = np.arange(len(selected))
    axis.barh(y, [a["known_common_variables_now"] for a in selected], color="#247b70", label="После дополнения")
    axis.barh(y, [a["known_common_variables_before"] for a in selected], color="#b7c1c8", label="До дополнения")
    axis.set_yticks(y, [a["name"] for a in selected])
    axis.invert_yaxis()
    axis.set(xlim=(0, len(ids)+.3), xlabel=f"Измерено признаков из {len(ids)} общего набора", title="Ранее исключённые участники: реальные наблюдения")
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output / "assets/coverage_expansion.png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    lines += ["![Покрытие ранее исключённых участников](assets/coverage_expansion.png)", "",
              "## Факторные решения после дополнения", "",
              "| Факторов | Было: 28 × 8 | Теперь: " + f"{len(frame)} × {len(ids)}" + " |", "|---|---:|---:|"]
    for model in preview["models"]:
        past = next((m for m in before["models"] if m["q"] == model["q"]), None)
        lines.append(f"| {model['q']} | {pct(past['variance_total']) if past else '—'} | {pct(model['variance_total'])} |")
    lines += ["", "Доли относятся к общей дисперсии оставленных стандартизованных переменных. Состав объектов изменился, поэтому разность процентов "
              "не является отдельным эффектом качества данных или долей нового фактора. В основном расчёте оставшиеся пропуски заполняются средним признака; "
              "сравнение альтернатив приведено ниже. Номера и знаки осей между решениями условны.", ""]
    display = copy.deepcopy(preview)
    for model in display["models"]:
        description = semantics["solutions"][str(model["q"])]
        lines += [f"### {model['q']} фактора — {pct(model['variance_total'])}", ""]
        for factor in model["factors"]:
            manual = next(f for f in description["factors"] if f["id"] == factor["id"])
            lines += [f"**{factor['id']}. {manual['name']} — {pct(factor['variance_share'])}.** {manual['meaning']}", ""]
            factor.update(automatic_name=factor["name"], name=manual["name"], label=manual["name"], interpretation=manual["meaning"])
        if description.get("difference"):
            lines += [description["difference"], ""]
        lines += [f"[Диаграмма нагрузок {model['q']} факторов](assets/factor_preview_q{model['q']}.png)", ""]
    lines += ["На диаграммах нагрузок строки — признаки, столбцы — факторы. Число и цвет показывают направление и силу связи. "
              "Один знак у нескольких признаков позволяет описать их совместную выраженность; разные знаки могут обозначать контраст. "
              "Меньшая выраженность свойства не создаёт его антоним. Такие оси описывают публичную коммуникацию в выбранном корпусе, а не доказанные частные убеждения.", "",
              "## Отношение к России", "",
              f"Теперь известны **{russia_known}/504** отдельных российских свойств у **{len(russia_actors)} участников**. "
              "Прямые высказывания о конкретных действиях России, сотрудничестве, санкциях и дистанцировании различаются по содержанию и роли говорящего. "
              "Одно событие с несколькими темами сохраняет один event_id.", ""]
    if preview["russia_review"]["status"] == "insufficient_measurement":
        lines += ["**Проверить отдельный российский фактор ≥2% всё ещё нельзя:** собственные российские индикаторы не прошли порог покрытия общей матрицы. "
                  "Это не нулевой процент и не доказательство отсутствия такой структуры. Заполнение российского блока из должностей или общих политических осей "
                  "выдало бы модельную догадку за новую информацию.", ""]
    else:
        lines += [f"Статус проверки: `{preview['russia_review']['status']}`. Кандидаты и их доли сохранены в полном сравнении факторов.", ""]
    lines += ["Для Кагаме добавлена оценка критики конкретного внешнеполитического действия России: в интервью 12 сентября 2023 года "
              "он отвергает вторжение одной страны в другую, обсуждая Украину, и одновременно говорит о нейтральной позиции Руанды. "
              "Это ограниченное проявление, а не общий антироссийский профиль. "
              "[Полный текст интервью](https://www.newtimes.co.rw/article/10916/news/politics/kagame-the-problem-is-not-between-me-and-tshisekedi/amp).", "",
              "Например, диверсификация поставщиков пшеницы у Нгиренте сохранена как контекст: она недостаточна для оценки ограничения проектов с Россией. "
              "Собственные объяснения голосования по Украине кодируются в пределах конкретного заявления; не становятся общей враждебностью или санкционной позицией. "
              "[Индивидуальные российские проявления и источники](russia_profile_evidence.md) · "
              "[План и границы дополнения](../../../research/rwanda_politicians_2021_2026/COVERAGE_EXPANSION_PLAN.json).", "",
              "## Можно ли заменять небольшие пропуски", "",
              "Да, в аналитической копии достаточно измеренной матрицы. Среднее признака прозрачно, но стягивает значения к центру и меняет корреляции. "
              "Регрессия использует связи признаков и тоже может их усиливать. Поэтому здесь сравниваются mean, регуляризованная BayesianRidge и KNN; "
              "обучение проводится на 75% объектов, а на остальных временно скрываются 12% известных оценок. Ни один метод не видит скрытую истину при обучении.", "",
              "| Заполнение | MAE | RMSE |", "|---|---:|---:|"]
    for method in imputation["validation"]["summary"]:
        lines.append(f"| {method['method']} | {method['pooled_mae']:.3f} | {method['pooled_rmse']:.3f} |")
    lines += ["", "Это ошибки случайно скрытых известных ячеек, не доказательство правильности настоящих неизвестных позиций. "
              "Среднее показало наименьшую ошибку в этой проверке, поэтому оснований автоматически переходить на регрессию нет. "
              "В настоящих пропусках важны служебная роль, доступность источников и содержание высказываний; они могут быть неслучайными. "
              "Дополнительно выполнены 10 стохастических регрессионных заполнений с отдельным обучением FA и согласованием осей. "
              "Разброс показывает чувствительность к восстановлению, а не доверительные интервалы истинных убеждений. "
              "[Полное сравнение заполнений и нагрузки](imputation_review_2026-10-15.md).", "",
              "Для q=3 медианное сходство профилей нагрузок при этих заполнениях составляет 0,962 / 0,896 / 0,849 по F1–F3. "
              "Для q=4 у F2 и F4 — только 0,530 и 0,492: дополнительная детализация особенно зависит от восстановления пустых ячеек. "
              "Это описательное сравнение направлений осей, не вероятность правильности модели.", "",
              "## Ограничения и рабочий выбор", "", semantics["recommendation"], ""]
    strict = imputation["coverage_sensitivity"]["scenarios"]
    first = strict[0]
    sd = first["diagnostics"]
    lines += [f"При исключении всех текущих оценок с контекстными оговорками ({first['removed_count']}) остаётся "
              f"**{sd['retained_objects']} × {sd['retained_scales']}**. Отдельная FA этой копии дала 43,05% при двух факторах и 55,98% при трёх. "
              "Четыре фактора при семи признаках не допускаются по числу ковариационных параметров. Оговорка не объявляет наблюдение ошибочным.", "",
              "В строгом варианте из общего набора ушёл признак программного изменения курса, а также Хареримана, Мпайимана и Виктуар Ингабире. "
              "Линия преемственности основного q=3 сохраняется заметно лучше остальных (сходство нагрузок 0,932 против 0,379 и 0,582). "
              "Профилактика в строгих решениях остаётся осью с одним сильным индикатором. Значит, усиление охвата позволяет обучить более строгую модель, "
              "но не подтверждает всю прежнюю структуру. Проценты двух матриц имеют разные знаменатели — 8 и 7 признаков — и не измеряют улучшение качества. "
              "[Строгая FA и подробное сопоставление](strict_without_flags/COMPARISON.md).", "",
              "В методическом приложении отдельно исключён весь слой последующих дополнений. После этого прежний допуск к FA не выполняется: "
              "источники дополнения существенно влияют на возможность и состав анализа.", "",
              "Выборка целевая, управленцы преобладают, международные дипломаты выступают в иной служебной роли. Разметка выполнена ассистентами по источникам, "
              "а не независимой группой человеческих экспертов. Расширенный отчёт, кластеры и слайды готовятся после выбора факторного решения.", "",
              "[Полные факторы и диагностика](factor_preview.md) · [Матрица с исходными пропусками](factor_matrix.json) · "
              "[Методика работы с пропусками](../../../docs/MISSING_DATA.md).", ""]
    text = "\n".join(lines)
    (output / "short_report.md").write_text(text, encoding="utf-8")
    mirrored = text.replace("(assets/", f"(../../studies/{IDENTIFIER}/output/assets/")
    for stem in ("coverage_by_actor", "factor_preview", "factor_matrix", "russia_profile_evidence", "imputation_review_2026-10-15"):
        mirrored = mirrored.replace(f"({stem}.", f"(../../studies/{IDENTIFIER}/output/{stem}.")
    mirrored = mirrored.replace("(strict_without_flags/", f"(../../studies/{IDENTIFIER}/output/strict_without_flags/")
    (RESEARCH / "SHORT_REPORT.md").write_text(mirrored.replace("(../../../", "(../../"), encoding="utf-8")
    _write(output / "factor_matrix.json", {"objects": frame.index.tolist(), "variables": [scale_map[sid] for sid in ids],
        "values": [[None if pd.isna(v) else float(v) for v in row] for row in frame.to_numpy()]})
    _write(output / "reproducibility.json", {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__, "seed": SEED, "ratings_fingerprint": fingerprint(rows), "ratings_basis": ratings_basis(study),
        "coverage_expansion": receipt, "numerical_preview_sha256": preview_sha,
        "imputation_review_sha256": hashlib.sha256((output / "imputation_review_2026-10-15.json").read_bytes()).hexdigest(),
        "strict_comparison_sha256": hashlib.sha256((output / "strict_without_flags/COMPARISON.json").read_bytes()).hexdigest(),
        "review_input_timing": "Imputation and strict-FA file receipts were taken before final delivery labels/artifact links changed; numerical ratings and their basis remain bound.",
        "source_review_and_prior_results": "attempts/independent_before_coverage_expansion"})
    # Keep the previous numerical audit under its immutable snapshot, and make the familiar MD link current.
    (output / "factor_sensitivity.md").write_text("# Актуальная проверка чувствительности\n\n"
        "[Сравнение заполнений и строгие сценарии допуска](imputation_review_2026-10-15.md).\n\n"
        "[Факторный анализ строгой матрицы 34×7](strict_without_flags/COMPARISON.md).\n\n"
        "[Предыдущий бутстреп матрицы 28×8](../attempts/independent_before_coverage_expansion/output/factor_sensitivity.md) "
        "относится к прежнему составу, а не к текущему расширению.\n", encoding="utf-8")
    _write(output / "factor_sensitivity.json", {"format_version": 2, "mode": "coverage_and_imputation_sensitivity",
        "current_review": "imputation_review_2026-10-15.json", "coverage_sensitivity": imputation["coverage_sensitivity"],
        "current_strict_factor_comparison": "strict_without_flags/COMPARISON.json",
        "previous_row_bootstrap": "../attempts/independent_before_coverage_expansion/output/factor_sensitivity.json",
        "previous_bootstrap_is_current": False})
    display["semantic_review"] = {"status": "provisional_assistant_interpretation", "file": semantic_path.relative_to(ROOT).as_posix(),
                                   "numerical_preview_sha256": preview_sha}
    display["limitations"] = [semantics["recommendation"],
        f"В основную FA вошли {len(retained)} из 42 участников; заполнения не увеличивают измеренное покрытие. Проверка чувствительности находится в коротком отчёте.",
        *display.get("limitations", [])]
    artifacts = [{"label": "Короткий отчёт: охват и факторы", "format": "md", "url": f"/api/studies/{IDENTIFIER}/files/short_report.md"},
                 {"label": "Пропуски: сравнение методов и устойчивость", "format": "md", "url": f"/api/studies/{IDENTIFIER}/files/imputation_review_2026-10-15.md"},
                 {"label": "Покрытие всех 42 участников", "format": "md", "url": f"/api/studies/{IDENTIFIER}/files/coverage_by_actor.md"},
                 {"label": "Россия: персональные основания", "format": "md", "url": f"/api/studies/{IDENTIFIER}/files/russia_profile_evidence.md"},
                 {"label": "Полные нагрузки и диагностика", "format": "md", "url": f"/api/studies/{IDENTIFIER}/files/factor_preview.md"},
                 {"label": "Диаграмма расширения охвата", "format": "png", "url": f"/api/studies/{IDENTIFIER}/files/assets/coverage_expansion.png"}]
    artifacts += [{"label": c["title"], "format": "png", "url": f"/api/studies/{IDENTIFIER}/files/{c['file']}"} for c in preview.get("charts", [])]
    store.update(IDENTIFIER, status="await_factor_selection", analysis=display, artifacts=artifacts)
    print(json.dumps({"objects": len(frame), "variables": len(ids), "known": known(rows), "newly_admitted": len(newly),
                      "russia_known": russia_known, "report_words": len(text.split())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
