"""One report content model for Markdown, HTML, PDF, DOCX, and slides."""
from __future__ import annotations

import html
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def paragraph(text: str) -> dict:
    return {"type": "paragraph", "text": str(text)}


def heading(text: str, level: int = 1) -> dict:
    return {"type": "heading", "text": str(text), "level": level}


def table(headers, rows) -> dict:
    return {"type": "table", "headers": [str(v) for v in headers],
            "rows": [[str(v) for v in row] for row in rows]}


def _comparison_prose(value) -> list[dict]:
    if isinstance(value, str):
        return [paragraph(value)] if value.strip() else []
    if isinstance(value, list):
        blocks = []
        for item in value:
            if isinstance(item, dict):
                text = item.get("text") or item.get("description") or item.get("summary")
                label = item.get("factor_id") or item.get("label")
                item = f"{label}: {text}" if label and text else text
            blocks.extend(_comparison_prose(item))
        return blocks
    return []


def _comparison_number(value, specification=".3f") -> str:
    return format(value, specification) if isinstance(value, (int, float)) and not isinstance(value, bool) else "—"


def comparison_overview(comparison: dict) -> list[dict]:
    """Make alternatives visible without overwriting the frozen q=2 baseline."""
    if not comparison or not comparison.get("models"):
        return []
    scope = comparison.get("scope", {})
    blocks = [heading("Сравнение моделей q=2, q=3 и q=4: краткий итог", 2),
              paragraph("Базовое решение прежнего отчёта q=2 сохранено для сопоставления. Ниже сравниваются модели q=2, q=3 и q=4 "
                        "на одной матрице. Увеличение q само по себе не доказывает более содержательное решение."),
              paragraph(f"Общая матрица сравнения: {scope.get('n_objects', '—')} объектов и {scope.get('n_scales', '—')} шкал. "
                        f"Для сопоставления составов групп используется одинаковое k={scope.get('fixed_cluster_count', 3)}; "
                        "численное и содержательное предпочтение k рассматриваются отдельно.")]
    if scope.get("ratings_reused") is True:
        blocks.append(paragraph("Во всех моделях повторно использована одна сохранённая матрица оценок; нового оценивания объектов не проводилось."))
    blocks.extend(_comparison_prose(comparison.get("executive_summary")))
    preferred = {row["q"]: row["k"] for row in comparison.get("selected_k", [])}
    blocks.append(table(["Факторов q", "Общая дисперсия", "CV log likelihood", "RMS остатков", "Численное / рабочее k"],
                        [[model.get("q", model.get("factor_count")), _comparison_number(model.get("variance_total"), ".1%"),
                          _comparison_number(model.get("dimension_fit", {}).get("heldout_log_likelihood_mean")),
                          _comparison_number(model.get("dimension_fit", {}).get("residual_rms_off_diagonal")),
                          f"{model.get('numerical_recommended_k', '—')} / " +
                          str(preferred.get(model.get("q"), model.get("cluster_count") if model.get("q") == 2 else "не выбрано"))]
                         for model in comparison["models"]]))
    blocks.extend(_comparison_prose(comparison.get("recommendation")))
    return blocks


def _comparison_contingency(matrix: dict, caption: str) -> list[dict]:
    if not matrix or not matrix.get("values"):
        return []
    return [paragraph(caption), table(["Исходная / новая группа"] + matrix["columns"],
                                     [[identity] + values for identity, values in zip(matrix["rows"], matrix["values"])])]


def _comparison_clusters(clusters: list[dict]) -> list[dict]:
    blocks = []
    for cluster in clusters:
        members = cluster.get("members", [])
        blocks += [heading(f"{cluster['id']} {cluster.get('name', '')}: {cluster.get('count', len(members))} объектов", 3)]
        blocks.extend(_comparison_prose(cluster.get("interpretation")))
        if cluster.get("factor_center"):
            blocks.append(paragraph("Центр в факторных координатах: " + "; ".join(
                f"{identity}={_comparison_number(value, '+.3f')}" for identity, value in cluster["factor_center"].items()) + "."))
        blocks.append(paragraph("Полный состав: " + ", ".join(item["name"] if isinstance(item, dict) else str(item) for item in members)))
        if cluster.get("representatives"):
            blocks.append(paragraph("Ближайшие к центру: " + ", ".join(item["name"] for item in cluster["representatives"])))
        traits = cluster.get("distinguishing_scales", [])
        if traits:
            blocks.append(table(["Шкала", "Среднее группы", "Среднее выборки", "Различие в σ"],
                                [[trait["name"], _comparison_number(trait.get("group_mean"), "+.2f"),
                                  _comparison_number(trait.get("overall_mean"), "+.2f"),
                                  _comparison_number(trait.get("difference"), "+.2f")] for trait in traits]))
    return blocks


def _comparison_changes(pair: dict) -> list[dict]:
    changed = [row for row in pair.get("transitions", []) if row.get("changed") is True]
    if pair.get("moved_count") is not None and len(changed) != pair["moved_count"]:
        raise ValueError("Число переходов не совпадает с полным списком изменивших группу объектов")
    if not changed:
        return [paragraph("После согласования меток переходов между группами нет.")]
    return [table(["Объект", "Исходная группа", "Согласованная новая группа"],
                  [[row["name"], row["from_cluster"], row["to_cluster_aligned"]] for row in changed])]


def comparison_details(comparison: dict) -> list[dict]:
    if not comparison or not comparison.get("models"):
        return []
    blocks = [heading("Сравнение факторных моделей: подробности"),
              paragraph("Номер и знак оси условны. При изменении q нагрузки поворачиваются заново: оси могут менять порядок, "
                        "направление и распределение содержания. F1 модели q=3 нельзя автоматически считать той же осью, "
                        "что F1 модели q=2. Соответствие проверяется по полным векторам нагрузок, а не по одинаковому номеру."),
              paragraph("Доли дисперсии относятся к модельному описанию стандартизованных шкал. Изменения разбиений "
                        "сопоставляются при фиксированном числе групп k. Метки групп согласованы по максимальному "
                        "пересечению составов; простая перестановка C1/C2/C3 не считается переходом объекта.")]
    for model in comparison["models"]:
        q = model.get("q", model.get("factor_count"))
        if q == 2:
            continue
        blocks.append(heading(f"Альтернативная модель q={q}", 2))
        narrative = model.get("narrative", {})
        blocks.extend(_comparison_prose(narrative.get("summary")))
        blocks.extend(_comparison_prose(narrative.get("new_meanings")))
        factors = model.get("factors", [])
        if factors:
            blocks.append(table(["Ось", "Интерпретация", "Доля дисперсии"],
                                [[factor["id"], factor.get("label") or factor.get("name", ""),
                                  _comparison_number(factor.get("variance_share"), ".1%")]
                                 for factor in factors]))
        for factor in factors:
            blocks.append(heading(f"q={q}, {factor['id']}: {factor.get('label') or factor.get('name', '')}", 3))
            blocks.extend(_comparison_prose(factor.get("interpretation")))
            rows = [[pole, trait["name"], _comparison_number(trait.get("loading"), "+.3f")]
                    for pole, key in (("Положительный", "positive_traits"), ("Отрицательный", "negative_traits"))
                    for trait in factor.get(key, [])]
            if rows:
                blocks.append(table(["Полюс", "Шкала", "Нагрузка"], rows))
        if model.get("loading_table"):
            identities = [factor["id"] for factor in factors] or list(model["loading_table"][0]["loadings"])
            blocks += [heading(f"Полная матрица нагрузок q={q}", 3),
                       table(["Шкала"] + identities,
                             [[row["name"]] + [_comparison_number(row["loadings"].get(identity), "+.3f") for identity in identities]
                              for row in model["loading_table"]])]
        blocks.extend(_comparison_prose(narrative.get("cluster_changes")))
        if model.get("clusters"):
            blocks.append(heading(f"Профили групп q={q} при фиксированном k={model.get('cluster_count', 3)}", 3))
            blocks.extend(_comparison_clusters(model["clusters"]))
    for match in comparison.get("factor_comparisons", []):
        source, target = match["reference_q"], match["target_q"]
        blocks.append(heading(f"Соответствие осей: q={source} и q={target}", 2))
        if match.get("matched_factors"):
            blocks.append(table([f"Ось q={source}", f"Ось q={target}", "Конгруэнтность", "Согласование знака"],
                                [[row["reference_factor"], row["target_factor"], _comparison_number(row.get("congruence")),
                                  "Обратное направление" if row.get("sign", 1) < 0 else "Одинаковое направление"]
                                 for row in match["matched_factors"]]))
        matrix = match.get("congruence_matrix", {})
        if matrix.get("values"):
            blocks.append(table([f"q={source} / q={target}"] + matrix["column_factors"],
                                [[identity] + [_comparison_number(value) for value in row]
                                 for identity, row in zip(matrix["row_factors"], matrix["values"])]))
        if match.get("unmatched_target_factors"):
            blocks.append(paragraph(f"Оси q={target} без отдельной пары в q={source}: " + ", ".join(match["unmatched_target_factors"]) +
                                    ". Отсутствие пары отражает различие размерности; содержательная новизна оценивается по нагрузкам."))
        if match.get("principal_angles_degrees"):
            blocks.append(paragraph("Главные углы между факторными подпространствами: " +
                                    ", ".join(_comparison_number(angle, ".1f") + "°" for angle in match["principal_angles_degrees"]) +
                                    ". Это геометрическая диагностика близости подпространств."))
        if match.get("procrustes", {}).get("relative_loading_residual") is not None:
            blocks.append(paragraph("Относительный остаток нагрузок после отображения Прокруста: " +
                                    _comparison_number(match["procrustes"]["relative_loading_residual"]) +
                                    ". Отображение сопоставляет подпространства и не заменяет исходные именованные оси varimax."))
    fixed = comparison.get("fixed_k3", [])
    if fixed:
        blocks += [heading("Составы групп при одинаковом k=3", 2),
                   table(["Модели q", "ARI", "NMI", "Сохранили группу", "Сменили группу"],
                         [[f"{row['reference_q']} → {row['target_q']}", _comparison_number(row.get("ari")),
                           _comparison_number(row.get("nmi")), row.get("unchanged_count", "—"), row.get("moved_count", "—")]
                          for row in fixed])]
        for row in fixed:
            blocks.extend(_comparison_contingency(row.get("contingency_aligned", {}),
                          f"Пересечение составов q={row['reference_q']} и q={row['target_q']} после согласования меток, k={row.get('k', 3)}."))
            blocks.append(heading(f"Все переходы: q={row['reference_q']} → q={row['target_q']}, k={row.get('k', 3)}", 3))
            blocks.extend(_comparison_changes(row))
    free = comparison.get("numerical_free_k", [])
    if free:
        blocks += [heading("Свободный выбор числа групп", 2),
                   paragraph("Численное предпочтение k — диагностическая рекомендация. Рабочий выбор дополнительно учитывает "
                             "содержательную различимость полных профилей; увеличение числа групп может лишь дробить близкие сочетания свойств.")]
        for row in free:
            blocks += [heading(f"Кандидаты k при q={row['q']}", 3),
                       paragraph(f"Численная рекомендация: k={row['recommended_k']}. " + str(row.get("selection_rule", ""))),
                       table(["k", "Силуэт", "Мин. группа", "Макс. группа", "Средний ARI", "Мин. медианный Jaccard"],
                             [[item["k"], _comparison_number(item.get("silhouette")), item.get("min_size", "—"),
                               item.get("max_size", "—"), _comparison_number(item.get("stability", {}).get("mean_ari")),
                               _comparison_number(item.get("stability", {}).get("min_cluster_median_jaccard"))]
                              for item in row.get("candidates", [])])]
            model = next((model for model in comparison["models"] if model.get("q") == row["q"]), {})
            if model.get("candidate_review"):
                blocks.append(table(["k", f"Содержательное сравнение при q={row['q']}"],
                                    [[item["k"], item["interpretation"]] for item in model["candidate_review"]]))
    selected = comparison.get("selected_k", [])
    if selected:
        blocks += [heading("Рабочие варианты с отдельно выбранным k", 2),
                   paragraph("Эти варианты дополняют сравнение при фиксированном k=3. Если число групп различается, "
                             "таблицы пересечений показывают дробление и объединение составов. Частичное сопоставление групп "
                             "не превращается в число объектов, якобы сменивших группу; ARI и NMI сравнивают разбиения целиком.")]
        for row in selected:
            q, k = row["q"], row["k"]
            blocks.append(heading(f"Рабочая типология q={q}, k={k}", 3))
            blocks.extend(_comparison_prose(row.get("selection_reason")))
            quality = row.get("quality", {})
            blocks.append(table(["Силуэт", "Средний ARI", "Медианный ARI", "Минимальный ARI", "Мин. медианный Jaccard"],
                                [[_comparison_number(quality.get(key)) for key in
                                  ("silhouette", "mean_ari", "median_ari", "min_ari", "min_cluster_median_jaccard")]]))
            for key, caption in (("relative_to_q2", "С базовой моделью q=2, k=3"),
                                 ("relative_to_fixed_k3", f"С моделью q={q} при фиксированном k=3")):
                relative = row.get(key, {})
                if relative:
                    blocks.append(paragraph(f"{caption}: ARI={_comparison_number(relative.get('ari'))}; "
                                            f"NMI={_comparison_number(relative.get('nmi'))}."))
                    blocks.extend(_comparison_contingency(relative.get("overlap", {}), caption + ": пересечение составов."))
            fixed_model = next((model for model in comparison["models"] if model.get("q") == q), {})
            if k == fixed_model.get("cluster_count"):
                blocks.append(paragraph("Рабочее k совпадает с фиксированным: полный состав и профили групп приведены выше для этой модели."))
            elif row.get("clusters"):
                blocks.extend(_comparison_clusters(row["clusters"]))
    preferred_pair = comparison.get("preferred_k4_comparison")
    if preferred_pair:
        source, target, k = preferred_pair["reference_q"], preferred_pair["target_q"], preferred_pair.get("k", 4)
        blocks += [heading(f"Согласованность рабочих типологий: q={source} и q={target}, k={k}", 2),
                   table(["ARI", "NMI", "Сохранили группу", "Сменили группу"],
                         [[_comparison_number(preferred_pair.get("ari")), _comparison_number(preferred_pair.get("nmi")),
                           preferred_pair.get("unchanged_count", "—"), preferred_pair.get("moved_count", "—")]])]
        blocks.extend(_comparison_contingency(preferred_pair.get("contingency_aligned", {}),
                      f"Пересечение рабочих типологий q={source} и q={target}, k={k}, после согласования меток."))
        blocks.append(heading(f"Все переходы между рабочими типологиями q={source} и q={target}, k={k}", 3))
        blocks.extend(_comparison_changes(preferred_pair))
    for chart in comparison.get("charts", []):
        file = str(chart["file"]).replace("\\", "/")
        if file.startswith("/") or ":" in file or ".." in file.split("/"):
            raise ValueError("Сравнительная диаграмма должна находиться внутри output")
        blocks += [heading(chart.get("title", "Сравнение моделей"), 2),
                   {"type": "image", "file": file, "caption": chart.get("caption") or chart.get("title", "")}]
    blocks.extend(_comparison_prose(comparison.get("limitations")))
    return blocks


def report_blocks(study: dict, analysis: dict, ratings: list[dict]) -> list[dict]:
    objects, scales = study["objects"], study["scales"]
    observed = sum(r.get("status") == "observed" for r in ratings)
    n_cells = len(objects) * len(scales)
    selection = study.get("selection_method") or "Целевая выборка по заданному пользователем описанию; не случайная выборка и не измеренный рейтинг известности."
    if isinstance(selection, dict):
        selection = selection.get("description", str(selection))
    blocks = [heading(study["title"], 0),
              paragraph(f"Область: {study['scope']}. Аспект: {study['aspect']}. Период: {study['date_from']} — {study['date_to']}."),
              heading("Основные результаты"),
              paragraph(f"Проанализировано {analysis['n_objects']} объектов по {analysis['n_scales']} пригодным шкалам. "
                        f"Факторная модель содержит {analysis['factor_count']} осей; суммарная описанная дисперсия стандартизованных переменных — "
                        f"{analysis['variance_total']:.1%}. Получено {analysis['cluster_count']} групп объектов. "
                        "Результат представляет разведочную модель публичных образов в выбранном корпусе источников."),
              paragraph("Группы выделены по численным профилям объектов. Их названия — интерпретации модели; они не доказывают общность реальных электоратов, личных качеств или наличие общественного спроса на незаполненные области."),
              heading("Методика"),
              paragraph(selection),
              paragraph("После утверждения объектов собраны досье с источниками, выделены определения, объединены почти одинаковые синонимы и нормализован уровень описания. Словарь шкал утверждается отдельно. Алиасы и решения о нормализации сохраняются."),
              paragraph("Каждый полюс шкалы оценён независимо от 0 до 5. Основная координата — разность положительного и отрицательного полюсов от −5 до +5; сумма хранится как показатель смешанности. Недостаток сведений обозначается пропуском. Численный ноль не заменяет неизвестное."),
              paragraph(f"Наблюдаемых оценок: {observed} из {n_cells} ({observed/max(1,n_cells):.1%}). "
                        f"Режим измерения: {study.get('measurement_mode', 'llm_exploratory')}. "
                        "Оценки LLM не являются ответами граждан или независимыми человеческими экспертными оценками. "
                        "Досье — выборка материалов, а не исчерпывающее описание объекта за десять лет."),
              paragraph("Используется вероятностная Gaussian факторная модель с диагональным специфическим шумом. Координаты — условные средние факторов; рассчитывается система по числу факторов без обращения вырожденной эмпирической корреляционной матрицы. Нагрузки и баллы согласованно поворачиваются ортогональным varimax."),
              paragraph("Доля дисперсии фактора вычислена как сумма квадратов его нагрузок, делённая на число стандартизованных признаков. Сумма долей описывает общую модельную дисперсию в этом решении; это не доля населения и не оценка правильности интерпретации. Конечные шкалы рассматриваются как непрерывная разведочная аппроксимация."),
              paragraph("Число факторов сопоставляется с диагностикой размерности, качеством модели и интерпретируемостью. Для кластеризации сравниваются несколько k, силуэт, размеры и устойчивость на подвыборках. В режиме LLM отдельно сравниваются содержательные профили всех разбиений: насколько новые группы различаются по свойствам, а не только по именам. Это модельное интерпретационное решение, не независимая экспертная проверка; k можно изменить и сравнить. t-SNE служит иллюстрацией соседства и не определяет кластеры."),
              heading("Словарь измерительных шкал"),
              table(["Шкала", "+5", "−5", "Определение"], [[s["name"], s["positive"], s["negative"], s["definition"]] for s in scales]),
              heading("Факторная структура"),
              table(["Ось", "Интерпретация", "Доля дисперсии"], [[f["id"], f.get("label") or f.get("name"), f"{f['variance_share']:.1%}"] for f in analysis["factors"]])]
    diagnostics = analysis.get("diagnostics", {})
    overview_at = next(index for index, block in enumerate(blocks)
                       if block.get("type") == "heading" and block.get("text") == "Методика")
    blocks[overview_at:overview_at] = comparison_overview(analysis.get("model_comparison", {}))
    insert_at = next(index for index, block in enumerate(blocks)
                     if block.get("type") == "heading" and block.get("text") == "Словарь измерительных шкал")
    provenance = study.get("scale_provenance", {})
    normalization = []
    if provenance:
        normalization = [heading("Происхождение и нормализация словаря", 2),
                         paragraph(f"Досье при подготовке словаря: {provenance.get('source_object_count', len(objects))}; "
                                   f"исходных описательных формулировок: {provenance.get('descriptor_count', 'не зафиксировано')}. "
                                   "Числа отражают сбор материалов и генерацию словаря, а не независимые экспертные голоса.")]
        review = provenance.get("manual_review", {})
        if review:
            normalization += [paragraph(f"До проверки: {review['initial_count']} шкал; после проверки: {review['final_count']}. "
                                        "Изменённый словарь отдельно утверждён пользователем.")]
        normalization += [paragraph(note) for note in review.get("notes", provenance.get("normalization_notes", []))]
        normalization += [table(["Шкала", "Сохранённые близкие формулировки", "Объектов происхождения"],
                               [[scale["name"], "; ".join(scale.get("aliases", [])), scale.get("descriptor_object_count", "—")]
                                for scale in scales]),
                          paragraph("Число объектов происхождения относится к материалам генерации словаря. Оно не измеряет выраженность свойства и не используется как дополнительный вес шкалы.")]
    quality = [heading("Полнота оценок и отбор для расчёта", 2),
               paragraph(f"Исходно задано {len(objects)} объектов и {len(scales)} шкал; "
                         f"в расчёт вошло {analysis['n_objects']} объектов и {analysis['n_scales']} шкал. "
                         f"Доля известных ячеек используемой матрицы: {diagnostics.get('retained_coverage', 0):.1%}. "
                         "Объекту требуется не менее 60% известных шкал и не менее четырёх оценок; "
                         "шкале — не менее 60% объектов и восьми оценок. В каждой итерации сначала проверяются шкалы, "
                         "затем объекты по оставшимся шкалам; отбор повторяется до стабильного состава. Постоянные шкалы исключаются. "
                         "После отбора требуется не менее 75% известных ячеек. Пропуски не считаются нейтральными ответами. "
                         "Исключение шкалы из-за нехватки оценок означает ограничение корпуса и измерения, а не отсутствие соответствующих идей в политическом пространстве.")]
    if diagnostics.get("mass_mean") is not None:
        quality.append(paragraph(f"Средняя сумма двух полюсов среди исходных наблюдаемых оценок: {diagnostics['mass_mean']:.2f} из 10. "
                                 f"В {diagnostics.get('zero_difference_with_mass', 0)} ячейках исходной матрицы разность равна нулю при ненулевой сумме. "
                                 "Такие сочетания отличаются от подтверждённой невыраженности обоих полюсов. Они могут отражать разные контексты или изменение позиции за период, а не внутреннюю психологическую противоречивость."))
    for key, caption in (("excluded_objects", "Исключённые из расчёта объекты"),
                         ("excluded_scales", "Исключённые из расчёта шкалы")):
        if diagnostics.get(key):
            quality += [heading(caption, 3), table(["Название", "Причина"],
                        [[row.get("name", row["id"]), row.get("reason", "")] for row in diagnostics[key]])]
    supplements = study.get("evidence_summary", {}).get("supplement_objects", 0)
    if supplements:
        quality.append(paragraph(f"Для {supplements} объектов выполнен однократный целевой добор источников по неизвестным шкалам. "
                                 "Повторно оценивались только неизвестные ячейки, первые наблюдаемые оценки сохранены. Исходные досье и первый проход доступны в исследовательских файлах."))
    completion_objects = study.get("evidence_summary", {}).get("completion_objects", 0)
    if completion_objects:
        quality.append(paragraph(f"Дополнительно для {completion_objects} неполных профилей пилота выполнен последний ограниченный аудит: "
                                 "поиск только по неизвестным ячейкам шкал общего ядра. Предыдущие наблюдаемые оценки не менялись; "
                                 "новые основания, остаточные пробелы и журнал аудита сохранены отдельно. Дальнейший автоматический добор не выполнялся."))
    blocks[insert_at:insert_at] = normalization + quality
    if analysis.get("dimension_candidates"):
        dimension_rows = analysis["dimension_candidates"]
        blocks += [heading("Сопоставление числа факторов", 2),
                   paragraph("Проверены альтернативы вероятностной модели на трёх отложенных разбиениях объектов. "
                             "Параллельный анализ перестановок относится к PCA и используется как диагностика размерности, "
                             "а не как тождественный критерий числа общих факторов. Правило и показатели каждого варианта сохранены в metadata.json.")]
        blocks.append(table(["Факторов", "Средняя log likelihood на отложенных объектах", "RMS остаточных корреляций"],
                      [[row.get("q", row.get("factor_count", "")),
                        f"{row.get('heldout_log_likelihood_mean', row.get('heldout_log_likelihood', 0)):.3f}",
                        f"{row.get('residual_rms_off_diagonal', 0):.3f}"]
                       for row in dimension_rows]))
    if analysis.get("interpretation_review"):
        blocks += [paragraph("Словесные выводы проверены по нагрузкам и измеренным шкалам. "
                             "Редакторские уточнения формулировок и основания проверки сохранены в metadata.json; численные оценки не менялись.")]
    charts = analysis.get("charts", [])
    if charts:
        for chart in charts[:1]:
            blocks.extend([{"type": "image", "file": chart["file"], "caption": chart.get("caption") or chart["title"]}])
    for factor in analysis["factors"]:
        blocks.extend([heading(f"{factor['id']} {factor.get('label') or factor.get('name')}", 2),
                       paragraph(factor.get("interpretation", ""))])
        rows = []
        for pole, key in (("Положительный", "positive_traits"), ("Отрицательный", "negative_traits")):
            rows += [[pole, t["name"], f"{t['loading']:+.3f}"] for t in factor.get(key, [])[:6]]
        blocks.append(table(["Полюс", "Шкала", "Нагрузка"], rows))
    if analysis.get("loading_table"):
        factor_ids = [factor["id"] for factor in analysis["factors"]]
        blocks += [heading("Полная матрица факторных нагрузок", 2),
                   paragraph("Приведены все шкалы, вошедшие в расчёт. Факторный анализ не прикрепляет каждую шкалу к единственной группе: "
                             "возможны перекрёстные нагрузки и слабая связь со всеми выделенными факторами. Для интерпретации выше используется порог |нагрузки| ≥ 0,35. "
                             "Знак относится к положительному свойству шкалы; направление каждой оси условно."),
                   table(["Шкала"] + factor_ids,
                         [[row["name"]] + [f"{row['loadings'][identity]:+.3f}" for identity in factor_ids]
                          for row in analysis["loading_table"]])]
    blocks.append(heading("Карты объектов и признаков"))
    for chart in charts[1:]:
        blocks.extend([heading(chart["title"], 2), {"type": "image", "file": chart["file"], "caption": chart.get("caption", "")}])
    blocks.extend([heading("Сравнение кластерных решений"),
                   table(["k", "Силуэт", "Мин. группа", "Макс. группа", "Средний ARI", "Мин. медианный Jaccard групп"],
                         [[r["k"], f"{r.get('silhouette', 0):.3f}", r.get("min_size", ""), r.get("max_size", ""),
                           f"{r.get('stability_mean_ari', 0):.3f}",
                           f"{r.get('stability', {}).get('min_cluster_median_jaccard', 0):.3f}"] for r in analysis.get("cluster_candidates", [])]),
                   paragraph(analysis.get("cluster_selection_reason") or analysis.get("diagnostics", {}).get("cluster_selection_reason", "Выбранное k рассматривается как рабочая типология и сопоставляется с альтернативами.")),
                   heading("Интерпретация групп")])
    review = analysis.get("cluster_content_review")
    if review:
        blocks.insert(len(blocks) - 1, table(["k", "Содержательное сравнение"],
                     [[row["k"], row["interpretation"]] for row in review.get("alternatives", [])]))
    for cluster in analysis["clusters"]:
        blocks.extend([heading(f"{cluster['id']} {cluster.get('name', '')}", 2),
                       paragraph(f"Размер: {cluster['count']} объектов ({cluster['count']/analysis['n_objects']:.1%} выборки). " + cluster.get("interpretation", "")),
                       paragraph("Состав: " + ", ".join(o["name"] if isinstance(o, dict) else str(o) for o in cluster["members"])),
                       paragraph("Ближайшие к центру представители: " + ", ".join(o["name"] for o in cluster.get("representatives", [])[:5]))])
        blocks.append(table(["Отличительная шкала", "Среднее группы", "Среднее выборки", "Различие в σ"],
                            [[s["name"], f"{s['group_mean']:+.2f}", f"{s['overall_mean']:+.2f}", f"{s['difference']:+.2f}"]
                             for s in cluster.get("distinguishing_scales", [])[:6]]))
    blocks.append(heading("Интерпретация изученного пространства"))
    blocks.append(paragraph(analysis.get("space_interpretation") or
                            "Структура задаётся совместными изменениями шкал публичного образа. Близость в факторном пространстве означает сходство оценённых профилей; дистанция показывает различия по выбранным измерениям. Кластеры описывают преобладающие сочетания этих измерений, а пограничные объекты могут разделять свойства нескольких групп."))
    blocks.append(paragraph("Слабая заполненность участка этой карты означает отсутствие соответствующего профиля в выбранной выборке и корпусе. Для вывода об идеях, отсутствующих в России по сравнению с другими странами, требуется международная сравнительная выборка, оценённая по тому же словарю. По одному российскому исследованию такой вывод не делается."))
    focus = str(study.get("focus") or "").strip().casefold()
    focused = [o for o in analysis.get("object_profiles", []) if focus and (focus in o["name"].casefold() or o["id"].casefold() == focus)]
    if focused:
        blocks.append(heading("Выбранные объекты подробнее"))
        for obj in focused:
            blocks.extend([heading(obj["name"], 2), paragraph(str(obj.get("profile", ""))),
                           paragraph(f"Кластер: {obj.get('cluster_id')}. Оценено шкал: {obj.get('observed_scales')}; пропусков: {obj.get('missing_scales')}."),
                           table(["Фактор", "Координата"], [[k, f"{v:+.3f}"] for k, v in obj["factor_scores"].items()])])
    blocks.extend(comparison_details(analysis.get("model_comparison", {})))
    blocks.extend([heading("Ограничения и воспроизводимость"),
                   paragraph("Выборка целевая и ограничена заданными критериями. Агрегирование десятилетнего периода может скрыть смену образа. Зависимые источники, смещения поисковой выдачи и оценщика могут влиять на результат. "
                             "Пропуски могут быть систематическими: публичные роли и тематическая освещённость влияют на доступность оценок. Заполнение средним не устраняет это смещение. "
                             "Сбор досье, оценивание, выбор k и словесная интерпретация моделью не являются независимыми проверками друг друга. Ни статистическая устойчивость, ни убедительность названия фактора сами по себе не подтверждают его внешнюю валидность."),
                   paragraph("Сохранены версии списков, подтверждения пользователя, источники, досье, поэлементные оценки с основаниями, матрицы и параметры модели. Повторный запуск при неизменном запросе использует кэш; это воспроизведение, а не независимое повторное измерение."),
                   paragraph(f"Численный расчёт: {analysis['n_objects']} объектов, {analysis['n_scales']} шкал, "
                             f"{analysis['factor_count']} факторов и {analysis['cluster_count']} кластеров. "
                             "Полные показатели качества, правила выбора размерности, пропуски, параметры и seed сохранены в metadata.json рядом с матрицами."),
                   heading("Объекты и источники")])
    warning_at = next(index for index, block in enumerate(blocks)
                      if block.get("type") == "heading" and block.get("text") == "Ограничения и воспроизводимость") + 1
    blocks[warning_at:warning_at] = [paragraph(warning) for warning in analysis.get("warnings", [])]
    for obj in objects:
        blocks.extend([heading(obj["name"], 2), paragraph(obj.get("role", "") + ". " + obj.get("inclusion_reason", ""))])
        for source in obj.get("sources", []):
            blocks.append({"type": "link", "text": source.get("title") or source.get("url"), "url": source.get("url", "")})
    return blocks


def markdown(blocks: list[dict]) -> str:
    parts = []
    safe = lambda text: str(text).replace("|", "\\|").replace("\n", " ")
    for block in blocks:
        kind = block["type"]
        if kind == "heading":
            parts.append("#" * (block["level"] + 1) + " " + block["text"])
        elif kind == "paragraph":
            parts.append(block["text"])
        elif kind == "table":
            parts.append("| " + " | ".join(map(safe, block["headers"])) + " |\n| " + " | ".join("---" for _ in block["headers"]) + " |\n" +
                         "\n".join("| " + " | ".join(map(safe, row)) + " |" for row in block["rows"]))
        elif kind == "image":
            parts.append(f"![{block['caption']}]({block['file']})")
        elif kind == "link":
            parts.append(f"[{block['text']}]({block['url']})")
    return "\n\n".join(parts) + "\n"


def html_document(blocks: list[dict]) -> str:
    esc, result = html.escape, []
    for block in blocks:
        kind = block["type"]
        if kind == "heading":
            tag = "h" + str(min(6, block["level"] + 1))
            result.append(f"<{tag}>{esc(block['text'])}</{tag}>")
        elif kind == "paragraph":
            result.append("<p>" + esc(block["text"]) + "</p>")
        elif kind == "table":
            result.append("<div class='table-wrap'><table><thead><tr>" + "".join("<th>" + esc(v) + "</th>" for v in block["headers"]) + "</tr></thead><tbody>" +
                          "".join("<tr>" + "".join("<td>" + esc(v) + "</td>" for v in row) + "</tr>" for row in block["rows"]) + "</tbody></table></div>")
        elif kind == "image":
            result.append(f"<figure><img src='{esc(block['file'], quote=True)}' alt='{esc(block['caption'], quote=True)}'><figcaption>{esc(block['caption'])}</figcaption></figure>")
        elif kind == "link":
            url = str(block.get("url", ""))
            if url.startswith(("https://", "http://")):
                result.append(f"<p><a href='{esc(url, quote=True)}' target='_blank' rel='noopener noreferrer'>{esc(block['text'])}</a></p>")
    css = "body{font:16px/1.65 Arial,sans-serif;color:#182a3c;max-width:1060px;margin:48px auto;padding:0 28px}h1{font-size:32px;line-height:1.25}h2{margin-top:42px;font-size:25px}h3{margin-top:30px;font-size:20px}p{margin:14px 0}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border:1px solid #d8e1e9;padding:10px;text-align:left;vertical-align:top}th{background:#edf3f6}.table-wrap{overflow:auto}img{width:100%;height:auto}figure{margin:24px 0}figcaption{font-size:13px;color:#526271}a{color:#146b73;overflow-wrap:anywhere}@media print{body{margin:0;max-width:none;font-size:11pt}h2,h3{break-after:avoid}figure,table{break-inside:avoid}}"
    return "<!doctype html><html lang='ru'><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>Отчёт исследования</title><style>" + css + "</style><body>" + "\n".join(result) + "</body></html>"


def write_report_bundle(output: Path) -> None:
    """Keep model-relative assets intact in the reproducibility download."""
    output = Path(output).resolve()
    with zipfile.ZipFile(output / "report_bundle.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        def add(file: Path, name: str) -> None:
            if file.is_file() and not file.is_symlink() and file.resolve().is_relative_to(output):
                archive.write(file, name)

        for name in ("report.md", "report.html", "report.pdf", "report.docx", "slides.pptx", "report_content.json"):
            add(output / name, name)
        for folder in ("assets", "models"):
            directory = output / folder
            if directory.is_symlink():
                continue
            for file in sorted(directory.rglob("*")):
                relative = file.relative_to(output)
                # Exclude hidden runtime material and nested archives, including
                # an old bundle that would recursively inflate the next export.
                if any(part.startswith(".") for part in relative.parts) or file.suffix.lower() in {".zip", ".7z", ".tar", ".gz"}:
                    continue
                add(file, relative.as_posix())
        for file in sorted(output.iterdir()):
            if file.suffix in {".csv", ".joblib"} or file.name == "metadata.json":
                add(file, "analysis/" + file.name)
        # Research records live beside output by design; scope these reads to
        # the study directory rather than the report's nested model folders.
        study_directory = output.parent.resolve()
        for name in ("study.json", "ratings.json"):
            file = study_directory / name
            if file.is_file() and not file.is_symlink() and file.resolve().is_relative_to(study_directory):
                archive.write(file, "research/" + name)
        for file in sorted((study_directory / "evidence").glob("*.json")):
            if file.is_file() and not file.is_symlink() and file.resolve().is_relative_to(study_directory):
                archive.write(file, "research/evidence/" + file.name)


def export_report(study: dict, analysis: dict, ratings: list[dict], output: Path,
                  root: Path, runtime_python: str | None = None, runtime_node: str | None = None) -> list[dict]:
    output.mkdir(parents=True, exist_ok=True)
    study = json.loads(json.dumps(study, ensure_ascii=False))
    study["evidence_summary"] = {"supplement_objects": len(list((output.parent / "evidence").glob("dossier_supplement-*.json"))),
                                 "completion_objects": len(list((output.parent / "evidence").glob("coverage_completion-*.json")))}
    for obj in study["objects"]:
        known = {str(item.get("url", "")).rstrip("/") for item in obj.get("sources", [])}
        obj["sources"] = list(obj.get("sources", []))
        for name in (obj["id"] + ".json", "dossier_supplement-" + obj["id"] + ".json", "coverage_completion-" + obj["id"] + ".json"):
            evidence = output.parent / "evidence" / name
            if not evidence.exists():
                continue
            dossier = json.loads(evidence.read_text(encoding="utf-8"))
            citations = list(dossier.get("retrieval_citations", [])) + dossier.get("supplement_provenance", {}).get("retrieval_citations", [])
            allowed = {str(url).rstrip("/") for url in citations}
            sources = list(dossier.get("sources", [])) + [{"url": fact.get("source_url"), "title": "Материал исследовательского досье"} for fact in dossier.get("facts", [])]
            for source in sources:
                url = str(source.get("url", "")).rstrip("/")
                if url in allowed and url not in known:
                    obj["sources"].append(source)
                    known.add(url)
    blocks = report_blocks(study, analysis, ratings)
    (output / "report.md").write_text(markdown(blocks), encoding="utf-8")
    (output / "report.html").write_text(html_document(blocks), encoding="utf-8")
    payload = {"study": study, "analysis": analysis, "blocks": blocks}
    (output / "report_content.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    python = runtime_python or os.environ.get("SEMANTIC_EXPORT_PYTHON") or sys.executable
    worker = root / "src/semantic_service/export_documents.py"
    subprocess.run([python, str(worker), str(output / "report_content.json"), str(output)],
                   check=True, capture_output=True, text=True, encoding="utf-8", timeout=180)
    node = runtime_node or os.environ.get("SEMANTIC_NODE") or shutil.which("node")
    if not node:
        raise ValueError("Для экспорта презентации нужен Node.js; задайте SEMANTIC_NODE")
    deck = subprocess.run([node, str(root / "src/semantic_service/presentation.mjs"),
                           str(output / "report_content.json"), str(output)],
                          capture_output=True, text=True, encoding="utf-8", timeout=420)
    if deck.returncode:
        raise ValueError("Ошибка экспорта презентации: " + deck.stderr[-1800:])
    write_report_bundle(output)
    label = {"md": "Отчёт Markdown", "html": "Открыть отчёт", "pdf": "Отчёт PDF", "docx": "Для Google Docs с диаграммами",
             "pptx": "Слайды PowerPoint", "zip": "Все форматы и диаграммы"}
    names = ["report.md", "report.html", "report.pdf", "report.docx", "slides.pptx", "report_bundle.zip"]
    return [{"label": label[Path(name).suffix[1:]], "format": Path(name).suffix[1:],
             "url": f"/api/studies/{study['id']}/files/{name}"} for name in names if (output / name).exists()]
