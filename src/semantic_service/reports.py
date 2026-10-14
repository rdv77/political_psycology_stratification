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
                   table(["k", "Силуэт", "Минимальная группа", "Максимальная группа"],
                         [[r["k"], f"{r.get('silhouette', 0):.3f}", r.get("min_size", ""), r.get("max_size", "")] for r in analysis.get("cluster_candidates", [])]),
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
    blocks.extend([heading("Ограничения и воспроизводимость"),
                   paragraph("Выборка целевая и ограничена заданными критериями. Агрегирование десятилетнего периода может скрыть смену образа. Зависимые источники, смещения поисковой выдачи и оценщика могут влиять на результат. Ни статистическая устойчивость, ни убедительность названия фактора сами по себе не подтверждают его внешнюю валидность."),
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


def export_report(study: dict, analysis: dict, ratings: list[dict], output: Path,
                  root: Path, runtime_python: str | None = None, runtime_node: str | None = None) -> list[dict]:
    output.mkdir(parents=True, exist_ok=True)
    study = json.loads(json.dumps(study, ensure_ascii=False))
    study["evidence_summary"] = {"supplement_objects": len(list((output.parent / "evidence").glob("dossier_supplement-*.json")))}
    for obj in study["objects"]:
        known = {str(item.get("url", "")).rstrip("/") for item in obj.get("sources", [])}
        obj["sources"] = list(obj.get("sources", []))
        for name in (obj["id"] + ".json", "dossier_supplement-" + obj["id"] + ".json"):
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
    with zipfile.ZipFile(output / "report_bundle.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("report.md", "report.html", "report.pdf", "report.docx", "slides.pptx", "report_content.json"):
            file = output / name
            if file.exists():
                archive.write(file, name)
        for file in (output / "assets").glob("*.png"):
            archive.write(file, "assets/" + file.name)
        for file in output.iterdir():
            if file.is_file() and (file.suffix in {".csv", ".joblib"} or file.name == "metadata.json"):
                archive.write(file, "analysis/" + file.name)
        for name in ("study.json", "ratings.json"):
            file = output.parent / name
            if file.exists():
                archive.write(file, "research/" + name)
        for file in (output.parent / "evidence").glob("*.json"):
            archive.write(file, "research/evidence/" + file.name)
    label = {"md": "Отчёт Markdown", "html": "Открыть отчёт", "pdf": "Отчёт PDF", "docx": "Для Google Docs с диаграммами",
             "pptx": "Слайды PowerPoint", "zip": "Все форматы и диаграммы"}
    names = ["report.md", "report.html", "report.pdf", "report.docx", "slides.pptx", "report_bundle.zip"]
    return [{"label": label[Path(name).suffix[1:]], "format": Path(name).suffix[1:],
             "url": f"/api/studies/{study['id']}/files/{name}"} for name in names if (output / name).exists()]
