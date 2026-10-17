"""Local cleaning export and a calibrated budget; never sends model requests.

Original CSVs, source text and per-document plans stay in .corpus_runtime.
Only aggregate measurements are optionally published in the repository.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.corpus_pipeline.service import CorpusPipeline, source_context, publication_date, digest
from src.corpus_pipeline.ingest import combine_publication_date
from src.corpus_pipeline.cleaning import annotate_document, summarize_annotations, candidate_families
from src.corpus_pipeline.context_policy import select_document_context, CONTEXT_POLICY_VERSION
from src.corpus_pipeline.excerpts import offline_encoding, select_spans_with_metadata
from src.corpus_pipeline.budget_forecast import forecast
from src.corpus_pipeline.preparation_cache import create_checkpoint, load_checkpoint


def packed(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def quantiles(values):
    values = sorted(values)
    return {str(p): values[min(len(values) - 1, int((len(values) - 1) * p / 100))] if values else None
            for p in (0, 25, 50, 75, 90, 95, 99, 100)}


def fragment(document, selection):
    result = {key: document.get(key, "") for key in
              ("id", "title", "url", "published_at", "source_name", "author", "language_hint")}
    result["excerpts"] = [{k: span[k] for k in ("start", "end", "text")} for span in selection["spans"]]
    return result


def cleaning_cache_hash(config, review_config):
    return digest({"config": config, "country_review_terms": review_config.get("country_review_terms", []),
                   "cleaning_version": hashlib.sha256((ROOT / "src/corpus_pipeline/cleaning.py").read_bytes()).hexdigest(),
                   "context_version": CONTEXT_POLICY_VERSION})


def repair_split_dates(pipeline, identifier):
    """Repair a pre-fix local import only before any model jobs exist.

    Both original CSV cells remain in metadata. The private change receipt
    records old/new parsed values; no response/evidence history is rewritten.
    """
    repairs = []
    with pipeline.connection() as db:
        for row in db.execute("SELECT id,data FROM documents WHERE corpus_id=?", (identifier,)):
            value = json.loads(row["data"])
            metadata = value.get("metadata", {})
            combined = combine_publication_date(metadata.get("Дата публикации", ""), metadata.get("Время публикации", ""))
            if " " not in combined or combined == value.get("published_at_raw"):
                continue
            expected = publication_date(combined)
            if expected == value.get("published_at"):
                continue
            repairs.append({"document_id": row["id"], "source_row": value["source_row"],
                            "previous_published_at": value.get("published_at"), "published_at": expected,
                            "published_at_raw": combined, "data": value})
        if repairs:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT COUNT(*) FROM jobs WHERE corpus_id=?", (identifier,)).fetchone()[0]:
                raise ValueError("Split-date repair cannot rewrite an already processed corpus")
            for repair in repairs:
                value = repair.pop("data")
                value.update(published_at=repair["published_at"], published_at_raw=repair["published_at_raw"],
                             publication_date_composition="explicit_date_and_time_no_timezone")
                db.execute("UPDATE documents SET data=? WHERE corpus_id=? AND id=?", (packed(value), identifier, repair["document_id"]))
            profile = json.loads(db.execute("SELECT profile FROM corpora WHERE id=?", (identifier,)).fetchone()[0])
            profile["field_mapping"]["published_at"] = "Дата публикации"
            profile["publication_date_composition"] = {"date_column": "Дата публикации", "time_column": "Время публикации", "timezone": None}
            db.execute("UPDATE corpora SET profile=? WHERE id=?", (packed(profile), identifier))
    if repairs:
        write(pipeline.runtime / "cleaning" / identifier / "split_date_repair.json", {
            "version": "explicit-split-publication-date-v1", "api_calls": 0, "raw_metadata_changed": False, "repairs": repairs})
    return len(repairs)


def prepare(root, identifier, pilot_snapshot, *, progress=None):
    pipeline = CorpusPipeline(root)
    dates_repaired = repair_split_dates(pipeline, identifier)
    corpus = pipeline.get_corpus(identifier)
    config = corpus["config"]
    pilot_raw = pilot_snapshot.read_bytes()
    pilot = json.loads(pilot_raw.decode("utf-8-sig"))
    review_config = {**config, "country_review_terms": pilot["corpus"]["config"].get("country_terms", [])}
    encoding = offline_encoding(pipeline.runtime / "tokenizer_cache")
    if encoding is None:
        raise ValueError("A verified local tokenizer is required for this budget calculation")
    directory = pipeline.runtime / "cleaning" / identifier
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = cleaning_cache_hash(config, review_config)
    cached = load_checkpoint(directory, corpus["source_sha256"], checkpoint_hash)
    with pipeline.connection() as db:
        rows = [] if cached is not None else db.execute("SELECT id,source_row,status,duplicate_of,data FROM documents WHERE corpus_id=? ORDER BY source_row", (identifier,)).fetchall()
        edges = [json.loads(row[0]) for row in db.execute("SELECT data FROM similar WHERE corpus_id=?", (identifier,))]
    frames, annotation_rows, statuses, exclusions = [], [], Counter(), Counter()
    types, languages, dates = Counter(), Counter(), []
    if cached is not None:
        frames, annotation_rows, statuses = cached
        if progress:
            progress({"phase": "cleaning_checkpoint_loaded", "documents": len(frames)})
    else:
        cleaned_temp = directory / "cleaned_unique.jsonl.tmp"
        index_temp = directory / "cleaning_index.jsonl.tmp"
        with cleaned_temp.open("w", encoding="utf-8") as cleaned, index_temp.open("w", encoding="utf-8") as index:
            for number, row in enumerate(rows, 1):
                statuses[row["status"]] += 1
                if row["status"] != "unique":
                    index.write(packed({"document_id": row["id"], "source_row": row["source_row"],
                                        "status": row["status"], "duplicate_of": row["duplicate_of"]}) + "\n")
                    continue
                document = json.loads(row["data"])
                annotation = annotate_document(document, review_config)
                annotation_rows.append(annotation)
                index.write(packed({"document_id": row["id"], "source_row": row["source_row"], **annotation}) + "\n")
                if annotation["status"] != "rule_excluded":
                    selected = {key: document.get(key, "") for key in
                                ("id", "source_row", "title", "text", "url", "published_at", "source_name", "author", "language_hint")}
                    selected.update(cleaning=annotation, source_context=source_context(document))
                    cleaned.write(packed(selected) + "\n")
                    frames.append((document, annotation))
                if progress and (number % 1000 == 0 or number == len(rows)):
                    progress({"phase": "local_cleaning", "completed": number, "total": len(rows)})
        cleaned_temp.replace(directory / "cleaned_unique.jsonl")
        index_temp.replace(directory / "cleaning_index.jsonl")
        create_checkpoint(directory, corpus["source_sha256"], checkpoint_hash)
    for annotation in annotation_rows:
        if annotation["status"] == "rule_excluded":
            exclusions[annotation["reason"]] += 1
    for document, annotation in frames:
        context = document.get("_source_context") or source_context(document)
        types[context["source_type"]] += 1
        languages[document.get("language_hint") or "unknown"] += 1
        if document.get("published_at"):
            dates.append(document["published_at"][:10])
    summary = summarize_annotations(annotation_rows)
    actor_counts = {actor["id"]: 0 for actor in config["objects"]}
    for document, annotation in frames:
        for actor_id in annotation["matched_actor_ids"]:
            actor_counts[actor_id] += 1
    families = candidate_families(edges)
    write(directory / "duplicate_candidates.json", families)
    plans = {"base1500": [], "adaptive6000": []}
    policy_counts, selected_tokens, original_tokens = Counter(), {name: [] for name in plans}, []
    expansion_counts = Counter()
    for number, (document, annotation) in enumerate(frames, 1):
        full_tokens = len(encoding.encode(document["text"], disallowed_special=()))
        if full_tokens <= 1500:
            # Both policies preserve a complete source that fits the base cap.
            # No repeated alias search or adaptive window construction is needed.
            base = {"spans": [{"start": 0, "end": len(document["text"]), "text": document["text"], "tokens": full_tokens}],
                    "metadata": {"max_tokens": 1500, "expanded": False, "reason": ["source_fits_base"]}}
            adaptive = base
        else:
            base = select_spans_with_metadata(document["text"], config["objects"], "mentions", 1500, encoding)
            adaptive = select_document_context(document, config["objects"], actor_counts, len(frames),
                                                {"base_max_tokens": 1500, "expanded_max_tokens": 6000}, encoding)
        metadata = adaptive["metadata"]
        reason = metadata.get("reason", ["base"])
        policy_counts["|".join(reason) if isinstance(reason, list) else str(reason)] += 1
        expansion_counts["expanded_documents"] += bool(metadata.get("expanded"))
        expansion_counts["uncovered_rare_actor_documents"] += bool(metadata.get("uncovered_rare_actor_ids"))
        for name, selection in (("base1500", base), ("adaptive6000", adaptive)):
            context = document.get("_source_context") or source_context(document)
            plans[name].append({"document": fragment(document, selection), "actor_ids": annotation["matched_actor_ids"],
                                "source_type": context["source_type"], "source_context": context,
                                "token_budget": selection["metadata"]["max_tokens"]})
            selected_tokens[name].append(sum(span["tokens"] for span in selection["spans"]))
        original_tokens.append(full_tokens)
        if progress and (number % 1000 == 0 or number == len(frames)):
            progress({"phase": "context_planning", "completed": number, "total": len(frames)})
    estimate = forecast(plans, pilot, config, encoding, progress=progress)
    actor_coverage = [{"id": actor["id"], "name": actor["name"], "unique_texts_with_name": actor_counts[actor["id"]]}
                      for actor in config["objects"]]
    public = {"version": "full-corpus-preparation-v1", "created_at": datetime.now(timezone.utc).isoformat(),
              "corpus_id": identifier, "source_sha256": corpus["source_sha256"], "raw_records": corpus["profile"]["rows_total"],
              "input_encoding": corpus["profile"]["encoding"], "csv_columns": len(corpus["profile"]["columns"]),
              "split_publication_date_time": {"repair_count_this_run": dates_repaired,
                                               "separate_raw_cells_preserved": True, "timezone_inferred": False},
              "import_status_counts": dict(statuses), "cleaning": summary, "rule_exclusions": dict(exclusions),
              "kept_for_semantic_screening": len(frames), "source_type_counts": dict(types),
              "language_hints_unverified": dict(languages), "publication_range": [min(dates), max(dates)] if dates else None,
              "near_duplicate_review": {"pairs": len(edges), "connected_candidate_families": len(families),
                                        "documents_in_candidates": len({i for family in families for i in family["member_ids"]}),
                                        "removed_automatically": 0},
              "actor_coverage_lexical": actor_coverage,
              "context": {"policy_version": CONTEXT_POLICY_VERSION, "base_max_tokens": 1500, "expanded_max_tokens": 6000,
                          "rare_threshold": max(5, int(len(frames) * .001)), "selection_reason_counts": dict(policy_counts),
                          **dict(expansion_counts),
                          "original_body_tokens": sum(original_tokens), "original_body_quantiles": quantiles(original_tokens),
                          "selected_body_tokens": {name: sum(v) for name, v in selected_tokens.items()},
                          "selected_body_quantiles": {name: quantiles(v) for name, v in selected_tokens.items()}},
              "budget_forecast": estimate, "api_calls": 0, "api_cost_usd": 0,
              "limitations": ["Local cleaning is not final semantic relevance classification.",
                              "Name and language hints do not establish identity or language; unknowns remain reviewable.",
                              "Near-duplicate candidate components are not proven equivalent stories or independent episodes.",
                              "Output counts, relevance and descriptor yield are forecasts from the earlier pilot, not measured on this file.",
                              "Final politician scores, interpretation and report generation are outside this API estimate."],
              "provenance": {"pilot_snapshot_sha256": hashlib.sha256(pilot_raw).hexdigest(),
                             "sources": {str(p.relative_to(root)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in (root / "scripts/prepare_full_corpus.py", root / "src/corpus_pipeline/context_policy.py",
                                                   root / "src/corpus_pipeline/cleaning.py", root / "src/corpus_pipeline/budget_forecast.py",
                                                   root / "src/corpus_pipeline/preparation_cache.py")}}}
    write(directory / "preparation_summary.json", public)
    return public


def render(result):
    lines = ["# Полный корпус Руанды: очистка и бюджет", "",
             "Очистка, выбор контекста и расчёт выполнены локально. Платных API-запросов не было. Исходные тексты, индексы исключений и очищенная выгрузка сохранены только на компьютере.", "",
             f"В файле **{result['raw_records']:,} записей**, а не 32 000; {result['csv_columns']} столбцов, кодировка {result['input_encoding']}. SHA256: `{result['source_sha256']}`.", "",
             "## Очистка", "", "| Этап | Записей |", "|---|---:|"]
    for name, value in result["import_status_counts"].items():
        lines.append(f"| {name} | {value:,} |")
    for name, value in result["rule_exclusions"].items():
        lines.append(f"| Исключено по правилу: {name} | {value:,} |")
    lines += [f"| Сохранено для смыслового отбора | {result['kept_for_semantic_screening']:,} |", "",
              "Убрано HTML-оформление и невидимые управляющие символы, нормализованы пробелы и Unicode. Пустые записи и точные текстовые повторы исключены из рабочей выгрузки с сохранением ссылок на исходные строки. Отрицания, числа и исходная речь не переписывались. Разные тексты по одной ссылке не объявлялись дублями.", "",
              f"Найдено {result['near_duplicate_review']['pairs']:,} пар возможных близких дублей, {result['near_duplicate_review']['connected_candidate_families']:,} связанных групп. Они оставлены для проверки: похожие публикации могут различаться говорящим, датой или утверждением. Автоматически исключено близких дублей: 0.", "",
              "Совпадение имени — ориентир для отбора, не подтверждение личности. Неизвестное имя и ненадёжная языковая метка не превращаются в отрицательный результат. Семантическая фильтрация моделью пока не выполнялась.", "",
              "## Покрытие утверждённых участников", "", "Считаются уникальные тексты с буквальным именем/утверждённым вариантом написания, а не самостоятельные события и не выраженность политических свойств.", "",
              "| Политик | Уникальных текстов |", "|---|---:|"]
    for actor in sorted(result["actor_coverage_lexical"], key=lambda a: (-a["unique_texts_with_name"], a["name"])):
        lines.append(f"| {actor['name']} | {actor['unique_texts_with_name']:,} |")
    lines += ["", f"Даты публикаций в рабочей выгрузке: {result['publication_range'][0]} — {result['publication_range'][1]}. Этот файл покрывает около 13 месяцев, а не весь утверждённый пятилетний период. Материалы 16–17.10.2026 относятся к техническому расширению источников; прежнее исследование с датой окончания 15.10.2026 не изменяется.", "",
              "В исходном CSV дата и время находятся в разных столбцах. Они явно объединены без придумывания часового пояса; исходные ячейки сохранены. Отсутствие слова Rwanda в теле текста оставлено признаком для проверки, а не автоматическим основанием отбрасывания: имя утверждённого политика может быть указано без названия страны.", "",
              "## Расширенный контекст", "",
              f"Обычные публикации: до 1500 токенов вокруг утверждённых имён. Для длинного интервью или редко представленного участника: до 6000 токенов. Порог редкости — не более {result['context']['rare_threshold']} уникальных текстов в очищенном корпусе. Расширение определяется содержанием и покрытием участника, а не отношением к России или желаемым факторным решением.", "",
              "Текст, помещающийся в лимит, сохраняется целиком. Для длинного источника используются отдельные исходные диапазоны; они не склеиваются в непрерывную цитату. Оригинал остаётся доступен для дополнительного прохода. Частота источников и выраженность признака остаются отдельными показателями.", "",
              "## Стоимость", "",
              "Ниже сценарный прогноз на реальном объёме файла, а не прежнее умножение 259 записей до 32 тысяч. Входные тексты токенизированы локально; выходы, доля принятых документов и выход свидетельств откалиброваны по предыдущему пилоту. Кэш не гарантирован, повторные попытки не включены. Полные агрегаты по сценариям и калибровочным группам — в JSON рядом с отчётом.", "",
              "| Контекст | Доля допущенных к извлечению | Отбор + извлечение, $ | С черновиком признаков, $ | При длинных ответах, $ |", "|---|---:|---:|---:|---:|"]
    for variant, data in result["budget_forecast"]["variants"].items():
        for scenario in data["pass_scenarios"]:
            fraction = scenario["pass_fraction"]
            core = scenario["core_cost_usd"]["mean"]
            total = scenario["with_describe_cost_usd"]["mean"]
            high = scenario["with_describe_cost_usd"]["p90"]
            def money(value):
                return f"{value:.2f}" if value is not None else "не установлено"
            lines.append(f"| {variant} | {fraction:.1%} | {money(core)} | {money(total)} | {money(high)} |")
    lines += ["", "Последний столбец суммирует эмпирические p90 длины ответа для каждого документа; это сценарий многословных ответов, а не 90%-ная граница общей цены. Доля допуска распределена равномерно по плану как допущение; реальный отбор может заметно отличаться от трёхдневного пилота.", "",
              "Прогноз относится к отбору, извлечению и отдельной подготовке черновика признаков. Окончательное оценивание политиков, интерпретация факторов и итоговые документы в него не включены.", "",
              "## Откуда списываются деньги", "",
              "Локальные очистка и подсчёт токенов не вызывают API. После явного запуска модельные запросы пойдут по OPENAI_API_KEY из локального окружения/файла .env и будут оплачены через API-аккаунт/проект владельца ключа. Подписка Codex не покрывает эти вызовы. Работа самого Codex при входе через ChatGPT использует лимиты его плана; это отдельный учёт.", "",
              "Основание: [официальная документация OpenAI об API-оплате](https://learn.chatgpt.com/docs/auth#sign-in-with-an-api-key), [тарифы API](https://developers.openai.com/api/docs/pricing). Ставки на 17.10.2026: gpt-4.1-mini — $0.40 за миллион входных / $1.60 выходных токенов; gpt-5.1 — $1.25 / $10.00. Баланс и название API-проекта по ключу здесь не установлены.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--pilot-snapshot", type=Path)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    pilot = args.pilot_snapshot or root / ".corpus_runtime/pilots/corpus-64405d243de9c4040bb6/full-v2/results_private.json"
    progress = lambda value: print(packed(value), flush=True)
    result = prepare(root, args.corpus_id, pilot, progress=progress)
    if args.publish:
        write(root / "reports/rwanda_full_corpus_preparation_2026_10_17.json", result)
        target = root / "research/rwanda_politicians_2021_2026/CORPUS_FULL_PREPARATION_2026-10-17.md"
        target.write_text(render(result), encoding="utf-8")
    print(packed({"phase": "complete", "corpus_id": result["corpus_id"], "raw_records": result["raw_records"],
                  "kept_for_semantic_screening": result["kept_for_semantic_screening"], "api_calls": 0}), flush=True)


if __name__ == "__main__":
    main()
