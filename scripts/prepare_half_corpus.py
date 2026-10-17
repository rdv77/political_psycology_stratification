"""Prepare a reproducible protected half-sample and its cost, entirely offline.

No queue, API client or environment key is used. Original data and the complete
cleaning export remain untouched. Texts, document IDs and sampling probabilities
stay in the ignored local runtime; only aggregates can be published.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_full_corpus import cleaning_cache_hash, fragment, packed, write
from src.corpus_pipeline.budget_forecast import forecast
from src.corpus_pipeline.context_policy import CONTEXT_POLICY_VERSION, select_document_context
from src.corpus_pipeline.excerpts import offline_encoding
from src.corpus_pipeline.models import estimate_cost
from src.corpus_pipeline.preparation_cache import load_checkpoint
from src.corpus_pipeline.sampling import sample_documents
from src.corpus_pipeline.service import CorpusPipeline

SEED = "rwanda-half-2026-10-17-v1"


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    checksum, count, size = hashlib.sha256(), 0, 0
    with temporary.open("wb") as stream:
        for row in rows:
            raw = (packed(row) + "\n").encode("utf-8")
            stream.write(raw)
            checksum.update(raw)
            count += 1
            size += len(raw)
    temporary.replace(path)
    return {"sha256": checksum.hexdigest(), "rows": count, "bytes": size}


def _profile(frames):
    profile = {key: Counter() for key in ("month", "source_type", "source_platform", "language_hint", "status", "actor_documents", "name_group")}
    for document, annotation in frames:
        profile["month"][document.get("published_at", "")[:7] or "unknown"] += 1
        context = document["_source_context"]
        profile["source_type"][context["source_type"]] += 1
        profile["source_platform"][context["source_platform"] or "unknown"] += 1
        profile["language_hint"][document.get("language_hint") or "unknown"] += 1
        profile["status"][annotation["status"]] += 1
        ids = annotation["matched_actor_ids"]
        for actor_id in ids:
            profile["actor_documents"][actor_id] += 1
        group = "any_other_actor" if set(ids) - {"paul-kagame"} else "kagame_only" if ids else "no_known_name"
        profile["name_group"][group] += 1
    return {key: dict(sorted(counts.items())) for key, counts in profile.items()}


def _matched_baseline(directory, corpus, pilot_raw, root):
    baseline = json.loads((directory / "preparation_summary.json").read_text(encoding="utf-8"))
    if baseline["corpus_id"] != corpus["id"] or baseline["source_sha256"] != corpus["source_sha256"]:
        raise ValueError("Full-corpus baseline belongs to another source")
    if baseline["provenance"]["pilot_snapshot_sha256"] != hashlib.sha256(pilot_raw).hexdigest():
        raise ValueError("Full and sample forecasts must use the same pilot snapshot")
    for relative, checksum in baseline["provenance"]["sources"].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != checksum:
            raise ValueError("Refresh the full forecast after changing its implementation")
    config = corpus["config"]
    forecast_data = baseline["budget_forecast"]
    for scenario in forecast_data["variants"]["adaptive6000"]["pass_scenarios"]:
        for stage in ("relevance", "extract", "describe"):
            row = scenario["describe"] if stage == "describe" else scenario["stages"][stage]
            model = forecast_data["describe_calibration"]["model"] if stage == "describe" else row["model"]
            if model != config["routes"][stage]:
                raise ValueError("Refresh the full forecast after changing model routes")
            input_rate = estimate_cost(model, 1_000_000, 0, pricing=config["pricing"]) / 1_000_000
            output_rate = estimate_cost(model, 0, 1_000_000, pricing=config["pricing"]) / 1_000_000
            if row["estimated_input_tokens"] is not None and row["estimated_output_tokens"]["mean"] is not None:
                expected = row["estimated_input_tokens"] * input_rate + row["estimated_output_tokens"]["mean"] * output_rate
                if not math.isclose(expected, row["cost_usd"]["mean"], rel_tol=1e-10, abs_tol=1e-10):
                    raise ValueError("Refresh the full forecast after changing prices")
    return baseline


def prepare(root, identifier, pilot_snapshot, *, seed=SEED, target_count=None, progress=None):
    pipeline = CorpusPipeline(root)
    corpus = pipeline.get_corpus(identifier)
    config = corpus["config"]
    if config["text_selection"] != {"mode": "mentions", "max_tokens": 1500, "compact_roster": True,
                                    "adaptive": True, "expanded_max_tokens": 6000}:
        raise ValueError("This comparison requires the existing adaptive1500/6000 policy")
    pilot_raw = pilot_snapshot.read_bytes()
    pilot = json.loads(pilot_raw.decode("utf-8-sig"))
    review_config = {**config, "country_review_terms": pilot["corpus"]["config"].get("country_terms", [])}
    cleaning = pipeline.runtime / "cleaning" / identifier
    cached = load_checkpoint(cleaning, corpus["source_sha256"], cleaning_cache_hash(config, review_config))
    if cached is None:
        raise ValueError("Complete and verify offline cleaning before sampling")
    frames, _, _ = cached
    baseline = _matched_baseline(cleaning, corpus, pilot_raw, root)
    if baseline["kept_for_semantic_screening"] != len(frames):
        raise ValueError("Baseline size no longer matches the cleaned source")
    before = _profile(frames)
    records = [{"id": doc["id"], "published_at": doc["published_at"],
                "source_type": doc["_source_context"]["source_type"],
                "language_hint": doc["language_hint"], "status": annotation["status"],
                "actor_ids": annotation["matched_actor_ids"]} for doc, annotation in frames]
    sample = sample_documents(records, approved_actor_ids=[actor["id"] for actor in config["objects"]],
                              primary_actor_id="paul-kagame", target_count=target_count, seed=seed)
    chosen = {row["id"] for row in sample["manifest"]["documents"] if row["selected"]}
    selected = [(doc, annotation) for doc, annotation in frames if doc["id"] in chosen]
    after = _profile(selected)
    directory = pipeline.runtime / "sampling" / identifier / "half-v1"
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {**sample["manifest"], "corpus_id": identifier, "source_sha256": corpus["source_sha256"],
                "cleaning_checkpoint_sha256": hashlib.sha256((cleaning / "cleaning_checkpoint.json").read_bytes()).hexdigest()}
    write(directory / "selection.json", manifest)
    write(directory / "selected_ids.json", sorted(chosen))
    exported = write_jsonl(directory / "selected_unique.jsonl", (
        {**{key: value for key, value in doc.items() if key != "_source_context"},
         "cleaning": annotation, "source_context": doc["_source_context"]} for doc, annotation in selected))
    if progress:
        progress({"phase": "sample_selected", "documents": len(selected), "original": len(frames),
                  "name_groups": after["name_group"]})
    encoding = offline_encoding(pipeline.runtime / "tokenizer_cache")
    if encoding is None:
        raise ValueError("Verified local tokenizer required; no downloads are attempted")
    plans, reasons, body_tokens, original_tokens, expansions = [], Counter(), 0, 0, 0
    uncovered_rare = 0
    # Full-corpus frequencies keep rarity and context comparable with the full
    # estimate; downsampling must not itself change who is labelled rare.
    actor_counts = {actor["id"]: before["actor_documents"].get(actor["id"], 0) for actor in config["objects"]}
    for number, (document, annotation) in enumerate(selected, 1):
        tokens = len(encoding.encode(document["text"], disallowed_special=()))
        original_tokens += tokens
        if tokens <= 1500:
            context = {"spans": [{"start": 0, "end": len(document["text"]), "text": document["text"], "tokens": tokens}],
                       "metadata": {"max_tokens": 1500, "expanded": False, "reason": ["source_fits_base"]}}
        else:
            context = select_document_context(document, config["objects"], actor_counts, len(frames),
                                               {"base_max_tokens": 1500, "expanded_max_tokens": 6000}, encoding)
        metadata = context["metadata"]
        body_tokens += sum(span["tokens"] for span in context["spans"])
        expansions += bool(metadata.get("expanded"))
        uncovered_rare += bool(metadata.get("uncovered_rare_actor_ids"))
        reason = metadata.get("reason", ["base"])
        reasons["|".join(reason) if isinstance(reason, list) else str(reason)] += 1
        source = document["_source_context"]
        plans.append({"document": fragment(document, context), "actor_ids": annotation["matched_actor_ids"],
                      "source_type": source["source_type"], "source_context": source, "token_budget": metadata["max_tokens"]})
        if progress and (number % 1000 == 0 or number == len(selected)):
            progress({"phase": "sample_context", "completed": number, "total": len(selected)})
    planned_artifact = write_jsonl(directory / "adaptive_context_plans.jsonl", plans)
    estimate = forecast({"half_adaptive6000": plans}, pilot, config, encoding, progress=progress)
    full_variant = baseline["budget_forecast"]["variants"]["adaptive6000"]
    half_variant = estimate["variants"]["half_adaptive6000"]
    comparisons = []
    for full, half in zip(full_variant["pass_scenarios"], half_variant["pass_scenarios"]):
        if full["name"] != half["name"] or full["pass_fraction"] != half["pass_fraction"]:
            raise ValueError("Mismatched full/sample forecast scenarios")
        total, sample_total = full["with_describe_cost_usd"]["mean"], half["with_describe_cost_usd"]["mean"]
        comparisons.append({"scenario": full["name"], "pass_fraction": full["pass_fraction"],
                            "full_with_describe_usd": full["with_describe_cost_usd"],
                            "half_with_describe_usd": half["with_describe_cost_usd"],
                            "saving_mean_usd": total - sample_total if total is not None and sample_total is not None else None,
                            "saving_mean_fraction": 1 - sample_total / total if total and sample_total is not None else None})
    coverage = [{"id": actor["id"], "name": actor["name"], "before": actor_counts[actor["id"]],
                 "after": after["actor_documents"].get(actor["id"], 0)} for actor in config["objects"]]
    public = {"version": "protected-half-corpus-v1", "created_at": datetime.now(timezone.utc).isoformat(),
              "corpus_id": identifier, "source_sha256": corpus["source_sha256"], "seed": seed,
              "original_unique": len(frames), "selected_unique": len(selected),
              "sampling": sample["audit"], "profile_before": before, "profile_after": after,
              "actor_coverage_lexical": coverage,
              "context": {"policy_version": CONTEXT_POLICY_VERSION, "rarity_reference": "complete_cleaned_corpus",
                          "rare_threshold": max(5, int(len(frames) * .001)), "base_max_tokens": 1500,
                          "expanded_max_tokens": 6000, "expanded_documents": expansions,
                          "uncovered_rare_actor_documents": uncovered_rare, "reason_counts": dict(reasons),
                          "original_selected_body_tokens": original_tokens, "supplied_selected_body_tokens": body_tokens},
              "budget_forecast": estimate, "full_comparison": comparisons,
              "api_calls": 0, "api_cost_usd": 0, "queue_modified": False, "canonical_study_modified": False,
              "private_artifacts": {"selected_unique": exported, "context_plans": planned_artifact,
                                    "selection_sha256": hashlib.sha256((directory / "selection.json").read_bytes()).hexdigest()},
              "provenance": {"pilot_snapshot_sha256": hashlib.sha256(pilot_raw).hexdigest(),
                             "full_forecast_sha256": hashlib.sha256((cleaning / "preparation_summary.json").read_bytes()).hexdigest(),
                             "sampling_source_sha256": hashlib.sha256((ROOT / "src/corpus_pipeline/sampling.py").read_bytes()).hexdigest(),
                             "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
              "limitations": ["A name match is not verified identity or own-position attribution.",
                              "Purposeful actor protection changes composition; raw sample counts are not corpus frequencies.",
                              "Language metadata are unverified; unknown-name records remain eligible for stratified selection.",
                              "The original and full cleaning export are preserved; exclusion from this sample is not irrelevance.",
                              "No selection on Russia stance, previous scores, factor solution or desired political conclusions.",
                              "Cost outputs/yield are transferred from a small pilot, not measured on this sample."]}
    write(directory / "summary.json", public)
    return public


def render(result):
    groups = result["profile_before"]["name_group"]
    after = result["profile_after"]["name_group"]
    central = next(row for row in result["full_comparison"] if row["scenario"] == "transferred_pilot_observed_positive")
    lines = ["# Руанда: половина корпуса с сохранением материалов о других политиках", "",
             f"Подготовлено **{result['selected_unique']:,} из {result['original_unique']:,} уникальных текстов**. Исходник и полный очищенный корпус сохранены. Отбор и расчёт выполнены локально: API-запросов — 0, стоимость подготовки — $0.", "",
             "## Принцип отбора", "",
             "Цель — сравнить утверждённых политиков и описать политическое пространство. Поэтому сохраняются все тексты, где буквальным поиском найден хотя бы один участник кроме Кагаме, в том числе совместные упоминания с Кагаме. Это защита сравнительного материала, а не подтверждение релевантности или авторства высказываний.", "",
             f"Таких текстов {groups.get('any_other_actor', 0):,}. Остаток до половины набирается из публикаций только о Кагаме и записей без утверждённых имён. Последние не объявляются нерелевантными: неизвестное имя остаётся отдельным слоем для смысловой проверки.", "",
             "Добор распределён по месяцу публикации × типу источника × языковой метке × статусу локальной проверки × наличию имени. Если остаток позволяет, каждая непустая страта получает хотя бы один текст; оставшиеся места распределяются пропорционально свободной ёмкости методом наибольших остатков. Поэтому малые страты могут иметь повышенную долю отбора. Внутри страты используется псевдослучайный порядок SHA256(seed, document_id), независимый от исходного порядка CSV.", "",
             f"Фиксированный seed: `{result['seed']}`. Позиция по России, прежние оценки и желаемые факторы в отборе не используются. Близкие дубли здесь не объявляются тождественными событиями; прежние кандидаты сохранены для проверки.", "",
             "| Группа текстов | До отбора | В половине |", "|---|---:|---:|"]
    for key, label in (("any_other_actor", "Есть участник кроме Кагаме"), ("kagame_only", "Только Кагаме"), ("no_known_name", "Без утверждённых имён")):
        lines.append(f"| {label} | {groups.get(key, 0):,} | {after.get(key, 0):,} |")
    lines += ["", "## Что сохраняется и как читать результат", "",
              "Материалы со всеми найденными именами других участников сохранены полностью. У Мпайиманы буквальных совпадений не найдено: сокращение корпуса этого пробела не исправляет. Новые алиасы и личности всё ещё требуют проверки. Языковые метки исходника не подтверждают фактический язык публикации.", "",
              "Половина намеренно усиливает представительство политиков, которых затмевает Кагаме. Доли текстов в ней не следует выдавать за доли всего медиапотока. В локальном индексе записаны причины и вероятности включения: 1 для защищённых текстов, n_h/N_h для добора внутри страты; для частотных оценок используются обратные веса. Если у страты нулевая квота, восстановить её частоты по выборке нельзя. Для содержательных характеристик число повторов и выраженность свойства остаются разными показателями.", "",
              "Для отобранных текстов остаётся прежний лимит контекста: до 1500 токенов, до 6000 для длинных интервью и редких участников. Редкость считается по полному очищенному корпусу, чтобы сокращение само не меняло правило расширения. Расширение не означает передачу всего интервью без лимита.", "",
              f"Расширенный контекст получили {result['context']['expanded_documents']:,} документов. Планируемых токенов тела текста: {result['context']['supplied_selected_body_tokens']:,}; инструкции, схемы ответа и ответы модели учитываются в цене отдельно.", "",
              "## Пересчитанный бюджет", "",
              f"Основной сценарий с черновиком признаков: **${central['half_with_describe_usd']['mean']:.2f} вместо ${central['full_with_describe_usd']['mean']:.2f}**, экономия **{central['saving_mean_fraction']:.1%}**. Цена рассчитана по фактически выбранным текстам и контекстам; сокращение числа публикаций вдвое не обещает сокращения цены ровно вдвое.", "",
              "| Доля допуска к извлечению | Полный корпус, $ | Половина, $ | Половина при длинных ответах, $ |", "|---|---:|---:|---:|"]
    for row in result["full_comparison"]:
        lines.append(f"| {row['pass_fraction']:.1%} | {row['full_with_describe_usd']['mean']:.2f} | {row['half_with_describe_usd']['mean']:.2f} | {row['half_with_describe_usd']['p90']:.2f} |")
    lines += ["", "В цену входят смысловой отбор, извлечение свидетельств и черновик описательных признаков. Финальное оценивание политиков, факторный и кластерный анализ, интерпретация, повторные прогоны и итоговые документы в этот прогноз не включены. Доли допуска и длины ответов перенесены из небольшого пилота; это сценарии, а не новые измерения качества. Последний столбец — сумма эмпирических p90 ответов, не доверительная граница общей цены.", "",
              "Платная обработка не запускалась, очередь сервиса не изменялась. API оплачивается через проект используемого ключа отдельно от подписки Codex, как описано в [предыдущем отчёте](CORPUS_FULL_PREPARATION_2026-10-17.md).", "",
              "## Покрытие участников", "", "Считаются тексты с именем, не независимые события и не оценки политических свойств.", "",
              "| Участник | Полный корпус | Половина |", "|---|---:|---:|"]
    for actor in sorted(result["actor_coverage_lexical"], key=lambda row: (-row["before"], row["name"])):
        lines.append(f"| {actor['name']} | {actor['before']:,} | {actor['after']:,} |")
    for dimension, title in (("month", "Месяцы"), ("source_type", "Типы источников"), ("language_hint", "Непроверенные языковые метки"), ("status", "Статусы локальной проверки")):
        lines += ["", f"## {title}", "", "| Категория | Полный корпус | Половина |", "|---|---:|---:|"]
        for value, count in result["profile_before"][dimension].items():
            lines.append(f"| {value} | {count:,} | {result['profile_after'][dimension].get(value, 0):,} |")
    lines += ["", "## Воспроизведение и файлы", "",
              "```powershell", f"python scripts/prepare_half_corpus.py --corpus-id {result['corpus_id']} --publish", "```", "",
              f"Локальный каталог: `.corpus_runtime/sampling/{result['corpus_id']}/half-v1/`. `selected_unique.jsonl` — рабочая половина с исходными текстами; `selected_ids.json` — список ID для следующего запуска relevance/extract через параметр document_ids; `selection.json` — полный индекс включения и вероятностей; `adaptive_context_plans.jsonl` — запланированные фрагменты. Кнопка подготовки без document_ids по-прежнему означает весь корпус: подготовленные ID нужно передать явно. Публичный Git содержит код, агрегаты и этот отчёт, без текстов, авторов и ключей.", "",
              "Период источников остаётся 01.10.2025–17.10.2026. Исходное пятилетнее исследование, его утверждённые шкалы и прежние оценки не изменялись.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--pilot-snapshot", type=Path)
    parser.add_argument("--target-count", type=int)
    parser.add_argument("--seed", default=SEED)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    pilot = args.pilot_snapshot or root / ".corpus_runtime/pilots/corpus-64405d243de9c4040bb6/full-v2/results_private.json"
    result = prepare(root, args.corpus_id, pilot, seed=args.seed, target_count=args.target_count,
                     progress=lambda value: print(packed(value), flush=True))
    if args.publish:
        write(root / "reports/rwanda_half_corpus_2026_10_17.json", result)
        (root / "research/rwanda_politicians_2021_2026/CORPUS_HALF_SELECTION_2026-10-17.md").write_text(render(result), encoding="utf-8")
    print(packed({"phase": "complete", "selected": result["selected_unique"], "api_calls": 0}), flush=True)


if __name__ == "__main__":
    main()
