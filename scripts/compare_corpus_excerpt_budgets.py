"""Offline token/cost and literal-evidence retention scenarios; never an API run.

Source texts and receipt IDs remain private. Outputs contain aggregates only.
Old classifications, output lengths and accepted facts are held fixed; none is
claimed to be reproduced by a future bounded-input model request.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import closing
import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.corpus_pipeline.excerpts import EXCERPT_POLICY_VERSION, offline_encoding, select_spans_with_metadata
from src.corpus_pipeline.models import make_task, estimate_cost
from src.corpus_pipeline.service import CorpusPipeline, compact_objects


def serialized(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def tokens(encoding, text):
    if hasattr(encoding, "encode_ordinary"):
        return len(encoding.encode_ordinary(text))
    return len(encoding.encode(text, disallowed_special=()))


def payload_parts(payload, encoding):
    """Known tokenizer components; provider rendering overhead stays residual."""
    user = payload["input"][0]["content"][0]["text"]
    schema = json.dumps(payload["text"]["format"]["schema"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {"user": tokens(encoding, user), "instructions": tokens(encoding, payload["instructions"]),
            "schema": tokens(encoding, schema)}


def intersect_spans(spans, start, end, text):
    return [{"start": max(s["start"], start), "end": min(s["end"], end),
             "text": text[max(s["start"], start):min(s["end"], end)]}
            for s in spans if max(s["start"], start) < min(s["end"], end)]


def retained_fact(fact, text, spans):
    """Recorded absolute quote and all evidence must fit ONE original range."""
    start, end = fact.get("quote_start"), fact.get("quote_end")
    quote, attribution = fact.get("quote"), fact.get("attribution_evidence")
    if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text)
            or not isinstance(quote, str) or not quote or text[start:end] != quote
            or not isinstance(attribution, str) or not attribution):
        return False, False
    date = fact.get("date_evidence")
    if date is not None and not isinstance(date, str):
        return False, False
    required = [quote, attribution] + ([date] if date else [])
    return any(s["start"] <= start and end <= s["end"] and all(v in s["text"] for v in required) for s in spans), True


def bounded_payload(job, spans, *, compact=False):
    context = json.loads(job["payload"]["input"][0]["content"][0]["text"])
    document = context["document"]
    old_text = document["text"]
    old_start = document.get("chunk_start", 0)
    for key in ("text", "chunk_start", "chunk_end"):
        document.pop(key, None)
    document["excerpts"] = [{k: s[k] for k in ("start", "end", "text")} for s in spans]
    objects = context["config"]["objects"]
    if compact and spans:
        context["config"]["objects"] = copy.deepcopy(compact_objects(objects, document))
    # Empty intersections occur only in the fixed-old-count comparator. They
    # are explicitly cost-only, invalid for actual bounded requests. A real
    # original passage lets the payload factory build identical schemas and
    # instructions; the hypothetical empty USER is then restored verbatim.
    factory_context = copy.deepcopy(context)
    if not spans:
        factory_context["document"]["excerpts"] = [{"start": old_start, "end": old_start + len(old_text), "text": old_text}]
    payload = make_task(job["stage"], factory_context, model=job["payload"]["model"],
                        max_output_tokens=job["payload"]["max_output_tokens"])
    payload["input"][0]["content"][0]["text"] = serialized(context)
    return payload, len(context["config"]["objects"]), compact and context["config"]["objects"] == objects


def _amount(model, inputs, outputs, pricing):
    return estimate_cost(model, inputs, outputs, pricing=pricing)


def compare(results, documents, encoding, *, selector=select_spans_with_metadata, target_count=32000, progress=None):
    if encoding is None:
        raise ValueError("Verified local tokenizer required; no approximate character counts substituted")
    if type(target_count) is not int or target_count < 1:
        raise ValueError("Positive integer target_count required")
    config = results["corpus"]["config"]
    objects, pricing = config["objects"], config["pricing"]
    docs = {d["id"]: d for d in documents if CorpusPipeline._rule_exclusion(d, config) is None}
    jobs = [j for j in results["jobs"] if j["active_recipe"] and j["stage"] in {"relevance", "extract"}]
    if any(j["status"] not in {"done", "review"} or j.get("input_tokens", 0) <= 0 for j in jobs):
        raise ValueError("Only completed billed baseline receipts can enter this controlled comparison")
    by_stage_doc = defaultdict(list)
    old_parts, decomposition = {}, {}
    for stage in ("relevance", "extract"):
        group = [j for j in jobs if j["stage"] == stage]
        breakdown = {"requests": len(group), "documents": 0, "api_input_tokens": 0, "api_output_tokens": 0,
                     "plain_document_tokens": 0, "serialized_document_delta_tokens": 0,
                     "fixed_input_including_api_overhead_tokens": 0, "observed_cost_usd": 0., "uncached_cost_usd": 0.,
                     "input_cost_usd": 0., "output_cost_usd": 0.}
        hidden = []
        for job in group:
            doc = job["context"]["document"]
            identifier = doc["id"]
            if identifier not in docs or "text" not in doc:
                raise ValueError("Baseline must use legacy full chunks in the eligible source frame")
            start = doc.get("chunk_start", 0)
            if docs[identifier]["text"][start:start + len(doc["text"])] != doc["text"]:
                raise ValueError("Baseline chunk source coordinates do not match imported text")
            by_stage_doc[(stage, identifier)].append(job)
            parts = payload_parts(job["payload"], encoding)
            old_parts[job["id"]] = parts
            hidden.append(job["input_tokens"] - parts["user"])
            context = json.loads(job["payload"]["input"][0]["content"][0]["text"])
            context["document"]["text"] = ""
            document_delta = parts["user"] - tokens(encoding, serialized(context))
            breakdown["api_input_tokens"] += job["input_tokens"]
            breakdown["api_output_tokens"] += job["output_tokens"]
            breakdown["plain_document_tokens"] += tokens(encoding, doc["text"])
            breakdown["serialized_document_delta_tokens"] += document_delta
            breakdown["fixed_input_including_api_overhead_tokens"] += job["input_tokens"] - document_delta
            breakdown["observed_cost_usd"] += job["cost_usd"]
            model = job["payload"]["model"]
            breakdown["uncached_cost_usd"] += _amount(model, job["input_tokens"], job["output_tokens"], pricing)
            breakdown["input_cost_usd"] += _amount(model, job["input_tokens"], 0, pricing)
            breakdown["output_cost_usd"] += _amount(model, 0, job["output_tokens"], pricing)
        breakdown["documents"] = sum(key[0] == stage for key in by_stage_doc)
        breakdown["non_USER_overhead_min_tokens"] = min(hidden) if hidden else None
        breakdown["non_USER_overhead_max_tokens"] = max(hidden) if hidden else None
        decomposition[stage] = breakdown
    if set(identifier for stage, identifier in by_stage_doc if stage == "relevance") != set(docs):
        raise ValueError("Baseline relevance must cover every eligible unique document, including negatives and review")
    job_map = {j["id"]: j for j in jobs}
    facts = [f for f in results.get("facts", []) if f.get("extraction_job_id") in job_map]
    descriptions = [j for j in results["jobs"] if j["active_recipe"] and j["stage"] == "describe"]
    describe = {"requests": len(descriptions), "input_tokens": sum(j["input_tokens"] for j in descriptions),
                "output_tokens": sum(j["output_tokens"] for j in descriptions),
                "input_fact_occurrences": sum(len(j["context"].get("facts", [])) for j in descriptions),
                "observed_cost_usd": sum(j["cost_usd"] for j in descriptions),
                "uncached_cost_usd": sum(_amount(j["payload"]["model"], j["input_tokens"], j["output_tokens"], pricing) for j in descriptions),
                "scenario": "held_constant_not_recomputed_from_retained_facts"}
    raw = results["corpus"]["profile"]["rows_total"]
    scale = target_count / raw
    variants = [("full", None, False)] + [(mode, cap, False) for mode in ("head", "mentions") for cap in (500, 1000, 1500)]
    variants += [("mentions", cap, True) for cap in (1000, 1500)]
    rows = []
    for mode, cap, compact in variants:
        selected, metadata = {}, []
        for identifier, doc in docs.items():
            choice = selector(doc["text"], objects, mode=mode, max_tokens=cap or 1500, encoding=encoding)
            spans = choice["spans"]
            if any(doc["text"][s["start"]:s["end"]] != s["text"] for s in spans):
                raise ValueError("Selected spans are not exact original source substrings")
            selected[identifier] = spans
            metadata.append(choice["metadata"])
        estimates = {name: {s: {"requests": 0, "input_tokens": 0, "USER_only_input_tokens": 0, "output_tokens": 0,
                                  "uncached_cost_usd": 0., "empty_intersections": 0, "roster_entries_total": 0,
                                  "full_roster_fallback_calls": 0} for s in ("relevance", "extract")}
                     for name in ("fixed_original_request_count", "one_request_per_document")}
        for (stage, identifier), group in by_stage_doc.items():
            for job in group:
                old = old_parts[job["id"]]
                if mode == "full":
                    predicted, user_only, roster, fallback, empty = job["input_tokens"], job["input_tokens"], len(objects), False, False
                else:
                    source = job["context"]["document"]
                    start = source.get("chunk_start", 0)
                    clipped = intersect_spans(selected[identifier], start, start + len(source["text"]), docs[identifier]["text"])
                    payload, roster, fallback = bounded_payload(job, clipped, compact=compact)
                    new = payload_parts(payload, encoding)
                    user_only = job["input_tokens"] + new["user"] - old["user"]
                    predicted = job["input_tokens"] + sum(new.values()) - sum(old.values())
                    empty = not clipped
                if predicted < 0 or user_only < 0:
                    raise ValueError("Negative token projection; tokenizer/payload accounting incompatible")
                out = estimates["fixed_original_request_count"][stage]
                out["requests"] += 1; out["input_tokens"] += predicted; out["USER_only_input_tokens"] += user_only
                out["output_tokens"] += job["output_tokens"]; out["roster_entries_total"] += roster
                out["full_roster_fallback_calls"] += bool(fallback); out["empty_intersections"] += empty
                out["uncached_cost_usd"] += _amount(job["payload"]["model"], predicted, job["output_tokens"], pricing)
            if mode != "full":
                reference = min(group, key=lambda j: j["context"]["document"].get("chunk_start", 0))
                payload, roster, fallback = bounded_payload(reference, selected[identifier], compact=compact)
                new = payload_parts(payload, encoding)
                residual = min(j["input_tokens"] - sum(old_parts[j["id"]].values()) for j in group)
                hidden = min(j["input_tokens"] - old_parts[j["id"]]["user"] for j in group)
                predicted, user_only = residual + sum(new.values()), hidden + new["user"]
                if predicted < 0 or user_only < 0:
                    raise ValueError("Negative per-document token projection")
                out = estimates["one_request_per_document"][stage]
                out["requests"] += 1; out["input_tokens"] += predicted; out["USER_only_input_tokens"] += user_only
                out["output_tokens"] += sum(j["output_tokens"] for j in group)
                out["roster_entries_total"] += roster; out["full_roster_fallback_calls"] += bool(fallback)
                out["uncached_cost_usd"] += _amount(reference["payload"]["model"], predicted, sum(j["output_tokens"] for j in group), pricing)
        retention = {name: Counter() for name in estimates}
        actor_retention = {name: defaultdict(Counter) for name in estimates}
        for fact in facts:
            identifier = fact["document_id"]
            text = docs[identifier]["text"]
            job = job_map[fact["extraction_job_id"]]
            source = job["context"]["document"]
            start = source.get("chunk_start", 0)
            clipped = intersect_spans(selected[identifier], start, start + len(source["text"]), text)
            for name, spans in (("fixed_original_request_count", clipped), ("one_request_per_document", selected[identifier])):
                kept, valid = retained_fact(fact, text, spans)
                count = retention[name]
                count["accepted_facts"] += 1; count["retained_facts"] += kept; count["invalid_recorded_quote_offsets"] += not valid
                if fact["kind"] in {"public_statement", "action"}:
                    count["own_position_facts"] += 1; count["retained_own_position_facts"] += kept
                count["total_" + fact["kind"]] += 1; count["retained_" + fact["kind"]] += kept
                actor = actor_retention[name][fact["actor_id"]]
                actor["accepted_facts"] += 1; actor["retained_facts"] += kept
                if fact["kind"] in {"public_statement", "action"}:
                    actor["own_position_facts"] += 1; actor["retained_own_position_facts"] += kept
        actor_rows, macro = {}, {}
        names = {o["id"]: o["name"] for o in objects}
        for scenario, counts in actor_retention.items():
            actor_rows[scenario] = [{"actor_id": key, "name": names[key], **dict(value)} for key, value in sorted(counts.items(), key=lambda x: names[x[0]])]
            rates = [c["retained_facts"] / c["accepted_facts"] for c in counts.values()]
            own_rates = [c["retained_own_position_facts"] / c["own_position_facts"] for c in counts.values() if c["own_position_facts"]]
            macro[scenario] = {"actors_with_baseline_facts": len(rates), "roster_actors_without_baseline_facts": len(objects) - len(rates),
                               "unweighted_actor_mean_fraction": sum(rates) / len(rates) if rates else None,
                               "actors_with_own_baseline_facts": len(own_rates),
                               "unweighted_own_actor_mean_fraction": sum(own_rates) / len(own_rates) if own_rates else None}
        for name in estimates:
            if mode == "full" and name == "one_request_per_document":
                estimates[name] = None
                continue
            total = sum(s["uncached_cost_usd"] for s in estimates[name].values())
            estimates[name] = {"stages": estimates[name], "core_cost_usd": total,
                               "core_plus_held_describe_usd": total + describe["uncached_cost_usd"],
                               "conditional_32000_raw_core_usd": total * scale,
                               "conditional_32000_raw_with_held_describe_usd": (total + describe["uncached_cost_usd"]) * scale}
        rows.append({"id": "full" if mode == "full" else f"{mode}_{cap}" + ("_compact" if compact else ""),
                     "mode": mode, "max_document_tokens": cap, "compact_roster": compact,
                     "selection": {"selected_body_tokens": sum(m["selected_tokens"] for m in metadata),
                                   "span_count": sum(m["span_count"] for m in metadata),
                                   "truncated_documents": sum(m["truncated"] for m in metadata),
                                   "no_mentions_head_fallback_documents": sum(m["no_mentions_fallback"] for m in metadata)},
                     "cost_scenarios": estimates, "literal_retention": {k: dict(v) for k, v in retention.items()},
                     "actor_retention": actor_rows, "macro_retention": macro})
        if progress is not None:
            current = rows[-1]
            scenario = estimates["one_request_per_document"] or estimates["fixed_original_request_count"]
            progress({"variant": current["id"], "core_cost_usd": scenario["core_cost_usd"],
                      "conditional_32000_raw_core_usd": scenario["conditional_32000_raw_core_usd"],
                      "retention": current["literal_retention"]["one_request_per_document"]})
    dated = sorted(str(d.get("published_at", ""))[:10] for d in documents if d.get("published_at"))
    return {"schema_version": "excerpt-budget-comparison-v1", "encoding": getattr(encoding, "name", "provided"),
            "excerpt_policy_version": EXCERPT_POLICY_VERSION,
            "source": {"corpus_id": results["corpus"]["id"], "raw_documents": raw, "eligible_unique_documents": len(docs), "target_raw_documents": target_count,
                       "publication_date_minmax": [dated[0], dated[-1]] if dated else None,
                       "conditional_scaling_factor": scale},
            "baseline": {"stages": decomposition, "describe_held_constant": describe,
                         "active_models": config["routes"], "pricing_per_million": pricing},
            "variants": rows, "pricing_source": "https://developers.openai.com/api/docs/pricing",
            "methodology": {"paid_calls": 0, "future_cache_guaranteed": False,
                "fixed_count_input": "I_API + delta tokenized(USER) + delta tokenized(instructions) + delta tokenized(canonical output schema)",
                "one_document_input": "min_group[I_API-T(USER)-T(instructions)-T(schema)] + new three components",
                "USER_only_control": "I_API + T(USER_new)-T(USER_old); retains all old non-USER overhead",
                "outputs": "Each scenario preserves the SUM of original output tokens, including multi-chunk groups",
                "relevance": "Old decisions frozen; all eligible classification documents retained, no negative/review pruning",
                "extraction": "Old positive-document groups retained; first original context keeps its original relevance metadata; no new classification is predicted",
                "fixed_empty_intersections": "Cost-only hypothetical empty USER excerpts, not valid actual requests",
                "retention": "Recorded exact global quote offset plus quote, attribution and non-null date evidence within ONE retained original span; lexical availability, not model recall or factual validation",
                "scale": "Conditional transfer of this259-record source composition; no probability CI, no full-research/scoring price",
                "payload_limits": "Local tokenizer differences estimate provider prompt rendering. Current payload factory/schema version is recorded; unseen rendering overhead assumed unchanged."}}


def render(result):
    b = result["baseline"]
    lines = ["# Ограничение текста: стоимость и сохранность свидетельств", "",
             "Офлайн-сравнение исходных оплаченных запросов. Новых ответов модели нет. Сохранность ниже означает наличие полного исходного свидетельства во фрагменте, а не качество нового ответа.", "",
             "## Измеренная база", "", "| Этап | Запросы / документы | Вход | Текст в JSON | Остальной вход | Выход | Без кэша, $ |", "|---|---:|---:|---:|---:|---:|---:|"]
    for stage, s in b["stages"].items():
        lines.append(f"| {stage} | {s['requests']} / {s['documents']} | {s['api_input_tokens']} | {s['serialized_document_delta_tokens']} | {s['fixed_input_including_api_overhead_tokens']} | {s['api_output_tokens']} | {s['uncached_cost_usd']:.5f} |")
    d = b["describe_held_constant"]
    lines += ["", f"Describe оставлен отдельно: {d['requests']} запроса, {d['input_tokens']} входных и {d['output_tokens']} выходных токенов, {d['input_fact_occurrences']} вхождений фактов; ${d['uncached_cost_usd']:.5f}. Эти факты и словарь заново не пересчитывались.", "",
              "Основной расчёт использует сохранённые стандартные ставки без гарантированного кэша: [официальная страница OpenAI](https://developers.openai.com/api/docs/pricing). Фактический расход исходных запросов с кэшем сохранён в JSON. Неизменные инструкции, схема, сведения о пространстве и список политиков стоят токены даже при коротком тексте.", "",
              "## Сценарии стоимости", "", "Все лимиты — токены тела документа, а не символы и не лимит всего запроса. Mentions выбирает до четырёх отдельных диапазонов, не склеивая их. Компактный список содержит только буквально найденные утверждённые имена/алиасы; при отсутствии совпадения сохраняется весь список.", "",
              "| Вариант | Старое число запросов, $ | Один запрос / документ, $ | Условные 32 000: ядро, $ | С сохранённым describe, $ |", "|---|---:|---:|---:|---:|"]
    for v in result["variants"]:
        fixed, single = v["cost_scenarios"]["fixed_original_request_count"], v["cost_scenarios"]["one_request_per_document"]
        actual = single or fixed
        single_text = f"{single['core_cost_usd']:.3f}" if single else "—"
        lines.append(f"| {v['id']} | {fixed['core_cost_usd']:.3f} | {single_text} | {actual['conditional_32000_raw_core_usd']:.2f} | {actual['conditional_32000_raw_with_held_describe_usd']:.2f} |")
    lines += ["", "## Сохранность исходных подтверждений", "", "| Вариант | Все факты | Собственные позиции/действия | Усечено документов |", "|---|---:|---:|---:|"]
    for v in result["variants"]:
        r = v["literal_retention"]["one_request_per_document"]
        lines.append(f"| {v['id']} | {r.get('retained_facts', 0)}/{r.get('accepted_facts', 0)} | {r.get('retained_own_position_facts', 0)}/{r.get('own_position_facts', 0)} | {v['selection']['truncated_documents']} |")
    chosen = {v["id"]: {a["actor_id"]: a for a in v["actor_retention"]["one_request_per_document"]} for v in result["variants"]}
    columns = ("head_500", "mentions_500", "mentions_1000", "mentions_1500")
    lines += ["", "## Проверка по каждому политику", "",
              "Среднее по всем строкам фактов может быть высоким из-за преобладания одного политика. Поэтому ниже отдельно приведён каждый политик с исходными фактами; макросреднее даёт им равный вес. Политики без исходных фактов не получают вымышленный нулевой процент.", "",
              "| Политик | Исходных фактов | Head 500 | Mentions 500 | Mentions 1000 | Mentions 1500 |", "|---|---:|---:|---:|---:|---:|"]
    for key, base in sorted(chosen["full"].items(), key=lambda x: x[1]["name"]):
        values = [str(chosen[column][key].get("retained_facts", 0)) for column in columns]
        lines.append(f"| {base['name']} | {base['accepted_facts']} | " + " | ".join(values) + " |")
    rates = [next(v for v in result["variants"] if v["id"] == key)["macro_retention"]["one_request_per_document"]["unweighted_actor_mean_fraction"] for key in columns]
    lines.append("| **Макросреднее по политикам с исходными фактами** | — | " + " | ".join(f"{r:.1%}" if r is not None else "—" for r in rates) + " |")
    nofacts = result["variants"][0]["macro_retention"]["one_request_per_document"]["roster_actors_without_baseline_facts"]
    lines += ["", f"У {nofacts} политиков утверждённого списка в этом источнике нет исходных принятых фактов; сохранность их материала этим сравнением не установлена. Компактные варианты имеют ту же текстовую сохранность, что соответствующие Mentions без сжатия списка."]
    louise = next((a for a in chosen["mentions_1500"].values() if "mushikiwabo" in a["name"].casefold()), None)
    if louise:
        lines += ["", f"Показательное ограничение: у {louise['name']} Mentions 1500 сохраняет лишь {louise.get('retained_facts', 0)}/{louise['accepted_facts']} ({louise.get('retained_facts', 0)/louise['accepted_facts']:.1%}) исходных строк. Высокий общий процент не означает сохранность материала каждого политика."]
    primary = next(v for v in result["variants"] if v["id"] == "mentions_1500_compact")["cost_scenarios"]["one_request_per_document"]
    lines += ["", "Рабочий стартовый вариант — Mentions 1500 с компактным утверждённым списком. Его нельзя считать универсальным режимом без потерь: после аудита покрытия для длинных интервью и редко представленных политиков следует отдельно сохранять полный текст или дополнительные исходные фрагменты. Автоматический адаптивный проход здесь не реализован.", "",
              f"Условная база для 32 000 с сохранённым describe — ${primary['conditional_32000_raw_with_held_describe_usd']:.2f}. Практический резерв $200–220 является отдельным бюджетным допущением на вариацию корпуса и дополнительные фрагменты, а не доверительным интервалом. Финальное оценивание политиков в эти суммы не включено."]
    lines += ["", "## Как читать результат", "",
              "Старое число запросов — контролируемое сравнение: выходы и все вызовы сохранены; пустые пересечения отдельных старых chunks используются только в расчёте цены. Для интеграции применим сценарий одного запроса на документ: повторный вход устраняется, но сумма старых выходов сохраняется. Невидимый служебный остаток берётся минимальным в группе. Это оценка токенизатором, а не новая оплаченная квитанция и не гарантированная верхняя цена.", "",
              "Часть входа меняется также из-за новых инструкций excerpt-режима и actor-enum схемы; эти разности учтены отдельно от USER. Компактный список сокращает повторяемую конфигурацию и схему, но не создаёт доказательство идентичности. Новый relevance на коротком тексте может отличаться от старого: отрицательные и неизвестные случаи здесь не выброшены ради экономии. Диагностика выбора, включая найденные вне excerpt имена, модели не передаётся. Строки фактов могут относиться к одному событию: проценты сохранности не являются числом независимых эпизодов.", "",
              f"Для 32 000 использован условный множитель {result['source']['conditional_scaling_factor']:.6f} = 32000/{result['source']['raw_documents']}. Исходная выгрузка охватывает лишь три дня ({result['source']['publication_date_minmax']}) и зависит от текущей повестки; она не репрезентирует весь пятилетний корпус. Состав другого корпуса по длине, языку, авторам, времени и событиям не известен. Это не доверительный интервал. Describe, оценивание политиков, человеческий аудит и повторные исследования не предсказаны; сохранённый describe — отдельная фиксированная ценовая предпосылка.", "",
              "Воспроизведение без API:", "", "```text", f"python scripts/compare_corpus_excerpt_budgets.py --corpus-id {result['source']['corpus_id']}", "```", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--target-count", type=int, default=32000)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    source_paths = (Path(__file__), ROOT / "src/corpus_pipeline/excerpts.py", ROOT / "src/corpus_pipeline/models.py", ROOT / "src/corpus_pipeline/service.py")
    source_hashes = {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    snapshot = args.snapshot or root / ".corpus_runtime/pilots" / args.corpus_id / "full-v2/results_private.json"
    raw = snapshot.read_bytes()
    results = json.loads(raw.decode("utf-8-sig"))
    if results["corpus"]["id"] != args.corpus_id:
        raise ValueError("Snapshot corpus mismatch")
    encoding = offline_encoding(root / ".corpus_runtime/tokenizer_cache")
    if encoding is None:
        raise ValueError("Offline tokenizer rank cache missing/invalid; no download attempted")
    with closing(sqlite3.connect((root / ".corpus_runtime/corpus.sqlite3").as_uri() + "?mode=ro", uri=True)) as db:
        documents = [json.loads(row[0]) for row in db.execute("SELECT data FROM documents WHERE corpus_id=? AND status='unique'", (args.corpus_id,))]
    result = compare(results, documents, encoding, target_count=args.target_count,
                     progress=lambda row: print(serialized(row), flush=True))
    final_hashes = {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    result["provenance"] = {"snapshot_sha256": hashlib.sha256(raw).hexdigest(),
                            "source_csv_sha256": results["corpus"]["source_sha256"],
                            "source_sha256": source_hashes,
                            "source_changed_during_run": [key for key in source_hashes if source_hashes[key] != final_hashes[key]]}
    json_path = root / "reports/corpus_excerpt_budget_2026_10_17.json"
    md_path = root / "research/rwanda_politicians_2021_2026/CORPUS_EXCERPT_BUDGET_2026-10-17.md"
    for path, text in ((json_path, json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"), (md_path, render(result))):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))
    print(serialized({"json": str(json_path), "markdown": str(md_path), "paid_calls": 0,
                      "variants": [{"id": v["id"], "cost": (v["cost_scenarios"]["one_request_per_document"] or v["cost_scenarios"]["fixed_original_request_count"])["core_cost_usd"]} for v in result["variants"]]}))


if __name__ == "__main__":
    main()
