"""Pure offline readiness for source-grounded scoring; never creates scores.

Literal quote groups save repeated context. They are NOT verified events or
independent episodes: the same sentence may be said twice, and different
sentences or translations from one event deliberately remain separate.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import re
import unicodedata

from .models import _eligible_fact


POLICY_VERSION = "literal-source-scoring-preparation-v1"
_OUTSIDE = {"historical_outside_window", "historical_out_of_scope", "outside_window",
            "outside_period", "historical_outside_scope", "out_of_scope"}


def packed(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(packed(value).encode("utf-8")).hexdigest()


def filter_eligible_fact(fact, config):
    """Reuse the measurement gate and respect explicit outside-window reviews."""
    return (_eligible_fact(fact, config)
            and str(fact.get("temporal_status", "")).casefold() not in _OUTSIDE)


def _literal(value):
    # No case folding, punctuation stripping, translation or negation removal.
    return " ".join(unicodedata.normalize("NFC", value).split())


def _approved_public_label(value, label):
    if (not isinstance(value, str) or not value.strip() or len(value) > 240
            or re.search(r"https?://|[\x00-\x1f]", value)):
        raise ValueError("Invalid approved " + label)
    return value


def _measure(content, encoding):
    size = len(content.encode("utf-8"))
    tokens = None if encoding is None else len(encoding.encode(content, disallowed_special=()))
    return {"utf8_bytes": size, "content_tokens": tokens}


def _sum_measure(rows):
    return {"utf8_bytes": sum(row["utf8_bytes"] for row in rows),
            "content_tokens": (sum(row["content_tokens"] for row in rows)
                               if rows and all(row["content_tokens"] is not None for row in rows)
                               else 0 if not rows else None)}


def _packet_plan(units, scales, encoding):
    """Content components only, not provider payloads or a monetary forecast."""
    unit_sizes = [_measure(packed(unit), encoding) for unit in units]
    unit_total = _sum_measure(unit_sizes)
    dictionary_size = _measure(packed({"scales": scales}), encoding)
    separate_scales = _sum_measure([_measure(packed({"scales": [scale]}), encoding) for scale in scales])
    count, scale_count = len(units), len(scales)

    def component_totals(unit_repetitions, dictionary_totals):
        tokens = (None if unit_total["content_tokens"] is None or dictionary_totals["content_tokens"] is None
                  else unit_repetitions * unit_total["content_tokens"] + count * dictionary_totals["content_tokens"])
        return {"utf8_bytes": unit_repetitions * unit_total["utf8_bytes"] + count * dictionary_totals["utf8_bytes"],
                "content_component_tokens": tokens}

    return {"protocol_optimized": False, "paid_requests_prepared": False,
            "content_representation": "complete_private_evidence_unit_and_dictionary_json_components",
            "tokenizer": getattr(encoding, "name", None) if encoding is not None else None,
            "tokenizer_available": encoding is not None,
            "unit_content_once": unit_total, "all_scales_content_once": dictionary_size,
            "one_unit_all_scales": {"hypothetical_packets": count,
                                    "hypothetical_property_evaluations": count * scale_count,
                                    **component_totals(1, dictionary_size)},
            "one_unit_one_scale": {"hypothetical_packets": count * scale_count,
                                   "hypothetical_property_evaluations": count * scale_count,
                                   **component_totals(scale_count, separate_scales)},
            "cost_usd": None, "output_tokens": None, "api_calls": 0,
            "limitations": ["Content component counts omit instructions, schema, message framing and output.",
                            "Component token counts are not the tokenization of an actual combined API payload.",
                            "No caching, model choice, retries or output allowance is assumed.",
                            "These deliberately unoptimized packet sketches require a separate pilot after dictionary approval.",
                            "Literal groups are not independent episodes and cannot yet be used as episode aggregation weights."]}


def prepare(snapshot, dictionary, *, encoding=None):
    """Return private source units and safe readiness, without I/O or mutation.

    Every approved actor and scale gets a not_scored/null cell. Evidence counts
    describe availability only; no keyword or draft-label match becomes a score.
    The caller may supply an already loaded LOCAL tokenizer; None means that
    exact token counts are unavailable, never zero-cost input.
    """
    config = snapshot["corpus"]["config"]
    objects = config["objects"]
    if not isinstance(objects, list) or not objects:
        raise ValueError("A nonempty approved actor roster is required")
    actors, names = [], {}
    for actor in objects:
        identifier = actor.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in names:
            raise ValueError("Approved actor IDs must be nonempty and unique")
        actors.append(identifier)
        names[identifier] = _approved_public_label(actor.get("name"), "actor name")
    if dictionary.get("stage_status") not in {"draft", "approved"}:
        raise ValueError("Dictionary stage_status must be explicitly draft or approved")
    scales = dictionary.get("scales")
    if not isinstance(scales, list) or not scales:
        raise ValueError("A nonempty candidate dictionary is required")
    scale_ids, safe_scales = set(), []
    for scale in scales:
        identifier = scale.get("id")
        if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,120}", identifier) or identifier in scale_ids:
            raise ValueError("Property IDs must be unique stable slugs")
        scale_ids.add(identifier)
        safe_scales.append({"id": identifier, "name": _approved_public_label(scale.get("name"), "property name")})

    facts_by_id, exclusions, input_counts = {}, Counter(), Counter()
    groups, eligible = defaultdict(list), []
    repeated_rows = 0
    for fact in snapshot["facts"]:
        identifier = fact.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("Each source fact needs a nonempty ID")
        if identifier in facts_by_id:
            if fact != facts_by_id[identifier]:
                raise ValueError("Conflicting duplicate fact ID")
            repeated_rows += 1
            continue
        facts_by_id[identifier] = fact
        actor = fact.get("actor_id")
        if actor in names:
            input_counts[actor] += 1
        if actor not in names:
            exclusions["unapproved_actor"] += 1
        elif not filter_eligible_fact(fact, config):
            exclusions["measurement_or_review_ineligible"] += 1
        elif (not isinstance(fact.get("quote"), str) or not fact["quote"].strip()
              or not isinstance(fact.get("document_id"), str) or not fact["document_id"]):
            exclusions["missing_literal_quote_or_document"] += 1
        else:
            eligible.append(fact)
            # None is an explicit grouping category, never source_date. A known
            # date never joins None or a different explicit event date.
            groups[(actor, _literal(fact["quote"]), fact.get("event_date"))].append(fact)

    order = {actor: i for i, actor in enumerate(actors)}
    units = []
    for (actor, quote, event_date), members in sorted(groups.items(), key=lambda item: (order[item[0][0]], digest(item[0]))):
        members = sorted(members, key=lambda fact: fact["id"])
        units.append({"unit_id": "literal-" + digest([actor, quote, event_date])[:28],
                      "unit_type": "literal_quote_group", "actor_id": actor,
                      "event_date": event_date, "normalized_quote": quote,
                      "normalization": "NFC_and_whitespace_only", "fact_ids": [fact["id"] for fact in members],
                      "member_facts": deepcopy(members), "member_count": len(members),
                      "episode_identity_verified": False, "source_independence_verified": False})

    by_actor = defaultdict(list)
    for unit in units:
        by_actor[unit["actor_id"]].append(unit)
    readiness = []
    for actor in actors:
        actor_units = by_actor[actor]
        members = [fact for unit in actor_units for fact in unit["member_facts"]]
        count = len(actor_units)
        readiness.append({"actor_id": actor, "name": names[actor], "source_facts": input_counts[actor],
                          "eligible_facts": len(members), "evidence_units": count,
                          "literal_duplicate_groups": sum(unit["member_count"] > 1 for unit in actor_units),
                          "literal_duplicate_member_excess": sum(unit["member_count"] - 1 for unit in actor_units),
                          "eligible_source_documents": len({fact["document_id"] for fact in members}),
                          "units_with_explicit_event_date": sum(unit["event_date"] is not None for unit in actor_units),
                          "units_with_unknown_event_date": sum(unit["event_date"] is None for unit in actor_units),
                          "availability": "none" if not count else "sparse" if count < 5 else "material_available",
                          "scoring_status": "not_scored", "factor_readiness": "not_evaluated_no_scores"})
    cells = [{"actor_id": actor, "scale_id": scale["id"], "status": "not_scored", "value": None}
             for actor in actors for scale in safe_scales]
    facts_sha = digest([facts_by_id[key] for key in sorted(facts_by_id)])
    dictionary_sha = digest(dictionary)
    metadata = {"policy_version": POLICY_VERSION, "facts_content_sha256": facts_sha,
                "dictionary_content_sha256": dictionary_sha, "roster_sha256": digest(objects),
                "evidence_units_content_sha256": digest(units),
                "grouping": "same_actor_same_NFC_whitespace_quote_same_explicit_date_or_null",
                "date_policy": "unknown_remains_null; publication_date_never_substitutes_event_date",
                "episode_identity_verified": False, "source_independence_verified": False,
                "api_calls": 0, "queue_modified": False, "corpus_config_modified": False,
                "canonical_study_modified": False, "source_receipts_modified": False}
    public = {"version": POLICY_VERSION, "corpus_id": snapshot["corpus"].get("id"),
              "source_sha256": snapshot["corpus"].get("source_sha256"),
              "dictionary_stage_status": dictionary["stage_status"], "provenance": metadata,
              "counts": {"approved_actors": len(actors), "candidate_properties": len(scales),
                         "unique_input_facts": len(facts_by_id), "repeated_identical_input_rows": repeated_rows,
                         "eligible_facts": len(eligible), "evidence_units": len(units),
                         "literal_duplicate_groups": sum(unit["member_count"] > 1 for unit in units),
                         "literal_duplicate_member_excess": len(eligible) - len(units),
                         "actors_with_eligible_material": sum(bool(by_actor[actor]) for actor in actors),
                         "actors_without_eligible_material": sum(not by_actor[actor] for actor in actors),
                         "unscored_cells": len(cells), "observed_cells": 0,
                         "exclusions": dict(sorted(exclusions.items()))},
              "availability_policy": {"sparse_if_evidence_units_below": 5,
                                      "purpose": "descriptive_warning_only_not_a_factor_analysis_admission_gate"},
              "actor_readiness": readiness, "scales": safe_scales, "cells": cells,
              "packet_sizing": _packet_plan(units, scales, encoding),
              "api_calls": 0, "new_cost_usd": 0, "scores_generated": False,
              "public_safety": {"quotes_published": False, "fact_ids_published": False,
                                "source_urls_published": False, "authors_published": False},
              "limitations": ["These are literal evidence units, not verified independent episodes.",
                              "Identical words with an unknown or identical explicit date may refer to different events.",
                              "Different quotes, paraphrases and translations of one event are not merged.",
                              "All original member facts, quotes, attribution and dates remain in the private export.",
                              "Counts measure material availability, never intensity, relevance to every property or media influence.",
                              "All cells are not_scored/null, including actors with no eligible material; no zero scores are inferred.",
                              "Sparse/available labels do not establish that a factor solution can be estimated.",
                              ("Dictionary approval is recorded; attribution/event review and an explicit scoring pilot are still required."
                               if dictionary["stage_status"] == "approved" else
                               "Dictionary approval, attribution/event review and an explicit scoring pilot are still required.")]}
    return {"private_units": units, "metadata": metadata, "public_readiness": public}


def render_readiness(public):
    """Render only the safe allowlisted aggregate, with no source-unit input."""
    count = public["counts"]
    dictionary_label = "утверждённых" if public["dictionary_stage_status"] == "approved" else "предлагаемых"
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["# Подготовка оценивания по источникам Руанды", "",
             "Это офлайн-подготовка следующего этапа. Новых оценок и обращений к API нет.", "",
             f"В реестре **{count['approved_actors']} участников** и **{count['candidate_properties']} {dictionary_label} свойств**. "
             f"Собственные допустимые материалы есть у **{count['actors_with_eligible_material']}**, "
             f"без них — **{count['actors_without_eligible_material']}**.", "",
             f"**{count['eligible_facts']} фрагментов** организованы в **{count['evidence_units']} единиц материала**. "
             f"Групп точных повторов — **{count['literal_duplicate_groups']}**; "
             f"повторных членов сверх первого — **{count['literal_duplicate_member_excess']}**.", "",
             "Объединены только цитаты одного человека, совпавшие после NFC-нормализации и приведения пробелов, "
             "с одинаковой явной датой или с неизвестной датой у всех членов. Неизвестная дата остаётся неизвестной. "
             "Разные известные даты и известная/неизвестная дата разделены.", "",
             "**Это не независимые эпизоды.** Одинаковые слова могли прозвучать несколько раз; "
             "переводы и разные цитаты об одном событии здесь не объединены. Исходные фрагменты и атрибуция сохранены локально.", "",
             f"Все **{count['unscored_cells']} клеток** будущей таблицы имеют `status=not_scored`, `value=null`. "
             "Упоминание темы не считается оценкой. Покрытие конкретных свойств и пригодность факторизации ещё не измерялись.", "",
             "## Доступность материала по участникам", "",
             "«Мало материала» означает менее пяти единиц и служит только предупреждением о доступности.", "",
             "| Участник | Допустимых фрагментов | Единиц материала | Документов | С явной датой | Доступность |",
             "|---|---:|---:|---:|---:|---|"]
    labels = {"none": "Нет", "sparse": "Мало материала", "material_available": "Материал есть"}
    for row in public["actor_readiness"]:
        lines.append("| " + " | ".join(map(cell, [row["name"], row["eligible_facts"], row["evidence_units"],
                          row["eligible_source_documents"], row["units_with_explicit_event_date"], labels[row["availability"]]])) + " |")
    sizing = public["packet_sizing"]
    lines += ["", "## Размер будущих запросов: предварительные сценарии", "",
              "Протокол пока не оптимизирован. Числа ниже описывают содержимое условных пакетов; "
              "они не являются точной сметой API. Инструкции, схема ответа, служебная обвязка, выходные токены и кеширование не учтены.", "",
              "| Сценарий | Условных пакетов | Проверок «единица × свойство» | Входных байт компонентов | Токенов компонентов |",
              "|---|---:|---:|---:|---:|"]
    for key, label in (("one_unit_all_scales", "Одна единица × все свойства"), ("one_unit_one_scale", "Одна единица × одно свойство")):
        row = sizing[key]
        lines.append("| " + " | ".join(map(cell, [label, row["hypothetical_packets"], row["hypothetical_property_evaluations"],
                          row["utf8_bytes"], row["content_component_tokens"] if row["content_component_tokens"] is not None else "Недоступно"])) + " |")
    lines += ["", f"Локальный токенизатор: `{sizing['tokenizer'] or 'недоступен'}`. "
              "Фактическая токенизация объединённого API-запроса может отличаться от суммы компонентов.", "",
              "Денежная смета не рассчитана: перед платным проходом нужны проверка событий и отдельный пилот оценивания по согласованному словарю. "
              "Новые расходы этой подготовки — **$0**. Старые результаты, очередь и лимит прогона $70 не изменены.", ""]
    return "\n".join(lines)
