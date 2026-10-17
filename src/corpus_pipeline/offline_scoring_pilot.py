"""Deterministic review sample only: no scoring, clients, queues or I/O."""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from datetime import date
import json
import re

from .scoring_preparation import digest, filter_eligible_fact


POLICY_VERSION = "offline-source-review-pilot-v1"
LANGUAGES = {"rw", "en", "fr", "sw", "mixed", "unknown", "other"}
ATTRIBUTIONS = {"direct_statement", "reported_statement", "reported_action"}
RUSSIA_WORDS = re.compile(r"(?<!\w)(?:russia(?:n|ns)?|russie|russe(?:s)?|uburusiya|abarusiya|rusiya|urusi|kirusi)(?!\w)", re.I)
OUTSIDE = {"historical_outside_window", "historical_out_of_scope", "outside_window",
           "outside_period", "historical_outside_scope", "out_of_scope"}


def _languages(jobs):
    labels = defaultdict(set)
    for job in jobs:
        if job.get("stage") != "relevance" or job.get("status") != "done" or job.get("active_recipe") is not True:
            continue
        result = job.get("result")
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except (ValueError, TypeError):
                result = {}
        language = result.get("language") if isinstance(result, dict) else None
        context = job.get("context")
        if isinstance(context, str):
            try:
                context = json.loads(context)
            except (ValueError, TypeError):
                context = {}
        document = context.get("document", {}) if isinstance(context, dict) else {}
        identifier = document.get("id") if isinstance(document, dict) else None
        # Real snapshots use a stage-specific unit_id; the original document ID
        # belongs to context.document.id. Older minimal exports can lack context.
        if not isinstance(identifier, str) or not identifier:
            identifier = job.get("unit_id")
        if identifier:
            labels[identifier].add(language if language in LANGUAGES else "unknown")
    # Conflicting active labels are not resolved with language_hint or recency.
    return {identifier: next(iter(values)) if len(values) == 1 else "unknown"
            for identifier, values in labels.items()}


def _features(unit, languages):
    members = unit["member_facts"]
    labels = {languages.get(fact["document_id"], "unknown") for fact in members}
    language = next(iter(labels)) if len(labels) == 1 else "unknown"
    tags = {"language:" + language, "date:" + ("unknown" if unit["event_date"] is None else "explicit"),
            "literal:" + ("repeat" if unit["member_count"] > 1 else "singleton")}
    for fact in members:
        basis = fact.get("attribution_basis")
        tags.add("attribution:" + (basis if basis in ATTRIBUTIONS else "other"))
    # This is a sampling cue in literal evidence, not an inferred position.
    if any(RUSSIA_WORDS.search(str(fact.get("quote", "")) + " " + str(fact.get("attribution_evidence", "")))
           for fact in members):
        tags.add("russia:lexical_candidate")
    return tags


def _control_reason(fact, config):
    temporal = str(fact.get("temporal_status", "")).casefold()
    event = fact.get("event_date")
    invalid_date = False
    if event:
        try:
            date.fromisoformat(event)
        except (ValueError, TypeError):
            invalid_date = True
    if temporal in OUTSIDE or (event and not invalid_date and
            (config.get("date_from") and event < config["date_from"] or
             config.get("date_to") and event > config["date_to"])):
        return "historical_outside_window"
    if invalid_date:
        return "invalid_event_date"
    if fact.get("attribution_basis") == "uncertain" or fact.get("is_own_position") is False or (
            fact.get("speaker_actor_id") is not None and fact["speaker_actor_id"] != fact.get("actor_id")):
        return "unresolved_or_foreign_attribution"
    if fact.get("attribution_basis") == "third_party_claim" or fact.get("kind") == "attributed_image":
        return "third_party_image_or_claim"
    if fact.get("measurement_status") == "context_only" or fact.get("kind") == "institutional_context" or fact.get("attribution_basis") == "context":
        return "institutional_context_without_own_position"
    if fact.get("review_status") in {"rejected", "excluded"} or fact.get("excluded"):
        return "rejected_or_excluded_review"
    if fact.get("review_status") == "unreviewed" or fact.get("reviewed") is False:
        return "unreviewed_attribution_or_scope"
    return "other_measurement_ineligible"


def build(snapshot, dictionary, prepared, *, max_units=60, max_controls=12, max_per_actor=3):
    """Build a private review package and an allowlisted public summary.

    The prepared object must originate from scoring_preparation.prepare using
    these exact facts, roster and dictionary. No selection becomes a rating.
    """
    if not (0 <= max_units <= 60 and 0 <= max_controls <= 12 and 1 <= max_per_actor <= 3):
        raise ValueError("Review sample limits exceeded")
    config = snapshot["corpus"]["config"]
    actors = config["objects"]
    actor_order = {actor["id"]: i for i, actor in enumerate(actors)}
    metadata = prepared["metadata"]
    facts = {fact["id"]: fact for fact in snapshot["facts"]}
    if (metadata["dictionary_content_sha256"] != digest(dictionary) or
            metadata["roster_sha256"] != digest(actors) or
            metadata["facts_content_sha256"] != digest([facts[key] for key in sorted(facts)]) or
            metadata["evidence_units_content_sha256"] != digest(prepared["private_units"])):
        raise ValueError("Prepared material does not match frozen inputs")
    units = prepared["private_units"]
    languages = _languages(snapshot.get("jobs", []))
    features = {unit["unit_id"]: _features(unit, languages) for unit in units}
    available = Counter(tag for tags in features.values() for tag in tags)
    by_actor = defaultdict(list)
    for unit in units:
        by_actor[unit["actor_id"]].append(unit)
    if max_units < len(by_actor):
        raise ValueError("Sample limit must preserve one unit per actor with material")
    selected, seen, covered, actor_counts = [], set(), Counter(), Counter()

    def rank(unit):
        tags = features[unit["unit_id"]]
        new = sum(1 / available[tag] for tag in sorted(tags) if not covered[tag])
        diversity = sum(1 / (available[tag] * (covered[tag] + 1)) for tag in sorted(tags))
        return (-new, -diversity, len(by_actor[unit["actor_id"]]), digest([POLICY_VERSION, unit["unit_id"]]))

    def take(unit):
        selected.append(unit)
        seen.add(unit["unit_id"])
        actor_counts[unit["actor_id"]] += 1
        covered.update(features[unit["unit_id"]])

    # Scarce actors get their mandatory first place before any extra place.
    for actor in sorted(by_actor, key=lambda value: (len(by_actor[value]), actor_order[value])):
        if len(selected) >= max_units:
            break
        take(min(by_actor[actor], key=rank))
    while len(selected) < max_units:
        candidates = [unit for unit in units if unit["unit_id"] not in seen and actor_counts[unit["actor_id"]] < max_per_actor]
        if not candidates:
            break
        take(min(candidates, key=rank))

    controls = []
    for fact in facts.values():
        if (fact.get("actor_id") in actor_order and not filter_eligible_fact(fact, config)
                and isinstance(fact.get("quote"), str) and fact["quote"].strip()
                and isinstance(fact.get("document_id"), str) and fact["document_id"]):
            controls.append({"reason_family": _control_reason(fact, config), "fact": fact})
    chosen_controls, reason_counts, control_actor_counts = [], Counter(), Counter()
    while controls and len(chosen_controls) < max_controls:
        chosen = min(controls, key=lambda row: (bool(reason_counts[row["reason_family"]]),
                     control_actor_counts[row["fact"]["actor_id"]], reason_counts[row["reason_family"]],
                     digest([POLICY_VERSION, row["fact"]["id"]])))
        controls.remove(chosen)
        chosen_controls.append(chosen)
        reason_counts[chosen["reason_family"]] += 1
        control_actor_counts[chosen["fact"]["actor_id"]] += 1

    review_fields = {"status": "not_reviewed", "attribution_verdict": None,
                     "capacity_verdict": None, "event_date_verdict": None,
                     "episode_identity_verdict": None, "independence_verdict": None,
                     "related_unit_ids": [], "notes": None}
    private = {"policy_version": POLICY_VERSION, "execution_allowed": False,
               "selected_units": [{**deepcopy(unit), "selection_features": sorted(features[unit["unit_id"]]),
                                    "review_status": "pending_source_event_review",
                                    "review_fields": deepcopy(review_fields)} for unit in selected],
               "excluded_controls": [{**deepcopy(row), "numeric_scoring_eligible": False,
                                      "review_fields": {"status": "not_reviewed", "semantic_rejection_confirmed": None,
                                                        "notes": None}} for row in chosen_controls],
               "calibration_holdout_split": None, "property_candidates": None,
               "api_calls": 0, "new_cost_usd": 0, "scores_generated": False}
    all_control_reasons = Counter(_control_reason(fact, config) for fact in facts.values()
                                 if fact.get("actor_id") in actor_order and not filter_eligible_fact(fact, config)
                                 and isinstance(fact.get("quote"), str) and fact["quote"].strip()
                                 and isinstance(fact.get("document_id"), str) and fact["document_id"])
    public = {"policy_version": POLICY_VERSION, "dictionary_stage_status": dictionary["stage_status"],
              "execution_allowed": False, "execution_block_reason": "user_requested_no_paid_calls_and_api_credit_unavailable",
              "counts": {"approved_actors": len(actors), "actors_with_material": len(by_actor),
                         "actors_represented": len(actor_counts), "available_units": len(units),
                         "selected_units": len(selected), "selected_controls": len(chosen_controls),
                         "available_controls": sum(all_control_reasons.values()), "observed_scores": 0},
              "limits": {"eligible_units": max_units, "excluded_controls": max_controls, "units_per_actor": max_per_actor},
              "feature_coverage": [{"feature": tag, "available_units": available[tag], "selected_units": covered[tag]}
                                   for tag in sorted(available)],
              "missing_available_features": sorted(tag for tag in available if not covered[tag]),
              "control_reason_coverage": [{"reason_family": reason, "available_controls": all_control_reasons[reason],
                                           "selected_controls": reason_counts[reason]} for reason in sorted(all_control_reasons)],
              "actor_cards": [{"name": actor["name"], "available_units": len(by_actor[actor["id"]]),
                               "selected_units": actor_counts[actor["id"]], "selected_excluded_controls": control_actor_counts[actor["id"]],
                               "scoring_status": "not_scored", "value": None} for actor in actors],
              "provenance": {**metadata, "selection_policy_sha256": digest({"version": POLICY_VERSION,
                             "max_units": max_units, "max_controls": max_controls, "max_per_actor": max_per_actor}),
                             "private_review_package_sha256": digest(private),
                             "active_relevance_languages_sha256": digest(languages)},
              "calibration_holdout_split": None, "api_calls": 0, "new_cost_usd": 0, "scores_generated": False,
              "limitations": ["This is a targeted review sample, not a population accuracy estimate.",
                              "Literal groups are not verified independent events; no calibration/holdout split is assigned.",
                              "Russia lexical candidates are sampling cues, never positions or numeric ratings.",
                              "Missing languages or risks are not invented to fill quotas.",
                              "Publication dates never replace unknown event dates.",
                              "Excluded controls remain outside numeric scoring.",
                              "Dictionary approval does not authorize paid execution."]}
    return {"private_package": private, "public_summary": public}


def render(public):
    """Render the aggregate summary without access to the private package."""
    count = public["counts"]
    lines = ["# Офлайн-пакет проверки оценивания Руанды", "",
             "Пользователь выбрал продолжение без платных запросов; баланс API пока не пополнен. "
             "Платный запуск отложен (`execution_allowed=false` в манифесте пакета). Новые расходы — **$0**.", "",
             f"Подготовлено **{count['selected_units']} единиц материала** для **{count['actors_represented']} участников** "
             f"и **{count['selected_controls']} отдельных исключённых контролей**. Все **{count['approved_actors']} карточки** сохранены; оценок нет.", "",
             "Сначала выбрана одна единица для каждого участника с пригодным материалом, начиная с самых редких. "
             "Остальные места заполняются для разнообразия языков, атрибуции, дат, буквальных повторов и упоминаний России. "
             "Не более трёх единиц на человека. Состав воспроизводим и зафиксирован контрольными суммами.", "",
             "Язык публикации взят только из завершённых активных проверок релевантности. Это сохранённая модельная разметка, "
             "а не новая проверка языка цитаты. Конфликты и отсутствие разметки дают `unknown`; "
             "языковые подсказки выгрузки не используются. Упоминание России служит только поводом проверить фрагмент, не оценкой позиции.", "",
             "Допустимость означает прохождение текущего входного фильтра; смысл, говорящий и связь с событием ещё подлежат проверке. "
             "В локальном пакете сохранён пустой шаблон такой проверки; все поля заключения остаются незаполненными.", "",
             "## Покрытие признаков отбора", "", "| Метка отбора | Доступно единиц | Выбрано |", "|---|---:|---:|"]
    labels = {"language:en": "Язык публикации: английский", "language:fr": "Язык публикации: французский",
              "language:rw": "Язык публикации: киньяруанда", "language:sw": "Язык публикации: суахили",
              "language:mixed": "Язык публикации: смешанный", "language:unknown": "Язык публикации: не установлен",
              "language:other": "Язык публикации: другой", "date:explicit": "Явная дата события из извлечения",
              "date:unknown": "Дата события не установлена", "literal:repeat": "Группа буквальных повторов",
              "literal:singleton": "Цитата без буквального повтора",
              "attribution:direct_statement": "Атрибуция извлечения: прямая речь",
              "attribution:reported_statement": "Атрибуция извлечения: пересказ высказывания",
              "attribution:reported_action": "Атрибуция извлечения: пересказ действия",
              "attribution:other": "Атрибуция извлечения: иная",
              "russia:lexical_candidate": "Упоминание России: лексический кандидат"}
    for row in public["feature_coverage"]:
        lines.append(f"| {labels.get(row['feature'], row['feature'])} | {row['available_units']} | {row['selected_units']} |")
    lines += ["", "## Исключённые контрольные примеры", "", "| Причина исключения | Доступно | Выбрано |", "|---|---:|---:|"]
    reasons = {"historical_outside_window": "Событие вне периода",
               "invalid_event_date": "Некорректная дата события",
               "third_party_image_or_claim": "Чужое описание или обвинение",
               "institutional_context_without_own_position": "Контекст без собственной позиции",
               "unresolved_or_foreign_attribution": "Неопределённый или иной говорящий",
               "rejected_or_excluded_review": "Отклонено при проверке",
               "unreviewed_attribution_or_scope": "Удержано до проверки",
               "other_measurement_ineligible": "Иное несоответствие входному фильтру"}
    for row in public["control_reason_coverage"]:
        lines.append(f"| {reasons.get(row['reason_family'], row['reason_family'])} | {row['available_controls']} | {row['selected_controls']} |")
    lines += ["", "Эти примеры уже отклонены текущим фильтром и предназначены для проверки его обоснованности. "
              "Их наличие не является независимой оценкой точности фильтра. В числовое оценивание они не допускаются. "
              "Отсутствующие риски не заменяются вымышленными примерами.", "", "## Участники", "",
              "| Участник | Доступно единиц | В пакете | Исключённых контролей |", "|---|---:|---:|---:|"]
    for row in public["actor_cards"]:
        name = row["name"].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {name} | {row['available_units']} | {row['selected_units']} | {row['selected_excluded_controls']} |")
    missing = ", ".join(public["missing_available_features"]) or "нет"
    lines += ["", f"Доступные, но не попавшие в пакет метки: {missing}.", "",
              "Буквальная группа ещё не является независимым эпизодом. Перед оцениванием нужны проверка говорящего, времени, "
              "служебной роли и связи материалов с одним событием. Поэтому калибровочная и проверочная части пока не разделены. "
              "Неизвестные даты остаются неизвестными; все оценки — `not_scored/null`.", "",
              "Исходные цитаты, идентификаторы фактов и документов остаются только в локальном закрытом пакете. "
              "Очередь, исходные квитанции, утверждённые данные и старые результаты не изменены. "
              "Для будущего платного пилота понадобятся отдельный лимит и пополнение API; прежний лимит $70 не расширяется. "
              "Пополнение баланса не запускает обработку автоматически.", "",
              "[Согласованный словарь](APPROVAL.md) · [Решение о расходах](execution_decision.json) · [План проверки](SCORING_PILOT_PLAN.md)", ""]
    return "\n".join(lines)
