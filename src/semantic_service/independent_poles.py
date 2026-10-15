"""Evidence-bound unipolar measurement; missing antipodes never discard a score.

Old paired measurements and dossiers are immutable inputs. New retrieval, raw
model replies, per-property support and decisions are stored separately.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

from .store import fingerprint, now


def variables(scales):
    result = []
    for scale in scales:
        poles = ["positive"] if scale.get("kind") == "unipolar" else ["positive", "negative"]
        for pole in poles:
            result.append({"id": scale["id"] if scale.get("kind") == "unipolar" else scale["id"] + "__" + pole,
                           "parent_scale_id": scale["id"], "pole": pole,
                           "name": scale["name"] if scale.get("kind") == "unipolar" else scale[pole],
                           "definition": scale["definition"],
                           "research_domain": scale.get("research_domain"),
                           "semantic_family_id": scale.get("semantic_family_id", scale["id"])})
    return result


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _raw_reply(provider, study, response):
    path = provider.store.directory(study["id"]) / "evidence/independent_responses" / (response["cache_key"] + ".json")
    _write(path, {key: response.get(key) for key in
                 ("data", "model", "task", "at", "usage", "citations", "system_fingerprint", "cache_key")})
    return path.relative_to(provider.store.directory(study["id"])).as_posix()


def assemble_dossier(provider, study, obj):
    """Reuse dated original evidence, without rewriting any earlier source file."""
    from .providers import fact_within_period
    directory = provider.store.directory(study["id"])
    candidates = [directory / "evidence" / (obj["id"] + ".json"),
                  directory / "evidence" / ("reviewed-" + obj["id"] + ".json"),
                  directory / "evidence" / ("dossier_supplement-" + obj["id"] + ".json")]
    facts, sources, seen, inputs = [], [], set(), []
    historical = False
    previous = study.get("measurement_protocol", {}).get("previous_snapshot")
    if previous:
        snapshot = (directory / previous / "study.json").resolve()
        if not snapshot.is_relative_to(directory.resolve()):
            raise ValueError("Historical dossier context escaped the study")
        original = json.loads(snapshot.read_text(encoding="utf-8"))
        historical = (all(original.get(k) == study.get(k) for k in ("scope", "aspect", "date_from", "date_to")) and
                      next((o for o in original.get("objects", []) if o["id"] == obj["id"]), None) == obj)
    if not historical:
        base = provider.dossier(study, obj)
        base_path = directory / base.get("evidence_file", "evidence/" + obj["id"] + ".json")
        candidates = [base_path]
        supplemental_path = directory / "evidence" / ("dossier_supplement-" + obj["id"] + ".json")
        if supplemental_path.exists():
            supplemental = json.loads(supplemental_path.read_text(encoding="utf-8"))
            original_basis = fingerprint({"basis": base.get("basis"), "facts": base.get("facts", [])})
            if supplemental.get("original_dossier_basis") == original_basis and supplemental.get("scales_basis") == fingerprint(study["scales"]):
                candidates.append(supplemental_path)
    for path in candidates:
        if not path.exists():
            continue
        dossier = json.loads(path.read_text(encoding="utf-8"))
        if dossier.get("object_id") != obj["id"]:
            raise ValueError("Independent dossier actor mismatch")
        inputs.append({"file": path.relative_to(directory).as_posix(), "sha256": fingerprint(dossier)})
        sources.extend(dossier.get("sources", []))
        for index, raw in enumerate(dossier.get("facts", [])):
            if not fact_within_period(raw, study):
                continue
            key = (str(raw.get("source_url", "")).rstrip("/"), " ".join(str(raw.get("text", "")).split()))
            if not key[0].startswith(("http://", "https://")) or not key[1] or key in seen:
                continue
            seen.add(key)
            facts.append(dict(raw, original_fact_file=path.relative_to(directory).as_posix(), original_fact_index=index))
    result = {"object_id": obj["id"], "facts": facts, "sources": sources, "input_files": inputs,
              "period": [study["date_from"], study["date_to"]], "measurement_model": "independent_poles"}
    result["basis"] = fingerprint(result)
    return result


def enrich_independent_dossier(provider, study, obj, dossier=None):
    """One comparable, preregistered search round for every actor, never fill to a target."""
    from .providers import TEXT, SOURCE, evidence_fact_schema, fact_within_period, grounded_facts, object_schema
    dossier = dossier or assemble_dossier(provider, study, obj)
    directory = provider.store.directory(study["id"])
    destination = directory / "evidence" / ("independent-retrieval-" + obj["id"] + ".json")
    ordinary = [v for v in variables(study["scales"]) if v["research_domain"] != "russia_relations"]
    basis = fingerprint({"version": 1, "input_basis": dossier["basis"], "properties": ordinary,
                         "scope": study["scope"], "aspect": study["aspect"],
                         "period": dossier["period"], "model": provider.search_model})
    if destination.exists():
        saved = json.loads(destination.read_text(encoding="utf-8"))
        if saved.get("retrieval_basis") == basis:
            return saved
    properties = [{k: v[k] for k in ("id", "name")} for v in ordinary]
    prompt = (f"Найди дополнительные первичные материалы о публичных политических действиях и заявлениях {obj['name']}. "
              f"Область: {study['scope']}. Только события {study['date_from']}—{study['date_to']}. "
              "Нужны 18–24 содержательных, неповторяющихся датированных эпизода, охватывающих разные характеристики. "
              "Это независимые однополюсные свойства, а не обязательные антонимы: одного подтверждения свойства достаточно "
              "для его измерения, неизвестное противоположное свойство не мешает. Сам не назначай числа. "
              "Ищи программы, конкретные решения, прямые интервью и выступления, предложения по экономике, социальным "
              "правам, выборам, институтам, международным правилам, экологии; также публичную аргументацию и стиль. "
              "Нужны как подтверждения, так и явные ограничения/контрпримеры. Должность, приговор, обвинения оппонентов "
              "и молчание не заменяют собственную политическую позицию. Различай объявленное намерение и реализацию. "
              "У каждого эпизода source_url из реальных retrieval citations, точная дата события в period, capacity, "
              "channel, event_id. topic_ids=[] для обычных тем. Пересказывай, не цитируй длинные фрагменты. "
              "Публичное служебное действие допустимо как действие, но не как личное убеждение. Не выдумывай недостающие данные. "
              "Россия исследуется отдельным поиском; здесь ищи общие политические признаки. "
              f"Свойства: {json.dumps(properties, ensure_ascii=False)}\n"
              f"Уже найдено (не повторяй): {json.dumps([f['text'] for f in dossier['facts']], ensure_ascii=False)}")
    schema = object_schema({"facts": {"type": "array", "items": evidence_fact_schema(study)},
                            "sources": {"type": "array", "items": SOURCE}, "limitations": TEXT})
    response = provider.request(study["id"], "independent-retrieval-" + obj["id"], prompt, schema, search=True, max_tokens=6500)
    response_file = _raw_reply(provider, study, response)
    accepted = [f for f in grounded_facts(response["data"], response.get("citations", [])) if fact_within_period(f, study)]
    known = {(f["source_url"].rstrip("/"), " ".join(f["text"].split())) for f in dossier["facts"]}
    additions = [dict(f, evidence_origin="retrieval_cited_not_independently_verified", retrieval_response=response_file)
                 for f in accepted if (f["source_url"].rstrip("/"), " ".join(f["text"].split())) not in known]
    result = copy.deepcopy(dossier)
    result.update(facts=dossier["facts"] + additions, retrieval_basis=basis,
                  retrieval={"response_file": response_file, "citations": response.get("citations", []),
                             "accepted_fact_count": len(accepted), "added_fact_count": len(additions),
                             "limitations": response["data"].get("limitations"), "at": response["at"]})
    result["basis"] = fingerprint({"input": dossier["basis"], "retrieval_basis": basis, "facts": result["facts"]})
    _write(destination, result)
    return result


def _unknown(reason):
    return {"status": "unknown", "value": None, "confidence": 0, "evidence_indices": [],
            "source_urls": [], "event_ids": [], "rationale": reason, "support_type": "unknown"}


def attach_reviewed_sources(provider, study, obj, dossier, config_key="reviewed_russia_supplement"):
    """Attach the pinned source-review layer in both CLI and normal service runs."""
    from .providers import fact_within_period
    config = study.get("independent_evidence", {}).get(config_key)
    if not config:
        return dossier
    root = Path(provider.root).resolve()
    supplement = (root / config["path"]).resolve()
    if not supplement.is_relative_to(root) or hashlib.sha256(supplement.read_bytes()).hexdigest() != config.get("sha256"):
        raise ValueError("Reviewed source layer escaped the repository or changed after approval")
    data = json.loads(supplement.read_text(encoding="utf-8"))
    allowed_topics = {scale["id"] for scale in study["scales"]} | {v["id"] for v in variables(study["scales"])}
    selected = []
    for raw in data.get("facts", []):
        if raw.get("actor_id", raw.get("object_id")) != obj["id"] or raw.get("eligibility") not in {"eligible", "eligible_direct"}:
            continue
        fact = dict(raw, period=raw.get("period", raw.get("date")))
        if (not fact_within_period(fact, study) or not fact.get("topic_ids") or
            any(sid not in allowed_topics for sid in fact["topic_ids"]) or
            not str(fact.get("source_url", "")).startswith(("https://", "http://")) or
            any(not isinstance(fact.get(k), str) or not fact[k].strip() for k in ("text", "event_id", "capacity"))):
            raise ValueError("Eligible reviewed fact has invalid attribution, date, topics or source")
        selected.append(dict(fact, evidence_origin="assistant_verified_source", source_review_file=supplement.relative_to(root).as_posix()))
    result = copy.deepcopy(dossier)
    seen = {(f.get("source_url"), f.get("text")): index for index, f in enumerate(result["facts"])}
    for fact in selected:
        key = (fact.get("source_url"), fact.get("text"))
        if key in seen:
            # Preserve indices and exclusions while binding an existing excerpt
            # to its actual reviewed layer, including future thematic scoring.
            existing = result["facts"][seen[key]]
            existing["evidence_origin"] = fact["evidence_origin"]
            existing.setdefault("source_review_file", fact["source_review_file"])
            existing["topic_ids"] = sorted(set((existing.get("topic_ids") or []) + fact["topic_ids"]))
        else:
            seen[key] = len(result["facts"])
            result["facts"].append(fact)
    result[config_key] = {"path": supplement.relative_to(root).as_posix(),
        "sha256": config["sha256"], "selected_fact_count": len(selected)}
    if result["facts"] == dossier["facts"] and result[config_key] == dossier.get(config_key):
        return result
    result["basis"] = fingerprint({"previous_basis": dossier["basis"], "facts": result["facts"],
                                   "reviewed_supplement": result[config_key]})
    return result


def apply_fact_review(provider, study, obj, dossier):
    """Keep known source errors and duplicate-event corrections out of future scoring."""
    config = study.get("independent_evidence", {}).get("fact_review_registry")
    if not config:
        return dossier
    root = Path(provider.root).resolve()
    path = (root / config["path"]).resolve()
    if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != config.get("sha256"):
        raise ValueError("Fact review registry changed or escaped the repository")
    registry = json.loads(path.read_text(encoding="utf-8"))
    rules = {(r["source_url"], r["text_sha256"]): r for r in registry["facts"] if r["object_id"] == obj["id"]}
    result = copy.deepcopy(dossier)
    for fact in result["facts"]:
        key = (fact.get("source_url"), hashlib.sha256(str(fact.get("text", "")).encode("utf-8")).hexdigest())
        if key not in rules:
            continue
        rule = rules[key]
        fact["fact_review"] = rule
        if rule.get("excluded"):
            fact["measurement_status"] = "excluded_known_source_issue"
        if rule.get("corrected_period"):
            fact.setdefault("original_period", fact.get("period"))
            fact["period"] = rule["corrected_period"]
        if rule.get("canonical_event_id"):
            fact.setdefault("original_event_id", fact.get("event_id"))
            fact["event_id"] = rule["canonical_event_id"]
    result["fact_review_registry"] = config
    if result["facts"] == dossier["facts"] and dossier.get("fact_review_registry") == config:
        return result
    result["basis"] = fingerprint({"input_basis": dossier["basis"], "registry": config, "facts": result["facts"]})
    return result


def score_independent_dossier(provider, study, obj, dossier):
    from .providers import TEXT, fact_within_period, object_schema
    directory = provider.store.directory(study["id"])
    evidence_file = "evidence/independent-dossier-" + obj["id"] + ".json"
    _write(directory / evidence_file, dossier)
    requested = variables(study["scales"])
    additional_exclusions = {(f["object_id"], f["source_url"], f["text_sha256"])
                             for f in study.get("independent_evidence", {}).get("additional_excluded_facts", [])}
    ordinary = [v for v in requested if v["research_domain"] != "russia_relations"]
    required = [v for v in requested if v["research_domain"] == "russia_relations"]
    groups = [ordinary[i:i+18] for i in range(0, len(ordinary), 18)]
    # Thematic constructs see only their own dated, personally attributed facts.
    groups.extend([v for v in required if v["parent_scale_id"] == sid]
                  for sid in dict.fromkeys(v["parent_scale_id"] for v in required))
    scores = {}
    for group_index, group in enumerate(groups):
        if not group:
            continue
        thematic = group[0]["research_domain"] == "russia_relations"
        thematic_topics = {group[0]["parent_scale_id"]} | {v["id"] for v in group}
        positions = [i for i, f in enumerate(dossier["facts"]) if fact_within_period(f, study) and
                     f.get("measurement_status") != "excluded_known_source_issue" and
                     (obj["id"], f.get("source_url"), hashlib.sha256(str(f.get("text", "")).encode("utf-8")).hexdigest()) not in additional_exclusions and
                     (not thematic or (thematic_topics.intersection(f.get("topic_ids", [])) and
                      f.get("evidence_origin") == "assistant_verified_source" and
                      f.get("measurement_status") != "context_only"))]
        if not positions:
            scores.update({v["id"]: _unknown("Нет допустимых датированных персональных свидетельств по этому свойству") for v in group})
            continue
        evidence = [dict({key: dossier["facts"][i].get(key) for key in
                         ("text", "source_url", "period", "event_id", "capacity", "channel")}, evidence_index=i)
                    for i in positions]
        ids = [v["id"] for v in group]
        rating_schema = object_schema({"variable_id": {"type": "string", "enum": ids},
            "status": {"type": "string", "enum": ["observed", "unknown"]},
            "value": {"type": ["number", "null"]}, "confidence": {"type": "number"},
            "support_type": {"type": "string", "enum": ["positive_manifestation", "explicit_absence", "unknown"]},
            "rationale": TEXT, "evidence_indices": {"type": "array", "items": {"type": "integer"}}})
        prompt = (f"Разведочная оценка публичных позиций и действий {obj['name']}, период {study['date_from']}—{study['date_to']}. "
            "Каждое свойство — САМОСТОЯТЕЛЬНАЯ переменная выраженности 0..5, не разность и не выбор антонима. "
            "Неизвестность другого свойства никак не отменяет известную оценку. Свойства могут сочетаться. "
            "По каждому id верни ровно один ответ. При прямом проявлении можно дать observed даже по одному эпизоду "
            "с ограниченной выраженностью: 1=единичное слабое/узкое проявление, 2=ясное ограниченное проявление, "
            "3=несколько последовательных эпизодов/устойчивый публичный акцент, 4=сильный повторяющийся акцент, "
            "5=определяющая устойчивая черта, подкреплённая несколькими независимыми эпизодами. "
            "0=ПРЯМО подтверждённая невыраженность ЭТОГО свойства в достаточном материале; для 0 support_type=explicit_absence. "
            "Отсутствие упоминания, молчание, неизвестность, другое свойство, должность или решение суда не доказывают ноль. "
            "Используй только facts: статус observed требует фактического соответствия смыслу свойства и evidence_indices "
            "(поле evidence_index исходных фактов, НЕ порядковый номер в этом сокращённом списке). "
            "Не додумывай программу по должности/партийности, не переноси действия государства или другого человека. "
            "Служебное действие допускает вывод только о публичном действии, не о частной симпатии. "
            "Контакт с Россией свидетельствует только о конкретном взаимодействии, не о поддержке войны/всех действий России. "
            "Если сведений нет: unknown,value=null,support_type=unknown. rationale объясняет связь конкретного эпизода "
            "с точным свойством и предел вывода, confidence0..1 — достаточность сведений. Не заполняй ради факторного анализа. "
            f"Свойства: {json.dumps(group, ensure_ascii=False)}\nФакты: {json.dumps(evidence, ensure_ascii=False)}")
        response = provider.request(study["id"], f"independent-ratings-{obj['id']}-{group_index}", prompt,
                     object_schema({"ratings": {"type": "array", "items": rating_schema}}), max_tokens=5000)
        raw_file = _raw_reply(provider, study, response)
        returned = response["data"]["ratings"]
        if len(returned) != len(ids) or {r["variable_id"] for r in returned} != set(ids):
            raise ValueError("Independent rating response has foreign, missing or duplicated IDs")
        allowed = set(positions)
        for row in returned:
            value = row.get("value")
            indices = row.get("evidence_indices", [])
            valid_indices = (bool(indices) and len(indices) == len(set(indices)) and
                             all(isinstance(i, int) and not isinstance(i, bool) and i in allowed for i in indices))
            numeric = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 5
            valid = row.get("status") == "observed" and numeric and valid_indices and (
                (value > 0 and row.get("support_type") == "positive_manifestation") or
                (value == 0 and row.get("support_type") == "explicit_absence"))
            score = {"status": "observed", "value": value, "confidence": max(0., min(1., float(row.get("confidence", 0)))),
                     "evidence_indices": indices, "source_urls": list(dict.fromkeys(dossier["facts"][i]["source_url"] for i in indices)),
                     "event_ids": list(dict.fromkeys(dossier["facts"][i].get("event_id", "") for i in indices)),
                     "rationale": row.get("rationale", ""), "support_type": row["support_type"]} if valid else _unknown(
                         row.get("rationale", "") if row.get("status") == "unknown" else "Отклонено проверкой диапазона, источников или основания нулевой оценки")
            score.update(provenance={"raw_response_file": raw_file, "model": response["model"], "at": response["at"],
                                     "cache_key": response["cache_key"], "original_proposal": row},
                         evidence_file=evidence_file, evidence_basis=dossier["basis"])
            scores[row["variable_id"]] = score
    rows = []
    for scale in study["scales"]:
        poles = ["positive"] if scale.get("kind") == "unipolar" else ["positive", "negative"]
        by_pole = {p: scores[scale["id"] if scale.get("kind") == "unipolar" else scale["id"] + "__" + p] for p in poles}
        count = sum(s["status"] == "observed" for s in by_pole.values())
        rows.append({"object_id": obj["id"], "scale_id": scale["id"], "pole_scores": by_pole,
                     "positive": by_pole["positive"]["value"], "negative": by_pole.get("negative", {}).get("value"),
                     "value": by_pole["positive"]["value"] if scale.get("kind") == "unipolar" else None,
                     "status": "observed" if count == len(poles) else "partial" if count else "unknown",
                     "difference": None, "mass": None, "measurement_model": "independent_poles",
                     "evidence_file": evidence_file, "evidence_basis": dossier["basis"], "at": now(),
                     "reason": "Каждое свойство оценивается независимо; разность и искусственное дополнение до пяти не вычисляются"})
    return rows


def rate_independent_object(provider, study, obj):
    dossier = assemble_dossier(provider, study, obj)
    if study.get("independent_evidence", {}).get("retrieval_rounds", 1):
        dossier = enrich_independent_dossier(provider, study, obj, dossier)
    dossier = attach_reviewed_sources(provider, study, obj, dossier)
    dossier = attach_reviewed_sources(provider, study, obj, dossier, "completion_source_supplement")
    dossier = attach_reviewed_sources(provider, study, obj, dossier, "coverage_expansion_source_supplement")
    dossier = apply_fact_review(provider, study, obj, dossier)
    return score_independent_dossier(provider, study, obj, dossier)
