"""Explain the evidence-to-ratings funnel from frozen local files; never score.

Only aggregate counts and approved roster labels are exported. The input can
contain private source IDs, URLs and quotes, but none is copied to the report.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.corpus_pipeline.scoring_preparation import filter_eligible_fact


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _counter(values):
    return dict(sorted(Counter(str(value) if value is not None else "missing" for value in values).items()))


def audit(facts, annotations, combined_review, profiles, roster, config):
    """Produce a deterministic, source-text-free forensic coverage report."""
    eligible = [f for f in facts if filter_eligible_fact(f, config)]
    study_config = {**config, "date_from": profiles["approved_window"]["from"],
                    "date_to": profiles["approved_window"]["to"]}
    study_candidates = [f for f in facts if filter_eligible_fact(f, study_config)]
    annotated_ids = {fid for row in annotations["annotations"] for fid in row["source_fact_ids"]}
    reviewed_by_actor, facts_by_actor, eligible_by_actor = defaultdict(list), defaultdict(list), defaultdict(list)
    for case in combined_review["cases"]:
        reviewed_by_actor[case["actor_id"]].append(case)
    for fact in facts:
        facts_by_actor[fact["actor_id"]].append(fact)
    for fact in eligible:
        eligible_by_actor[fact["actor_id"]].append(fact)
    strict = profiles["scenarios"]["strict_dated"]
    loose = profiles["scenarios"]["content_supported_unknown_time_sensitivity"]
    strict_cells = Counter(c["actor_id"] for c in strict["cells"] if c["status"] == "observed")
    loose_cells = Counter(c["actor_id"] for c in loose["cells"] if c["status"] == "observed")
    rows = []
    for actor in roster:
        aid = actor["id"]
        raw, candidates, cases = facts_by_actor[aid], eligible_by_actor[aid], reviewed_by_actor[aid]
        known_exact = [f for f in candidates if f.get("event_date")]
        known_interval = [f for f in candidates if not f.get("event_date") and f.get("event_date_from") and f.get("event_date_to")]
        rows.append({
            "actor_id": aid, "name": actor["name"],
            "all_extracted_facts": len(raw),
            "eligible_candidates": len(candidates),
            "candidate_literal_groups": len({f["literal_material_group_id"] for f in candidates}),
            "candidate_source_documents": len({f["document_id"] for f in candidates}),
            "candidate_exact_event_dates_before_full_semantic_validation": len(known_exact),
            "candidate_interval_event_dates": len(known_interval),
            "candidate_unknown_event_time": len(candidates) - len(known_exact) - len(known_interval),
            "candidate_source_date_present": sum(bool(f.get("source_date")) for f in candidates),
            "native_reviewed_facts": sum(f["id"] in annotated_ids for f in raw),
            "native_reviewed_eligible_candidates": sum(f["id"] in annotated_ids for f in candidates),
            "eligible_candidates_not_semantically_reviewed": sum(f["id"] not in annotated_ids for f in candidates),
            "native_reviewed_cases": len(cases),
            "reviewed_cases_with_provisional_ratings": sum(bool(c["ratings"]) for c in cases),
            "provisional_case_property_values": sum(len(c["ratings"]) for c in cases),
            "strict_actor_property_values": strict_cells[aid],
            "unknown_time_sensitivity_actor_property_values": loose_cells[aid],
        })
    case_rating_by_date = Counter()
    case_rating_by_verdict = Counter()
    for case in combined_review["cases"]:
        case_rating_by_date[case["event_date"]["status"]] += len(case["ratings"])
        case_rating_by_verdict[case["verdict"]] += len(case["ratings"])
    candidate_dated = sum(bool(f.get("event_date") or f.get("event_date_from")) for f in eligible)
    selected_cases = sum(c.get("sample_kind") != "excluded_control" for c in combined_review["cases"])
    strict_observed = {(c["actor_id"], c["scale_id"]) for c in strict["cells"] if c["status"] == "observed"}
    loose_observed = {(c["actor_id"], c["scale_id"]) for c in loose["cells"] if c["status"] == "observed"}
    ranked = sorted(rows, key=lambda row: -row["eligible_candidates"])
    result = {
        "version": "rwanda-scoring-bottleneck-forensic-audit-v1",
        "method": "read_only_counts_of_frozen_derived_facts_and_validated_review",
        "counts": {
            "all_extracted_facts": len(facts),
            "eligible_candidates_after_general_checks": len(eligible),
            "eligible_candidates_within_study_window_gate": len(study_candidates),
            "candidate_difference_from_corpus_vs_study_end_date": len(eligible) - len(study_candidates),
            "candidate_literal_groups": len({f["literal_material_group_id"] for f in eligible}),
            "candidate_source_documents": len({f["document_id"] for f in eligible}),
            "candidate_actors": len({f["actor_id"] for f in eligible}),
            "candidate_known_event_date_or_interval": candidate_dated,
            "candidate_unknown_event_time": len(eligible) - candidate_dated,
            "native_reviewed_facts": len(annotated_ids),
            "native_reviewed_eligible_candidates": sum(f["id"] in annotated_ids for f in eligible),
            "eligible_candidates_not_semantically_reviewed": sum(f["id"] not in annotated_ids for f in eligible),
            "native_reviewed_cases": len(combined_review["cases"]),
            "native_reviewed_selected_cases": selected_cases,
            "reviewed_cases_with_provisional_ratings": sum(bool(c["ratings"]) for c in combined_review["cases"]),
            "provisional_case_property_values": sum(len(c["ratings"]) for c in combined_review["cases"]),
            "reviewed_episode_groups": profiles["counts"]["reviewed_episode_groups"],
            "strict_numeric_case_values": strict["counts"]["numeric_case_ratings_before_episode_deduplication"],
            "strict_episode_property_values": strict["counts"]["episode_property_values_after_deduplication"],
            "strict_actor_property_values": strict["counts"]["observed_cells"],
            "unknown_time_sensitivity_numeric_case_values": loose["counts"]["numeric_case_ratings_before_episode_deduplication"],
            "unknown_time_sensitivity_episode_property_values": loose["counts"]["episode_property_values_after_deduplication"],
            "unknown_time_sensitivity_actor_property_values": loose["counts"]["observed_cells"],
            "all_roster_actors": len(roster), "approved_properties": len(profiles["scales"]),
        },
        "candidate_review_coverage": {
            "facts_fraction": sum(f["id"] in annotated_ids for f in eligible) / len(eligible) if eligible else None,
            "is_random_representative_sample": False,
            "corpus_semantic_accuracy_estimated": False,
        },
        "evidence_imbalance": {
            "top_two_actor_ids": [row["actor_id"] for row in ranked[:2]],
            "top_two_candidate_facts": sum(row["eligible_candidates"] for row in ranked[:2]),
            "top_two_fraction_of_candidates": sum(row["eligible_candidates"] for row in ranked[:2]) / len(eligible) if eligible else None,
            "actors_without_candidate_facts": sum(row["eligible_candidates"] == 0 for row in rows),
            "actors_with_one_to_nine_candidate_facts": sum(0 < row["eligible_candidates"] < 10 for row in rows),
        },
        "sensitivity_membership": {
            "strict_cells_not_observed_in_unknown_time_sensitivity": [
                {"actor_id": actor_id, "scale_id": scale_id} for actor_id, scale_id in sorted(strict_observed - loose_observed)],
            "unknown_time_sensitivity_only_observed_cell_count": len(loose_observed - strict_observed),
            "loose_cells_held_for_unresolved_event_independence": loose["counts"]["cells_held_for_event_independence"],
            "explanation_ru": "Сценарий с неизвестным временем не является надмножеством строгих ячеек: добавление разных значений без подтверждённого разделения событий удерживает некоторые оценки вместо механического усреднения.",
        },
        "windows": {"corpus_extraction": {"from": config["date_from"], "to": config["date_to"]},
                    "profile_research": profiles["approved_window"]},
        "all_fact_kind_counts": _counter(f.get("kind") for f in facts),
        "eligible_candidate_attribution_basis_counts": _counter(f.get("attribution_basis") for f in eligible),
        "eligible_candidate_temporal_status_counts": _counter(f.get("temporal_status") for f in eligible),
        "provisional_rating_counts_by_reviewed_date_status": dict(sorted(case_rating_by_date.items())),
        "provisional_rating_counts_by_reviewed_verdict": dict(sorted(case_rating_by_verdict.items())),
        "actors": rows,
        "interpretation_ru": [
            "Число 11285 обозначает извлечённые фрагменты, а 39 — уникальные ячейки человек × признак после содержательного оценивания небольшой выборки и агрегации. Это разные единицы счёта.",
            "Главное узкое место — отсутствует содержательное оценивание подавляющего большинства пригодных кандидатов. Общие проверки происхождения и риска не являются оценкой всех 38 свойств.",
            "Текущий обновляющий скрипт принимает validated review, а не полный facts_private.json. Поэтому оставшиеся пригодные факты не могли повлиять на оценки автоматически.",
            "Строгий профиль дополнительно требует подтверждённого времени эпизода. Дата публикации не равна дате события; отсутствие точного дня можно разрешать обоснованным интервалом, если источник подтверждает включение в период.",
            "Сценарий с неизвестными датами показывает потери от временной неопределённости, но не подтверждает включение события в исследуемый период.",
            "Дубли одного события и несколько свидетельств одного свойства у одного человека сворачиваются в одну итоговую ячейку. Поэтому использовать весь массив не означает получить столько же итоговых чисел.",
            "Неизвестное свойство не становится нулём. При точечной новостной выборке и 38 узких свойствах отсутствие публикаций о теме ограничивает сопоставимость даже после массового оценивания.",
        ],
        "next_steps_no_api_ru": [
            "Сгруппировать все 7021 кандидата по утверждённым участникам, буквальным повторам, сюжетам и ролям; подсказки поиска использовать только для отбора опор, не для выставления баллов.",
            "Оценивать здесь досье участников по всем 38 рубрикам, начиная с редких участников и недопокрытых свойств. Сохранять ссылку на конкретные опоры, дату или интервал, роль, основания балла и неизвестного.",
            "Расширять контекст локально для неясного говорящего и времени. Прямо проверенные интервалы включать в основную матрицу, не подставляя дату публикации автоматически.",
            "После оценивания полного доступного покрытия повторить неизменённые критерии допуска. Если плотной матрицы нет, отдельно согласовать другой измерительный дизайн или дополнительные источники; не гарантировать факторное решение заранее.",
        ],
        "api_calls": 0, "new_api_cost_usd": 0,
        "ratings_generated": False, "private_source_ids_published": False,
        "raw_quotes_published": False, "source_urls_published": False,
        "limitations": [
            "Eligibility is a candidate gate, not a semantic truth certificate.",
            "Literal grouping is not validated episode deduplication.",
            "Review coverage counts are descriptive; selected cases were purposive, not a probability sample.",
            "Unreviewed candidate event dates may still be semantically wrong even when lexical date checks passed.",
        ],
    }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    paths = {
        "facts": root / ".corpus_runtime/corrected_evidence_2026_10_18/facts_private.json",
        "annotations": root / ".corpus_runtime/corrected_evidence_2026_10_18/annotations_private.json",
        "combined_review": root / "reports/rwanda_chat_combined_review_2026_10_18.json",
        "profiles": root / "reports/rwanda_source_reanalysis_2026_10_18.json",
        "study": root / "studies/rwanda-politicians-2021-2026/study.json",
    }
    data = {key: read(path) for key, path in paths.items()}
    result = audit(data["facts"], data["annotations"], data["combined_review"], data["profiles"],
                   data["study"]["objects"], {"date_from": "2021-10-15", "date_to": "2026-10-17",
                                                    "aspect": "public_positions"})
    result["provenance"] = {key + "_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()
                            for key, path in paths.items()}
    output = args.output or root / "reports/rwanda_scoring_bottleneck_2026_10_18.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["counts"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
