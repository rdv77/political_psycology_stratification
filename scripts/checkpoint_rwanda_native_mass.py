"""Compute conservative profiles at actual complete native reading boundaries."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.corpus_pipeline.native_mass import audit_journal, read_json, review_bundle, checkpoint_changes, source_intensity_bounds, bound_changes, stopping_decision
from src.corpus_pipeline.source_rating_update import build_profiles, attach_factor_comparison


def run():
    directory = ROOT / ".corpus_runtime/native_mass_2026_10_18"
    summary = audit_journal(ROOT)
    dossiers = read_json(ROOT / ".corpus_runtime/evidence_dossiers_2026_10_18/dossiers_private.json")
    dossiers["queue_rows"] = read_json(directory / "queue_private.json")["rows"]
    dictionary = read_json(ROOT / "research/rwanda_politicians_2021_2026/source_scales_v2/approved_scales.json")
    study = read_json(ROOT / "studies/rwanda-politicians-2021-2026/study.json")
    historical = read_json(ROOT / "studies/rwanda-politicians-2021-2026/output/factor_preview.json")
    timing = read_json(directory / "timing_overrides_private.json") if (directory / "timing_overrides_private.json").exists() else []
    numbers = {int(r["case_id"][1:]) for r in summary["cases"]}
    # Preserve previous full-source adjudications. Re-reading their extracted
    # passages is logged but must not duplicate or replace the fuller evidence.
    baseline = [read_json(ROOT / "reports/rwanda_chat_combined_review_2026_10_18.json"),
                read_json(ROOT / "reports/rwanda_native_probe_2026_10_18.json")]
    prior_fact_ids = set()
    for folder in ("semantic_review_2026_10_17", "semantic_extension_2026_10_18", "bottleneck_probe_2026_10_18"):
        for path in (ROOT / ".corpus_runtime" / folder / "cases").glob("*.json"):
            prior_fact_ids.update(f["id"] for f in read_json(path)["member_facts"])
    prior_passage_cases = {r["case_id"] for r in dossiers["queue_rows"] if r["fact_id"] in prior_fact_ids}
    contiguous = 0
    while contiguous + 1 in numbers:
        contiguous += 1
    boundaries = ([751] if contiguous >= 751 else []) + list(range(1251, contiguous + 1, 500))
    exhausted = contiguous == len(dossiers["queue_rows"])
    if exhausted and contiguous not in boundaries:
        boundaries.append(contiguous)
    checkpoints, previous, latest, previous_bounds = [], None, None, None
    for boundary in boundaries:
        bundle = review_bundle(summary, dossiers, dictionary, last_case=boundary, timing_overrides=timing)
        bundle["cases"] = [c for c in bundle["cases"] if c["case_id"] not in prior_passage_cases]
        profiles = build_profiles(study["objects"], dictionary, baseline + [bundle])
        attach_factor_comparison(profiles, study["objects"], dictionary, historical)
        bounds = source_intensity_bounds(profiles)
        block_size = boundary - (checkpoints[-1]["through_case"] if checkpoints else 0)
        row = {"through_case": boundary, "reviewed_cases_in_block": block_size,
               "profiles_aggregated": True, "scenarios": {}}
        for key, scenario in profiles["scenarios"].items():
            row["scenarios"][key] = {"counts": scenario["counts"],
                                     "factor_admission": scenario["factor_admission"],
                                     "changes": checkpoint_changes(previous["scenarios"][key]["cells"], scenario["cells"]) if previous else None}
            row["scenarios"][key]["informative_cells_including_bounds"] = bounds[key]["informative_cells"]
            row["scenarios"][key]["bound_changes"] = bound_changes(previous_bounds[key]["cells"], bounds[key]["cells"]) if previous_bounds else None
        checkpoints.append(row)
        previous, latest = profiles, profiles
        previous_bounds = bounds
        if exhausted and boundary == contiguous:
            (ROOT / "reports/rwanda_native_mass_review_bundle_2026_10_18.json").write_text(
                json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    processed_ids = {r["case_id"] for r in summary["cases"]}
    rare_actor_ids = {a["actor_id"] for a in dossiers["private_actor_dossiers"]
                      if len(a["eligible_fact_ids"]) <= 10}
    rare_queue_complete = all(r["case_id"] in processed_ids for r in dossiers["queue_rows"]
                              if dossiers["private_fact_index"][r["fact_id"]]["fact"]["actor_id"] in rare_actor_ids)
    for row in checkpoints:
        row["substantive_changes"] = sum(
            (s["changes"]["substantive_changes"] + s["bound_changes"]["substantive_cells"])
            for s in row["scenarios"].values() if s["changes"] is not None)
        # The first 751 cases establish a baseline, never a stable 500-case block.
        row["stopping_criterion_met"] = stopping_decision(
            checkpoints[:checkpoints.index(row) + 1], rare_actor_queue_exhausted=rare_queue_complete)
    stopped = bool(checkpoints and checkpoints[-1]["stopping_criterion_met"])
    result = {"version": "native-mass-conservative-checkpoints-v1", "checkpoints": checkpoints,
              "native_journal_content_sha256": summary["native_journal_content_sha256"],
              "reviewed_contiguous_fragments": contiguous,
              "processed_contiguous_fragments": contiguous,
              "new_api_cost_usd": 0, "all_42_cards_retained": True,
              "prior_full_source_adjudications_preserved": True,
              "passage_cases_not_double_counted_against_prior_fact_ids": len(prior_passage_cases & {r["case_id"] for r in summary["cases"]}),
              "early_stopping_demonstrated": stopped,
              "all_eligible_fragments_processed": exhausted,
              "completion_reason": "eligible_queue_exhausted" if exhausted else "stability_rule" if stopped else "reading_in_progress",
              "rare_actor_queue_exhausted": rare_queue_complete,
              "stopping_rule": "Two consecutive complete 500-case blocks; no new/lost point values, no >=0.5 point changes, no >=0.5 endpoint changes in either time scenario; all <=10-fragment actor queues exhausted.",
              "limitations_ru": ["Применена прежняя консервативная агрегация; непроверенная независимость разных эпизодов удерживает численное значение.",
                                 "Новые даты и интервалы требуют отдельной смысловой проверки, дата публикации не заменяет дату события."]}
    for name, value in [("rwanda_native_mass_checkpoints_2026_10_18.json", result),
                        ("rwanda_native_mass_profiles_2026_10_18.json", latest),
                        ("rwanda_native_mass_bounds_2026_10_18.json", previous_bounds)]:
        if value is not None:
            (ROOT / "reports" / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"reviewed": contiguous, "checkpoints": [{"through": c["through_case"], "scenarios": {k: v["counts"]["observed_cells"] for k, v in c["scenarios"].items()}} for c in checkpoints]}, ensure_ascii=False))


if __name__ == "__main__":
    run()
