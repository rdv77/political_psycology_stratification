"""Compute conservative profiles at actual complete native reading boundaries."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.corpus_pipeline.native_mass import audit_journal, read_json, review_bundle, checkpoint_changes
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
    checkpoints, previous, latest = [], None, None
    for boundary in boundaries:
        bundle = review_bundle(summary, dossiers, dictionary, last_case=boundary, timing_overrides=timing)
        bundle["cases"] = [c for c in bundle["cases"] if c["case_id"] not in prior_passage_cases]
        profiles = build_profiles(study["objects"], dictionary, baseline + [bundle])
        attach_factor_comparison(profiles, study["objects"], dictionary, historical)
        row = {"through_case": boundary, "reviewed_cases_in_block": 751 if boundary == 751 else 500,
               "profiles_aggregated": True, "scenarios": {}}
        for key, scenario in profiles["scenarios"].items():
            row["scenarios"][key] = {"counts": scenario["counts"],
                                     "factor_admission": scenario["factor_admission"],
                                     "changes": checkpoint_changes(previous["scenarios"][key]["cells"], scenario["cells"]) if previous else None}
        checkpoints.append(row)
        previous, latest = profiles, profiles
    result = {"version": "native-mass-conservative-checkpoints-v1", "checkpoints": checkpoints,
              "reviewed_contiguous_fragments": contiguous,
              "new_api_cost_usd": 0, "all_42_cards_retained": True,
              "prior_full_source_adjudications_preserved": True,
              "passage_cases_not_double_counted_against_prior_fact_ids": len(prior_passage_cases & {r["case_id"] for r in summary["cases"]}),
              "early_stopping_demonstrated": False,
              "limitations_ru": ["Применена прежняя консервативная агрегация; непроверенная независимость разных эпизодов удерживает численное значение.",
                                 "Новые даты и интервалы требуют отдельной смысловой проверки, дата публикации не заменяет дату события."]}
    for name, value in [("rwanda_native_mass_checkpoints_2026_10_18.json", result),
                        ("rwanda_native_mass_profiles_2026_10_18.json", latest)]:
        if value is not None:
            (ROOT / "reports" / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"reviewed": contiguous, "checkpoints": [{"through": c["through_case"], "scenarios": {k: v["counts"]["observed_cells"] for k, v in c["scenarios"].items()}} for c in checkpoints]}, ensure_ascii=False))


if __name__ == "__main__":
    run()
