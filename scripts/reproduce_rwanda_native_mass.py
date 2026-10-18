"""Reproduce public profile aggregation and conditional FA without raw corpus/API."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.semantic_service import analysis  # initialize scientific runtime consistently
from src.corpus_pipeline.native_mass import read_json, source_intensity_bounds
from src.corpus_pipeline.source_rating_update import build_profiles
from src.corpus_pipeline.native_factor_sensitivity import run_sensitivity


def run():
    dictionary = read_json(ROOT / "research/rwanda_politicians_2021_2026/source_scales_v2/approved_scales.json")
    study = read_json(ROOT / "studies/rwanda-politicians-2021-2026/study.json")
    bundles = [read_json(ROOT / "reports" / name) for name in (
        "rwanda_chat_combined_review_2026_10_18.json", "rwanda_native_probe_2026_10_18.json",
        "rwanda_native_mass_review_bundle_2026_10_18.json")]
    result = build_profiles(study["objects"], dictionary, bundles)
    expected = read_json(ROOT / "reports/rwanda_native_mass_profiles_2026_10_18.json")
    for key, scenario in result["scenarios"].items():
        if scenario["cells"] != expected["scenarios"][key]["cells"]:
            raise AssertionError(f"Public profile reproduction differs: {key}")
    bounds = source_intensity_bounds(result)
    if bounds != read_json(ROOT / "reports/rwanda_native_mass_bounds_2026_10_18.json"):
        raise AssertionError("Public interval profiles differ")
    factor = run_sensitivity(bounds["content_supported_unknown_time_sensitivity"]["cells"])
    old = read_json(ROOT / "reports/rwanda_native_mass_factor_sensitivity_2026_10_18.json")
    if factor["panel"] != old["panel"]:
        raise AssertionError("Availability panel differs")
    for a, b in zip(factor["scenarios"]["midpoint_scenario"], old["scenarios"]["midpoint_scenario"]):
        if a["status"] != b["status"] or abs(a["variance_total"] - b["variance_total"]) > 1e-6:
            raise AssertionError("Conditional factor comparison differs")
    print(json.dumps({"public_profiles_reproduced": True, "all_42_by_38_cells_checked": True,
                      "conditional_factor_comparison_reproduced": True, "new_api_cost_usd": 0}))


if __name__ == "__main__":
    run()
