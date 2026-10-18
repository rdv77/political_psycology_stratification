"""Publish the progress of source reading, without claiming complete scoring."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.corpus_pipeline.native_mass import audit_journal

if __name__ == "__main__":
    result = audit_journal(ROOT)
    checkpoint_file = ROOT / "reports/rwanda_native_mass_checkpoints_2026_10_18.json"
    if checkpoint_file.exists():
        checkpoint = json.loads(checkpoint_file.read_text(encoding="utf-8"))
        if checkpoint.get("native_journal_content_sha256") == result["native_journal_content_sha256"]:
            result["profile_aggregation_completed"] = True
            result["mass_scoring_completed"] = checkpoint["all_eligible_fragments_processed"]
            result["early_stopping_demonstrated"] = checkpoint["early_stopping_demonstrated"]
            result["completion_reason"] = checkpoint["completion_reason"]
            result["all_1596_cells_have_numeric_scores"] = False
            result["canonical_42_actor_factor_analysis_performed"] = False
    target = ROOT / "reports/rwanda_native_mass_2026_10_18.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in {"cases", "limitations_ru"}}, ensure_ascii=False))
