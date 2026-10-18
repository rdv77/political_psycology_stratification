"""Publish the progress of source reading, without claiming complete scoring."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.corpus_pipeline.native_mass import audit_journal

if __name__ == "__main__":
    result = audit_journal(ROOT)
    target = ROOT / "reports/rwanda_native_mass_2026_10_18.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in {"cases", "limitations_ru"}}, ensure_ascii=False))
