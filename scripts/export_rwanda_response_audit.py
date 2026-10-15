"""Export only the public pilot's model responses actually referenced by its rows."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "studies/rwanda-politicians-2021-2026"


def main():
    rows = json.loads((DIRECTORY / "ratings.json").read_text(encoding="utf-8"))
    needed = {(row.get("model"), row.get("at")) for row in rows
              if row.get("model") != "not_scored_no_eligible_evidence"}
    destination = DIRECTORY / "evidence/model_responses"
    destination.mkdir(exist_ok=True)
    index = []
    fields = {"data", "model", "task", "at", "usage", "citations", "system_fingerprint", "cache_key"}
    for file in (ROOT / ".service_cache").glob("*.json"):
        record = json.loads(file.read_text(encoding="utf-8"))
        if (record.get("model"), record.get("at")) not in needed or not str(record.get("task", "")).startswith("ratings-"):
            continue
        public = {key: value for key, value in record.items() if key in fields}
        (destination / file.name).write_text(json.dumps(public, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        index.append({"model": record["model"], "at": record["at"], "task": record["task"],
                      "file": "evidence/model_responses/" + file.name,
                      "cache_key": record.get("cache_key")})
    (DIRECTORY / "evidence/model_response_index.json").write_text(json.dumps({
        "purpose": "Raw model proposals, including partial poles; not accepted measurements. Canonical ratings.json and source audit determine usable cells.",
        "responses": sorted(index, key=lambda row: (row["at"], row["task"]))}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"exported_responses": len(index), "needed_model_timestamps": len(needed)}))


if __name__ == "__main__":
    main()
