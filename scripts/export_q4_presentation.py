"""Export the reader-facing q4 presentation without rebuilding the full report.

Uses the saved q4/k4 model, approved dictionary, reviewed prose and labelled
charts. No rating, search or factor fitting is performed. Default: readiness;
--apply: preserve the presentation preference and replace the deck and bundle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import pandas as pd

from src.semantic_service.store import StudyStore


ROOT = Path(__file__).resolve().parents[1]
IDENTIFIER = "russian-politicians-2016-2026"
NARRATIVE = ROOT / "research/russian_politicians_2016_2026/PRESENTATION_Q4_CONTENT.json"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_configuration(study: dict, output: Path) -> dict:
    folder = output / "models/q4/selected_k4"
    model = load(folder / "metadata.json")
    narrative = load(NARRATIVE)
    if model["factor_count"] != 4 or model["cluster_count"] != 4:
        raise ValueError("The requested presentation requires the saved q4/k4 solution")
    if digest(output / "matrix.csv") != digest(folder / "matrix.csv"):
        raise ValueError("The presentation and report use different measured matrices")
    if {row["id"] for row in narrative["factors"]} != {row["id"] for row in model["factors"]}:
        raise ValueError("The narrative does not cover every factor")
    actual = {row["id"]: row for row in model["clusters"]}
    if {row["id"] for row in narrative["clusters"]} != set(actual):
        raise ValueError("The narrative does not cover every group")
    for row in narrative["clusters"]:
        if row["count"] != actual[row["id"]]["count"]:
            raise ValueError("A narrative group count differs from the computed membership")
    expected = {member["id"]: cluster["id"] for cluster in model["clusters"] for member in cluster["members"]}
    labels = pd.read_csv(folder / "cluster_labels.csv").set_index("object_id")["cluster_id"].to_dict()
    if expected != labels or len(expected) != model["n_objects"]:
        raise ValueError("The narrative source and chart labels disagree")
    manifest = load(output / "assets/presentation_q4/manifest.json")
    if manifest["factor_count"] != 4 or manifest["cluster_count"] != 4:
        raise ValueError("The chart manifest has another model")
    for name, checksum in manifest["source_sha256"].items():
        path = (output / name).resolve()
        if not path.is_relative_to(output.resolve()) or digest(path) != checksum:
            raise ValueError("The plotted source changed: " + name)
    for chart in manifest["charts"]:
        path = (output / chart["file"]).resolve()
        if not path.is_relative_to(output.resolve()) or not path.is_file():
            raise ValueError("A presentation chart is missing")
    return {"mode": "interpretive", "factor_count": 4, "cluster_count": 4,
            "model_file": "models/q4/selected_k4/metadata.json",
            "narrative_file": "presentation_content.json", "narrative_sha256": digest(NARRATIVE),
            "chart_manifest": "assets/presentation_q4/manifest.json",
            "input_receipts": {field: study["approvals"][field]["fingerprint"] for field in ("objects", "scales")},
            "input_scope": {key: study[key] for key in ("scope", "aspect", "date_from", "date_to")},
            "matrix_sha256": digest(output / "matrix.csv")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    store = StudyStore(ROOT / "studies")
    study = store.require_approved(IDENTIFIER)
    output = store.directory(IDENTIFIER) / "output"
    config = checked_configuration(study, output)
    print(json.dumps({"ready": True, "apply": args.apply, "factor_count": 4, "cluster_count": 4}), flush=True)
    if not args.apply:
        return
    preserved = {name: digest(output / name) for name in ("report.md", "report.html", "report.pdf", "report.docx")}
    preserved["ratings.json"] = digest(output.parent / "ratings.json")
    payload = load(output / "report_content.json")
    payload["study"]["presentation_config"] = config
    (output / "presentation_content.json").write_bytes(NARRATIVE.read_bytes())
    (output / "report_content.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    from src.semantic_service.server import Application
    from src.semantic_service.reports import write_report_bundle
    import os
    Application.configure_export_runtime()
    subprocess.run([os.environ["SEMANTIC_NODE"], str(ROOT / "src/semantic_service/presentation.mjs"),
                    str(output / "report_content.json"), str(output)], check=True, timeout=420)
    artifacts = study["artifacts"]
    for artifact in artifacts:
        if artifact.get("format") == "pptx":
            artifact["label"] = "Презентация: четыре фактора"
    store.update(IDENTIFIER, presentation_config=config, artifacts=artifacts)
    write_report_bundle(output)
    for name, checksum in preserved.items():
        path = output.parent / name if name == "ratings.json" else output / name
        if digest(path) != checksum:
            raise RuntimeError("An unrelated report or measured input changed: " + name)
    print(json.dumps({"complete": True, "full_report_and_ratings_preserved": True}), flush=True)


if __name__ == "__main__":
    main()
