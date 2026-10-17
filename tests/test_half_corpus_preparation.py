"""Integration coverage for private sampling exports and the selected cost plan."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts.prepare_half_corpus import _matched_baseline, prepare
from src.corpus_pipeline.models import DEFAULT_PRICING, estimate_cost


class ByteEncoding:
    name = "test-byte-encoding"

    def encode(self, text, **kwargs):
        return list(text.encode("utf-8"))


def variant(multiplier=1):
    return {"pass_scenarios": [
        {"name": name, "pass_fraction": fraction,
         "with_describe_cost_usd": {"mean": 10 * multiplier, "p90": 15 * multiplier}}
        for name, fraction in (("assumed_lower_50pct", .5),
                               ("transferred_pilot_observed_positive", .9), ("assumed_all_pass", 1.))]}


class HalfCorpusPreparationTests(unittest.TestCase):
    def test_export_preserves_full_frame_and_forecasts_only_selected_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / ".corpus_runtime"
            cleaning = runtime / "cleaning" / "synthetic"
            cleaning.mkdir(parents=True)
            (cleaning / "cleaning_checkpoint.json").write_text("checkpoint", encoding="utf-8")
            original = cleaning / "preparation_summary.json"
            original.write_text("original baseline", encoding="utf-8")
            pilot = root / "pilot.json"
            pilot.write_text(json.dumps({"corpus": {"config": {}}}), encoding="utf-8")
            config = {"objects": [{"id": "paul-kagame", "name": "Paul Kagame"},
                                   {"id": "diane-rwigara", "name": "Diane Rwigara"}],
                      "text_selection": {"mode": "mentions", "max_tokens": 1500,
                                         "compact_roster": True, "adaptive": True, "expanded_max_tokens": 6000}}
            corpus = {"id": "synthetic", "source_sha256": "a" * 64, "config": config}
            pipeline = SimpleNamespace(runtime=runtime, get_corpus=lambda identifier: corpus)
            frames = []
            for i in range(8):
                ids = ["paul-kagame", "diane-rwigara"] if i == 0 else ["paul-kagame"] if i > 1 else []
                document = {"id": f"PRIVATE-ID-{i}", "source_row": i + 2, "title": "PRIVATE_TITLE",
                            "text": "PRIVATE_BODY " * (200 if i == 0 else 2), "url": "https://private.invalid",
                            "published_at": "2026-10-17T12:30:00", "source_name": "PRIVATE_PUBLISHER",
                            "author": "PRIVATE_AUTHOR", "language_hint": "en",
                            "_source_context": {"source_type": "article", "source_platform": "media",
                                                "author_is_verified_actor": False}}
                annotation = {"matched_actor_ids": ids, "status": "needs_semantic_review" if not ids else "ready_for_semantic_filter",
                              "reason": "test", "flags": []}
                frames.append((document, annotation))
            baseline = {"kept_for_semantic_screening": 8,
                        "budget_forecast": {"variants": {"adaptive6000": variant(2)}}}
            expanded = Mock(return_value={"spans": [{"start": 0, "end": len(frames[0][0]["text"]),
                                                     "text": frames[0][0]["text"], "tokens": len(frames[0][0]["text"])}],
                                          "metadata": {"expanded": True, "max_tokens": 6000, "reason": ["rare_actor"]}})
            estimate = Mock(return_value={"variants": {"half_adaptive6000": variant()}})
            with patch("scripts.prepare_half_corpus.CorpusPipeline", return_value=pipeline), \
                 patch("scripts.prepare_half_corpus.load_checkpoint", return_value=(frames, [], {})), \
                 patch("scripts.prepare_half_corpus._matched_baseline", return_value=baseline), \
                 patch("scripts.prepare_half_corpus.offline_encoding", return_value=ByteEncoding()), \
                 patch("scripts.prepare_half_corpus.select_document_context", expanded), \
                 patch("scripts.prepare_half_corpus.forecast", estimate), \
                 patch("src.corpus_pipeline.models.OpenAIResponsesHTTP.call", side_effect=AssertionError("No API")):
                result = prepare(root, "synthetic", pilot)
            self.assertEqual(result["selected_unique"], 4)
            self.assertEqual(result["api_calls"], 0)
            self.assertFalse(result["queue_modified"])
            self.assertEqual(original.read_text(encoding="utf-8"), "original baseline")
            self.assertEqual(result["context"]["rarity_reference"], "complete_cleaned_corpus")
            self.assertEqual(expanded.call_args.args[3], 8)
            self.assertEqual(expanded.call_args.args[2]["paul-kagame"], 7)
            self.assertEqual(expanded.call_args.args[2]["diane-rwigara"], 1)
            selected_plans = estimate.call_args.args[0]["half_adaptive6000"]
            self.assertEqual(len(selected_plans), 4)
            self.assertIn("PRIVATE-ID-0", {plan["document"]["id"] for plan in selected_plans})
            directory = runtime / "sampling" / "synthetic" / "half-v1"
            ids = json.loads((directory / "selected_ids.json").read_text(encoding="utf-8"))
            exported = [json.loads(line) for line in (directory / "selected_unique.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(set(ids), {doc["id"] for doc in exported})
            self.assertEqual(set(ids), {plan["document"]["id"] for plan in selected_plans})
            self.assertTrue(all("_source_context" not in doc and "source_context" in doc for doc in exported))
            self.assertNotIn("PRIVATE", json.dumps(result))
            self.assertEqual(result["full_comparison"][1]["saving_mean_fraction"], .5)

    def test_baseline_from_other_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "preparation_summary.json").write_text(json.dumps({"corpus_id": "other", "source_sha256": "different"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "another source"):
                _matched_baseline(directory, {"id": "current", "source_sha256": "expected"}, b"pilot", directory)

    def test_baseline_cannot_mix_changed_routes_or_prices_into_sampling_savings(self):
        routes = {"relevance": "gpt-4.1-mini", "extract": "gpt-5.1", "describe": "gpt-5.1"}
        scenario = {"stages": {}}
        for stage, model in routes.items():
            row = {"model": model, "estimated_input_tokens": 1_000_000,
                   "estimated_output_tokens": {"mean": 10},
                   "cost_usd": {"mean": estimate_cost(model, 1_000_000, 10)}}
            if stage == "describe":
                scenario["describe"] = row
            else:
                scenario["stages"][stage] = row
        baseline = {"corpus_id": "current", "source_sha256": "expected",
                    "provenance": {"pilot_snapshot_sha256": hashlib.sha256(b"pilot").hexdigest(), "sources": {}},
                    "budget_forecast": {"describe_calibration": {"model": "gpt-5.1"},
                                        "variants": {"adaptive6000": {"pass_scenarios": [scenario]}}}}
        corpus = {"id": "current", "source_sha256": "expected",
                  "config": {"routes": routes, "pricing": copy.deepcopy(DEFAULT_PRICING)}}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "preparation_summary.json").write_text(json.dumps(baseline), encoding="utf-8")
            self.assertEqual(_matched_baseline(directory, corpus, b"pilot", directory), baseline)
            changed_routes = copy.deepcopy(corpus)
            changed_routes["config"]["routes"]["relevance"] = "gpt-5.1"
            with self.assertRaisesRegex(ValueError, "changing model routes"):
                _matched_baseline(directory, changed_routes, b"pilot", directory)
            changed_prices = copy.deepcopy(corpus)
            changed_prices["config"]["pricing"]["gpt-4.1-mini"]["input"] = 99
            with self.assertRaisesRegex(ValueError, "changing prices"):
                _matched_baseline(directory, changed_prices, b"pilot", directory)


if __name__ == "__main__":
    unittest.main()
