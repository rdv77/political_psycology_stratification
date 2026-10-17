"""Synthetic planned payloads and empirical calibration; no API or files."""
import copy
import json
import unittest
from unittest.mock import patch

from src.corpus_pipeline.budget_forecast import _Counter, forecast
from src.corpus_pipeline.models import make_task


class Bytes:
    name = "test-byte-encoding"
    def encode_ordinary(self, value):
        return list(value.encode("utf-8"))
    def encode(self, value, **kwargs):
        return self.encode_ordinary(value)


class CorpusBudgetForecastTests(unittest.TestCase):
    def fixture(self):
        encoding = Bytes()
        config = {"scope": "Synthetic", "aspect": "public_positions", "objects": [{"id": "a", "name": "Alpha North", "aliases": []}],
                  "languages": ["en"], "date_from": "2021-01-01", "date_to": "2026-12-31", "country_terms": [],
                  "routes": {"relevance": "gpt-4.1-mini", "extract": "gpt-5.1", "describe": "gpt-5.1"},
                  "text_selection": {"mode": "mentions", "max_tokens": 1500, "compact_roster": True}}
        counter = _Counter(encoding)
        jobs, facts = [], []
        for i in range(6):
            text = "Alpha North proposes a specific reform."
            doc = {"id": "doc" + str(i), "text": text, "source_type": "article"}
            for stage in ("relevance", "extract"):
                context = {"config": config, "document": doc}
                if stage == "extract":
                    context["relevance"] = {"status": "relevant", "actor_ids": ["a"], "language": "en", "reason": "PRIVATE_PILOT_REASON"}
                    context["relevance_job_id"] = "PRIVATE_JOB"
                payload = make_task(stage, context, model=config["routes"][stage])
                jobs.append({"id": stage + str(i), "stage": stage, "status": "done", "active_recipe": True,
                    "context": copy.deepcopy(context), "payload": payload, "input_tokens": sum(counter.parts(payload).values()) + 37,
                    "output_tokens": (10 + i * 10) if stage == "relevance" else (100 + i * 100),
                    "result": {"status": "relevant" if i < 3 else "irrelevant"}})
            facts.append({"id": "f" + str(i), "actor_id": "a", "kind": "public_statement", "attribution_basis": "direct_statement",
                          "extraction_job_id": "extract" + str(i), "quote": text, "claim": "Proposal", "event_date": None})
        for index, group in enumerate((facts[:3], facts[3:])):
            context = {"config": config, "facts": group}
            payload = make_task("describe", context, model="gpt-5.1")
            jobs.append({"id": "describe" + str(index), "stage": "describe", "status": "done", "active_recipe": True,
                         "context": context, "payload": payload, "input_tokens": sum(counter.parts(payload).values()) + 37,
                         "output_tokens": 100 + index * 100})
        pilot = {"corpus": {"config": config}, "jobs": jobs, "facts": facts}
        body = "Alpha North proposes something."
        plan = {"document": {"id": "PRIVATE_NEW_DOC", "excerpts": [{"start": 0, "end": len(body), "text": body}],
                            "source_type": "article", "text_selection": {"matched_actor_ids": ["PRIVATE_DIAGNOSTIC"]}},
                "actor_ids": ["a"], "source_type": "article", "token_budget": 1500,
                "metadata_private": {"author": "PRIVATE_AUTHOR", "text": "PRIVATE_RAW"}}
        return {"base1500": [plan, copy.deepcopy(plan)]}, pilot, config, encoding

    def test_exact_known_parts_plus_same_model_residual_and_explicit_pass_scenarios(self):
        args = self.fixture()
        before = copy.deepcopy(args[:3])
        result = forecast(*args)
        stage = result["variants"]["base1500"]["known_inputs"]["relevance"]
        self.assertEqual(stage["calibration_residual"]["mean"], 37)
        self.assertEqual(stage["projected_api_input_tokens"]["mean"], stage["known_tokenized_sum"] + 2 * 37)
        scenarios = result["variants"]["base1500"]["pass_scenarios"]
        self.assertEqual([s["pass_fraction"] for s in scenarios], [.5, .5, 1])
        self.assertEqual(scenarios[-1]["stages"]["extract"]["expected_requests"], 2)
        self.assertEqual(args[:3], before)

    def test_output_mean_median_p90_are_measured_proxies_and_descriptor_packing_explicit(self):
        result = forecast(*self.fixture())
        output = result["variants"]["base1500"]["output_proxy_tokens_if_all_pass"]
        self.assertEqual(output["relevance"], {"mean": 70, "median": 70, "p90": 110})
        self.assertEqual(output["extract"], {"mean": 700, "median": 700, "p90": 1100})
        full = result["variants"]["base1500"]["pass_scenarios"][-1]
        self.assertEqual(full["describe"]["expected_eligible_facts"], 2)
        self.assertEqual(full["describe"]["assumed_groups_of_60"], 1)
        self.assertEqual(result["describe_calibration"]["pilot_calls"], 2)
        self.assertIn("not a p90 total", result["methodology"]["p90_warning"])

    def test_sparse_kind_and_long_context_use_disclosed_fallback_and_progress(self):
        plans, pilot, config, encoding = self.fixture()
        item = copy.deepcopy(plans["base1500"][0])
        text = "Alpha North " + "x" * 2000
        item["document"]["excerpts"] = [{"start": 0, "end": len(text), "text": text}]
        item["source_type"] = "post"
        events = []
        result = forecast({"adaptive6000": [item]}, pilot, config, encoding, progress=events.append)
        self.assertEqual(result["variants"]["adaptive6000"]["output_pool_fallback_counts"]["extract"], {"stage": 1})
        self.assertIn("planned_body_above_selected_pilot_max_extract", {f["code"] for f in result["flags"]})
        self.assertEqual(events, [{"variant": "adaptive6000", "processed": 1, "planned_documents": 1}])

    def test_source_context_is_extract_only_and_inputs_unchanged(self):
        plans, pilot, config, encoding = self.fixture()
        source_context = {"source_type": "comment", "source_platform": "social", "author_is_verified_actor": True}
        for item in plans["base1500"]:
            item["source_context"] = dict(source_context)
        before = copy.deepcopy((plans, pilot, config))
        captured = []
        def capture(stage, context, **kwargs):
            captured.append((stage, copy.deepcopy(context)))
            return make_task(stage, context, **kwargs)
        with patch("src.corpus_pipeline.budget_forecast.make_task", side_effect=capture):
            forecast(plans, pilot, config, encoding)
        for stage, context in captured:
            if stage == "extract":
                self.assertEqual({key: context["document"][key] for key in source_context}, source_context)
            elif stage == "relevance":
                self.assertEqual(context["document"]["source_type"], "article")
                self.assertNotIn("source_platform", context["document"])
                self.assertNotIn("author_is_verified_actor", context["document"])
        self.assertEqual((plans, pilot, config), before)

    def test_missing_model_or_descriptor_calibration_never_becomes_free(self):
        plans, pilot, config, encoding = self.fixture()
        pilot["jobs"] = [j for j in pilot["jobs"] if j["stage"] != "describe"]
        config["routes"]["extract"] = "gpt-4.1-mini"  # Explicitly priced, but not calibrated for this stage.
        result = forecast(plans, pilot, config, encoding)
        full = result["variants"]["base1500"]["pass_scenarios"][-1]
        self.assertIsNone(full["stages"]["extract"]["cost_usd"]["mean"])
        self.assertIsNone(full["describe"]["cost_usd"]["mean"])
        self.assertIsNone(full["with_describe_cost_usd"]["mean"])

    def test_no_api_no_private_output_and_no_automatic_negative_from_missing_name(self):
        plans, pilot, config, encoding = self.fixture()
        plans["base1500"][0]["actor_ids"] = []
        with patch("src.corpus_pipeline.models.httpx.Client", side_effect=AssertionError("API forbidden")):
            result = forecast(plans, pilot, config, encoding)
        text = json.dumps(result, ensure_ascii=False, allow_nan=False)
        self.assertNotIn("PRIVATE_", text)
        self.assertEqual(result["paid_calls"], 0)
        self.assertEqual(result["variants"]["base1500"]["known_inputs"]["relevance"]["requests"], 2)
        self.assertEqual(result["variants"]["base1500"]["pass_scenarios"][-1]["stages"]["extract"]["expected_requests"], 2)
        self.assertIn("no_local_actor_for_hypothetical_extract", {f["code"] for f in result["flags"]})


if __name__ == "__main__":
    unittest.main()
