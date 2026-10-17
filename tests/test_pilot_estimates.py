"""Document-level weighting and explicit unknowns, with no files or API calls."""
from copy import deepcopy
import json
import unittest

from src.corpus_pipeline.pilot_estimates import estimate_costs


def obs(identifier, stratum, *, relevance=1., extract=2., facts=3, **changes):
    result = {"id": identifier, "stratum": stratum, "random_sample": True,
              "relevance_complete": True, "relevant": True, "review": False,
              "relevance_cost_usd": relevance / 2, "relevance_uncached_cost_usd": relevance,
              "extraction_complete": True, "extraction_random": True,
              "extraction_cost_usd": extract / 2, "extraction_uncached_cost_usd": extract, "facts": facts}
    result.update(changes)
    return result


class PilotEstimateTests(unittest.TestCase):
    def test_population_weighting_preserves_document_cost_and_relevance_association(self):
        pop = [{"id": str(i), "stratum": "short" if i < 8 else "long"} for i in range(10)]
        rows = [obs("0", "short", relevance=1, extract=2, facts=2),
                obs("1", "short", relevance=1, relevant=False, extraction_complete=False, facts=0),
                obs("8", "long", relevance=5, extract=10, facts=10),
                obs("9", "long", relevance=5, extract=10, facts=10)]
        original = deepcopy(rows)
        result = estimate_costs(pop, rows, raw_count=20, target_count=1000, descriptor_cost_usd=4, descriptor_input_facts=8)
        primary = result["forecasts"]["scenarios"][0]
        # Eligible count500; classify=.8*1+.2*5=1.8, extract=.8*.5*2+.2*1*10=2.8.
        self.assertAlmostEqual(primary["stages"]["classification"]["baseline_uncached_usd"], 900)
        self.assertAlmostEqual(primary["stages"]["extraction"]["baseline_uncached_usd"], 1400)
        self.assertAlmostEqual(primary["stages"]["core"]["baseline_uncached_usd"], 2300)
        self.assertAlmostEqual(primary["stages"]["core"]["observed_cache_usd"], 1150)
        self.assertAlmostEqual(primary["estimated_fact_yield"]["observed_relevant"], 1400)
        self.assertAlmostEqual(primary["descriptor_forecast_usd"]["observed_relevant"], 700)
        self.assertEqual([s["design_weight"] for s in result["strata"]], [1., 4.])
        self.assertEqual(rows, original)
        json.dumps(result, allow_nan=False)

    def test_targeted_extras_and_nonrandom_extractions_are_not_baseline(self):
        pop = [{"id": str(i), "stratum": "s"} for i in range(4)]
        rows = [obs("0", "s", relevance=1, extract=2),
                obs("1", "s", relevance=999, extract=999, random_sample=False),
                obs("2", "s", relevance=1, extract=888, extraction_random=False),
                {"id": "3", "stratum": "s"}]
        result = estimate_costs(pop, rows, raw_count=4, target_count=1000)
        primary = result["forecasts"]["scenarios"][0]
        self.assertEqual(primary["stages"]["core"]["baseline_uncached_usd"], 3000)
        self.assertEqual(result["measured"]["reported_costs"]["extraction_uncached_cost_usd"]["known_sum_usd"], 1889)
        self.assertEqual(result["measured"]["extraction_complete_random_positive"], 1)

    def test_missing_extraction_remains_unknown_not_targeted_or_free(self):
        pop = [{"id": "a", "stratum": "s"}, {"id": "b", "stratum": "s"}]
        rows = [obs("a", "s", extraction_complete=False), obs("b", "s", random_sample=False)]
        result = estimate_costs(pop, rows, raw_count=2)
        primary = result["forecasts"]["scenarios"][0]
        self.assertIsNone(primary["stages"]["extraction"]["baseline_uncached_usd"])
        self.assertIsNone(primary["stages"]["core"]["baseline_uncached_usd"])
        self.assertIsNone(primary["estimated_fact_yield"]["observed_relevant"])
        self.assertIsNone(result["descriptor_forecast"]["observed_usd_per_input_fact"])
        self.assertIn("no_random_extraction", {f["code"] for f in result["flags"]})

    def test_global_conditional_fallback_is_population_positive_weighted(self):
        pop = [{"id": str(i), "stratum": "a" if i < 8 else "b" if i < 10 else "c"} for i in range(12)]
        rows = [obs("0", "a", extract=2), obs("8", "b", extract=10),
                obs("10", "c", extraction_complete=False)]
        result = estimate_costs(pop, rows, raw_count=12)
        # Global conditional=(8*2+2*10)/(8+2), not unweighted6.
        self.assertAlmostEqual(result["forecasts"]["global_extraction_conditional"]["baseline_uncached_usd"], 3.6)
        c = next(s for s in result["strata"] if s["stratum"] == "c")
        self.assertAlmostEqual(c["extraction_conditional_effective"]["baseline_uncached_usd"], 3.6)
        self.assertEqual(c["extraction_conditional_source"]["baseline_uncached_usd"], "global_random_conditional")

    def test_review_ambiguity_is_scenario_not_ci_and_eligible_fractions_are_explicit(self):
        pop = [{"id": str(i), "stratum": "s"} for i in range(4)]
        rows = [obs("0", "s", extract=4),
                obs("1", "s", relevant=False, review=True, extraction_complete=False)]
        result = estimate_costs(pop, rows, raw_count=8, target_count=1000)
        scenarios = result["forecasts"]["scenarios"]
        self.assertEqual([s["eligible_count"] for s in scenarios], [500, 750, 1000])
        self.assertEqual(scenarios[0]["stages"]["extraction"]["baseline_uncached_usd"], 1000)
        self.assertEqual(scenarios[0]["stages"]["extraction"]["review_upper_uncached_usd"], 2000)
        self.assertEqual(scenarios[1]["fraction_basis"], "assumption")
        self.assertEqual(result["methodology"]["interval_kind"], "scenario_only_not_probability_or_confidence_interval")
        self.assertFalse(result["descriptor_forecast"]["included_in_core"])
        self.assertNotIn("budget_usd", result)

    def test_uncovered_stratum_and_missing_cost_cannot_silently_be_zero(self):
        pop = [{"id": "a", "stratum": "a"}, {"id": "b", "stratum": "b"}]
        result = estimate_costs(pop, [obs("a", "a")], raw_count=2)
        self.assertIsNone(result["forecasts"]["scenarios"][0]["stages"]["core"]["baseline_uncached_usd"])
        self.assertIn("uncovered_classification_stratum", {f["code"] for f in result["flags"]})
        row = obs("b", "b")
        del row["relevance_uncached_cost_usd"]
        result = estimate_costs(pop, [obs("a", "a"), row], raw_count=2)
        self.assertIsNone(result["forecasts"]["scenarios"][0]["stages"]["classification"]["baseline_uncached_usd"])
        self.assertIsNotNone(result["forecasts"]["scenarios"][0]["stages"]["classification"]["observed_cache_usd"])

    def test_incomplete_random_rows_are_excluded_and_flagged(self):
        pop = [{"id": "a", "stratum": "s"}, {"id": "b", "stratum": "s"}]
        result = estimate_costs(pop, [obs("a", "s"), obs("b", "s", relevance_complete=False, relevance=999)], raw_count=2)
        self.assertEqual(result["strata"][0]["design_weight"], 1)
        self.assertEqual(result["strata"][0]["complete_case_expansion_weight"], 2)
        self.assertIn("incomplete_random_classification", {f["code"] for f in result["flags"]})

    def test_invalid_frame_accounting_and_nonfinite_values_are_rejected(self):
        pop = [{"id": "a", "stratum": "s"}]
        for rows, changes in (([obs("x", "s")], {}), ([obs("a", "wrong")], {}),
                              ([obs("a", "s"), obs("a", "s")], {}),
                              ([obs("a", "s", relevance=float("nan"))], {}),
                              ([obs("a", "s", random_sample=1)], {}),
                              ([obs("a", "s", facts=-1)], {}), ([], {"raw_count": 0}),
                              ([], {"descriptor_cost_usd": -1}), ([], {"target_count": True})):
            with self.subTest(rows=rows, changes=changes), self.assertRaises(ValueError):
                estimate_costs(pop, rows, **{"raw_count": 1, **changes})


if __name__ == "__main__":
    unittest.main()
