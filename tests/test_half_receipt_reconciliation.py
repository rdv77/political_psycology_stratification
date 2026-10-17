"""Frozen billing reconciliation; synthetic receipts only."""
import copy
import json
import unittest
from unittest.mock import patch

from scripts.reconcile_half_receipts import reconcile
from src.corpus_pipeline.models import DEFAULT_PRICING, estimate_cost
from src.corpus_pipeline.service import digest


class ReceiptReconciliationTests(unittest.TestCase):
    def fixture(self):
        jobs, receipts = [], []
        for index in range(3):
            payload = {"model": "gpt-5.1", "input": "PRIVATE_SOURCE" + str(index)}
            cost = estimate_cost("gpt-5.1", 1000, 100, cached_input_tokens=800)
            jobs.append({"id": "PRIVATE_JOB" + str(index), "payload": payload, "stage": "extract", "active_recipe": True,
                "status": "done", "started_at": "2026-10-17T12:00:0" + str(index), "cost_usd": cost,
                "input_tokens": 1000, "output_tokens": 100, "response_id": "PRIVATE_RESPONSE" + str(index)})
            if index:
                receipts.append({"request_sha256": digest(payload), "response_id": "PRIVATE_RESPONSE" + str(index),
                    "model": "gpt-5.1-2025-11-13", "actual_service_tier": "default",
                    "usage": {"input_tokens": 1000, "output_tokens": 100, "input_tokens_details": {"cached_tokens": 800}}})
        snapshot = {"manifest": {"pricing": copy.deepcopy(DEFAULT_PRICING)}, "jobs": jobs, "phase_trace": {"stage": "extract"}}
        return snapshot, receipts

    def test_cached_alias_cost_matches_and_expected_legacy_is_not_free(self):
        snapshot, receipts = self.fixture()
        before = copy.deepcopy((snapshot, receipts))
        with patch("src.corpus_pipeline.models.httpx.Client", side_effect=AssertionError("API forbidden")):
            result = reconcile(snapshot, receipts, pre_wrapper_calls=1)
        public = result["public"]
        self.assertEqual(public["fully_priced_matched_cost_difference_usd"], 0)
        self.assertEqual(public["by_stage"]["extract"]["pre_wrapper_without_receipt_jobs"], 1)
        self.assertGreater(public["totals"]["job_recorded_cost_usd"], public["receipt_recomputed_cost_usd"])
        self.assertEqual(public["issue_counts"], {"pre_wrapper_without_receipt": 1})
        self.assertEqual((snapshot, receipts), before)
        text = json.dumps(public)
        for private in ("PRIVATE_", "request_sha256", "response_id"):
            self.assertNotIn(private, text)

    def test_duplicate_response_deduplicates_but_unknown_usage_is_not_zero(self):
        snapshot, receipts = self.fixture()
        receipts.append(copy.deepcopy(receipts[0]))
        receipts.append({"request_sha256": "orphan-private-hash", "response_id": None, "model": "gpt-5.1", "usage": None})
        result = reconcile(snapshot, receipts, pre_wrapper_calls=1)["public"]
        expected = sum(job["cost_usd"] for job in snapshot["jobs"][1:])
        self.assertAlmostEqual(result["receipt_priced_cost_usd_known"], expected)
        self.assertIsNone(result["receipt_recomputed_cost_usd"])
        self.assertEqual(result["duplicate_response_records"], 1)
        self.assertEqual(result["unpriced_receipt_units"], 1)
        self.assertEqual(result["issue_counts"]["receipt_without_snapshot_job"], 1)

    def test_usage_response_cost_mismatch_and_reserves_are_separate(self):
        snapshot, receipts = self.fixture()
        snapshot["jobs"][1].update(input_tokens=999, response_id="PRIVATE_WRONG_RESPONSE", cost_usd=.1)
        snapshot["jobs"][2].update(status="uncertain", reserved_usd=.2)
        result = reconcile(snapshot, receipts, pre_wrapper_calls=0)
        public = result["public"]
        self.assertEqual(public["issue_counts"]["response_id_mismatch_or_ambiguous"], 1)
        self.assertEqual(public["issue_counts"]["job_usage_mismatch"], 1)
        self.assertEqual(public["issue_counts"]["job_cost_mismatch"], 1)
        self.assertAlmostEqual(public["totals"]["outstanding_reserved_usd"], .2)
        self.assertEqual(public["issue_counts"]["job_without_receipt"], 1)
        self.assertTrue(any(item.get("job_id") == "PRIVATE_JOB1" for item in result["private"]["issues"]))

    def test_conflicting_duplicate_and_nonstandard_tier_invalidate_complete_total(self):
        snapshot, receipts = self.fixture()
        conflict = copy.deepcopy(receipts[0])
        conflict["usage"]["output_tokens"] = 500
        receipts.extend([conflict, None])
        receipts[1]["actual_service_tier"] = "priority"
        public = reconcile(snapshot, receipts, pre_wrapper_calls=1)["public"]
        self.assertIsNone(public["receipt_recomputed_cost_usd"])
        self.assertEqual(public["issue_counts"]["conflicting_duplicate_response_id"], 1)
        self.assertEqual(public["issue_counts"]["nonstandard_service_tier"], 1)
        self.assertEqual(public["issue_counts"]["invalid_receipt"], 1)
        with self.assertRaisesRegex(ValueError, "Frozen manifest.pricing"):
            reconcile({"jobs": []}, [])


if __name__ == "__main__":
    unittest.main()
