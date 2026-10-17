"""Safe receipt fields and unchanged billing exceptions with MockTransport."""
import copy
import json
from pathlib import Path
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

import httpx

from src.corpus_pipeline.models import ModelResponseError, make_task
from src.corpus_pipeline.receipt_client import ReceiptingClient
from src.corpus_pipeline.service import digest


class ReceiptClientTests(unittest.TestCase):
    def setUp(self):
        base = Path(__file__).resolve().parents[1] / ".service_runtime/receipt_client_tests"
        base.mkdir(parents=True, exist_ok=True)
        temp = tempfile.TemporaryDirectory(dir=base)
        self.addCleanup(temp.cleanup)
        self.ledger = Path(temp.name) / "receipts.jsonl"
        self.payload = make_task("relevance", {"config": {"objects": [{"id": "a", "name": "Alpha North"}]},
            "document": {"text": "Alpha North PRIVATE_SOURCE text."}}, model="gpt-4.1-mini")

    def response(self, *, refusal=False):
        content = {"type": "refusal", "refusal": "PRIVATE_SOURCE sk-PRIVATE_KEY"} if refusal else {
            "type": "output_text", "text": json.dumps({"status": "relevant", "actor_ids": ["a"],
                "language": "en", "reason": "PRIVATE_GENERATED_TEXT"})}
        return {"id": "resp_test", "model": "gpt-4.1-mini", "status": "completed", "service_tier": "default",
                "usage": {"input_tokens": 200, "output_tokens": 20, "total_tokens": 220,
                    "input_tokens_details": {"cached_tokens": 100}, "private_debug": "PRIVATE_SOURCE"},
                "output": [{"type": "message", "status": "completed", "content": [content]}]}

    def records(self):
        return [json.loads(line) for line in self.ledger.read_text(encoding="utf-8").splitlines()]

    def test_default_tier_original_hash_immutability_and_allowlisted_metadata(self):
        original = copy.deepcopy(self.payload)
        self.payload["service_tier"] = "priority"
        original = copy.deepcopy(self.payload)
        requests = []
        def handler(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json=self.response(), headers={
                "x-ratelimit-remaining-tokens": "999", "retry-after": "1", "set-cookie": "sk-PRIVATE_KEY"})
        client = ReceiptingClient("sk-PRIVATE_KEY", self.ledger, transport=httpx.MockTransport(handler))
        self.addCleanup(client.close)
        result = client.call(self.payload)
        self.assertEqual(self.payload, original)
        self.assertEqual(requests[0]["service_tier"], "default")
        self.assertEqual(len(requests), 1)
        receipt = self.records()[0]
        self.assertEqual(receipt["request_sha256"], digest(original))
        self.assertEqual(receipt["wire_request_sha256"], digest(requests[0]))
        self.assertEqual(receipt["actual_service_tier"], "default")
        self.assertEqual(receipt["usage"]["input_tokens_details"]["cached_tokens"], 100)
        self.assertEqual(receipt["rate_headers"], {"x-ratelimit-remaining-tokens": "999", "retry-after": "1"})
        self.assertEqual(result["response_id"], "resp_test")
        for secret in ("PRIVATE_SOURCE", "PRIVATE_GENERATED_TEXT", "PRIVATE_KEY", "set-cookie", "private_debug"):
            self.assertNotIn(secret, self.ledger.read_text(encoding="utf-8"))

    def test_refusal_records_paid_usage_and_preserves_exception(self):
        calls = []
        def handler(request):
            calls.append(1)
            return httpx.Response(200, json=self.response(refusal=True))
        client = ReceiptingClient("sk-PRIVATE_KEY", self.ledger, transport=httpx.MockTransport(handler))
        self.addCleanup(client.close)
        with self.assertRaises(ModelResponseError) as raised:
            client.call(self.payload)
        self.assertEqual(raised.exception.error_code, "refusal")
        self.assertEqual(raised.exception.usage["input_tokens"], 200)
        self.assertEqual(self.records()[0]["adapter_error_code"], "refusal")
        self.assertEqual(self.records()[0]["usage"]["output_tokens"], 20)
        self.assertEqual(len(calls), 1)
        self.assertNotIn("PRIVATE_SOURCE", self.ledger.read_text(encoding="utf-8"))

    def test_http_error_and_timeout_receipts_do_not_log_message_or_retry(self):
        for timeout in (False, True):
            with self.subTest(timeout=timeout):
                calls = []
                def handler(request):
                    calls.append(1)
                    if timeout:
                        raise httpx.ReadTimeout("sk-PRIVATE_KEY PRIVATE_SOURCE", request=request)
                    return httpx.Response(429, json={"error": {"code": "rate_limit_exceeded", "type": "rate_limit_error",
                        "message": "sk-PRIVATE_KEY PRIVATE_SOURCE"}}, headers={"retry-after": "2"})
                client = ReceiptingClient("sk-PRIVATE_KEY", self.ledger, transport=httpx.MockTransport(handler))
                try:
                    with self.assertRaises(ModelResponseError) as raised:
                        client.call(self.payload)
                finally:
                    client.close()
                receipt = self.records()[-1]
                self.assertEqual(len(calls), 1)
                self.assertIsNone(receipt["usage"])
                self.assertEqual(receipt["http_status"], None if timeout else 429)
                self.assertEqual(receipt["provider_error_code"], None if timeout else "rate_limit_exceeded")
                self.assertEqual(raised.exception.uncertain, timeout)
                self.assertNotIn("PRIVATE_", self.ledger.read_text(encoding="utf-8"))

    def test_shared_jsonl_appends_are_complete_from_distinct_clients(self):
        barrier = threading.Barrier(4)
        def call(index):
            def handler(request):
                barrier.wait(timeout=5)
                return httpx.Response(200, json=self.response())
            client = ReceiptingClient("sk-PRIVATE_KEY", self.ledger, transport=httpx.MockTransport(handler))
            try:
                return client.call(self.payload)["response_id"]
            finally:
                client.close()
        with ThreadPoolExecutor(max_workers=4) as pool:
            self.assertEqual(len(list(pool.map(call, range(4)))), 4)
        self.assertEqual(len(self.records()), 4)


if __name__ == "__main__":
    unittest.main()
