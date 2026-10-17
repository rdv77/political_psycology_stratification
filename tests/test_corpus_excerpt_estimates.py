"""Controlled request deltas and literal retention; no API or private sources."""
import copy
import json
import unittest

from scripts.compare_corpus_excerpt_budgets import bounded_payload, compare, payload_parts, retained_fact, render
from src.corpus_pipeline.models import make_task


class ByteEncoding:
    name = "synthetic-byte-encoding"
    def encode(self, text, **kwargs):
        return list(text.encode("utf-8"))
    encode_ordinary = encode


class ExcerptEstimatesTests(unittest.TestCase):
    def fixture(self):
        encoding = ByteEncoding()
        objects = [{"id": "a", "name": "Alpha North", "aliases": ["A. North", "UnusedAlias"]},
                   {"id": "b", "name": "Beta South", "aliases": []}]
        config = {"objects": objects, "scope": "Synthetic positions", "aspect": "public_positions",
                  "routes": {"relevance": "gpt-4.1-mini", "extract": "gpt-5.1", "describe": "gpt-5.1"},
                  "pricing": {"gpt-4.1-mini": {"input": .4, "cached_input": .1, "output": 1.6},
                              "gpt-5.1": {"input": 1.25, "cached_input": .125, "output": 10}}}
        text = 'Alpha North said: "PRIVATE_POSITION". 2025-01-01.' + ' background ' * 100
        docs = [{"id": "d1", "text": text, "published_at": "2025-01-01"},
                {"id": "d2", "text": "Beta South ordinary news.", "published_at": "2025-01-02"}]
        jobs = []
        for stage, identifier, start, end in (("relevance", "d1", 0, 700), ("relevance", "d1", 700, len(text)),
                                              ("relevance", "d2", 0, len(docs[1]["text"])), ("extract", "d1", 0, 700)):
            doc = next(d for d in docs if d["id"] == identifier)
            context = {"config": copy.deepcopy(config), "document": {"id": identifier, "text": doc["text"][start:end],
                       "chunk_start": start, "chunk_end": end, "author": "PRIVATE_AUTHOR"}}
            model = config["routes"][stage]
            payload = make_task(stage, context, model=model)
            parts = payload_parts(payload, encoding)
            jobs.append({"id": "job" + str(len(jobs)), "stage": stage, "active_recipe": True, "status": "done",
                         "context": context, "payload": payload, "input_tokens": sum(parts.values()) + 37,
                         "output_tokens": 17, "cost_usd": .001, "result": {"status": "irrelevant" if identifier == "d2" else "relevant"}})
        quote = 'Alpha North said: "PRIVATE_POSITION".'
        fact = {"actor_id": "a", "kind": "public_statement", "document_id": "d1", "extraction_job_id": "job3",
                "quote": quote, "quote_start": 0, "quote_end": len(quote), "attribution_evidence": "Alpha North said:", "date_evidence": None}
        results = {"corpus": {"id": "corpus-synthetic", "config": config, "profile": {"rows_total": 2}},
                   "jobs": jobs, "facts": [fact]}
        return results, docs, encoding

    def test_baseline_exact_one_document_removes_repeat_overhead_but_preserves_outputs_and_negatives(self):
        inputs = self.fixture()
        before = copy.deepcopy(inputs[:2])
        result = compare(*inputs)
        baseline = result["variants"][0]["cost_scenarios"]["fixed_original_request_count"]
        self.assertEqual(baseline["stages"]["relevance"]["input_tokens"], sum(j["input_tokens"] for j in inputs[0]["jobs"] if j["stage"] == "relevance"))
        variant = next(v for v in result["variants"] if v["id"] == "head_500")
        one = variant["cost_scenarios"]["one_request_per_document"]["stages"]["relevance"]
        fixed = variant["cost_scenarios"]["fixed_original_request_count"]["stages"]["relevance"]
        self.assertEqual((fixed["requests"], one["requests"]), (3, 2))
        self.assertEqual(fixed["output_tokens"], one["output_tokens"])
        self.assertEqual(one["output_tokens"], 51)
        self.assertGreater(fixed["input_tokens"], one["input_tokens"])
        self.assertEqual(inputs[:2], before)
        for s in result["baseline"]["stages"].values():
            self.assertEqual(s["api_input_tokens"], s["serialized_document_delta_tokens"] + s["fixed_input_including_api_overhead_tokens"])

    def test_quote_and_all_support_must_share_one_span_with_correct_global_offset(self):
        text = "Quote here. gap Attribution here. gap Date2025"
        fact = {"quote": "Quote here.", "quote_start": 0, "quote_end": 11, "attribution_evidence": "Attribution here.", "date_evidence": "Date2025"}
        whole = [{"start": 0, "end": len(text), "text": text}]
        self.assertEqual(retained_fact(fact, text, whole), (True, True))
        separated = [{"start": 0, "end": 11, "text": text[:11]}, {"start": 16, "end": len(text), "text": text[16:]}]
        self.assertEqual(retained_fact(fact, text, separated), (False, True))
        self.assertEqual(retained_fact(dict(fact, quote_start=1), text, whole), (False, False))

    def test_compact_keeps_canonical_name_only_approved_matched_aliases_and_nohit_full_fallback(self):
        results, docs, encoding = self.fixture()
        job = results["jobs"][0]
        matched_text = "A. North says policy."
        matched = [{"start": 0, "end": len(matched_text), "text": matched_text}]
        payload, count, fallback = bounded_payload(job, matched, compact=True)
        objects = json.loads(payload["input"][0]["content"][0]["text"])["config"]["objects"]
        self.assertEqual(objects, [{"id": "a", "name": "Alpha North", "aliases": ["A. North"]}])
        self.assertEqual(count, 1)
        self.assertFalse(fallback)
        nohit_text = "No known names."
        nohit = [{"start": 0, "end": len(nohit_text), "text": nohit_text}]
        payload, count, fallback = bounded_payload(job, nohit, compact=True)
        self.assertEqual(count, 2)
        self.assertTrue(fallback)

    def test_aggregate_privacy_and_macro_do_not_turn_uncovered_actors_into_zero_accuracy(self):
        results, docs, encoding = self.fixture()
        output = compare(results, docs, encoding)
        public = json.dumps(output, ensure_ascii=False, allow_nan=False)
        self.assertNotIn("PRIVATE_", public)
        for key in ('"document_id"', '"payload"', '"context"', '"author"', '"quote"'):
            self.assertNotIn(key, public)
        macro = output["variants"][0]["macro_retention"]["one_request_per_document"]
        self.assertEqual(macro["actors_with_baseline_facts"], 1)
        self.assertEqual(macro["roster_actors_without_baseline_facts"], 1)
        self.assertEqual(macro["unweighted_actor_mean_fraction"], 1)
        self.assertIn("Макросреднее", render(output))
        self.assertFalse(output["methodology"]["future_cache_guaranteed"])


if __name__ == "__main__":
    unittest.main()
