"""Provider and provenance regressions with fake data and no network access."""
from __future__ import annotations

import concurrent.futures
import copy
import json
import os
import queue
import shutil
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.semantic_service.providers import ModelProvider, TEXT, grounded_facts, object_schema
from src.semantic_service.server import Application
from src.semantic_service.store import StudyStore, fingerprint


ROOT = Path(__file__).resolve().parents[1]
TEMP_PARENT = ROOT / ".service_runtime"


class ServiceProviderTests(unittest.TestCase):
    def setUp(self):
        TEMP_PARENT.mkdir(exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix="providers-test-", dir=TEMP_PARENT))
        self.addCleanup(self.remove_test_directory)
        self.environment = patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-key-never-sent",
            "PPLX_API_KEY": "test-search-key-never-sent",
            "SEMANTIC_MODEL": "test-model",
            "SEMANTIC_SEARCH_MODEL": "test-search-model",
            "SEMANTIC_INPUT_USD_PER_M": "2",
            "SEMANTIC_OUTPUT_USD_PER_M": "8",
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.dotenv = patch("src.semantic_service.providers.load_dotenv")
        self.dotenv.start()
        self.addCleanup(self.dotenv.stop)
        self.network = patch("src.semantic_service.providers.httpx.Client",
                             side_effect=AssertionError("External I/O is forbidden in provider tests"))
        self.client = self.network.start()
        self.addCleanup(self.network.stop)
        self.store = StudyStore(self.root / "studies")
        self.study = self.store.create({"title": "Synthetic provider regression", "target_count": 12})
        self.provider = ModelProvider(self.root, self.store)

    def remove_test_directory(self):
        target = self.root.resolve()
        intended = TEMP_PARENT.resolve()
        if not target.is_relative_to(intended) or target == intended:
            raise AssertionError("Test cleanup target escaped its intended directory")
        shutil.rmtree(target)

    @staticmethod
    def objects():
        return [{"id": f"o{i}", "name": f"Synthetic object {i}", "sources": []}
                for i in range(12)]

    @staticmethod
    def scales():
        return [{"id": f"s{i}", "name": f"Synthetic scale {i}",
                 "positive": f"Test pole A {i}", "negative": f"Test pole B {i}",
                 "definition": "Synthetic rubric for regression tests only", "aliases": []}
                for i in range(6)]

    def approve(self, app):
        identifier = self.study["id"]
        app.store.edit(identifier, "objects", self.objects())
        app.store.approve(identifier, "objects")
        app.store.edit(identifier, "scales", self.scales())
        app.store.approve(identifier, "scales")
        return app.store.get(identifier)

    def enable_mock_response(self, post_side_effect=None):
        self.client.side_effect = None
        transport = self.client.return_value.__enter__.return_value
        response = SimpleNamespace(status_code=200, json=lambda: {
            "choices": [{"message": {"content": '{"answer":"cached"}'},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        })
        transport.post.return_value = response
        transport.post.side_effect = post_side_effect
        return transport, response

    def test_model_sources_cannot_authorize_their_own_fact_urls(self):
        invented = {"text": "Synthetic invented claim", "source_url": "https://invented.invalid/source"}
        data = {"facts": [invented],
                "sources": [{"title": "Model-written citation", "url": invented["source_url"]}]}
        self.assertEqual(grounded_facts(data, []), [])
        self.assertEqual(grounded_facts(data, ["https://retrieved.example/document"]), [])
        self.client.assert_not_called()

    def test_fact_requires_a_retrieval_citation_and_nonempty_text(self):
        retrieved = {"text": "Synthetic supported claim", "source_url": "https://retrieved.example/document/"}
        data = {"facts": [retrieved, {"text": "", "source_url": retrieved["source_url"]},
                          {"text": "Unsafe scheme", "source_url": "file:///private"}]}
        self.assertEqual(grounded_facts(data, ["https://retrieved.example/document"]), [retrieved])
        self.client.assert_not_called()

    def test_identical_concurrent_requests_make_one_provider_call(self):
        transport, response = self.enable_mock_response()
        def delayed_post(*args, **kwargs):
            time.sleep(.04)
            return response
        transport.post.side_effect = delayed_post
        barrier = threading.Barrier(4)
        def request():
            barrier.wait(timeout=3)
            return self.provider.request(self.study["id"], "same-task", "Same synthetic prompt",
                                         object_schema({"answer": TEXT}), max_tokens=1000)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: request(), range(4)))
        self.assertEqual(transport.post.call_count, 1)
        self.assertTrue(all(result["data"] == {"answer": "cached"} for result in results))
        self.assertEqual(len({result["cache_key"] for result in results}), 1)
        self.assertEqual(self.store.get(self.study["id"])["usage"]["calls"], 1)
        request_again = self.provider.request(self.study["id"], "same-task", "Same synthetic prompt",
                                              object_schema({"answer": TEXT}), max_tokens=1000)
        self.assertEqual(request_again["data"], {"answer": "cached"})
        self.assertEqual(transport.post.call_count, 1)
        self.assertEqual(list(self.provider.cache.glob("*.tmp")), [])

    def test_inflight_reservations_prevent_parallel_budget_oversubscription(self):
        self.store.update(self.study["id"], budget_limit_usd=.012)
        entered, release = threading.Event(), threading.Event()
        transport, response = self.enable_mock_response()
        def blocked_post(*args, **kwargs):
            entered.set()
            if not release.wait(timeout=5):
                raise AssertionError("Test did not release the mocked provider")
            return response
        transport.post.side_effect = blocked_post
        barrier, rejected = threading.Barrier(3), queue.Queue()
        def request(index):
            barrier.wait(timeout=3)
            try:
                self.provider.request(self.study["id"], f"task-{index}", f"Synthetic prompt {index}",
                                      object_schema({"answer": TEXT}), max_tokens=1000)
                return "completed"
            except ValueError:
                rejected.put("rejected")
                return "rejected"
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(request, index) for index in range(3)]
            try:
                self.assertTrue(entered.wait(timeout=3))
                self.assertEqual([rejected.get(timeout=3) for _ in range(2)], ["rejected", "rejected"])
            finally:
                release.set()
            results = [future.result(timeout=3) for future in futures]
        self.assertEqual(results.count("completed"), 1)
        self.assertEqual(transport.post.call_count, 1)
        self.assertEqual(self.provider.reserved.get(self.study["id"], 0), 0)

    def test_edited_scale_definitions_cannot_reuse_imported_ratings(self):
        app = Application(root=self.root, studies_dir=self.store.root)
        study = self.approve(app)
        ratings = [{"object_id": obj["id"], "scale_id": scale["id"],
                    "status": "observed", "positive": 4, "negative": 0}
                   for obj in study["objects"] for scale in study["scales"]]
        app.import_ratings(study["id"], ratings)
        changed = copy.deepcopy(study["scales"])
        changed[0]["definition"] = "A different synthetic construct under the same identifier"
        app.store.edit(study["id"], "scales", changed)
        app.store.approve(study["id"], "scales")
        analyze, export = Mock(), Mock()
        with patch.dict(sys.modules, {
            "src.semantic_service.analysis": SimpleNamespace(analyze_study=analyze),
            "src.semantic_service.reports": SimpleNamespace(export_report=export),
        }):
            app.run_analysis(study["id"], {"use_imported_ratings": True})
            app.jobs[study["id"]].join(timeout=3)
        self.assertFalse(app.jobs[study["id"]].is_alive())
        stopped = app.store.get(study["id"])
        self.assertEqual(stopped["status"], "error")
        self.assertIn("импорт", stopped["error"].lower())
        analyze.assert_not_called()
        export.assert_not_called()
        self.client.assert_not_called()

    def test_invalid_or_reversed_calendar_dates_are_rejected(self):
        for start, end in (("2026-99-99", "2027-01-01"),
                           ("yesterday", "tomorrow"),
                           ("2026-12-01", "20261015")):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                self.store.create({"title": "Invalid synthetic period", "date_from": start, "date_to": end})
        self.client.assert_not_called()

    def test_dossier_completion_order_does_not_change_the_scale_prompt(self):
        study = dict(self.study, objects=self.objects())
        prompts = []
        def dossier(_study, obj):
            return {"object_id": obj["id"], "descriptors": ["Synthetic descriptor " + obj["id"]],
                    "facts": [{"text": "Synthetic fixture", "source_url": "https://fixture.example/source"}]}
        def request(_identifier, _task, prompt, _schema, **kwargs):
            prompts.append((_task, prompt))
            return {"data": {"scales": self.scales(), "normalization_notes": []},
                    "model": "test-model", "at": "test"}
        with patch.object(self.provider, "dossier", side_effect=dossier), \
             patch.object(self.provider, "request", side_effect=request):
            for reverse in (False, True):
                def completion_order(futures, reverse=reverse):
                    return iter(list(futures)[::-1] if reverse else list(futures))
                with patch("src.semantic_service.providers.concurrent.futures.as_completed",
                           side_effect=completion_order):
                    self.provider.generate_scales(study)
        self.assertEqual([task for task, _ in prompts], ["scales", "scales-normalize", "scales", "scales-normalize"])
        self.assertEqual(prompts[0][1], prompts[2][1])
        self.assertEqual(prompts[1][1], prompts[3][1])
        self.client.assert_not_called()

    def test_unknown_model_rating_never_becomes_numeric_zero(self):
        study = dict(self.study, scales=self.scales())
        response = {"data": {"ratings": [{"scale_id": "s0", "status": "unknown",
                    "positive": 5, "negative": 0, "evidence_indices": [0],
                    "reason": "Synthetic insufficient evidence", "confidence": .1}]},
                    "model": "test-model", "at": "test"}
        with patch.object(self.provider, "dossier", return_value={"facts": [
                {"text": "Synthetic evidence", "source_url": "https://fixture.example/source"}]}), \
             patch.object(self.provider, "request", return_value=response):
            rows = self.provider.rate_object(study, self.objects()[0])
        self.assertEqual(len(rows), 6)
        for row in rows:
            self.assertEqual(row["status"], "unknown")
            for field in ("positive", "negative", "difference", "mass"):
                self.assertIsNone(row[field])
        self.client.assert_not_called()

    def measurement_fixture(self, count=26):
        scales = [dict(self.scales()[0], id=f"s{i}", name=f"Synthetic scale {i}") for i in range(count)]
        study = dict(self.study, scales=scales, aspect="Публичный политический образ")
        obj = self.objects()[0]
        dossier = {"object_id": obj["id"], "basis": "synthetic-original-basis",
                   "facts": [{"text": "Synthetic original fact", "source_url": "https://fixture.example/original",
                              "period": "2024"}],
                   "sources": [{"title": "Original fixture", "url": "https://fixture.example/original"}]}
        evidence_dir = self.store.directory(study["id"]) / "evidence"
        evidence_dir.mkdir(exist_ok=True)
        original = evidence_dir / (obj["id"] + ".json")
        original.write_text(json.dumps(dossier), encoding="utf-8")
        return study, obj, dossier, evidence_dir, original

    @staticmethod
    def prompt_scales(prompt):
        return json.loads(prompt.split("Шкалы: ", 1)[1].split("\nФакты: ", 1)[0])

    @staticmethod
    def fixture_ratings(scales, unknown=(), evidence_index=0):
        return {"data": {"ratings": [{"scale_id": scale["id"],
                 "status": "unknown" if scale["id"] in unknown else "observed",
                 "positive": 3, "negative": 1, "confidence": .8,
                 "reason": "Synthetic regression fixture", "evidence_indices": [evidence_index]}
                for scale in scales]}, "model": "test-model", "at": "test-time"}

    @staticmethod
    def fixture_supplement():
        return {"data": {"facts": [{"text": "Synthetic newly retrieved fact",
                    "source_url": "https://fixture.example/addition", "period": "2025"}],
                    "sources": [{"title": "Retrieved fixture", "url": "https://fixture.example/addition"}],
                    "limitations": "Synthetic coverage remains incomplete"},
                "citations": ["https://fixture.example/addition"], "model": "test-search-model",
                "at": "test-time", "cache_key": "test-retrieval-cache-key"}

    def test_targeted_enrichment_rerates_only_unknown_scales_in_affected_chunks(self):
        study, obj, dossier, directory, original = self.measurement_fixture()
        original_bytes = original.read_bytes()
        unknown = {"s1", "s2", "s3", "s4", "s24", "s25"}
        calls = []
        def request(_identifier, task, prompt, _schema, **kwargs):
            calls.append((task, prompt, kwargs))
            if task.startswith("dossier-supplement-"):
                self.assertEqual(kwargs, {"search": True, "max_tokens": 8000})
                target = json.loads(prompt.split("Недостаточно покрытые шкалы: ", 1)[1].split("\nИсходные факты:", 1)[0])
                self.assertEqual({scale["id"] for scale in target}, unknown)
                return self.fixture_supplement()
            scales = self.prompt_scales(prompt)
            self.assertTrue(prompt.startswith("Разведочная оценка публичного образа Synthetic object 0. Период "))
            if task.startswith("ratings-supplement-"):
                self.assertTrue(all(scale["id"] in unknown for scale in scales))
                return self.fixture_ratings(scales, {"s25"}, evidence_index=1)
            return self.fixture_ratings(scales, unknown)
        with patch.object(self.provider, "dossier", return_value=dossier), \
             patch.object(self.provider, "request", side_effect=request):
            rows = self.provider.rate_object(study, obj)
            again = self.provider.rate_object(study, obj)
        self.assertEqual(rows, again)
        self.assertEqual([task for task, _, _ in calls if task.startswith("dossier-supplement-")], ["dossier-supplement-o0"])
        self.assertEqual([task for task, _, _ in calls[:6]], [
            "ratings-o0-0", "ratings-o0-12", "ratings-o0-24", "dossier-supplement-o0",
            "ratings-supplement-o0-0", "ratings-supplement-o0-24"])
        self.assertEqual([row["scale_id"] for row in rows], [scale["id"] for scale in study["scales"]])
        self.assertEqual(sum(row["status"] == "observed" for row in rows), 25)
        self.assertEqual(rows[0]["rating_pass"], "initial")
        self.assertEqual(rows[1]["rating_pass"], "supplement")
        self.assertEqual(rows[1]["evidence_indices"], [1])
        self.assertEqual(rows[1]["sources"], ["https://fixture.example/addition"])
        self.assertIsNone(rows[-1]["positive"])
        self.assertIsNone(rows[-1]["difference"])
        self.assertEqual(original.read_bytes(), original_bytes)
        initial = json.loads((directory / "scoring_initial-o0.json").read_text(encoding="utf-8"))
        supplement = json.loads((directory / "dossier_supplement-o0.json").read_text(encoding="utf-8"))
        self.assertEqual(sum(row["status"] == "observed" for row in initial["ratings"]), 20)
        self.assertEqual(initial["scales_basis"], fingerprint(study["scales"]))
        self.assertEqual(rows[1]["evidence_basis"], supplement["basis"])
        self.assertEqual(supplement["supplement_provenance"]["maximum_retrieval_rounds"], 1)
        self.assertEqual(list(directory.glob("*.tmp")), [])
        self.client.assert_not_called()

    def test_enrichment_threshold_includes_exactly_twenty_percent(self):
        for count, unknown_count, expected_calls in ((6, 1, 0), (5, 1, 1)):
            with self.subTest(count=count):
                study, obj, dossier, _, _ = self.measurement_fixture(count)
                response = self.fixture_ratings(study["scales"], {f"s{i}" for i in range(unknown_count)})
                with patch.object(self.provider, "dossier", return_value=dossier), \
                     patch.object(self.provider, "request", return_value=response), \
                     patch.object(self.provider, "enrich_dossier", return_value={"supplement_facts": []}) as enrich:
                    rows = self.provider.rate_object(study, obj)
                self.assertEqual(enrich.call_count, expected_calls)
                self.assertEqual(sum(row["status"] == "unknown" for row in rows), unknown_count)
        self.client.assert_not_called()

    def test_supplement_rejects_self_citations_deduplicates_and_invalidates_on_scale_edit(self):
        study, obj, dossier, directory, original = self.measurement_fixture()
        original_bytes = original.read_bytes()
        response = self.fixture_supplement()
        response["citations"].append("https://fixture.example/original")
        response["data"]["facts"].extend([
            {"text": "  Synthetic  ORIGINAL fact ", "source_url": "https://fixture.example/original/", "period": "2024"},
            {"text": "Synthetic newly retrieved fact", "source_url": "https://fixture.example/addition/", "period": "2025"},
            {"text": "Synthetic unsupported invention", "source_url": "https://invented.invalid/source", "period": "2025"}])
        response["data"]["sources"].append({"title": "Self-authorizing fake", "url": "https://invented.invalid/source"})
        with patch.object(self.provider, "request", return_value=response) as request:
            first = self.provider.enrich_dossier(study, obj, dossier, study["scales"][:2])
            cached = self.provider.enrich_dossier(study, obj, dossier, study["scales"][3:5])
            self.assertEqual(request.call_count, 1)
            self.assertEqual(first, cached)
            edited = copy.deepcopy(study)
            edited["scales"][0]["definition"] = "Changed semantic meaning under the same identifier"
            changed = self.provider.enrich_dossier(edited, obj, dossier, edited["scales"][:2])
        self.assertEqual(request.call_count, 2)
        self.assertNotEqual(first["basis"], changed["basis"])
        self.assertEqual(len(first["facts"]), 2)
        self.assertEqual(len(first["supplement_facts"]), 1)
        self.assertEqual(first["supplement_provenance"]["accepted_fact_count"], 3)
        self.assertEqual(first["supplement_provenance"]["added_fact_count"], 1)
        self.assertFalse(any("invented.invalid" in source["url"] for source in first["sources"]))
        self.assertEqual(original.read_bytes(), original_bytes)
        self.assertEqual(first["original_dossier_file"], "evidence/o0.json")
        self.assertEqual(first["scales_basis"], fingerprint(study["scales"]))
        self.client.assert_not_called()

    def test_optional_enrichment_failure_preserves_first_pass_and_unknown_nulls(self):
        study, obj, dossier, directory, _ = self.measurement_fixture(5)
        with patch.object(self.provider, "dossier", return_value=dossier), \
             patch.object(self.provider, "request", return_value=self.fixture_ratings(study["scales"], {"s4"})), \
             patch.object(self.provider, "enrich_dossier", side_effect=ValueError("Synthetic exhausted budget")) as enrich:
            rows = self.provider.rate_object(study, obj)
        self.assertEqual(enrich.call_count, 1)
        self.assertEqual(sum(row["status"] == "observed" for row in rows), 4)
        self.assertEqual(rows[0]["positive"], 3)
        self.assertIsNone(rows[-1]["positive"])
        self.assertIn("Synthetic exhausted budget", rows[-1]["enrichment_error"])
        self.assertTrue((directory / "scoring_initial-o0.json").exists())
        self.client.assert_not_called()

    def test_supplement_rating_failure_preserves_successful_chunks(self):
        study, obj, dossier, _, _ = self.measurement_fixture()
        unknown = {"s1", "s2", "s3", "s4", "s24", "s25"}
        def request(_identifier, task, prompt, _schema, **kwargs):
            if task.startswith("dossier-supplement-"):
                return self.fixture_supplement()
            if task == "ratings-supplement-o0-24":
                raise ValueError("Synthetic final chunk budget stop")
            return self.fixture_ratings(self.prompt_scales(prompt),
                                         () if task.startswith("ratings-supplement-") else unknown)
        with patch.object(self.provider, "dossier", return_value=dossier), \
             patch.object(self.provider, "request", side_effect=request):
            rows = self.provider.rate_object(study, obj)
        self.assertEqual(sum(row["status"] == "observed" for row in rows), 24)
        self.assertEqual(rows[1]["rating_pass"], "supplement")
        for row in rows[-2:]:
            self.assertEqual(row["status"], "unknown")
            self.assertIsNone(row["positive"])
            self.assertIn("budget stop", row["enrichment_error"])
        self.client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
