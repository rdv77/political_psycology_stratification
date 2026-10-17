"""Corpus HTTP integration against a disposable fake core; no model or search API."""
from __future__ import annotations

import copy
import hashlib
import http.client
import json
from pathlib import Path
import shutil
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from unittest.mock import patch

from src.semantic_service.server import Application, Handler


ROOT = Path(__file__).resolve().parents[1]


class QuietHandler(Handler):
    def log_message(self, format, *args):
        pass


class FakeCorpus:
    def __init__(self, root):
        self.root = root
        self.records = {}
        self.calls = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.fail = False
        self.escape_export = False

    def model_config(self):
        return {"configured": True, "routes": {"relevance": "fake-cheap"},
                "pricing": {"fake-cheap": {"input": .1, "output": .2}}}

    def list_corpora(self):
        return [copy.deepcopy(value) for value in self.records.values()]

    def get_corpus(self, identifier):
        if identifier not in self.records:
            raise FileNotFoundError(identifier)
        return copy.deepcopy(self.records[identifier])

    def import_csv(self, path, title="", config=None, mapping=None):
        self.calls.append(("import", path, copy.deepcopy(config), mapping))
        identifier = "fixture-corpus"
        self.records[identifier] = {"id": identifier, "title": title, "source_name": path.name,
            "profile": {"rows_total": 3, "rows_with_text": 2, "empty_rows": 1},
            "config": config or {}, "counts": {"documents": 3, "unique_documents": 1, "exact_duplicates": 1},
            "jobs": {"pending": 0, "done": 0}, "usage": {"estimated_usd": 0}}
        return self.get_corpus(identifier)

    def documents(self, identifier, limit=50, offset=0, status=None):
        self.get_corpus(identifier)
        self.calls.append(("documents", limit, offset, status))
        return {"documents": [{"id": "document-1", "title": "<img src=x onerror=alert(1)>",
            "text": "Вымышленный исходный материал", "status": "unique"}], "total": 1, "limit": limit, "offset": offset}

    def configure(self, identifier, config):
        self.calls.append(("configure", copy.deepcopy(config)))
        self.records[identifier]["config"].update(copy.deepcopy(config))
        return self.get_corpus(identifier)

    def prepare(self, identifier, stage="relevance", limit=100):
        self.calls.append(("prepare", stage, limit))
        self.records[identifier]["jobs"]["pending"] = limit
        return {"created": limit, "stage": stage, "total_estimated_upper_usd": .2}

    def run(self, identifier, max_jobs=5, budget_usd=1.0):
        self.calls.append(("run", max_jobs, budget_usd))
        self.started.set()
        if not self.release.wait(3):
            raise ValueError("Synthetic wait timed out")
        if self.fail:
            raise ValueError("Synthetic model unavailable")
        self.records[identifier]["jobs"] = {"pending": 0, "done": max_jobs}
        return {"done": max_jobs}

    def export(self, identifier):
        self.calls.append(("export", identifier))
        directory = self.root / ".corpus_runtime" / "exports" / identifier
        directory.mkdir(parents=True, exist_ok=True)
        path = (self.root / "private.json") if self.escape_export else (directory / "facts.json")
        path.write_text('{"facts":[]}', encoding="utf-8")
        return {"corpus_id": identifier, "directory": str(directory),
                "files": [{"name": "facts.json", "path": str(path)}]}


class CorpusHTTPTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="corpus-http-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "frontend").mkdir()
        for name in ("corpus.html", "corpus.js", "corpus.css"):
            shutil.copyfile(ROOT / "frontend" / name, self.root / "frontend" / name)
        self.input = self.root / "fixture.csv"
        self.input.write_text("title,text\nFictional,Fictional fixture only\n", encoding="utf-8")
        self.app = Application(root=self.root)
        self.core = FakeCorpus(self.root)
        self.app._corpus = self.core
        self.external = patch("src.semantic_service.providers.httpx.Client",
                              side_effect=AssertionError("External calls forbidden in corpus HTTP tests"))
        self.client = self.external.start()
        self.addCleanup(self.external.stop)
        self.addCleanup(self.client.assert_not_called)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), QuietHandler)
        self.server.daemon_threads = True
        self.server.app = self.app
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": .02}, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)
        self.port = self.server.server_address[1]

    def stop_server(self):
        self.core.release.set()
        for job in self.app.corpus_jobs.values():
            job["thread"].join(timeout=3)
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)

    def request(self, path, method="GET", data=None, headers=None, raw=None):
        headers = dict(headers or {})
        if data is not None:
            raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
            headers.setdefault("Content-Type", "application/json")
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        try:
            connection.request(method, path, body=raw, headers=headers)
            response = connection.getresponse()
            body = response.read()
            actual = dict((k.lower(), v) for k, v in response.getheaders())
            if "application/json" in actual.get("content-type", ""):
                body = json.loads(body.decode("utf-8"))
            return response.status, body, actual
        finally:
            connection.close()

    def import_fixture(self, **extra):
        code, body, _ = self.request("/api/corpora", "POST", {"path": str(self.input), "title": "Fixture only", **extra})
        self.assertEqual(code, 201, body)
        return body["id"]

    def wait_job(self, identifier):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            code, data, _ = self.request("/api/corpora/" + identifier)
            self.assertEqual(code, 200)
            if data.get("http_job", {}).get("status") != "running":
                return data
            time.sleep(.02)
        self.fail("Corpus HTTP job did not complete")

    def test_static_page_has_csp_and_assets_and_existing_studies_still_work(self):
        for path, content_type in (("/corpus", "text/html"), ("/corpus.js", "javascript"), ("/corpus.css", "text/css")):
            code, body, headers = self.request(path)
            self.assertEqual(code, 200)
            self.assertIn(content_type, headers["content-type"])
            if path == "/corpus":
                self.assertIn("script-src 'self'", headers["content-security-policy"])
                self.assertNotIn("unsafe-inline", headers["content-security-policy"])
        code, body, _ = self.request("/api/studies")
        self.assertEqual((code, body), (200, {"studies": []}))

    def test_import_prepare_configuration_and_pagination_never_run_api(self):
        identifier = self.import_fixture(mapping={"text": "text"})
        code, data, _ = self.request("/api/corpora")
        self.assertEqual(code, 200)
        self.assertEqual(len(data["corpora"]), 1)
        self.assertTrue(data["model_config"]["configured"])
        code, documents, _ = self.request(f"/api/corpora/{identifier}/documents?limit=7&offset=2&status=unique")
        self.assertEqual(code, 200)
        self.assertEqual(self.core.calls[-1], ("documents", 7, 2, "unique"))
        self.assertEqual(documents["documents"][0]["title"], "<img src=x onerror=alert(1)>")
        code, _, _ = self.request(f"/api/corpora/{identifier}/configure", "POST", {"config": {"scope": "Fixture scope"}})
        self.assertEqual(code, 200)
        code, jobs, _ = self.request(f"/api/corpora/{identifier}/prepare", "POST", {"stage": "extract", "limit": 3})
        self.assertEqual((code, jobs["created"]), (200, 3))
        self.assertFalse(any(call[0] == "run" for call in self.core.calls))
        code, error, _ = self.request(f"/api/corpora/{identifier}/documents?limit=10000")
        self.assertEqual(code, 400)

    def test_real_core_import_prepare_and_export_are_offline_and_wire_compatible(self):
        self.app._corpus = None
        with patch("src.corpus_pipeline.models.OpenAIResponsesHTTP.call",
                   side_effect=AssertionError("The offline workflow must not call the API")) as external:
            identifier = self.import_fixture(config={"scope": "Fictional fixture scope",
                "aspect": "public_positions", "objects": [{"id": "actor-1", "name": "Fictional Actor", "aliases": []}]})
            code, corpus, _ = self.request(f"/api/corpora/{identifier}")
            self.assertEqual(code, 200)
            self.assertEqual(corpus["counts"]["documents"], 1)
            code, documents, _ = self.request(f"/api/corpora/{identifier}/documents")
            self.assertEqual(code, 200)
            self.assertIn("Fictional fixture only", documents["documents"][0]["text"])
            code, prepared, _ = self.request(f"/api/corpora/{identifier}/prepare", "POST", {"stage": "relevance", "limit": 1})
            self.assertEqual(code, 200, prepared)
            self.assertEqual(prepared["created"], 1)
            self.assertEqual(prepared["api_calls"], 0)
            code, exported, _ = self.request(f"/api/corpora/{identifier}/export", "POST", {})
            self.assertEqual(code, 200)
            self.assertIn("facts.json", [x["name"] for x in exported["files"]])
            code, facts, _ = self.request(next(x["url"] for x in exported["files"] if x["name"] == "facts.json"))
            self.assertEqual(code, 200)
            self.assertEqual(facts, [])
            external.assert_not_called()

    def test_study_copy_requires_approved_objects_and_never_inherits_scale_approval(self):
        study = self.app.store.create({"title": "Fictional study", "target_count": 12})
        objects = [{"id": "p", "name": "Вымышленный участник", "name_original": "Fictional Actor"}]
        self.app.store.update(study["id"], objects=objects, scales=[{"id": "s", "name": "Fixture rubric"}])
        code, body, _ = self.request("/api/corpora", "POST", {"path": str(self.input), "study_id": study["id"]})
        self.assertEqual(code, 400)
        self.assertIn("утвердить", body["error"])
        self.app.store.approve(study["id"], "objects")
        source = self.app.store.directory(study["id"]) / "study.json"
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        identifier = self.import_fixture(study_id=study["id"], config={"scales_approved": True})
        config = self.core.get_corpus(identifier)["config"]
        self.assertFalse(config["scales_approved"])
        self.assertEqual(config["objects"][0]["aliases"], ["Fictional Actor"])
        self.assertEqual(config["languages"], [])
        self.assertEqual(config["scales"][0]["id"], "s")
        self.assertEqual(before, hashlib.sha256(source.read_bytes()).hexdigest())

    def test_run_is_explicit_background_and_excludes_parallel_mutations(self):
        identifier = self.import_fixture()
        code, data, _ = self.request(f"/api/corpora/{identifier}/run", "POST", {"max_jobs": 2, "budget_usd": .3})
        self.assertEqual(code, 202)
        self.assertEqual(data["http_job"]["status"], "running")
        self.assertTrue(self.core.started.wait(1))
        for action in ("run", "configure", "prepare", "export"):
            code, error, _ = self.request(f"/api/corpora/{identifier}/{action}", "POST", {})
            self.assertEqual(code, 400)
            self.assertIn("дождитесь", error["error"])
        self.core.release.set()
        complete = self.wait_job(identifier)
        self.assertEqual(complete["http_job"]["status"], "complete")
        self.assertEqual(complete["jobs"]["done"], 2)
        self.assertEqual([c for c in self.core.calls if c[0] == "run"], [("run", 2, .3)])

    def test_rwanda_copy_uses_explicit_position_mode_and_verified_language_plan(self):
        identifier = "rwanda-politicians-2021-2026"
        study = self.app.store.create({"title": "Fictional Rwanda fixture", "target_count": 12,
            "aspect": "Публичный политический образ"}, identifier=identifier)
        self.app.store.update(identifier, objects=[{"id": "fiction", "name": "Fixture", "name_original": "Fictional Actor"}])
        self.app.store.approve(identifier, "objects")
        plan = self.root / "research/rwanda_politicians_2021_2026/corpus_search/language_plan.json"
        plan.parent.mkdir(parents=True)
        plan.write_text(json.dumps({"core_collection_languages": ["Kinyarwanda", "French", "English", "Kiswahili"],
            "mandatory_country_context": {"query_terms": ["Rwanda", "#Rwanda", "u Rwanda", "uRwanda"]}}), encoding="utf-8")
        source = self.app.store.directory(identifier) / "study.json"
        original = source.read_bytes()
        corpus = self.import_fixture(study_id=study["id"])
        config = self.core.get_corpus(corpus)["config"]
        self.assertEqual(config["aspect"], "public_positions")
        self.assertEqual(config["languages"], ["rw", "fr", "en", "sw"])
        self.assertEqual(config["country_terms"], ["Rwanda", "#Rwanda", "u Rwanda", "uRwanda"])
        self.assertFalse(config["scales_approved"])
        self.assertEqual(source.read_bytes(), original)

    def test_failed_background_run_is_not_reported_as_success(self):
        identifier = self.import_fixture()
        self.core.fail = True
        self.core.release.set()
        code, _, _ = self.request(f"/api/corpora/{identifier}/run", "POST", {"max_jobs": 1, "budget_usd": .1})
        self.assertEqual(code, 202)
        data = self.wait_job(identifier)
        self.assertEqual(data["http_job"]["status"], "error")
        self.assertIn("Synthetic model unavailable", data["http_job"]["error"])

    def test_only_registered_runtime_exports_can_be_downloaded(self):
        identifier = self.import_fixture()
        code, data, _ = self.request(f"/api/corpora/{identifier}/export", "POST", {})
        self.assertEqual(code, 200)
        self.assertNotIn("path", data["files"][0])
        code, body, headers = self.request(data["files"][0]["url"])
        self.assertEqual(code, 200)
        self.assertEqual(body, {"facts": []})
        self.assertIn("attachment", headers["content-disposition"])
        code, _, _ = self.request(f"/api/corpora/{identifier}/files/%2e%2e%2fprivate.json")
        self.assertEqual(code, 404)
        self.core.escape_export = True
        code, body, _ = self.request(f"/api/corpora/{identifier}/export", "POST", {})
        self.assertEqual(code, 400)
        self.assertIn("внутри", body["error"])

    def test_local_origin_csv_only_and_positive_api_limits_remain_required(self):
        for path in ("relative.csv", str(self.root / ".env")):
            code, body, _ = self.request("/api/corpora", "POST", {"path": path})
            self.assertEqual(code, 400)
            self.assertIn("CSV", body["error"])
        code, _, _ = self.request("/api/corpora", "POST", {"path": str(self.input)}, headers={"Origin": "https://untrusted.example"})
        self.assertEqual(code, 400)
        self.assertEqual(self.core.calls, [])
        identifier = self.import_fixture()
        for values in ({"max_jobs": 0}, {"max_jobs": True}, {"max_jobs": 1001}, {"budget_usd": 0}, {"budget_usd": float("nan")}):
            code, _, _ = self.request(f"/api/corpora/{identifier}/run", "POST", values)
            self.assertEqual(code, 400)
        self.assertFalse(any(c[0] == "run" for c in self.core.calls))
        code, _, _ = self.request(f"/api/corpora/{identifier}/configure", "POST", {"config": []})
        self.assertEqual(code, 400)


if __name__ == "__main__":
    unittest.main()
