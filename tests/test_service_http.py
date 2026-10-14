"""Real local HTTP routes in disposable stores; no paid or external requests."""
from __future__ import annotations

import http.client
import json
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from src.semantic_service.server import Application, Handler
from src.semantic_service.store import BUSY


ROOT = Path(__file__).resolve().parents[1]


class QuietHandler(Handler):
    def log_message(self, format, *args):
        pass


class ServiceHTTPTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.app = Application(root=ROOT, studies_dir=Path(self.temporary.name))
        # These tests exercise actual missing-provider behavior, never a synthetic
        # replacement measurement. Even accidentally attempted external I/O fails.
        self.app.provider.openai_key = ""
        self.app.provider.search_key = ""
        self.external = patch(
            "src.semantic_service.providers.httpx.Client",
            side_effect=AssertionError("External model calls are forbidden in HTTP tests"),
        )
        self.external_client = self.external.start()
        self.addCleanup(self.external.stop)
        self.addCleanup(self.external_client.assert_not_called)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), QuietHandler)
        self.server.daemon_threads = True
        self.server.app = self.app
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            kwargs={"poll_interval": .03},
            daemon=True,
        )
        self.thread.start()
        self.addCleanup(self.stop_server)
        self.port = self.server.server_address[1]

    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)
        for job in self.app.jobs.values():
            job.join(timeout=3)

    def request(self, path, method="GET", data=None, headers=None, raw=None):
        actual_headers = dict(headers or {})
        body = raw
        if data is not None:
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            actual_headers.setdefault("Content-Type", "application/json")
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        try:
            connection.request(method, path, body=body, headers=actual_headers)
            response = connection.getresponse()
            content = response.read()
            response_headers = {key.lower(): value for key, value in response.getheaders()}
            value = json.loads(content.decode("utf-8")) if "application/json" in response_headers.get("content-type", "") else content
            return response.status, value, response_headers
        finally:
            connection.close()

    def create_study(self):
        status, study, _ = self.request("/api/studies", "POST", {
            "title": "Тест интерфейса: только вымышленные объекты",
            "scope": "Одноразовое тестовое пространство",
            "aspect": "Проверка маршрутов",
            "target_count": 12,
            "objects_text": "\n".join(f"Вымышленный объект {i}" for i in range(1, 13)),
            "date_from": "2025-01-01", "date_to": "2026-01-01",
        })
        self.assertEqual(status, 201)
        return study

    def wait_for_idle(self, identifier):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status, study, _ = self.request(f"/api/studies/{identifier}")
            self.assertEqual(status, 200)
            if study["status"] not in BUSY:
                return study
            time.sleep(.015)
        self.fail("Background HTTP job did not finish within five seconds")

    def prepare_candidates(self):
        study = self.create_study()
        status, _, _ = self.request(f"/api/studies/{study['id']}/discover", "POST", {})
        self.assertEqual(status, 202)
        result = self.wait_for_idle(study["id"])
        self.assertEqual(result["status"], "await_objects_approval")
        return result

    def prepare_approved(self):
        study = self.prepare_candidates()
        base = f"/api/studies/{study['id']}"
        status, _, _ = self.request(base + "/approve-objects", "POST", {})
        self.assertEqual(status, 200)
        scales = [{"id": f"s{i}", "name": f"Тестовая шкала {i}",
                   "positive": f"Тестовый положительный полюс {i}",
                   "negative": f"Тестовый отрицательный полюс {i}",
                   "definition": "Одноразовая синтетическая рубрика для проверки HTTP, не исследовательские данные",
                   "aliases": []} for i in range(6)]
        status, _, _ = self.request(base + "/scales", "PUT", {"scales": scales})
        self.assertEqual(status, 200)
        status, approved, _ = self.request(base + "/approve-scales", "POST", {})
        self.assertEqual(status, 200)
        return approved

    def test_config_and_static_files_have_expected_types_without_secrets(self):
        status, config, headers = self.request("/api/config")
        self.assertEqual(status, 200)
        self.assertFalse(config["provider_configured"])
        self.assertFalse(config["search_configured"])
        self.assertEqual(headers["cache-control"], "no-store")
        self.assertTrue({"md", "html", "pdf", "docx", "pptx", "zip"}.issubset(config["export_formats"]))
        for key in ("openai_key", "search_key", "OPENAI_API_KEY", "PPLX_API_KEY"):
            self.assertNotIn(key, config)
        for path, mime in (("/", "text/html"), ("/app.js", "javascript"), ("/styles.css", "text/css")):
            with self.subTest(path=path):
                status, data, headers = self.request(path)
                self.assertEqual(status, 200)
                self.assertGreater(len(data), 100)
                self.assertIn(mime, headers["content-type"])
                self.assertEqual(headers["x-content-type-options"], "nosniff")

    def test_explicit_discovery_is_persistent_and_never_invents_sources(self):
        study = self.prepare_candidates()
        self.assertEqual(len(study["objects"]), 12)
        self.assertEqual(study["objects"][0]["name"], "Вымышленный объект 1")
        self.assertTrue(all(obj["sources"] == [] for obj in study["objects"]))
        self.assertFalse(study["approvals"]["objects"]["approved"])
        status, listing, _ = self.request("/api/studies")
        self.assertEqual(status, 200)
        self.assertIn(study["id"], {row["id"] for row in listing["studies"]})
        status, persisted, _ = self.request(f"/api/studies/{study['id']}")
        self.assertEqual(status, 200)
        self.assertEqual(persisted["objects"], study["objects"])

    def test_http_workflow_requires_both_gates_and_editing_invalidates_results(self):
        draft = self.create_study()
        base = f"/api/studies/{draft['id']}"
        for action in ("run", "generate-scales", "approve-objects", "approve-scales"):
            with self.subTest(action=action):
                status, error, _ = self.request(base + "/" + action, "POST", {})
                self.assertEqual(status, 400)
                self.assertIn("error", error)
        study = self.prepare_approved()
        base = f"/api/studies/{study['id']}"
        self.assertTrue(study["approvals"]["objects"]["approved"])
        self.assertTrue(study["approvals"]["scales"]["approved"])
        edited = study["objects"]
        edited[0]["role"] = "Уточнённая тестовая роль"
        status, result, _ = self.request(base + "/objects", "PUT", {"objects": edited})
        self.assertEqual(status, 200)
        self.assertFalse(result["approvals"]["objects"]["approved"])
        self.assertFalse(result["approvals"]["scales"]["approved"])
        self.assertEqual(result["scales"], [])
        self.assertIsNone(result["analysis"])
        self.assertEqual(result["artifacts"], [])
        status, _, _ = self.request(base + "/run", "POST", {})
        self.assertEqual(status, 400)

    def test_missing_evidence_produces_an_error_instead_of_fabricated_analysis(self):
        study = self.prepare_approved()
        status, _, _ = self.request(f"/api/studies/{study['id']}/run", "POST", {})
        self.assertEqual(status, 202)
        stopped = self.wait_for_idle(study["id"])
        self.assertEqual(stopped["status"], "error")
        self.assertTrue(stopped["error"])
        self.assertIsNone(stopped["analysis"])
        self.assertEqual(stopped["artifacts"], [])
        self.assertEqual(stopped["usage"]["calls"], 0)

    def test_cross_origin_and_nonlocal_hosts_are_rejected_before_mutation(self):
        before = {row["id"] for row in self.app.store.list()}
        for headers in ({"Origin": "https://unrelated.example"}, {"Host": "unrelated.example"}):
            with self.subTest(headers=headers):
                status, error, _ = self.request("/api/studies", "POST", {"title": "Must not exist"}, headers=headers)
                self.assertEqual(status, 400)
                self.assertIn("error", error)
        after = {row["id"] for row in self.app.store.list()}
        self.assertEqual(before, after)

    def test_files_are_confined_to_output_and_report_policy_disables_scripts(self):
        study = self.create_study()
        output = self.app.store.directory(study["id"]) / "output"
        output.mkdir()
        (output / "report.html").write_text("<html><body>Тестовый отчёт</body></html>", encoding="utf-8")
        (output / "scores.csv").write_text("object,factor\nexample,1\n", encoding="utf-8")
        base = f"/api/studies/{study['id']}/files/"
        status, data, headers = self.request(base + "report.html")
        self.assertEqual(status, 200)
        self.assertIn("Тестовый отчёт", data.decode("utf-8"))
        self.assertIn("default-src 'none'", headers["content-security-policy"])
        self.assertNotIn("script-src", headers["content-security-policy"])
        status, _, headers = self.request(base + "scores.csv")
        self.assertEqual(status, 200)
        self.assertIn("attachment", headers["content-disposition"])
        for name in ("%2e%2e/study.json", "../../.env", "bad.exe"):
            with self.subTest(name=name):
                status, error, _ = self.request(base + name)
                self.assertEqual(status, 400)
                self.assertIn("error", error)
        status, _, _ = self.request(base + "absent.pdf")
        self.assertEqual(status, 404)

    def test_invalid_json_and_nonobject_bodies_return_actionable_errors(self):
        for raw in (b"[]", b"{broken"):
            with self.subTest(raw=raw):
                status, error, _ = self.request("/api/studies", "POST", raw=raw, headers={"Content-Type": "application/json"})
                self.assertEqual(status, 400)
                self.assertIn("error", error)
        status, error, _ = self.request("/api/studies", "POST", raw=b"x", headers={"Content-Type": "text/plain"})
        self.assertEqual(status, 400)
        self.assertIn("application/json", error["error"])


if __name__ == "__main__":
    unittest.main()
