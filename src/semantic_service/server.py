"""Local-only HTTP application. Run: python -m src.semantic_service.server."""
from __future__ import annotations

import argparse
import copy
import json
import mimetypes
import os
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from .providers import ModelProvider
from .store import BUSY, StudyStore, fingerprint, ratings_basis, validate_objects


ROOT = Path(__file__).resolve().parents[2]


class Application:
    def __init__(self, root: Path = ROOT, studies_dir: Path | None = None):
        self.root = Path(root)
        self.store = StudyStore(studies_dir or self.root / "studies")
        self.provider = ModelProvider(self.root, self.store)
        self.jobs = {}
        self.job_lock = threading.Lock()
        self._corpus = None
        self.corpus_jobs = {}
        self.corpus_job_lock = threading.RLock()
        self.seed_id = "russian-politicians-2016-2026"
        for row in self.store.list():
            study = self.store.get(row["id"])
            if study["status"] in BUSY:
                self.store.update(study["id"], status="interrupted", error="Сервер был остановлен. Запустите нужный этап снова; неизменные запросы читаются из кэша.")
        self.import_seed()
        self.configure_export_runtime()

    @staticmethod
    def configure_export_runtime():
        dependency = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies"
        defaults = {"SEMANTIC_EXPORT_PYTHON": dependency / "python/python.exe",
                    "SEMANTIC_NODE": dependency / "node/bin/node.exe",
                    "SEMANTIC_NODE_MODULES": dependency / "node/node_modules"}
        for key, path in defaults.items():
            if not os.environ.get(key) and path.exists():
                os.environ[key] = str(path)

    def import_seed(self):
        seed = self.root / "research/russian_politicians_2016_2026/candidates.json"
        if not seed.exists() or (self.store.directory(self.seed_id) / "study.json").exists():
            return
        data = json.loads(seed.read_text(encoding="utf-8"))
        spec = data["study"]
        study = self.store.create(spec, identifier=self.seed_id)
        study.update(objects=validate_objects(data["objects"]), status="await_objects_approval",
                     selection_method=data.get("selection_method", "Целевая разнообразная выборка"),
                     seed_provenance="research/russian_politicians_2016_2026/candidates.json",
                     selection_limitations=data.get("limitations", []),
                     progress={"stage": "objects", "message": "Подготовлен список кандидатов; проверьте и утвердите состав", "current": len(data["objects"]), "total": len(data["objects"])})
        self.store.save(study)

    def config(self):
        return {"provider_configured": self.provider.configured,
                "search_configured": bool(self.provider.search_key), "model": self.provider.model,
                "search_model": self.provider.search_model,
                "export_formats": ["md", "html", "pdf", "docx", "pptx", "zip"],
                "seed_id": self.seed_id if (self.store.directory(self.seed_id) / "study.json").exists() else None,
                "local_only": True, "budget_limit_usd": 20,
                "markdown_note": "Markdown импортируется через Файл → Открыть в Google Docs; для встроенных диаграмм удобнее DOCX. ZIP содержит Markdown и папку изображений."}

    def launch(self, identifier: str, status: str, worker):
        with self.job_lock:
            if identifier in self.jobs and self.jobs[identifier].is_alive():
                raise ValueError("Этап уже выполняется")
            self.store.update(identifier, status=status, error=None)
            def run():
                try:
                    worker()
                except Exception as exc:
                    message = str(exc)
                    # Provider clients sanitize their errors; no request/credential dumps.
                    self.store.update(identifier, status="error", error=message[:1800])
                    self.store.log(identifier, "Этап остановлен: " + message[:1000])
                    error_file = self.store.directory(identifier) / "last_error.log"
                    error_file.write_text(traceback.format_exc(), encoding="utf-8")
            job = threading.Thread(target=run, name="study-" + identifier, daemon=True)
            self.jobs[identifier] = job
            job.start()
        return self.store.get(identifier)

    def corpus(self):
        # Existing studies and tests do not require the optional corpus module.
        with self.corpus_job_lock:
            if self._corpus is None:
                from src.corpus_pipeline.service import CorpusPipeline
                self._corpus = CorpusPipeline(self.root)
            return self._corpus

    def corpus_config(self, payload):
        config = copy.deepcopy(payload["config"]) if payload.get("config") is not None else {}
        if not isinstance(config, dict):
            raise ValueError("Настройки корпуса должны быть JSON-объектом")
        if payload.get("study_id"):
            study = self.store.get(payload["study_id"])
            if not self.store.is_approved(study, "objects"):
                raise ValueError("В выбранном исследовании сначала нужно утвердить список объектов")
            snapshot = {key: copy.deepcopy(study.get(key, "")) for key in
                        ("scope", "aspect", "date_from", "date_to")}
            if snapshot["aspect"] not in {"public_positions", "public_image"}:
                snapshot["aspect"] = "public_positions"
            snapshot["objects"] = [{"id": obj["id"], "name": obj["name"],
                "aliases": list(dict.fromkeys([*obj.get("aliases", []),
                    *([obj["name_original"]] if obj.get("name_original") else [])]))}
                for obj in study["objects"]]
            snapshot.update(languages=copy.deepcopy(study.get("languages", [])),
                            scales=copy.deepcopy(study.get("scales", [])))
            if study["id"] == "rwanda-politicians-2021-2026":
                language_plan = self.root / "research/rwanda_politicians_2021_2026/corpus_search/language_plan.json"
                if language_plan.is_file():
                    plan = json.loads(language_plan.read_text(encoding="utf-8"))
                    codes = {"Kinyarwanda": "rw", "French": "fr", "English": "en", "Kiswahili": "sw"}
                    snapshot["languages"] = [codes[name] for name in plan["core_collection_languages"]]
                    snapshot["country_terms"] = list(plan["mandatory_country_context"]["query_terms"])
            snapshot.update(config)
            # Approval in an earlier study is not approval of this corpus rubric.
            snapshot["scales_approved"] = False
            config = snapshot
        return config

    def corpus_status(self, identifier):
        result = copy.deepcopy(self.corpus().get_corpus(identifier))
        with self.corpus_job_lock:
            job = self.corpus_jobs.get(identifier)
            if job:
                result["http_job"] = {k: v for k, v in job.items() if k != "thread"}
        return result

    def corpus_idle(self, identifier):
        self.corpus().get_corpus(identifier)
        with self.corpus_job_lock:
            if self.corpus_jobs.get(identifier, {}).get("status") == "running":
                raise ValueError("Корпус сейчас обрабатывается; дождитесь завершения запуска")

    def corpus_run(self, identifier, payload):
        with self.corpus_job_lock:
            self.corpus_idle(identifier)
            maximum, budget = payload.get("max_jobs", 5), payload.get("budget_usd", 1.0)
            if isinstance(maximum, bool) or not isinstance(maximum, int) or not 1 <= maximum <= 1000:
                raise ValueError("Лимит запуска: целое число от 1 до 1000 заданий")
            if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not 0 < budget <= 10000:
                raise ValueError("Задайте положительный бюджет API не более 10000 USD")
            state = {"status": "running", "action": "run", "error": None}
            self.corpus_jobs[identifier] = state
            def work():
                try:
                    result = self.corpus().run(identifier, max_jobs=maximum, budget_usd=budget)
                    with self.corpus_job_lock:
                        state.update(status="complete", result=result)
                except Exception as exc:
                    with self.corpus_job_lock:
                        state.update(status="error", error=str(exc)[:1800])
            state["thread"] = threading.Thread(target=work, name="corpus-" + identifier, daemon=True)
            state["thread"].start()
        return self.corpus_status(identifier)

    def corpus_export(self, identifier):
        self.corpus_idle(identifier)
        result = self.corpus().export(identifier)
        runtime = (self.root / ".corpus_runtime").resolve()
        files, saved = [], {}
        for item in result.get("files", []):
            path = Path(item["path"]).resolve()
            name = str(item.get("name") or path.name)
            if (not path.is_relative_to(runtime) or not path.is_file() or name in {".", ".."} or name != Path(name).name
                    or "/" in name or "\\" in name or name in saved
                    or path.suffix.lower() not in {".json", ".jsonl", ".csv", ".md", ".zip"}):
                raise ValueError("Экспорт должен содержать доступные файлы внутри локальной папки корпуса")
            saved[name] = str(path.relative_to(runtime))
            from urllib.parse import quote
            files.append({"name": name, "url": f"/api/corpora/{quote(identifier, safe='')}/files/{quote(name, safe='')}"})
        manifests = runtime / "http_exports"
        manifests.mkdir(parents=True, exist_ok=True)
        # Identifier validation remains owned by the core before any file access.
        key = fingerprint(identifier)
        temporary = manifests / (key + ".tmp")
        temporary.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")
        temporary.replace(manifests / (key + ".json"))
        return {"corpus_id": identifier, "files": files}

    def corpus_export_file(self, identifier, name):
        self.corpus().get_corpus(identifier)
        runtime = (self.root / ".corpus_runtime").resolve()
        manifest = runtime / "http_exports" / (fingerprint(identifier) + ".json")
        files = json.loads(manifest.read_text(encoding="utf-8"))
        if name not in files:
            raise FileNotFoundError(name)
        path = (runtime / files[name]).resolve()
        if not path.is_relative_to(runtime):
            raise ValueError("Недоступный файл корпуса")
        return path

    def discover(self, identifier: str):
        def work():
            study = self.store.get(identifier)
            self.store.log(identifier, "Подбираются объекты и источники", "discover", 0, study["target_count"])
            objects = self.provider.discover(study)
            latest = self.store.get(identifier)
            latest.update(objects=objects, scales=[], analysis=None, artifacts=[], status="await_objects_approval",
                          approvals={"objects": {"approved": False}, "scales": {"approved": False}})
            self.store.save(latest)
            self.store.log(identifier, "Проверьте и утвердите список объектов", "objects", len(objects), len(objects))
        return self.launch(identifier, "discovering", work)

    def generate_scales(self, identifier: str):
        study = self.store.get(identifier)
        if not self.store.is_approved(study, "objects"):
            raise ValueError("Перед подготовкой шкал утвердите список объектов")
        def work():
            current = self.store.get(identifier)
            if not self.store.is_approved(current, "objects"):
                raise ValueError("Список объектов изменился; сначала утвердите его новую версию")
            scales, provenance = self.provider.generate_scales(current,
                progress=lambda message, current, total: self.store.log(identifier, message, "sources", current, total))
            self.store.update(identifier, scales=scales, scale_provenance=provenance,
                              status="await_scales_approval", analysis=None, artifacts=[])
            self.store.log(identifier, "Проверьте словарь и смысл полюсов, затем утвердите шкалы", "scales", len(scales), len(scales))
        return self.launch(identifier, "generating_scales", work)

    def run_analysis(self, identifier: str, options: dict):
        self.store.require_approved(identifier)
        def work():
            study = self.store.require_approved(identifier)
            ratings_path = self.store.directory(identifier) / "ratings.json"
            if options.get("use_imported_ratings"):
                if not ratings_path.exists():
                    raise ValueError("Нет импортированной матрицы оценок")
                current_basis = {"objects": fingerprint(study["objects"]), "scales": fingerprint(study["scales"])}
                if study.get("measurement_model") == "independent_poles":
                    current_basis["measurement_model"] = "independent_poles"
                if study.get("imported_ratings_basis") != current_basis:
                    raise ValueError("Импортированные оценки относятся к другой версии объектов или шкал; импортируйте их заново")
                ratings = json.loads(ratings_path.read_text(encoding="utf-8"))
            else:
                basis = ratings_basis(study)
                receipt = study.get("ratings_cache")
                ratings = None
                if (options.get("force_rerate") is not True and isinstance(receipt, dict)
                        and receipt.get("basis") == basis and ratings_path.is_file()):
                    try:
                        saved = json.loads(ratings_path.read_text(encoding="utf-8"))
                        if isinstance(saved, list) and receipt.get("ratings_sha256") == fingerprint(saved):
                            ratings = saved
                    except (OSError, ValueError):
                        # Missing, malformed or stale material cannot authorize reuse.
                        pass
                if ratings is None:
                    # Invalidate the old receipt before the provider writes progress.
                    self.store.update(identifier, ratings_cache=None)
                    self.store.log(identifier, "Начато оценивание по утверждённым шкалам", "ratings", 0, len(study["objects"]))
                    ratings = self.provider.rate_all(study, progress=lambda message, current, total:
                                                    self.store.log(identifier, message, "ratings", current, total))
                    pending = ratings_path.with_suffix(".json.tmp")
                    pending.write_text(json.dumps(ratings, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
                    pending.replace(ratings_path)
                    self.store.update(identifier, ratings_cache={"basis": basis,
                                      "ratings_sha256": fingerprint(ratings), "mode": "provider_ratings"})
                else:
                    self.store.log(identifier, "Сохранённая матрица оценок проверена и используется повторно", "ratings",
                                   len(study["objects"]), len(study["objects"]))
            if study.get("delivery_mode") == "factor_preview":
                self.store.log(identifier, "Сравниваются факторные решения; готовится короткий отчёт", "factor_preview", 0, 1)
                from .factor_preview import preview_factors
                output = self.store.directory(identifier) / "output"
                preview = preview_factors(study, ratings, output)
                artifacts = [{"label": "Короткий отчёт: сравнение факторов", "format": "md",
                              "url": f"/api/studies/{identifier}/files/factor_preview.md"},
                             {"label": "Нагрузки и диагностика всех решений", "format": "json",
                              "url": f"/api/studies/{identifier}/files/factor_preview.json"}]
                for chart in preview.get("charts", []):
                    artifacts.append({"label": chart["title"], "format": "png",
                                      "url": f"/api/studies/{identifier}/files/{chart['file']}"})
                insufficient = preview.get("measurement_status") == "insufficient_data"
                self.store.update(identifier, status="needs_measurement" if insufficient else "await_factor_selection", analysis=preview,
                                  artifacts=artifacts, error=None)
                self.store.log(identifier,
                               "Недостаточно сопоставимых измерений. Короткий аудит покрытия готов" if insufficient else
                               "Короткий отчёт готов. Выберите факторное решение перед расширенным отчётом и слайдами",
                               "measurement_review" if insufficient else "factor_selection", 1, 1)
                return
            self.store.log(identifier, "Проверка матрицы, факторный и кластерный анализ", "analysis", 0, 1)
            from .analysis import analyze_study
            output = self.store.directory(identifier) / "output"
            analysis = analyze_study(study, ratings, output,
                                     factor_count=options.get("factor_count") or None,
                                     cluster_count=options.get("cluster_count") or None)
            if not options.get("use_imported_ratings") and self.provider.configured:
                self.store.log(identifier, "Содержательно сравниваются варианты кластеризации", "cluster_review", 0, 1)
                choice = self.provider.choose_clusters(self.store.get(identifier), analysis)
                if not options.get("cluster_count") and choice["recommended_k"] != analysis["cluster_count"]:
                    analysis = analyze_study(study, ratings, output,
                                             factor_count=analysis["factor_count"],
                                             cluster_count=choice["recommended_k"])
                analysis["cluster_content_review"] = choice
                analysis["cluster_selection_reason"] = choice["reason"]
                analysis["diagnostics"]["cluster_selection"]["selection_authority"] = "user_override" if options.get("cluster_count") else "semantic_and_numerical_review"
                analysis["diagnostics"]["cluster_selection"]["content_recommended_k"] = choice["recommended_k"]
                if options.get("cluster_count"):
                    analysis["cluster_selection_reason"] = f"Пользователь задал k={analysis['cluster_count']}. Содержательное сравнение: " + choice["reason"]
                if not choice["discrete_structure_supported"]:
                    analysis["warnings"].append("Содержательное сравнение не подтверждает естественные дискретные классы: группы используются как описательная типология непрерывного пространства.")
                self.store.log(identifier, "Интерпретируются нагрузки и полные профили групп", "interpretation", 0, 1)
                analysis = self.provider.interpret(self.store.get(identifier), analysis)
                (output / "metadata.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
            self.store.update(identifier, analysis=analysis)
            self.store.log(identifier, "Формируются отчёт, PDF и слайды", "export", 0, 1)
            from .reports import export_report
            study = self.store.get(identifier)
            artifacts = export_report(study, analysis, ratings, output, self.root)
            self.store.update(identifier, status="complete", artifacts=artifacts, error=None)
            self.store.log(identifier, "Исследование завершено. Отчёт и файлы готовы", "complete", 1, 1)
        return self.launch(identifier, "running", work)

    def import_ratings(self, identifier: str, ratings: list):
        study = self.store.require_approved(identifier)
        if study["status"] in BUSY:
            raise ValueError("Дождитесь окончания текущего этапа")
        if not isinstance(ratings, list) or len(ratings) > 200 * 80:
            raise ValueError("Некорректный формат оценок")
        objects, scales = {o["id"] for o in study["objects"]}, {s["id"] for s in study["scales"]}
        seen = set()
        for row in ratings:
            pair = (row.get("object_id"), row.get("scale_id"))
            if pair[0] not in objects or pair[1] not in scales or pair in seen:
                raise ValueError("Неизвестный или повторный объект/шкала в оценках")
            seen.add(pair)
        destination = self.store.directory(identifier) / "ratings.json"
        destination.write_text(json.dumps(ratings, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        imported_basis = {"objects": fingerprint(study["objects"]), "scales": fingerprint(study["scales"])}
        if study.get("measurement_model") == "independent_poles":
            imported_basis["measurement_model"] = "independent_poles"
        self.store.update(identifier, measurement_mode="imported_ratings", imported_ratings_count=len(ratings),
                          ratings_cache=None,
                          imported_ratings_basis=imported_basis)
        return self.store.get(identifier)


class Handler(BaseHTTPRequestHandler):
    server_version = "SemanticSpace/0.1"

    @property
    def app(self) -> Application:
        return self.server.app

    def log_message(self, format, *args):
        # Paths and status only, never request bodies or authorization data.
        print("[HTTP] " + format % args, flush=True)

    def json_response(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def body(self):
        length = int(self.headers.get("Content-Length", "0"))
        if not 0 < length <= 8_000_000:
            if length == 0:
                return {}
            raise ValueError("Запрос слишком большой")
        if not self.headers.get("Content-Type", "").startswith("application/json"):
            raise ValueError("Ожидается application/json")
        result = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(result, dict):
            raise ValueError("Ожидается JSON-объект")
        return result

    def local_request(self):
        host = self.headers.get("Host", "")
        port = self.server.server_address[1]
        if host not in {f"127.0.0.1:{port}", f"localhost:{port}"}:
            raise ValueError("Разрешены только запросы к локальному адресу сервиса")
        origin = self.headers.get("Origin")
        if origin and origin not in {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}:
            raise ValueError("Запрос с другого сайта отклонён")

    def do_GET(self):
        try:
            self.local_request()
            path = urlparse(self.path).path
            if path == "/api/config":
                self.app.import_seed()
                return self.json_response(self.app.config())
            if path == "/api/studies":
                return self.json_response({"studies": self.app.store.list()})
            parts = path.strip("/").split("/")
            if parts[:2] == ["api", "corpora"]:
                if len(parts) == 2:
                    return self.json_response({"corpora": self.app.corpus().list_corpora(),
                                               "model_config": self.app.corpus().model_config()})
                if len(parts) == 3:
                    return self.json_response(self.app.corpus_status(parts[2]))
                if len(parts) == 4 and parts[3] == "documents":
                    query = parse_qs(urlparse(self.path).query)
                    limit, offset = int(query.get("limit", [50])[0]), int(query.get("offset", [0])[0])
                    if not 1 <= limit <= 200 or offset < 0:
                        raise ValueError("Размер страницы должен быть от 1 до 200; смещение неотрицательное")
                    return self.json_response(self.app.corpus().documents(parts[2], limit=limit, offset=offset,
                        status=query.get("status", [None])[0]))
                if len(parts) == 5 and parts[3] == "files":
                    return self.send_file(self.app.corpus_export_file(parts[2], unquote(parts[4])), download=True)
                return self.json_response({"error": "Не найдено"}, 404)
            if len(parts) == 3 and parts[:2] == ["api", "studies"]:
                return self.json_response(self.app.store.get(parts[2]))
            if len(parts) >= 5 and parts[:2] == ["api", "studies"] and parts[3] == "files":
                base = (self.app.store.directory(parts[2]) / "output").resolve()
                relative = unquote("/".join(parts[4:]))
                file = (base / relative).resolve()
                if not file.is_relative_to(base) or file.suffix.lower() not in {".pdf", ".pptx", ".docx", ".md", ".html", ".zip", ".png", ".csv", ".json"}:
                    raise ValueError("Недоступный файл")
                return self.send_file(file, download=file.suffix.lower() in {".pdf", ".pptx", ".docx", ".md", ".zip", ".csv"})
            allowed = {"/": "index.html", "/index.html": "index.html", "/app.js": "app.js", "/styles.css": "styles.css",
                       "/corpus": "corpus.html", "/corpus.html": "corpus.html",
                       "/corpus.js": "corpus.js", "/corpus.css": "corpus.css"}
            if path in allowed:
                return self.send_file(self.app.root / "frontend" / allowed[path])
            self.json_response({"error": "Не найдено"}, 404)
        except FileNotFoundError:
            self.json_response({"error": "Файл или исследование не найдено"}, 404)
        except (ValueError, KeyError) as exc:
            self.json_response({"error": str(exc)}, 400)

    def send_file(self, file: Path, download=False):
        data = file.read_bytes()
        self.send_response(200)
        content = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
        if file.suffix in {".html", ".js", ".css", ".md"}:
            content += "; charset=utf-8"
        self.send_header("Content-Type", content)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-cache")
        if file.suffix == ".html" and "output" in file.parts:
            self.send_header("Content-Security-Policy", "default-src 'none'; img-src 'self' data:; style-src 'unsafe-inline'; frame-ancestors 'self'")
        elif file.name == "corpus.html":
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
        if download:
            self.send_header("Content-Disposition", 'attachment; filename="' + file.name + '"')
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        return self.mutate("POST")

    def do_PUT(self):
        return self.mutate("PUT")

    def mutate(self, method):
        try:
            self.local_request()
            payload = self.body()
            parts = urlparse(self.path).path.strip("/").split("/")
            if method == "POST" and parts[:2] == ["api", "corpora"]:
                if len(parts) == 2:
                    path = Path(str(payload.get("path", "")))
                    if not path.is_absolute() or path.suffix.lower() != ".csv":
                        raise ValueError("Укажите абсолютный путь к локальному CSV-файлу")
                    return self.json_response(self.app.corpus().import_csv(path,
                        title=payload.get("title", ""), config=self.app.corpus_config(payload),
                        mapping=payload.get("mapping")), 201)
                if len(parts) == 4:
                    identifier, action = parts[2:]
                    with self.app.corpus_job_lock:
                        self.app.corpus_idle(identifier)
                        if action == "configure":
                            return self.json_response(self.app.corpus().configure(identifier, self.app.corpus_config(payload)))
                        if action == "prepare":
                            return self.json_response(self.app.corpus().prepare(identifier,
                                stage=payload.get("stage", "relevance"), limit=payload.get("limit", 100)))
                        if action == "run":
                            return self.json_response(self.app.corpus_run(identifier, payload), 202)
                        if action == "export":
                            return self.json_response(self.app.corpus_export(identifier))
                return self.json_response({"error": "Неизвестное действие корпуса"}, 404)
            if method == "POST" and parts == ["api", "studies"]:
                return self.json_response(self.app.store.create(payload), 201)
            if len(parts) != 4 or parts[:2] != ["api", "studies"]:
                return self.json_response({"error": "Не найдено"}, 404)
            identifier, action = parts[2:]
            self.app.store.get(identifier)
            if method == "PUT" and action in {"objects", "scales"}:
                return self.json_response(self.app.store.edit(identifier, action, payload[action]))
            if method == "POST" and action in {"approve-objects", "approve-scales"}:
                return self.json_response(self.app.store.approve(identifier, action.split("-", 1)[1]))
            if method == "POST" and action == "discover":
                return self.json_response(self.app.discover(identifier), 202)
            if method == "POST" and action == "generate-scales":
                return self.json_response(self.app.generate_scales(identifier), 202)
            if method == "POST" and action == "run":
                return self.json_response(self.app.run_analysis(identifier, payload), 202)
            if method == "POST" and action == "import-ratings":
                return self.json_response(self.app.import_ratings(identifier, payload["ratings"]))
            self.json_response({"error": "Неизвестное действие"}, 404)
        except FileNotFoundError:
            self.json_response({"error": "Исследование не найдено"}, 404)
        except (ValueError, KeyError, TypeError) as exc:
            self.json_response({"error": str(exc)}, 400)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--studies-dir", type=Path)
    args = parser.parse_args()
    app = Application(studies_dir=args.studies_dir)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.app = app
    print(f"Semantic Space: http://127.0.0.1:{server.server_address[1]}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
