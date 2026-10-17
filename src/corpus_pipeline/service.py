"""Versioned local corpus store, durable tasks, and explicitly budgeted model runs.

Import, profiling, duplicate detection, preparation and export never call an API.
Original documents live in an ignored runtime directory, outside research results.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import threading
import uuid
import unicodedata

from .ingest import inspect_csv, iter_documents, exact_fingerprint, near_duplicate_pairs
from .identity import IDENTITY_POLICY_VERSION, ground_relevance
from .evidence import EVIDENCE_POLICY_VERSION, normalize_extraction
from .rubrics import RUBRIC_POLICY_VERSION, normalize_discovery, intensity_rubric_issues
from .excerpts import EXCERPT_POLICY_VERSION, offline_encoding, select_spans_with_metadata
from .source_spans import document_texts, fact_span
from .context_policy import CONTEXT_POLICY_VERSION, select_document_context, actor_mentions
from .models import (DEFAULT_PRICING, DEFAULT_ROUTES, ModelResponseError, COMMON, PROMPTS, SCHEMAS,
                     OpenAIResponsesHTTP, estimate_cost, make_task,
                     payload_input_bound, validate_result, _eligible_fact, EXCERPT_INSTRUCTIONS)


VERSION = "corpus-v1"
STAGES = ("relevance", "extract", "describe", "score")
MAX_OUTPUT = {"relevance": 600, "extract": 4096, "describe": 6000, "score": 3000}
LANGUAGES = {"kinyarwanda": "rw", "french": "fr", "english": "en", "kiswahili": "sw",
             "swahili": "sw", "киньяруанда": "rw", "французский": "fr",
             "английский": "en", "суахили": "sw"}


def now():
    return datetime.now(timezone.utc).isoformat()


def packed(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(packed(value).encode("utf-8")).hexdigest()


def chunks(text, width=9000, overlap=350):
    """Cover the entire text with stable, overlapping windows; never truncate it."""
    start = 0
    while start < len(text):
        end = min(start + width, len(text))
        if end < len(text):
            boundary = text.rfind("\n", start + width // 2, end)
            if boundary > start:
                end = boundary + 1
        yield start, end, text[start:end]
        if end == len(text):
            break
        start = end - overlap


def candidates(text, objects):
    result = []
    for actor in objects:
        aliases = [actor["name"]] + actor.get("aliases", [])
        if any(re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", text, re.IGNORECASE)
               for alias in aliases if alias):
            result.append(actor["id"])
    return result


def compact_objects(objects, document):
    """Supply only approved, literally mentioned actors; no inferred aliases.

    A passage without a known name keeps the full roster for manual/model
    review. It is never silently classified as an unrelated publication.
    """
    def normalized(value):
        return " ".join(unicodedata.normalize("NFC", value).casefold().split())
    texts = [normalized(text) for text in document_texts(document)]
    result = []
    for actor in objects:
        def present(label):
            label = normalized(label)
            return bool(label) and any(re.search(r"(?<!\w)" + re.escape(label) + r"(?!\w)", text)
                                       for text in texts)
        aliases = [alias for alias in actor.get("aliases", []) if present(alias)]
        if present(actor["name"]) or aliases:
            result.append({"id": actor["id"], "name": actor["name"], "aliases": aliases})
    return result or objects


def publication_date(value):
    """Normalize explicit export dates without inventing a timezone or event date."""
    value = str(value or "").replace("\u00a0", " ").strip()
    try:
        if re.fullmatch(r"\d{2}\.\d{2}\.\d{4}(?: \d{2}:\d{2}(?::\d{2})?)?", value):
            pattern = "%d.%m.%Y" + (" %H:%M:%S" if value.count(":") == 2 else " %H:%M" if ":" in value else "")
            return datetime.strptime(value, pattern).isoformat(timespec="seconds") if ":" in value else datetime.strptime(value, pattern).date().isoformat()
        if re.match(r"^\d{4}-\d{2}-\d{2}(?:[T ]|$)", value):
            return datetime.fromisoformat(value.replace("Z", "+00:00")).isoformat() if len(value) > 10 else date.fromisoformat(value).isoformat()
    except ValueError:
        pass
    return value


def source_context(document):
    """Only coarse content provenance; never forward demographic metadata."""
    metadata = document.get("metadata", {})
    raw = str(next((metadata[k] for k in ("Тип", "Тип сообщения", "message_type", "source_type") if metadata.get(k)), "")).casefold()
    kind = "comment" if raw in {"комментарий", "comment", "reply"} else "repost" if raw in {"репост", "репосты с дополнением", "репост с дополнением", "repost", "retweet", "quote tweet"} else "article" if raw in {"статья", "article", "news"} else "post" if raw in {"пост", "post"} else "unknown"
    platform = str(metadata.get("Тип площадки", "")).casefold()
    return {"source_type": kind, "source_platform": "social" if platform == "соцсеть" else "media" if platform in {"сми", "медиа"} else "unknown",
            "author_is_verified_actor": False}


class CorpusPipeline:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.runtime = self.root / ".corpus_runtime"
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.db_path = self.runtime / "corpus.sqlite3"
        self.run_lock = threading.Lock()
        with self.connection() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS corpora (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL, created_at TEXT NOT NULL,
                    source_name TEXT NOT NULL, source_sha256 TEXT NOT NULL,
                    profile TEXT NOT NULL, config TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY, corpus_id TEXT NOT NULL, source_row INTEGER NOT NULL,
                    data TEXT NOT NULL, text TEXT NOT NULL, content_hash TEXT NOT NULL,
                    status TEXT NOT NULL, duplicate_of TEXT);
                CREATE INDEX IF NOT EXISTS documents_corpus ON documents(corpus_id, source_row);
                CREATE TABLE IF NOT EXISTS similar (
                    corpus_id TEXT NOT NULL, left_id TEXT NOT NULL, right_id TEXT NOT NULL,
                    data TEXT NOT NULL, PRIMARY KEY(corpus_id,left_id,right_id));
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, corpus_id TEXT NOT NULL, stage TEXT NOT NULL,
                    unit_id TEXT NOT NULL, recipe TEXT NOT NULL, context TEXT NOT NULL,
                    payload TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT,
                    result TEXT, error TEXT, response_id TEXT,
                    input_tokens INTEGER NOT NULL DEFAULT 0, output_tokens INTEGER NOT NULL DEFAULT 0,
                    cost_usd REAL NOT NULL DEFAULT 0, reserved_usd REAL NOT NULL DEFAULT 0);
                CREATE INDEX IF NOT EXISTS job_queue ON jobs(corpus_id,status,created_at);
                CREATE TABLE IF NOT EXISTS facts (
                    id TEXT PRIMARY KEY, corpus_id TEXT NOT NULL, job_id TEXT NOT NULL,
                    actor_id TEXT NOT NULL, document_id TEXT NOT NULL, data TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS facts_corpus ON facts(corpus_id, actor_id);
                CREATE TABLE IF NOT EXISTS fact_reviews (
                    id TEXT PRIMARY KEY, corpus_id TEXT NOT NULL, fact_id TEXT NOT NULL,
                    created_at TEXT NOT NULL, reviewer TEXT NOT NULL, policy TEXT NOT NULL, data TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS fact_reviews_corpus ON fact_reviews(corpus_id,fact_id,created_at);
                CREATE TABLE IF NOT EXISTS scales (
                    id TEXT NOT NULL, corpus_id TEXT NOT NULL, job_id TEXT NOT NULL,
                    data TEXT NOT NULL, PRIMARY KEY(corpus_id,job_id,id));
                CREATE TABLE IF NOT EXISTS ratings (
                    corpus_id TEXT NOT NULL, job_id TEXT NOT NULL, actor_id TEXT NOT NULL,
                    scale_id TEXT NOT NULL, data TEXT NOT NULL,
                    PRIMARY KEY(corpus_id,job_id,actor_id,scale_id));
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, corpus_id TEXT NOT NULL, started_at TEXT NOT NULL,
                    budget_usd REAL NOT NULL, max_jobs INTEGER NOT NULL, data TEXT);
            """)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.db_path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def model_config(self):
        return {"configured": bool(os.environ.get("OPENAI_API_KEY")),
                "routes": dict(DEFAULT_ROUTES), "pricing": DEFAULT_PRICING,
                "paid_calls_require_explicit_run": True}

    def _config(self, supplied=None):
        config = {"scope": "", "aspect": "public_positions", "objects": [], "languages": [],
                  "date_from": "", "date_to": "", "country_terms": [], "scales": [],
                  "scales_approved": False, "routes": dict(DEFAULT_ROUTES),
                  "pricing": json.loads(packed(DEFAULT_PRICING)), "budget_limit_usd": None,
                  "text_selection": {"mode": "full", "max_tokens": 1000, "compact_roster": False}}
        supplied = supplied or {}
        if not isinstance(supplied, dict):
            raise ValueError("Настройки корпуса должны быть JSON-объектом")
        # Credentials and arbitrary execution options are never persisted.
        unknown = set(supplied) - set(config) - {"study_id", "rubric_version"}
        if unknown:
            raise ValueError("Неизвестные поля настроек корпуса: " + ", ".join(sorted(unknown)))
        config.update(supplied)
        policy = config["text_selection"]
        if (not isinstance(policy, dict) or set(policy) - {"mode", "max_tokens", "compact_roster", "adaptive", "expanded_max_tokens"}
                or policy.get("mode") not in {"full", "head", "mentions"}
                or type(policy.get("max_tokens")) is not int or not 100 <= policy["max_tokens"] <= 16000
                or type(policy.get("compact_roster", False)) is not bool
                or type(policy.get("adaptive", False)) is not bool):
            raise ValueError("text_selection: mode full/head/mentions, max_tokens 100–16000, compact_roster true/false")
        if policy.get("adaptive") and (policy["mode"] != "mentions"
                or type(policy.get("expanded_max_tokens", 6000)) is not int
                or not policy["max_tokens"] <= policy.get("expanded_max_tokens", 6000) <= 6000):
            raise ValueError("Расширенный контекст: режим mentions и expanded_max_tokens от базового лимита до 6000")
        cap = config["budget_limit_usd"]
        if cap is not None and (isinstance(cap, bool) or not isinstance(cap, (int, float))
                                or not math.isfinite(cap) or cap <= 0):
            raise ValueError("Общий бюджет корпуса должен быть положительным числом или null")
        if config["aspect"] not in {"public_positions", "public_image"}:
            raise ValueError("aspect: public_positions или public_image")
        if not isinstance(config["scope"], str) or len(config["scope"]) > 6000:
            raise ValueError("Описание пространства должно быть строкой до 6000 символов")
        for field in ("objects", "languages", "country_terms", "scales"):
            if not isinstance(config[field], list):
                raise ValueError(f"{field}: требуется список")
        ids = set()
        for actor in config["objects"]:
            if (not isinstance(actor, dict) or not isinstance(actor.get("name"), str)
                    or not actor["name"].strip() or not re.fullmatch(r"[A-Za-z0-9_-]{1,120}", str(actor.get("id", "")))
                    or actor["id"] in ids):
                raise ValueError("Объекты должны иметь уникальные id и непустые имена")
            ids.add(actor["id"])
            if not isinstance(actor.get("aliases", []), list) or any(not isinstance(x, str) for x in actor.get("aliases", [])):
                raise ValueError("Варианты имени должны быть списком строк")
        if len(ids) > 200:
            raise ValueError("Поддерживается до 200 объектов в одном исследовании")
        for field in ("languages", "country_terms"):
            if any(not isinstance(x, str) or not x.strip() for x in config[field]):
                raise ValueError(f"{field}: требуются непустые строки")
        for field in ("date_from", "date_to"):
            if config[field]:
                date.fromisoformat(config[field])
        if config["date_from"] and config["date_to"] and config["date_from"] > config["date_to"]:
            raise ValueError("Начало периода позже его окончания")
        for stage in STAGES:
            if not isinstance(config["routes"].get(stage), str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,150}", config["routes"][stage]):
                raise ValueError("Задайте допустимую модель для каждого этапа")
        if not isinstance(config["scales_approved"], bool):
            raise ValueError("scales_approved должен быть true или false")
        scale_ids = set()
        if len(config["scales"]) > 80:
            raise ValueError("Для одного решения поддерживается до 80 самостоятельных признаков")
        for scale in config["scales"]:
            if (not isinstance(scale, dict) or not re.fullmatch(r"[A-Za-z0-9_-]{1,120}", str(scale.get("id", "")))
                    or scale["id"] in scale_ids or not scale.get("name") or not scale.get("definition")):
                raise ValueError("Признаки должны иметь уникальные id, название и определение")
            scale_ids.add(scale["id"])
            if config["scales_approved"] and not all(isinstance(scale.get(k), str) and scale[k].strip()
                                                      for k in ("anchor0", "anchor3", "anchor5")):
                raise ValueError("Перед согласованием укажите наблюдаемые критерии 0, 3 и 5 каждой шкалы")
        if config["scales_approved"] and not config["scales"]:
            raise ValueError("Нельзя согласовать пустой словарь")
        # Reject unknown/invalid rates before any potential model request.
        for model in config["routes"].values():
            estimate_cost(model, 1, 1, pricing=config["pricing"])
        return config

    def _recipe(self, stage, config):
        base = {key: config.get(key) for key in
                ("scope", "objects", "languages", "date_from", "date_to", "country_terms")}
        selection = config.get("text_selection", {"mode": "full"})
        if selection["mode"] != "full" or selection.get("compact_roster"):
            # Missing/default full selection preserves existing recipes and paid
            # receipts; any bounded policy has its own downstream lineage.
            stable_selection = {k: v for k, v in selection.items() if k not in {"adaptive", "expanded_max_tokens"}}
            if selection.get("adaptive"):
                stable_selection.update(adaptive=True, expanded_max_tokens=selection.get("expanded_max_tokens", 6000),
                                        context_policy_version=CONTEXT_POLICY_VERSION)
            base["text_selection"] = {**stable_selection, "version": EXCERPT_POLICY_VERSION,
                                      "instructions": EXCERPT_INSTRUCTIONS}
            if selection["mode"] != "full":
                encoding = offline_encoding(self.runtime / "tokenizer_cache")
                base["text_selection"]["tokenizer"] = encoding.name if encoding else "utf8-byte-upper-bound"
        if stage != "relevance":
            base["aspect"] = config["aspect"]
            base["upstream_relevance"] = self._recipe("relevance", config)
        if stage in {"describe", "score"}:
            base["upstream_extract"] = self._recipe("extract", config)
        if stage == "score":
            base.update(scales=config["scales"], approved=config["scales_approved"],
                        rubric_version=config.get("rubric_version", "1"))
        return digest({"version": VERSION, "stage": stage, "config": base,
                       "task_contract": {"instructions": COMMON + PROMPTS[stage], "schema": SCHEMAS[stage],
                                         "identity_policy": IDENTITY_POLICY_VERSION,
                                         **({"evidence_policy": EVIDENCE_POLICY_VERSION} if stage != "relevance" else {}),
                                         **({"rubric_policy": RUBRIC_POLICY_VERSION} if stage in {"describe", "score"} else {})},
                       "model": config["routes"][stage], "max_output_tokens": MAX_OUTPUT[stage]})

    def _task_config(self, stage, config):
        keys = ["scope", "aspect", "objects", "languages", "date_from", "date_to", "country_terms"]
        if stage == "score":
            keys += ["scales"]
        return {key: config[key] for key in keys}

    def import_csv(self, path, title="", config=None, mapping=None):
        source = Path(path).expanduser().resolve()
        if source.suffix.lower() != ".csv" or not source.is_file():
            raise ValueError("Выберите существующий локальный CSV-файл")
        profile = inspect_csv(source)
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        if profile.get("source_sha256") != source_hash:
            raise ValueError("Исходный файл изменился во время проверки; повторите импорт")
        identifier = "corpus-" + digest({"source": source_hash, "mapping": mapping,
                                         "version": VERSION})[:20]
        with self.connection() as db:
            if db.execute("SELECT 1 FROM corpora WHERE id=?", (identifier,)).fetchone():
                return self.get_corpus(identifier)
        configuration = self._config(config)
        archive = self.runtime / "originals"
        archive.mkdir(exist_ok=True)
        snapshot = archive / (source_hash + ".csv")
        if not snapshot.exists():
            pending = snapshot.with_suffix("." + uuid.uuid4().hex + ".tmp")
            shutil.copyfile(source, pending)
            if hashlib.sha256(pending.read_bytes()).hexdigest() != source_hash:
                pending.unlink()
                raise ValueError("Исходный файл изменился во время импорта; повторите импорт")
            pending.replace(snapshot)
        profile.update(import_version=VERSION, source_sha256=source_hash)
        with self.connection() as db:
            db.execute("INSERT INTO corpora VALUES(?,?,?,?,?,?,?)", (identifier, title or source.stem,
                       now(), source.name, source_hash, packed(profile), packed(configuration)))
            representatives = {}
            for item in iter_documents(snapshot, mapping=mapping):
                document_id = "doc-" + digest([source_hash, item["source_row"], mapping, VERSION])[:28]
                text = item.get("text", "")
                fingerprint = exact_fingerprint(text) if text.strip() else ""
                duplicate = representatives.get(fingerprint) if fingerprint else None
                status = "empty" if not fingerprint else "exact_duplicate" if duplicate else "unique"
                item.update(id=document_id, actor_candidates=candidates(text, configuration["objects"]),
                            published_at_raw=item.get("published_at", ""))
                item["published_at"] = publication_date(item["published_at_raw"])
                db.execute("INSERT INTO documents VALUES(?,?,?,?,?,?,?,?)",
                           (document_id, identifier, item["source_row"], packed(item), text, fingerprint, status, duplicate))
                if fingerprint and not duplicate:
                    representatives[fingerprint] = document_id
            # Only compact shingles are retained by the candidate finder; raw
            # metadata and all document texts need not coexist in Python memory.
            unique_docs = ({"id": row[0], "text": row[1]} for row in db.execute(
                "SELECT id,text FROM documents WHERE corpus_id=? AND status='unique' ORDER BY source_row", (identifier,)))
            similar = near_duplicate_pairs(unique_docs)
            db.executemany("INSERT OR IGNORE INTO similar VALUES(?,?,?,?)",
                           [(identifier, x["left_id"], x["right_id"], packed(x)) for x in similar])
        return self.get_corpus(identifier)

    def list_corpora(self):
        with self.connection() as db:
            identifiers = [row[0] for row in db.execute("SELECT id FROM corpora ORDER BY created_at DESC")]
        return [self.get_corpus(identifier) for identifier in identifiers]

    def _row(self, identifier):
        with self.connection() as db:
            row = db.execute("SELECT * FROM corpora WHERE id=?", (identifier,)).fetchone()
        if row is None:
            raise FileNotFoundError(identifier)
        result = dict(row)
        result["profile"], result["config"] = json.loads(result["profile"]), json.loads(result["config"])
        return result

    def _facts(self, identifier, config, actor_id=None):
        with self.connection() as db:
            rows = db.execute("""SELECT f.data FROM facts f JOIN jobs j ON f.job_id=j.id
                WHERE f.corpus_id=? AND j.status='done' AND j.recipe=? ORDER BY f.id""",
                              (identifier, self._recipe("extract", config))).fetchall()
        facts = [json.loads(row[0]) for row in rows]
        with self.connection() as db:
            reviews = db.execute("SELECT * FROM fact_reviews WHERE corpus_id=? ORDER BY created_at,id", (identifier,)).fetchall()
        annotations = {}
        for row in reviews:
            annotations.setdefault(row["fact_id"], {}).update(json.loads(row["data"]))
            annotations[row["fact_id"]].update(review_id=row["id"], review_policy=row["policy"])
        facts = [dict(fact, **annotations.get(fact["id"], {})) for fact in facts]
        return [fact for fact in facts if actor_id is None or fact["actor_id"] == actor_id]

    def review_facts(self, identifier, decisions, *, reviewer, policy):
        """Append local, auditable annotations without modifying model receipts."""
        if not reviewer or not policy or not isinstance(decisions, list):
            raise ValueError("Укажите рецензента, версию правил и список решений")
        corpus = self._row(identifier)
        active = {fact["id"] for fact in self._facts(identifier, corpus["config"])}
        allowed = {"kind", "attribution_basis", "temporal_status", "measurement_status", "review_status", "review_note", "event_date", "date_evidence"}
        count = 0
        with self.connection() as db:
            for decision in decisions:
                if not isinstance(decision, dict) or decision.get("fact_id") not in active:
                    raise ValueError("Решения относятся только к активным свидетельствам корпуса")
                annotation = decision.get("annotation")
                if not isinstance(annotation, dict) or not annotation or set(annotation) - allowed or not annotation.get("review_note"):
                    raise ValueError("Укажите допустимую аннотацию и основание проверки")
                if "kind" in annotation and annotation["kind"] not in {"public_statement", "action", "attributed_image", "institutional_context"}:
                    raise ValueError("Неизвестный тип свидетельства")
                if "event_date" in annotation and annotation["event_date"] is not None:
                    raise ValueError("Локальная проверка может снять дату, но не выдумывать новую")
                identifier_review = "review-" + digest([identifier, decision["fact_id"], reviewer, policy, annotation])[:28]
                count += db.execute("INSERT OR IGNORE INTO fact_reviews VALUES(?,?,?,?,?,?,?)",
                                    (identifier_review, identifier, decision["fact_id"], now(), reviewer, policy, packed(annotation))).rowcount
        return {"annotations_added": count, "api_calls": 0, "raw_model_receipts_modified": False}

    def review_draft_scales(self, identifier, corrections, *, reviewer):
        """Edit unapproved rubric text; original API proposals stay in job receipts."""
        if not reviewer or not isinstance(corrections, list):
            raise ValueError("Укажите рецензента и список правок черновика")
        corpus = self._row(identifier)
        count = 0
        with self.connection() as db:
            for correction in corrections:
                row = db.execute("""SELECT s.* FROM scales s JOIN jobs j ON s.job_id=j.id
                    WHERE s.corpus_id=? AND s.id=? AND j.status='done' AND j.recipe=?""",
                                 (identifier, correction.get("scale_id"), self._recipe("describe", corpus["config"]))).fetchone()
                if row is None or not correction.get("review_note"):
                    raise ValueError("Нужен активный черновик и объяснение правки")
                value = json.loads(row["data"])
                if not value.get("draft"):
                    raise ValueError("Этот редактор изменяет только неутверждённые предложения")
                fields = correction.get("anchors", {})
                if not fields or set(fields) - {"anchor0", "anchor3", "anchor5"} or any(not isinstance(v, str) or not v.strip() for v in fields.values()):
                    raise ValueError("Допустимы непустые критерии 0/3/5")
                value.update(fields)
                if intensity_rubric_issues(value):
                    raise ValueError("Критерий выраженности не может измерять повторяемость")
                value.update(rubric_reviewer=reviewer, rubric_review_note=correction["review_note"], approved=False)
                db.execute("UPDATE scales SET data=? WHERE corpus_id=? AND job_id=? AND id=?", (packed(value), identifier, row["job_id"], row["id"]))
                count += 1
        return {"drafts_corrected": count, "api_calls": 0, "approved": False, "raw_model_receipts_modified": False}

    def _ratings(self, identifier, config):
        """Only ratings tied to the exact current evidence snapshot are active."""
        facts = self._facts(identifier, config)
        snapshots = {actor["id"]: digest([fact for fact in facts if fact["actor_id"] == actor["id"]])
                     for actor in config["objects"]}
        with self.connection() as db:
            rows = db.execute("""SELECT r.data FROM ratings r JOIN jobs j ON r.job_id=j.id
                WHERE r.corpus_id=? AND j.status='done' AND j.recipe=? ORDER BY r.actor_id,r.scale_id""",
                              (identifier, self._recipe("score", config))).fetchall()
        return [value for value in (json.loads(row[0]) for row in rows)
                if value.get("evidence_snapshot_sha") == snapshots.get(value["actor_id"])]

    def get_corpus(self, identifier):
        result = self._row(identifier)
        with self.connection() as db:
            statuses = dict(db.execute("SELECT status,COUNT(*) FROM documents WHERE corpus_id=? GROUP BY status", (identifier,)))
            jobs = dict(db.execute("SELECT status,COUNT(*) FROM jobs WHERE corpus_id=? GROUP BY status", (identifier,)))
            usage = dict(db.execute("""SELECT COALESCE(SUM(input_tokens),0) input_tokens,
                COALESCE(SUM(output_tokens),0) output_tokens, COALESCE(SUM(cost_usd),0) estimated_usd,
                COALESCE(SUM(CASE WHEN status IN ('running','uncertain') THEN reserved_usd ELSE 0 END),0) uncertain_reserved_usd
                FROM jobs WHERE corpus_id=?""", (identifier,)).fetchone())
            counts = {"documents": sum(statuses.values()), "unique_documents": statuses.get("unique", 0),
                      "exact_duplicates": statuses.get("exact_duplicate", 0), "empty_documents": statuses.get("empty", 0)}
            counts["near_duplicate_candidates"] = db.execute("SELECT COUNT(*) FROM similar WHERE corpus_id=?", (identifier,)).fetchone()[0]
            draft_rows = db.execute("""SELECT s.data FROM scales s JOIN jobs j ON s.job_id=j.id
                WHERE s.corpus_id=? AND j.status='done' AND j.recipe=? ORDER BY j.finished_at DESC,s.id""",
                                   (identifier, self._recipe("describe", result["config"]))).fetchall()
            latest = db.execute("SELECT id,stage,status,error FROM jobs WHERE corpus_id=? ORDER BY created_at DESC LIMIT 15", (identifier,)).fetchall()
        counts["facts"] = len(self._facts(identifier, result["config"]))
        counts["ratings"] = len(self._ratings(identifier, result["config"]))
        draft_scales = [json.loads(row[0]) for row in draft_rows]
        counts["scales"] = len(draft_scales)
        result.update(counts=counts, jobs=jobs, usage=usage, draft_scales=draft_scales,
                      recent_jobs=[dict(x) for x in latest], canonical_study_modified=False,
                      limitations=["Почти дубли являются кандидатами на проверку и не удаляются автоматически.",
                                   "Объединение независимых событий требует содержательной проверки; число документов не является силой свойства."])
        return result

    def documents(self, identifier, limit=50, offset=0, status=None):
        corpus = self._row(identifier)
        limit, offset = int(limit), int(offset)
        if not 1 <= limit <= 200 or offset < 0:
            raise ValueError("limit: 1–200; offset: неотрицательное число")
        where, parameters = "corpus_id=?", [identifier]
        if status:
            if status not in {"unique", "exact_duplicate", "empty"}:
                raise ValueError("Неизвестный статус документа")
            where += " AND status=?"
            parameters.append(status)
        with self.connection() as db:
            total = db.execute("SELECT COUNT(*) FROM documents WHERE " + where, parameters).fetchone()[0]
            rows = db.execute("SELECT * FROM documents WHERE " + where + " ORDER BY source_row LIMIT ? OFFSET ?",
                              parameters + [limit, offset]).fetchall()
        documents = []
        for row in rows:
            value = json.loads(row["data"])
            value.update(status=row["status"], duplicate_of=row["duplicate_of"],
                         actor_candidates=candidates(value["text"], corpus["config"]["objects"]))
            documents.append(value)
        return {"documents": documents, "total": total, "limit": limit, "offset": offset}

    def configure(self, identifier, config):
        if self.run_lock.locked():
            raise ValueError("Сначала дождитесь завершения текущего запуска")
        current = self._row(identifier)
        merged = dict(current["config"])
        merged.update(config)
        # Editing the dictionary invalidates its approval unless explicitly approved in this request.
        if "scales" in config and config["scales"] != current["config"]["scales"]:
            merged["scales_approved"] = config.get("scales_approved", False)
        configuration = self._config(merged)
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM jobs WHERE corpus_id=? AND status='running'", (identifier,)).fetchone():
                raise ValueError("Сначала дождитесь завершения работающего задания")
            db.execute("UPDATE corpora SET config=? WHERE id=?", (packed(configuration), identifier))
            for stage in STAGES:
                db.execute("UPDATE jobs SET status='obsolete' WHERE corpus_id=? AND stage=? AND status='pending' AND recipe!=?",
                           (identifier, stage, self._recipe(stage, configuration)))
        return self.get_corpus(identifier)

    @staticmethod
    def _rule_exclusion(document, config):
        terms = config.get("country_terms", [])
        if terms and not any(re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", document["text"], re.IGNORECASE) for term in terms):
            return "country_not_matched"
        raw_date = str(document.get("published_at", ""))[:10]
        try:
            dated = date.fromisoformat(raw_date).isoformat()
        except ValueError:
            return None  # An unknown date is reviewed by the model, never fabricated.
        if config.get("date_from") and dated < config["date_from"] or config.get("date_to") and dated > config["date_to"]:
            return "outside_period"
        return None

    def prepare(self, identifier, stage="relevance", limit=100, *, document_ids=None, fact_ids=None):
        if stage not in STAGES or not 1 <= int(limit) <= 5000:
            raise ValueError("Неизвестный этап или limit вне 1–5000")
        corpus, created, excluded = self._row(identifier), 0, 0
        config = corpus["config"]
        if fact_ids is not None and (stage != "describe" or not isinstance(fact_ids, (list, tuple, set))
                                     or not fact_ids or any(not isinstance(x, str) for x in fact_ids)):
            raise ValueError("Выбор свидетельств поддерживается только для describe; задайте непустые ID")
        selected = None
        if document_ids is not None:
            if stage not in {"relevance", "extract"} or not isinstance(document_ids, (list, tuple, set)):
                raise ValueError("Выбор документов поддерживается для relevance/extract")
            selected = set(document_ids)
            if not selected or any(not isinstance(x, str) for x in selected):
                raise ValueError("Задайте непустой список ID документов")
            with self.connection() as db:
                known = {row[0] for row in db.execute("SELECT id FROM documents WHERE corpus_id=? AND status='unique'", (identifier,))}
            if not selected <= known:
                raise ValueError("Пилот может выбирать только уникальные документы этого корпуса")
        if not config["scope"].strip() or not config["objects"]:
            raise ValueError("Перед смысловой обработкой задайте пространство и список объектов")
        if stage == "score" and not config["scales_approved"]:
            raise ValueError("Перед оцениванием согласуйте определения и критерии шкал")
        recipe = self._recipe(stage, config)
        contexts = []
        if stage == "relevance":
            policy = config.get("text_selection", {"mode": "full"})
            encoding = offline_encoding(self.runtime / "tokenizer_cache") if policy["mode"] != "full" else None
            with self.connection() as db:
                rows = db.execute("SELECT data FROM documents WHERE corpus_id=? AND status='unique' ORDER BY source_row", (identifier,)).fetchall()
            documents = [json.loads(row[0]) for row in rows]
            actor_counts, eligible_count = {}, 0
            if policy.get("adaptive"):
                actor_counts = {obj["id"]: 0 for obj in config["objects"]}
                for document in documents:
                    if not self._rule_exclusion(document, config):
                        eligible_count += 1
                        for actor_id in actor_mentions(document["text"], config["objects"]):
                            actor_counts[actor_id] += 1
            for document in documents:
                if selected is not None and document["id"] not in selected:
                    continue
                if self._rule_exclusion(document, config):
                    excluded += 1
                    continue
                if policy["mode"] != "full":
                    if policy.get("adaptive"):
                        selection = select_document_context(document, config["objects"], actor_counts, eligible_count,
                            {"base_max_tokens": policy["max_tokens"], "expanded_max_tokens": policy.get("expanded_max_tokens", 6000)}, encoding)
                    else:
                        selection = select_spans_with_metadata(document["text"], config["objects"],
                                                              mode=policy["mode"], max_tokens=policy["max_tokens"], encoding=encoding)
                    fragment = {key: document.get(key, "") for key in
                                ("id", "title", "url", "published_at", "source_name", "author", "language_hint")}
                    fragment["excerpts"] = [{key: span[key] for key in ("start", "end", "text")}
                                            for span in selection["spans"]]
                    fragment["text_selection"] = selection["metadata"]
                    contexts.append((f"{document['id']}:excerpts", {"document": fragment}))
                    continue
                for start, end, text in chunks(document["text"]):
                    fragment = {key: document.get(key, "") for key in
                                ("id", "title", "url", "published_at", "source_name", "author", "language_hint")}
                    fragment.update(text=text, chunk_start=start, chunk_end=end)
                    contexts.append((f"{document['id']}:{start}", {"document": fragment}))
        elif stage == "extract":
            with self.connection() as db:
                rows = db.execute("SELECT * FROM jobs WHERE corpus_id=? AND stage='relevance' AND status='done' AND recipe=? ORDER BY created_at",
                                  (identifier, self._recipe("relevance", config))).fetchall()
            for row in rows:
                result = json.loads(row["result"])
                if result.get("status") == "relevant":
                    context = json.loads(row["context"])
                    if selected is not None and context["document"]["id"] not in selected:
                        continue
                    with self.connection() as db:
                        original = db.execute("SELECT data FROM documents WHERE corpus_id=? AND id=?", (identifier, context["document"]["id"])).fetchone()
                    document = dict(context["document"], **source_context(json.loads(original[0])))
                    contexts.append((row["unit_id"], {"document": document,
                                                      "relevance": result, "relevance_job_id": row["id"]}))
        else:
            facts = self._facts(identifier, config)
            if not facts:
                raise ValueError("Сначала получите проверяемые фрагменты этапа extract")
            if stage == "describe":
                if fact_ids is not None and not set(fact_ids) <= {fact["id"] for fact in facts}:
                    raise ValueError("Выбирайте только активные свидетельства этого корпуса")
                facts = [fact for fact in facts if _eligible_fact(fact, config)
                         and (fact_ids is None or fact["id"] in set(fact_ids))]
                if not facts:
                    raise ValueError("Нет свидетельств выбранного режима измерения для словаря")
                # Hierarchical discovery: disjoint deterministic groups; no first-N truncation.
                for index in range(0, len(facts), 60):
                    group = facts[index:index + 60]
                    contexts.append(("facts-" + digest(group)[:20],
                                     {"facts": group, "facts_snapshot_sha": digest(group), "draft_only": True}))
            else:
                for actor in config["objects"]:
                    group = [fact for fact in facts if fact["actor_id"] == actor["id"]]
                    if group:
                        contexts.append((actor["id"], {"facts": group, "actor_id": actor["id"],
                                                       "facts_snapshot_sha": digest(group)}))
        with self.connection() as db:
            for unit, context in contexts:
                context["config"] = self._task_config(stage, config)
                if stage in {"relevance", "extract"} and config.get("text_selection", {}).get("compact_roster"):
                    context["config"]["objects"] = compact_objects(config["objects"], context["document"])
                payload = make_task(stage, context, model=config["routes"][stage], max_output_tokens=MAX_OUTPUT[stage])
                job_id = "job-" + digest({"corpus": identifier, "recipe": recipe, "unit": unit, "context": context})[:32]
                cursor = db.execute("""INSERT OR IGNORE INTO jobs
                    (id,corpus_id,stage,unit_id,recipe,context,payload,status,created_at)
                    VALUES(?,?,?,?,?,?,?,'pending',?)""",
                                    (job_id, identifier, stage, unit, recipe, packed(context), packed(payload), now()))
                created += cursor.rowcount
                if created >= int(limit):
                    break
            queue = db.execute("SELECT payload FROM jobs WHERE corpus_id=? AND status='pending'", (identifier,)).fetchall()
        upper = sum(estimate_cost(json.loads(row[0])["model"], payload_input_bound(json.loads(row[0])),
                                  json.loads(row[0])["max_output_tokens"], pricing=config["pricing"]) for row in queue)
        return {"created": created, "stage": stage, "jobs": self.get_corpus(identifier)["jobs"],
                "rule_exclusions": excluded, "total_estimated_upper_usd": round(upper, 6),
                "api_calls": 0}

    def _persist_result(self, db, identifier, job, result, context):
        stage = job["stage"]
        if stage == "extract":
            for fact in result["facts"]:
                document = context["document"]
                span = fact_span(document, fact)
                if span is None:
                    raise ValueError("Persisted evidence must fit one original source span")
                start = span["start"] + span["text"].index(fact["quote"])
                fact_id = "fact-" + digest([identifier, document["id"], start, fact["actor_id"], fact["quote"], fact["kind"], job["recipe"]])[:28]
                value = dict(fact, id=fact_id, document_id=document["id"], source_url=document.get("url", ""),
                             source_date=document.get("published_at", ""), quote_start=start,
                             quote_end=start + len(fact["quote"]), extraction_job_id=job["id"],
                             event_status="requires_review")
                db.execute("INSERT OR IGNORE INTO facts VALUES(?,?,?,?,?,?)", (fact_id, identifier, job["id"], fact["actor_id"], document["id"], packed(value)))
        elif stage == "describe":
            for scale in result["scales"]:
                scale_id = "trait-" + digest([job["id"], scale["id"]])[:20]
                value = dict(scale, id=scale_id, draft=True)
                db.execute("INSERT OR REPLACE INTO scales VALUES(?,?,?,?)", (scale_id, identifier, job["id"], packed(value)))
        elif stage == "score":
            for rating in result["ratings"]:
                value = dict(rating, provisional=True, scoring_job_id=job["id"],
                             evidence_snapshot_sha=context["facts_snapshot_sha"])
                db.execute("INSERT OR REPLACE INTO ratings VALUES(?,?,?,?,?)",
                           (identifier, job["id"], rating["actor_id"], rating["scale_id"], packed(value)))

    def run(self, identifier, max_jobs=5, budget_usd=1.0, *, client=None):
        if isinstance(max_jobs, bool) or isinstance(budget_usd, bool):
            raise ValueError("Лимит и бюджет должны быть числами")
        max_jobs, budget_usd = int(max_jobs), float(budget_usd)
        if not 1 <= max_jobs <= 1000 or not math.isfinite(budget_usd) or budget_usd <= 0:
            raise ValueError("Задайте лимит 1–1000 заданий и положительный бюджет этого запуска")
        if not self.run_lock.acquire(blocking=False):
            raise ValueError("Обработка корпуса уже выполняется; дождитесь её завершения")
        owned_client = None
        try:
            if client is None:
                key = os.environ.get("OPENAI_API_KEY", "")
                if not key:
                    raise ValueError("Задайте OPENAI_API_KEY в локальном .env или окружении сервера")
                client = owned_client = OpenAIResponsesHTTP(key)
            return self._run(identifier, max_jobs, budget_usd, client)
        finally:
            try:
                if owned_client is not None:
                    owned_client.close()
            finally:
                self.run_lock.release()

    def _run(self, identifier, max_jobs, budget_usd, client):
        corpus = self._row(identifier)
        config = corpus["config"]
        run_id, used, attempted = "run-" + uuid.uuid4().hex, 0.0, 0
        summary = {"run_id": run_id, "completed": 0, "review": 0, "failed": 0, "uncertain": 0,
                   "budget_usd": budget_usd, "stopped_reason": "queue_empty"}
        with self.connection() as db:
            db.execute("INSERT INTO runs VALUES(?,?,?,?,?,NULL)", (run_id, identifier, now(), budget_usd, max_jobs))
        while attempted < max_jobs:
            with self.connection() as db:
                db.execute("BEGIN IMMEDIATE")
                latest_config = json.loads(db.execute("SELECT config FROM corpora WHERE id=?", (identifier,)).fetchone()[0])
                if latest_config != config:
                    summary["stopped_reason"] = "configuration_changed"
                    break
                row = db.execute("SELECT * FROM jobs WHERE corpus_id=? AND status='pending' ORDER BY created_at,id LIMIT 1", (identifier,)).fetchone()
                if row is None:
                    break
                job = dict(row)
                if job["recipe"] != self._recipe(job["stage"], config):
                    db.execute("UPDATE jobs SET status='obsolete' WHERE id=?", (job["id"],))
                    continue
                context, payload = json.loads(job["context"]), json.loads(job["payload"])
                if job["stage"] == "score" and digest(self._facts(identifier, config, context["actor_id"])) != context["facts_snapshot_sha"]:
                    db.execute("UPDATE jobs SET status='obsolete' WHERE id=?", (job["id"],))
                    continue
                reservation = estimate_cost(payload["model"], payload_input_bound(payload), payload["max_output_tokens"], pricing=config["pricing"])
                if used + reservation > budget_usd + 1e-12:
                    summary["stopped_reason"] = "budget_reservation"
                    break
                if config.get("budget_limit_usd") is not None:
                    committed = db.execute("""SELECT COALESCE(SUM(cost_usd),0) +
                        COALESCE(SUM(CASE WHEN status IN ('running','uncertain') THEN reserved_usd ELSE 0 END),0)
                        FROM jobs WHERE corpus_id=?""", (identifier,)).fetchone()[0]
                    if committed + reservation > config["budget_limit_usd"] + 1e-12:
                        summary["stopped_reason"] = "corpus_budget_reservation"
                        break
                db.execute("UPDATE jobs SET status='running',started_at=?,reserved_usd=? WHERE id=?", (now(), reservation, job["id"]))
            attempted += 1
            response, usage, charge, status, error = None, {}, 0.0, "done", None
            try:
                response = client.call(payload)
                usage = response["usage"]
                charge = estimate_cost(payload["model"], usage["input_tokens"], usage["output_tokens"],
                                       cached_input_tokens=usage.get("input_tokens_details", {}).get("cached_tokens", 0), pricing=config["pricing"])
                data = (normalize_discovery(response["data"], context, validate_result) if job["stage"] == "describe"
                        else normalize_extraction(response["data"], context, validate_result) if job["stage"] == "extract"
                        else validate_result(job["stage"], response["data"], context))
                if job["stage"] == "relevance":
                    data = ground_relevance(data, context)
                    allowed = {LANGUAGES.get(x.casefold(), x.casefold()) for x in config["languages"]}
                    actual = LANGUAGES.get(data.get("language", "").casefold(), data.get("language", "").casefold())
                    if allowed and actual not in allowed:
                        data = dict(data, status="review" if actual in {"", "unknown", "und", "mixed"} else "irrelevant",
                                    reason="Язык требует проверки" if actual in {"", "unknown", "und", "mixed"} else "Язык вне разрешённого списка")
                    if data["status"] == "review":
                        status = "review"
            except ModelResponseError as exc:
                usage = getattr(exc, "usage", None) or {}
                known_http_rejection = (getattr(exc, "error_code", None) == "http_error"
                                        and isinstance(getattr(exc, "status", None), int)
                                        and 400 <= exc.status < 500)
                status = ("uncertain" if getattr(exc, "uncertain", False) or (not usage and not known_http_rejection)
                          else "review" if usage else "error")
                error = str(exc) + (f" HTTP {exc.status}" if isinstance(getattr(exc, "status", None), int) else "")
                if usage and "input_tokens" in usage and "output_tokens" in usage:
                    charge = estimate_cost(payload["model"], usage["input_tokens"], usage["output_tokens"],
                                           cached_input_tokens=usage.get("input_tokens_details", {}).get("cached_tokens", 0), pricing=config["pricing"])
                response = {"response_id": getattr(exc, "response_id", None), "data": getattr(exc, "data", None)}
            except (ValueError, KeyError, TypeError):
                status, error = "review", "Ответ требует проверки структуры, цитат или атрибуции"
            except Exception:
                # Unexpected transport/worker failure is not a political unknown.
                status, error = "uncertain", "Запрос прерван; проверьте оплату перед повторным запуском"
            used += reservation if status == "uncertain" else charge
            with self.connection() as db:
                result = response.get("data") if response else None
                if status == "done":
                    try:
                        self._persist_result(db, identifier, job, data, context)
                        result = data
                        if job["stage"] in {"extract", "describe"}:
                            result = dict(data, raw_model_result=response["data"])
                    except (ValueError, KeyError, TypeError):
                        status, error = "review", "Результат не прошёл проверку связей с источниками"
                elif job["stage"] == "relevance" and status == "review" and error is None:
                    result = data
                db.execute("""UPDATE jobs SET status=?,finished_at=?,result=?,error=?,response_id=?,
                    input_tokens=?,output_tokens=?,cost_usd=? WHERE id=?""",
                           (status, now(), packed(result) if result is not None else None, error,
                            response.get("response_id") if response else None, usage.get("input_tokens", 0),
                            usage.get("output_tokens", 0), charge, job["id"]))
            summary[{"done": "completed", "review": "review", "error": "failed", "uncertain": "uncertain"}[status]] += 1
            if status == "uncertain":
                summary["stopped_reason"] = "uncertain_request"
                break
        else:
            summary["stopped_reason"] = "job_limit"
        summary.update(attempted=attempted, accounted_or_reserved_usd=round(used, 6))
        with self.connection() as db:
            db.execute("UPDATE runs SET data=? WHERE id=?", (packed(summary), run_id))
        return summary

    def retry_errors(self, identifier):
        """Explicit retry for known failures only; uncertain charges require reconciliation."""
        self._row(identifier)
        with self.connection() as db:
            cursor = db.execute("UPDATE jobs SET status='pending',error=NULL WHERE corpus_id=? AND status='error' AND cost_usd=0", (identifier,))
        return {"retry_queued": cursor.rowcount, "uncertain_retried": False}

    def recover_interrupted(self, identifier):
        """Explicit crash recovery: never silently replay possibly billed requests."""
        self._row(identifier)
        if self.run_lock.locked():
            raise ValueError("Запуск ещё работает; дождитесь его завершения")
        with self.connection() as db:
            cursor = db.execute("""UPDATE jobs SET status='uncertain',error=?,finished_at=?
                WHERE corpus_id=? AND status='running'""",
                                ("Прерванное задание: проверьте оплату; автоматически не повторяется", now(), identifier))
        return {"marked_uncertain": cursor.rowcount, "api_calls": 0}

    def export(self, identifier):
        corpus = self.get_corpus(identifier)
        config = corpus["config"]
        facts = self._facts(identifier, config)
        ratings = self._ratings(identifier, config)
        with self.connection() as db:
            similar = [json.loads(row[0]) for row in db.execute("SELECT data FROM similar WHERE corpus_id=?", (identifier,))]
        values = {"corpus_summary.json": corpus, "facts.json": facts,
                  "draft_scales.json": corpus["draft_scales"], "ratings.json": ratings,
                  "near_duplicate_candidates.json": similar}
        snapshot = digest(values)
        values["provenance.json"] = {"corpus_id": identifier, "corpus_snapshot_sha": snapshot,
                                      "evidence_snapshot_sha": digest(facts),
                                      "source_sha256": corpus["source_sha256"], "pipeline_version": VERSION,
                                      "recipes": {stage: self._recipe(stage, config) for stage in STAGES},
                                      "ratings_are_provisional": True, "canonical_study_modified": False}
        directory = self.runtime / "exports" / identifier / snapshot[:20]
        if not directory.exists():
            pending = directory.with_name(directory.name + "." + uuid.uuid4().hex + ".pending")
            pending.mkdir(parents=True)
            for name, value in values.items():
                (pending / name).write_text(packed(value) + "\n", encoding="utf-8")
            pending.rename(directory)
        return {"directory": str(directory), "corpus_id": identifier,
                "files": [{"name": name, "path": str(directory / name)} for name in values],
                "corpus_snapshot_sha": snapshot}
