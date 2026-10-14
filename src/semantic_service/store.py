"""Durable study records and explicit, version-bound review gates."""
from __future__ import annotations

import copy
import hashlib
import json
import re
import threading
import uuid
from datetime import date, datetime, timezone
from pathlib import Path


BUSY = {"discovering", "generating_scales", "running"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def ratings_basis(study: dict) -> str:
    """Bind reusable measurements to the exact space, period and approved rubric."""
    return fingerprint({key: study.get(key) for key in
                        ("scope", "aspect", "date_from", "date_to", "objects", "scales")})


def validate_objects(objects: list[dict]) -> list[dict]:
    if not isinstance(objects, list) or not 1 <= len(objects) <= 200:
        raise ValueError("Нужен список от 1 до 200 объектов")
    result, names, ids = [], set(), set()
    for index, raw in enumerate(objects):
        item = dict(raw)
        name = str(item.get("name", "")).strip()
        if not name or len(name) > 250 or name.casefold() in names:
            raise ValueError("Имена объектов должны быть непустыми и уникальными")
        identifier = str(item.get("id") or f"object-{index + 1}")
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", identifier) or identifier in ids:
            raise ValueError("Некорректный или повторный идентификатор объекта")
        item.update(id=identifier, name=name)
        names.add(name.casefold())
        ids.add(identifier)
        result.append(item)
    return result


def validate_scales(scales: list[dict]) -> list[dict]:
    if not isinstance(scales, list) or not 6 <= len(scales) <= 80:
        raise ValueError("Нужно от 6 до 80 шкал")
    result, ids, pairs = [], set(), set()
    for index, raw in enumerate(scales):
        item = dict(raw)
        identifier = str(item.get("id") or f"scale-{index + 1}")
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", identifier) or identifier in ids:
            raise ValueError("Некорректный или повторный идентификатор шкалы")
        for field in ("name", "positive", "negative", "definition"):
            item[field] = str(item.get(field, "")).strip()
            if not item[field] or len(item[field]) > 1800:
                raise ValueError(f"Заполните поле шкалы: {field}")
        pair = tuple(sorted((item["positive"].casefold(), item["negative"].casefold())))
        if pair[0] == pair[1] or pair in pairs:
            raise ValueError("Полюса должны различаться; дублирующиеся шкалы нужно объединить")
        item["id"] = identifier
        item.setdefault("aliases", [])
        item.setdefault("anchors", {"0": "Нет выраженности при достаточных сведениях", "3": "Устойчиво заметно", "5": "Определяющая черта публичного образа"})
        ids.add(identifier)
        pairs.add(pair)
        result.append(item)
    return result


class StudyStore:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()

    def directory(self, identifier: str) -> Path:
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", identifier):
            raise ValueError("Некорректный идентификатор исследования")
        return self.root / identifier

    def save(self, study: dict) -> dict:
        with self.lock:
            study = copy.deepcopy(study)
            study["updated_at"] = now()
            directory = self.directory(study["id"])
            directory.mkdir(parents=True, exist_ok=True)
            temp = directory / "study.json.tmp"
            temp.write_text(json.dumps(study, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            temp.replace(directory / "study.json")
            return study

    def get(self, identifier: str) -> dict:
        with self.lock:
            return json.loads((self.directory(identifier) / "study.json").read_text(encoding="utf-8"))

    def list(self) -> list[dict]:
        result = []
        for file in self.root.glob("*/study.json"):
            try:
                study = json.loads(file.read_text(encoding="utf-8"))
                result.append({key: study.get(key) for key in ("id", "title", "status", "updated_at", "scope")})
            except (ValueError, OSError):
                continue
        return sorted(result, key=lambda item: item.get("updated_at") or "", reverse=True)

    def create(self, payload: dict, identifier: str | None = None) -> dict:
        title = str(payload.get("title") or payload.get("scope") or "Новое исследование").strip()
        if not title or len(title) > 300:
            raise ValueError("Название должно содержать от 1 до 300 символов")
        target = int(payload.get("target_count") or 70)
        if not 12 <= target <= 120:
            raise ValueError("Для пилота выберите от 12 до 120 объектов")
        study = {"id": identifier or "study-" + uuid.uuid4().hex[:12], "title": title,
                 "scope": str(payload.get("scope") or title), "query": str(payload.get("query") or ""),
                 "aspect": str(payload.get("aspect") or "Публичный образ"),
                 "date_from": str(payload.get("date_from") or "2016-10-15"),
                 "date_to": str(payload.get("date_to") or "2026-10-15"),
                 "focus": str(payload.get("focus") or ""), "objects_text": str(payload.get("objects_text") or ""),
                 "target_count": target, "status": "draft", "objects": [], "scales": [],
                 "approvals": {"objects": {"approved": False}, "scales": {"approved": False}},
                 "progress": {"stage": "draft", "message": "Задайте область и объекты исследования", "current": 0, "total": 0},
                 "analysis": None, "artifacts": [], "error": None, "logs": [], "created_at": now(),
                 "measurement_mode": "llm_exploratory", "budget_limit_usd": 20.0,
                 "usage": {"calls": 0, "input_tokens": 0, "output_tokens": 0, "estimated_usd": 0.0}}
        try:
            start_date = date.fromisoformat(study["date_from"])
            end_date = date.fromisoformat(study["date_to"])
        except ValueError:
            raise ValueError("Даты должны быть в формате ГГГГ-ММ-ДД") from None
        if start_date > end_date:
            raise ValueError("Начало периода должно предшествовать окончанию")
        study["date_from"], study["date_to"] = start_date.isoformat(), end_date.isoformat()
        return self.save(study)

    def update(self, identifier: str, **fields) -> dict:
        with self.lock:
            study = self.get(identifier)
            study.update(fields)
            return self.save(study)

    def log(self, identifier: str, message: str, stage: str | None = None,
            current: int = 0, total: int = 0) -> dict:
        with self.lock:
            study = self.get(identifier)
            study["logs"] = (study.get("logs", []) + [{"at": now(), "message": message}])[-200:]
            if stage:
                study["progress"] = {"stage": stage, "message": message, "current": current, "total": total}
            return self.save(study)

    def edit(self, identifier: str, field: str, records: list[dict]) -> dict:
        with self.lock:
            study = self.get(identifier)
            if study["status"] in BUSY:
                raise ValueError("Дождитесь завершения текущего этапа")
            validator = validate_objects if field == "objects" else validate_scales
            validated = validator(records)
            if fingerprint(validated) == fingerprint(study[field]):
                return study
            study[field] = validated
            study["approvals"][field] = {"approved": False}
            if field == "objects":
                study["scales"] = []
                study["approvals"]["scales"] = {"approved": False}
            study.update(status="await_objects_approval" if field == "objects" else "await_scales_approval",
                         analysis=None, artifacts=[], error=None)
            study.pop("imported_ratings_basis", None)
            study.pop("imported_ratings_count", None)
            study.pop("ratings_cache", None)
            return self.save(study)

    def approve(self, identifier: str, field: str, actor: str = "user_interface") -> dict:
        with self.lock:
            study = self.get(identifier)
            if study["status"] in BUSY:
                raise ValueError("Дождитесь завершения текущего этапа")
            if field == "scales" and not self.is_approved(study, "objects"):
                raise ValueError("Сначала утвердите список объектов")
            (validate_objects if field == "objects" else validate_scales)(study[field])
            study["approvals"][field] = {"approved": True, "at": now(), "actor": actor,
                                         "fingerprint": fingerprint(study[field])}
            study.update(status="objects_approved" if field == "objects" else "ready", error=None)
            return self.save(study)

    @staticmethod
    def is_approved(study: dict, field: str) -> bool:
        receipt = study.get("approvals", {}).get(field, {})
        return receipt.get("approved") is True and receipt.get("fingerprint") == fingerprint(study[field])

    def require_approved(self, identifier: str) -> dict:
        study = self.get(identifier)
        if not self.is_approved(study, "objects") or not self.is_approved(study, "scales"):
            raise ValueError("До оценивания нужно утвердить список объектов и словарь шкал")
        return study
