"""Private, allowlisted HTTP billing receipts; one request and no retries.

Payloads, generated prose, HTTP error messages, credentials and general headers
are never written. The original payload digest matches service.digest(payload).
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import threading

import httpx

from .models import ModelResponseError, OpenAIResponsesHTTP


_APPEND_LOCK = threading.Lock()
_RATE_HEADERS = (
    "x-ratelimit-limit-requests", "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-requests", "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens", "retry-after",
)


def _digest(payload):
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _label(value):
    return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:/-]{1,180}", value) else None


def _usage(value):
    if not isinstance(value, dict):
        return None
    result = {key: value[key] for key in ("input_tokens", "output_tokens", "total_tokens")
              if type(value.get(key)) is int and value[key] >= 0}
    for key, fields in (("input_tokens_details", ("cached_tokens",)),
                        ("output_tokens_details", ("reasoning_tokens",))):
        details = value.get(key)
        if isinstance(details, dict):
            result[key] = {field: details[field] for field in fields
                           if type(details.get(field)) is int and details[field] >= 0}
    return result or None


class ReceiptingClient:
    """OpenAIResponsesHTTP with standard-tier requests and safe JSONL receipts.

    transport/base_url are injection points for offline HTTP tests. Each worker
    should own one instance. A shared process-wide lock serializes appends from
    all instances. No automatic network retry, including after a receipt error.
    """
    def __init__(self, api_key, ledger_path, *, transport=None,
                 base_url="https://api.openai.com/v1", timeout=120.):
        self.ledger_path = Path(ledger_path)
        self._state = threading.local()
        self._http = httpx.Client(transport=transport, timeout=timeout, follow_redirects=False,
                                 event_hooks={"response": [self._response_hook]})
        try:
            self._delegate = OpenAIResponsesHTTP(api_key, client=self._http, base_url=base_url, timeout=timeout)
        except Exception:
            self._http.close()
            raise

    def _response_hook(self, response):
        state = getattr(self._state, "receipt", None)
        if state is None:
            return
        state["http_status"] = response.status_code
        state["rate_headers"] = {key: response.headers[key][:200] for key in _RATE_HEADERS if key in response.headers}
        # Reading here consumes the existing response, never issues another HTTP
        # request. The delegate can then parse the cached bytes normally.
        response.read()
        try:
            body = response.json()
        except (ValueError, UnicodeError):
            return
        if not isinstance(body, dict):
            return
        state.update(response_id=_label(body.get("id")), model=_label(body.get("model")) or state["model"],
                     usage=_usage(body.get("usage")), actual_service_tier=_label(body.get("service_tier")))
        error = body.get("error")
        if isinstance(error, dict):
            state["provider_error_code"] = _label(error.get("code"))
            state["provider_error_type"] = _label(error.get("type"))

    def _append(self, receipt):
        with _APPEND_LOCK:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with self.ledger_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(receipt, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")

    def call(self, payload):
        wire = copy.deepcopy(payload)
        wire["service_tier"] = "default"
        receipt = {"timestamp": datetime.now(timezone.utc).isoformat(),
                   "request_sha256": _digest(payload), "wire_request_sha256": _digest(wire),
                   "response_id": None, "model": _label(payload.get("model")), "usage": None,
                   "requested_service_tier": "default", "actual_service_tier": None,
                   "http_status": None, "rate_headers": {},
                   "provider_error_code": None, "provider_error_type": None}
        self._state.receipt = receipt
        try:
            result = self._delegate.call(wire)
        except Exception as exc:
            if isinstance(exc, ModelResponseError):
                receipt["response_id"] = _label(exc.response_id) or receipt["response_id"]
                receipt["model"] = _label(exc.model) or receipt["model"]
                receipt["usage"] = _usage(exc.usage) or receipt["usage"]
                receipt["adapter_error_code"] = _label(exc.error_code)
            else:
                receipt["adapter_error_code"] = type(exc).__name__
            try:
                self._append(receipt)
            except OSError as logging_error:
                # Preserve the original billing exception and its usage even
                # if the disk fails; never mask it or replay its request.
                exc.receipt_error_type = type(logging_error).__name__
            raise
        else:
            receipt["response_id"] = _label(result.get("response_id")) or receipt["response_id"]
            receipt["model"] = _label(result.get("model")) or receipt["model"]
            receipt["usage"] = _usage(result.get("usage")) or receipt["usage"]
            # On a successful HTTP call a failed append propagates; the service
            # conservatively retains an uncertain reservation and stops.
            self._append(receipt)
            return result
        finally:
            self._state.receipt = None

    def close(self):
        try:
            self._delegate.close()
        finally:
            self._http.close()
