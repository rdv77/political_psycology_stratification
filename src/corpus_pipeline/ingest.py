"""Read-only CSV ingestion and conservative duplicate-review candidates.

No source values are evaluated or coerced to numeric types. ``source_row`` is
the logical CSV record number (header = 1), not a physical newline number.
Mapping keys are canonical fields; values are original CSV header names or
``None`` to disable a field. ``text`` is accepted as an alias for ``message``.
The module needs only the Python 3.10 standard library.
"""

from __future__ import annotations

import codecs
import csv
import hashlib
import re
import threading
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


_FIELD_ALIASES = {
    "source_id": ("Id сообщения", "Идентификатор сообщения", "message_id", "post_id", "tweet_id", "source_id", "id", "№"),
    "url": ("Ссылка на сообщение", "Ссылка", "URL", "link", "permalink", "message_url", "post_url"),
    "title": ("Заголовок", "title", "headline", "subject"),
    "message": ("Текст сообщения", "Сообщение", "Текст", "message", "text", "body", "content", "full_text", "article_text"),
    "ocr_text": ("Автораспознанный текст", "Распознанный текст", "OCR", "ocr_text", "recognized_text", "auto_recognized_text"),
    "published_at": ("Дата публикации", "Время публикации", "Дата", "published_at", "published", "publication_date", "date", "created_at", "timestamp"),
    "source_name": ("Площадка", "Источник", "source_name", "source", "platform", "outlet", "site"),
    "author": ("Кто пишет", "Автор", "author", "author_name", "username", "user_name"),
    "language_hint": ("Язык", "Язык сообщения", "language_hint", "language", "lang", "language_code"),
}
_CSV_LOCK = threading.RLock()
_MAX_FIELD_SIZE = 32 * 1024 * 1024
_BLOCK_TAGS = {"p", "div", "br", "li", "ul", "ol", "blockquote", "section", "article", "h1", "h2", "h3", "h4", "table", "tr", "td", "pre"}
_HIDDEN_TAGS = {"script", "style", "template"}


class _VisibleHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _HIDDEN_TAGS:
            self.hidden.append(tag)
        elif not self.hidden and tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if not self.hidden and tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self.hidden:
            if tag == self.hidden[-1]:
                self.hidden.pop()
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.hidden:
            self.parts.append(data)


def normalize_text(text: str | None) -> str:
    """Remove HTML presentation, preserve Unicode letters and negations.

    Scripts/styles are discarded as text, never run. Normalization uses NFC,
    not accent stripping (in particular, Cyrillic й must remain й).
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        raise TypeError("text must be a string or None")
    parser = _VisibleHTML()
    parser.feed(text)
    parser.close()
    visible = unicodedata.normalize("NFC", "".join(parser.parts))
    visible = visible.replace("\r\n", "\n").replace("\r", "\n")
    visible = visible.replace("\ufeff", "").replace("\u200b", "")
    visible = "".join(c for c in visible if c in "\n\t" or unicodedata.category(c) != "Cc")
    lines = [re.sub(r"[^\S\n]+", " ", line).strip() for line in visible.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def exact_fingerprint(text: str | None) -> str:
    """SHA256 of visible NFC text with whitespace collapsed; case is retained."""
    canonical = re.sub(r"\s+", " ", normalize_text(text)).strip()
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _Layout:
    encoding: str
    delimiter: str
    quotechar: str
    columns: tuple[str, ...]
    sha256: str
    warnings: tuple[str, ...]


def _next_record(reader: Any) -> list[str]:
    # csv.field_size_limit is process-global. Restore it before yielding to a
    # caller, and serialize our own readers while temporarily raising it.
    with _CSV_LOCK:
        previous = csv.field_size_limit()
        try:
            csv.field_size_limit(max(previous, _MAX_FIELD_SIZE))
            return next(reader)
        finally:
            csv.field_size_limit(previous)


def _layout(path: str | Path) -> _Layout:
    path = Path(path)
    digest = hashlib.sha256()
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    utf8_ok = True
    prefix = b""
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            if not prefix:
                prefix = chunk[:4]
            digest.update(chunk)
            if utf8_ok:
                try:
                    decoder.decode(chunk, final=False)
                except UnicodeDecodeError:
                    utf8_ok = False
        if utf8_ok:
            try:
                decoder.decode(b"", final=True)
            except UnicodeDecodeError:
                utf8_ok = False
    warnings: list[str] = []
    if prefix.startswith((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)):
        encoding = "utf-32"
    elif prefix.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        encoding = "utf-16"
    elif prefix.startswith(codecs.BOM_UTF8):
        encoding = "utf-8-sig"
    elif utf8_ok:
        encoding = "utf-8"
    else:
        encoding = "cp1251"
        warnings.append("No Unicode BOM or valid UTF-8: cp1251 is a heuristic; confirm the source encoding.")
    with path.open("r", encoding=encoding, newline="", errors="strict") as stream:
        sample = stream.read(65536)
    if not sample.strip():
        return _Layout(encoding, ",", '"', (), digest.hexdigest(), ("CSV has no header or data.",))
    # Headers usually determine the separator more reliably than a sample
    # ending in the middle of a quoted multiline article.
    first_line = sample.splitlines()[0]
    try:
        dialect = csv.Sniffer().sniff(first_line, delimiters=";,\t|")
        delimiter, quotechar = dialect.delimiter, dialect.quotechar
    except csv.Error:
        delimiter, quotechar = ",", '"'
        warnings.append("Delimiter could not be inferred; using comma (possibly a single-column CSV).")
    with path.open("r", encoding=encoding, newline="", errors="strict") as stream:
        reader = csv.reader(stream, delimiter=delimiter, quotechar=quotechar, strict=True)
        try:
            columns = tuple(_next_record(reader))
        except StopIteration:
            columns = ()
        except csv.Error as exc:
            raise ValueError("Invalid CSV header") from exc
    if len(set(columns)) != len(columns):
        warnings.append("Duplicate header names: metadata preserves every value using numbered suffixes.")
    if any(not x.strip() for x in columns):
        warnings.append("Empty column names are retained in metadata using generated column labels.")
    return _Layout(encoding, delimiter, quotechar, columns, digest.hexdigest(), tuple(warnings))


def _header_key(value: str) -> str:
    value = unicodedata.normalize("NFC", value).casefold().replace("ё", "е")
    return "".join(c for c in value if c.isalnum()) or value.strip()


def _mapping(columns: tuple[str, ...], mapping: Mapping[str, str | None] | None = None) -> dict[str, str | None]:
    lookup: dict[str, str] = {}
    for name in columns:
        lookup.setdefault(_header_key(name), name)
    result: dict[str, str | None] = {}
    for field, aliases in _FIELD_ALIASES.items():
        result[field] = next((lookup[_header_key(a)] for a in aliases if _header_key(a) in lookup), None)
    for field, column in (mapping or {}).items():
        if field == "text":
            field = "message"
        if field not in result:
            raise ValueError(f"Unknown mapping field: {field}")
        if column is not None and column not in columns:
            raise ValueError(f"Mapping column does not exist: {column}")
        result[field] = column
    return result


def _records(path: str | Path, layout: _Layout) -> Iterator[tuple[int, list[str]]]:
    with Path(path).open("r", encoding=layout.encoding, newline="", errors="strict") as stream:
        reader = csv.reader(stream, delimiter=layout.delimiter, quotechar=layout.quotechar, strict=True)
        try:
            _next_record(reader)
            row_number = 2
            while True:
                row = _next_record(reader)
                yield row_number, row
                row_number += 1
        except StopIteration:
            return
        except csv.Error as exc:
            raise ValueError(f"Invalid CSV near physical line {reader.line_num}") from exc


def _metadata(columns: tuple[str, ...], row: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for i in range(max(len(columns), len(row))):
        name = columns[i] if i < len(columns) and columns[i] else f"__column_{i + 1}__"
        candidate, suffix = name, 2
        while candidate in result:
            candidate = f"{name} [{suffix}]"
            suffix += 1
        result[candidate] = row[i] if i < len(row) else ""
    return result


def _compose(values: list[tuple[str, str]], *, cleaned: bool) -> str:
    sections: list[str] = []
    seen: set[str] = set()
    for label, raw in values:
        text = normalize_text(raw) if cleaned else raw
        if not text.strip():
            continue
        key = re.sub(r"\s+", " ", text).strip()
        if cleaned and key in seen:
            continue
        seen.add(key)
        sections.append(f"=== {label} ===\n{text}")
    return "\n\n".join(sections)


def inspect_csv(path: str | Path) -> dict[str, Any]:
    """Return counts/schema/hash only; never include full texts or row samples."""
    layout = _layout(path)
    mapping = _mapping(layout.columns)
    positions = {name: layout.columns.index(name) for name in layout.columns}
    total = with_text = primary_text_rows = empty = wrong_width = 0
    for _, row in _records(path, layout):
        total += 1
        if not any(x.strip() for x in row):
            empty += 1
            continue
        wrong_width += len(row) != len(layout.columns)
        visible = {field: normalize_text(row[positions[column]])
                   for field in ("title", "message", "ocr_text")
                   if (column := mapping[field]) is not None and positions[column] < len(row)}
        primary_text_rows += bool(visible.get("message"))
        if any(visible.values()):
            with_text += 1
    warnings = list(layout.warnings)
    if wrong_width:
        warnings.append(f"Nonempty records with a different column count: {wrong_width}; missing/extra values are preserved.")
    if not any(mapping[x] for x in ("title", "message", "ocr_text")):
        warnings.append("No recognized text columns; supply an explicit mapping to iter_documents.")
    if mapping["message"] is None:
        warnings.append("Primary message column was not recognized; titles/OCR alone do not confirm full-text import. Supply an explicit message mapping before analysis.")
    elif with_text and not primary_text_rows:
        warnings.append("Primary message column has no visible text; text-bearing records contain titles/OCR only. Confirm the source export before analysis.")
    return {"encoding": layout.encoding, "delimiter": layout.delimiter, "columns": list(layout.columns),
            "rows_total": total, "rows_with_text": with_text, "empty_rows": empty,
            "field_mapping": mapping, "warnings": warnings, "source_sha256": layout.sha256}


def combine_publication_date(date_value: str, time_value: str) -> str:
    """Combine explicit split date/time only; never invent a day or timezone.

    Source metadata retains both original cells. A full timestamp, time-only
    export, malformed value or ambiguous date is returned unchanged.
    """
    date_value, time_value = str(date_value or "").strip(), str(time_value or "").strip()
    if (re.fullmatch(r"(?:\d{2}\.\d{2}\.\d{4}|\d{4}-\d{2}-\d{2})", date_value)
            and re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d(?::[0-5]\d)?", time_value)):
        return date_value + " " + time_value
    return date_value


def iter_documents(path: str | Path, mapping: Mapping[str, str | None] | None = None) -> Iterator[dict[str, Any]]:
    """Yield every parsed record, preserving original column values as strings.

    ``raw_text`` combines unmodified title/message/OCR fields with labels;
    ``text`` combines visible cleaned fields, omitting identical sections.
    Empty and metadata-only records are yielded with empty ``text``; the caller
    can keep an explicit empty status instead of losing source-row provenance.
    """
    layout = _layout(path)
    fields = _mapping(layout.columns, mapping)
    positions = {name: layout.columns.index(name) for name in layout.columns}
    for number, row in _records(path, layout):
        def cell(field: str) -> str:
            column = fields[field]
            return row[positions[column]] if column is not None and positions[column] < len(row) else ""
        parts = [("TITLE", cell("title")), ("MESSAGE", cell("message")), ("OCR", cell("ocr_text"))]
        time_column = next((column for column in ("Время публикации", "publication_time", "time")
                            if column in positions and column != fields["published_at"]), None)
        published = combine_publication_date(cell("published_at"),
            row[positions[time_column]] if time_column is not None and positions[time_column] < len(row) else "")
        yield {"source_row": number, "source_id": cell("source_id").strip(), "url": cell("url").strip(),
               "title": normalize_text(cell("title")), "raw_text": _compose(parts, cleaned=False),
               "text": _compose(parts, cleaned=True), "published_at": published,
               "source_name": cell("source_name").strip(), "author": cell("author").strip(),
               "language_hint": cell("language_hint").strip(), "metadata": _metadata(layout.columns, row)}


_NEGATION_WORDS = frozenset("no not never neither nor without cannot can't won't isn't aren't wasn't weren't don't doesn't didn't ne pas jamais sans aucun aucune non не нет ни никогда без нельзя si sio siyo hapana hakuna nta ntabwo ntaho".split())
_TOKEN = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)
_MIN_TOKENS = 40
_ANCHORS = 16
_BUCKET_SIZE = 32
_MAX_CANDIDATES = 64
_MAX_MATCHES = 8


def _negations(tokens: list[str]) -> Counter[str]:
    return Counter(t for t in tokens if t in _NEGATION_WORDS or re.match(r"^(?:nt[aiu]|ha[ktmyu])", t))


def near_duplicate_pairs(documents: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Find review candidates using bounded bottom-k shingle LSH, not all pairs.

    Require >=40 tokens, >=25 distinct five-token shingles and Jaccard >=.85.
    A changed negation or numeric token multiset prevents a duplicate proposal.
    This is deliberately conservative, not exhaustive: 32 representatives per
    anchor, 64 comparisons and at most 8 returned neighbours per new document.
    Same-topic text and short slogans are not sufficient. No records are deleted.
    The negation guard is multilingual but is not a complete contradiction model.
    """
    buckets: dict[bytes, list[int]] = defaultdict(list)
    # Packed 128-bit hashes avoid retaining a Python set/object per shingle for
    # every article. Only the current article needs the membership-test set.
    stored: list[tuple[str, bytes, Counter[str], Counter[str]]] = []
    identifiers: set[str] = set()
    pairs: list[dict[str, Any]] = []
    for document in documents:
        identifier = document.get("id")
        if not isinstance(identifier, (str, int)) or isinstance(identifier, bool) or not str(identifier):
            raise ValueError("Each document needs a nonempty string or integer id")
        identifier = str(identifier)
        if identifier in identifiers:
            raise ValueError("Document ids must be unique")
        identifiers.add(identifier)
        tokens = _TOKEN.findall(normalize_text(document.get("text", "")).casefold().replace("’", "'"))
        if len(tokens) < _MIN_TOKENS:
            continue
        shingles = {hashlib.blake2b("\x1f".join(tokens[i:i + 5]).encode("utf-8"), digest_size=16).digest()
                    for i in range(len(tokens) - 4)}
        if len(shingles) < 25:
            continue
        negations = _negations(tokens)
        numbers = Counter(number for token in tokens for number in re.findall(r"\d+", token))
        ordered = sorted(shingles)
        anchors = ordered[:_ANCHORS]
        candidates = Counter(i for anchor in anchors for i in buckets.get(anchor, []))
        matches: list[tuple[float, int]] = []
        for previous, shared in candidates.most_common(_MAX_CANDIDATES):
            if shared < 2:
                continue
            _, other, other_negations, other_numbers = stored[previous]
            if negations != other_negations or numbers != other_numbers:
                continue
            other_size = len(other) // 16
            if min(len(shingles), other_size) / max(len(shingles), other_size) < .85:
                continue
            intersection = sum(other[offset:offset + 16] in shingles for offset in range(0, len(other), 16))
            similarity = intersection / (len(shingles) + other_size - intersection)
            if similarity >= .85:
                matches.append((similarity, previous))
        for similarity, previous in sorted(matches, key=lambda x: (-x[0], x[1]))[:_MAX_MATCHES]:
            pairs.append({"left_id": stored[previous][0], "right_id": identifier,
                          "similarity": round(similarity, 6), "kind": "near_duplicate_candidate"})
        index = len(stored)
        stored.append((identifier, b"".join(ordered), negations, numbers))
        for anchor in anchors:
            bucket = buckets[anchor]
            bucket.append(index)
            if len(bucket) > _BUCKET_SIZE:
                del bucket[0]
    return pairs
