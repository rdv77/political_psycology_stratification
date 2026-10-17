"""Lexical calendar precision shared by extraction and offline corrections.

This checks the supplied date evidence, never a publication timestamp. An
explicit calendar day is necessary, but not sufficient, to establish when the
particular actor said or did something. Semantic event binding remains a
separate source-review task.
"""
from __future__ import annotations

from datetime import date
import re
import unicodedata


_MONTHS = {
    1: ("january", "janvier", "mutarama", "januari", "jan", "janv"),
    2: ("february", "février", "fevrier", "gashyantare", "februari", "feb", "févr", "fevr"),
    3: ("march", "mars", "werurwe", "machi", "mar"),
    4: ("april", "avril", "mata", "aprili", "apr", "avr"),
    5: ("may", "mai", "gicurasi", "mei"),
    6: ("june", "juin", "kamena", "juni", "jun"),
    7: ("july", "juillet", "nyakanga", "julai", "jul", "juil"),
    8: ("august", "août", "aout", "kanama", "agosti", "aug"),
    9: ("september", "septembre", "nzeri", "septemba", "sep", "sept"),
    10: ("october", "octobre", "ukwakira", "oktoba", "oct"),
    11: ("november", "novembre", "ugushyingo", "novemba", "nov"),
    12: ("december", "décembre", "decembre", "ukuboza", "desemba", "dec", "déc")}


def event_date_precision_problem(fact):
    """Return a QA code if evidence does not state the selected exact day.

    The caller validates ISO syntax before calling. A year/month cannot become
    its first day. No weekday, 'today' or day/month is completed from metadata.
    Exact source adjudications may separately resolve linked dates explicitly.
    """
    event = date.fromisoformat(fact["event_date"])
    evidence = " ".join(unicodedata.normalize("NFC", str(fact.get("date_evidence") or "")).split()).casefold()
    year, month, day = event.year, event.month, event.day
    numeric = re.findall(r"(?<!\d)(\d{4})-(\d{1,2})-(\d{1,2})(?!\d)", evidence)
    numeric += [(y, m, d) for d, m, y in re.findall(r"(?<!\d)(\d{1,2})[./](\d{1,2})[./](\d{4})(?!\d)", evidence)]
    if numeric:
        return None if any(tuple(map(int, item)) == (year, month, day) for item in numeric) else "date_value_mismatch"
    candidates = set()
    for number, labels in _MONTHS.items():
        # News datelines commonly shorten months: '2 Dec 2025', 'Dec.4,
        # 2025', or French '2 déc. 2025'. A period is punctuation, not a
        # missing date component. The day and year remain mandatory.
        words = "(?:" + "|".join(re.escape(label) for label in labels) + r")\.?"
        # 19 May 2026 / 19 mai 2026 / 19 Gicurasi 2026 / 19 Mei 2026
        for d, y in re.findall(r"(?<!\d)(\d{1,2})(?:st|nd|rd|th|er)?\s+" + words +
                               r"\s*,?\s*(\d{4})(?!\d)", evidence):
            candidates.add((int(y), number, int(d)))
        # May 19, 2026, including prose 'May 19 2026'.
        for d, y in re.findall(r"(?<!\w)" + words + r"\s*(\d{1,2})(?:st|nd|rd|th)?\s*,?\s*(\d{4})(?!\d)", evidence):
            candidates.add((int(y), number, int(d)))
    if candidates:
        return None if (year, month, day) in candidates else "date_value_mismatch"
    return "date_precision_not_exact_day"
