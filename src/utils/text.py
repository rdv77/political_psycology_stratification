from __future__ import annotations

import re
from typing import Iterable, List, Set

try:
    import snowballstemmer  # lightweight Russian stemmer
    _ru_stemmer = snowballstemmer.stemmer('russian')
except Exception:  # optional dependency
    snowballstemmer = None
    _ru_stemmer = None


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def to_lower_strip(items: Iterable[str]) -> List[str]:
    return [normalize_whitespace(s).lower() for s in items if s and normalize_whitespace(s)]


_RU_STOPWORDS: Set[str] = set(
    [
        "и","в","во","на","по","для","к","ко","от","до","из","у","со","с","о","об","при",
        "что","как","же","ли","не","ни","да","но","то","же","бы","же","а","или","либо",
        "это","той","та","тот","те","эта","эти","того","той","этом","этой","этим","этими",
        "который","которая","которые","которых","которого",
        "ориентирована","ориентированная","ориентация","ориентированный","направлена","направленный",
    ]
)


def russian_stems(text: str) -> List[str]:
    cleaned = normalize_whitespace(re.sub(r"[\.,;:!\?\(\)\[\]\{\}\"'«»“”„–—-]", " ", text.lower()))
    tokens = [t for t in cleaned.split(" ") if t and t not in _RU_STOPWORDS]
    if _ru_stemmer is None:
        # Fallback: simple suffix stripping for common adjectival/verb endings
        stems = [re.sub(r"(ыми|ими|ами|ями|ый|ий|ой|ая|яя|ое|ее|ого|его|ому|ему|ыми|ими|ых|их|ую|юю|ать|ять|еть|ить|ться|тся|ла|ло|ли|ем|ет|ют|ут|ит|ят)$", "", t) for t in tokens]
        return [s for s in stems if s]
    return [s for s in _ru_stemmer.stemWords(tokens) if s and s not in _RU_STOPWORDS]


def canonicalize_trait(text: str) -> str:
    stems = russian_stems(text)
    # Sort to canonical order and drop duplicates
    uniq = []
    seen = set()
    for s in sorted(stems):
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return " ".join(uniq)


