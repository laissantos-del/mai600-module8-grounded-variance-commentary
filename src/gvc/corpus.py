"""
Corpus loading and per-type chunking.

A single document section is the chunk unit. Policy and definition content is kept
at the rule or term level so that a threshold is never separated from its exception,
while memos are kept by section because cause and quantification usually sit in
adjacent sentences.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

# Retrieval axes search disjoint sub-corpora. Module 7 measured what happens when
# they do not: the owner register polluted the causes axis and crowded out real
# driver memos.
RULES_CATS = {"rules"}
CAUSE_CATS = {"causes", "distractor", "precedent"}


def load_corpus(path: str | Path) -> tuple[list[dict], pd.DataFrame]:
    """Return the raw documents and a flat one-row-per-section chunk frame."""
    docs = json.loads(Path(path).read_text())
    rows = []
    for d in docs:
        for s in d["sections"]:
            rows.append(
                {
                    "doc_id": d["doc_id"],
                    "doc_title": d["doc_title"],
                    "doc_type": d["doc_type"],
                    "category": d["category"],
                    "section": s["section"],
                    "chunk_text": s["text"],
                    "published_date": d["published_date"],
                    "effective_from": d["effective_from"],
                    "effective_to": d["effective_to"],
                    "version": d["version"],
                    "supersedes": d["supersedes"],
                    "superseded_by": d["superseded_by"],
                    "owner": d["owner"],
                    "entity_scope": ", ".join(d["entity_scope"]),
                    "topic": ", ".join(d["topic"]),
                }
            )
    return docs, pd.DataFrame(rows)


def is_missing(v) -> bool:
    """True for None, NaN, pandas NA, and empty strings.

    Necessary because Arrow-backed string columns represent a JSON null as float
    NaN, which is truthy. A plain `if v else None` guard therefore lets NaN through
    and `date.fromisoformat` raises. Colab's older pandas kept None as None, so this
    only appears locally, which is precisely why the parity check exists.
    """
    if v is None:
        return True
    try:
        if pd.isna(v):
            return True
    except (TypeError, ValueError):
        pass
    return isinstance(v, str) and not v.strip()


def as_date(s) -> date | None:
    return None if is_missing(s) else date.fromisoformat(str(s))


def month_end(period: str) -> date:
    """Close date for a YYYY-MM period."""
    y, m = map(int, period.split("-"))
    first_of_next = date(y + (m == 12), (m % 12) + 1, 1)
    return date.fromordinal(first_of_next.toordinal() - 1)


def admissible(chunk: pd.Series, close: date) -> bool:
    """Point-in-time admissibility.

    A chunk is retrievable for a close period only if it existed by then and the
    close date falls inside its effective window. Without this a September variance
    could be explained by a November memo, which is lookahead leakage: the
    commentary would be unreproducible at the time it was supposedly written.
    """
    pub = as_date(chunk["published_date"])
    ef = as_date(chunk["effective_from"])
    et = as_date(chunk["effective_to"])
    if pub is None or pub > close:
        return False
    if ef and close < ef:
        return False
    if et and close >= et:
        return False
    return True
