"""
Scoring.

The numeric-fidelity checker needed three corrections during Module 7, each from a
real false result, and all three are preserved here:

  1. ISO dates and bare account or cost-centre codes are not figures. Flagging
     "2023-06" as a fabricated number failed a correct answer.
  2. "$495K" and "$1.8M" are figures. Scale suffixes must be expanded, or real
     commentary style is scored as fabrication.
  3. A number is grounded if it matches the verified figures OR any number in the
     retrieved context. Correctly quoting the $250,000 policy threshold is not a
     fabrication. This mirrors attribution to identified sources.

Stated limitation, chosen deliberately: a bare fabricated integer with no currency
symbol, comma, or percent sign is not checked, so that the metric has zero false
positives.
"""

from __future__ import annotations

import re

_SCALE = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}


def financial_numbers(text: str) -> list[tuple[str, float]]:
    """Extract (kind, value) for genuine financial tokens only."""
    text = re.sub(r"\d{4}-\d{2}(?:-\d{2})?", " ", text)      # strip ISO dates
    out: list[tuple[str, float]] = []

    for m in re.findall(r"[\d,]+(?:\.\d+)?\s?%", text):       # percentages
        out.append(("pct", round(float(re.sub(r"[^\d.]", "", m)), 1)))
    text = re.sub(r"[\d,]+(?:\.\d+)?\s?%", " ", text)         # consume them

    pattern = r"\$\s?[\d,]+(?:\.\d+)?\s?[KkMmBb]?|\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\s?[KkMmBb]?"
    for m in re.findall(pattern, text):
        value = float(re.sub(r"[^\d.]", "", m))
        value *= _SCALE.get(m.strip()[-1].lower(), 1)
        out.append(("usd", value))
    return out


def grounded_sets(row, context_text: str = "") -> tuple[set[float], set[float]]:
    """Numbers the answer may legitimately contain."""
    usd: set[float] = set()
    pct: set[float] = set()
    for k, v in row.verified_figures().items():
        if isinstance(v, (int, float)):
            (pct if k.endswith("pct") or k.endswith("pct_threshold") else usd).add(
                round(abs(float(v)), 2)
            )
    for kind, val in financial_numbers(context_text):
        (pct if kind == "pct" else usd).add(val)
    return usd, pct


def numeric_fidelity(answer: str, row, context_text: str = "") -> tuple[bool, list]:
    """True iff every financial number traces to a provided source."""
    usd, pct = grounded_sets(row, context_text)
    unmatched = []
    for kind, val in financial_numbers(answer):
        target, tol = (pct, 0.1) if kind == "pct" else (usd, 1.0)
        if not any(abs(val - t) <= tol for t in target):
            unmatched.append((kind, val))
    return len(unmatched) == 0, unmatched


def abstained(answer: str) -> bool:
    return "unsupported" in answer.lower()


def retrieval_hit(row, causes) -> bool | None:
    """None where the case has no gold document, since a hit is undefined."""
    if row.gold_doc is None:
        return None
    return row.gold_doc in list(causes["doc_id"])


def classify_outcome(score_text: str) -> str:
    """Failure indicators are checked FIRST.

    A mixed string such as "hit=False, numeric=PASS" is a failure overall. Checking
    for PASS first counted it as a pass, which is a bug Module 7 caught by testing
    edge cases rather than by reading the code.
    """
    s = str(score_text)
    if "FAIL" in s or "False" in s:
        return "FAIL"
    if "PASS" in s or "True" in s:
        return "PASS"
    return "mixed"
