"""
Prompt assembly.

Each context block is labelled with its authority, which reinforces the separation
inside the prompt itself. Note what is absent: no request to calculate anything.
"""

from __future__ import annotations

import textwrap

import pandas as pd


def _fig_lines(row) -> str:
    return "\n".join(f"  {k}: {v}" for k, v in row.verified_figures().items())


def build_grounded_prompt(row, rules: pd.DataFrame, causes: pd.DataFrame) -> str:
    """The B3/B4 prompt: verified figures, retrieved rules, retrieved causes."""
    rules_ctx = "\n".join(
        f"  [{i + 1}] {r.doc_id} {r.section}: {r.chunk_text}"
        for i, (_, r) in enumerate(rules.iterrows())
    ) or "  (none provided)"

    base = len(rules)
    causes_ctx = "\n".join(
        f"  [{base + i + 1}] {r.doc_id} {r.section} (published {r.published_date}): {r.chunk_text}"
        for i, (_, r) in enumerate(causes.iterrows())
    ) or "  (none provided)"

    return textwrap.dedent(f"""\
    SYSTEM
    You write month-end variance commentary for an FP&A team.

    ABSOLUTE RULES
    1. NEVER calculate, estimate, or restate a number. Every figure you may use is
       supplied under VERIFIED FIGURES, including the materiality thresholds.
       Reproduce them exactly. If a figure you need is not supplied, say so.
    2. State a cause ONLY if a DRIVER EVIDENCE passage supports it. If none does,
       set the cause to "unsupported" and escalate to the cost-centre owner. An
       unsupported cause is a more serious error than no cause at all.
    3. Every factual sentence carries a citation [n] resolving to a passage below.
    4. Apply the materiality rule as written in POLICY and cite the clause used.
    5. No speculation and no recommendations beyond the escalation path.

    VERIFIED FIGURES  (authority: variance engine, treat as fact)
    {_fig_lines(row)}

    POLICY  (authority: policy corpus)
    {rules_ctx}

    DRIVER EVIDENCE  (authority: driver corpus, may be empty)
    {causes_ctx}

    TASK
    Write the commentary for the row above in under 80 words: a headline with the
    dollar and percent variance, the cause (or "unsupported"), the accountable
    owner, and citations.
    """)


def build_ungrounded_prompt(row) -> str:
    """The B1 prompt: identical figures, no retrieval, an open 'why'.

    This is what a bare LLM already does, and it is the honest baseline for the
    claim that grounding adds something.
    """
    return textwrap.dedent(f"""\
    You write FP&A variance commentary. Using these figures, write a one-sentence
    headline with the dollar and percent variance and explain the most likely CAUSE.
    Keep it under 80 words.

    FIGURES
    {_fig_lines(row)}
    """)


def build_longcontext_prompt(row, all_chunks: pd.DataFrame) -> str:
    """The B2 control: same figures, whole corpus in context, no retrieval.

    Tests whether retrieval is doing work or whether context alone suffices. Li et
    al. (2024) found long context can beat retrieval when sufficiently resourced,
    which is precisely the resource a small local model lacks.
    """
    corpus = "\n".join(
        f"  [{i + 1}] {r.doc_id} {r.section}: {r.chunk_text}"
        for i, (_, r) in enumerate(all_chunks.iterrows())
    )
    return textwrap.dedent(f"""\
    SYSTEM
    You write month-end variance commentary for an FP&A team.

    ABSOLUTE RULES
    1. NEVER calculate a number. Use only VERIFIED FIGURES, exactly.
    2. State a cause ONLY if a DOCUMENT passage supports it, otherwise say
       "unsupported" and escalate to the cost-centre owner.
    3. Every factual sentence carries a citation [n].

    VERIFIED FIGURES  (authority: variance engine)
    {_fig_lines(row)}

    DOCUMENTS  (the entire corpus, unfiltered)
    {corpus}

    TASK
    Write the commentary for the row above in under 80 words with citations.
    """)


def deterministic_template(row) -> str:
    """B0: no model at all. Numerically perfect, causally empty. The floor."""
    direction = "over" if row.variance_usd > 0 else "under"
    return (
        f"{row.account} was ${abs(row.variance_usd):,.0f} {direction} budget "
        f"({row.variance_pct:+.1f}%) in {row.period} for {row.cost_centre}. "
        f"Commentary required: {row.commentary_required}."
    )
