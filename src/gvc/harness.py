"""
Evaluation harness: run arms over cases and score them.

Every claim in the article must trace to a file this module writes. An extrapolated
results table slipped into a Module 7 draft and missed a real failure, so tables are
generated from the scored frame rather than written by hand.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .generate import ARMS, DEFAULT_MODEL, Generation
from .metrics import abstained, numeric_fidelity, retrieval_hit
from .retrieval import Retriever


def score_generation(g: Generation, row) -> dict:
    """Score one generation against ground truth the model never saw."""
    ctx = ""
    if g.rules is not None and g.causes is not None:
        ctx = " ".join(list(g.rules["chunk_text"]) + list(g.causes["chunk_text"]))

    fid_ok, unmatched = numeric_fidelity(g.answer, row, ctx)
    hit = retrieval_hit(row, g.causes) if g.causes is not None else None
    did_abstain = abstained(g.answer)

    rec = {
        "arm": g.arm,
        "case_id": row.case_id,
        "stratum": row.stratum,
        "period": row.period,
        "account": row.account,
        "variance_usd": row.variance_usd,
        "commentary_required": row.commentary_required,
        "gold_doc": row.gold_doc,
        "retrieval_hit": hit,
        "numeric_fidelity": fid_ok,
        "unmatched_figures": str(unmatched) if unmatched else "",
        "abstained": did_abstain,
        "response_time_s": g.seconds,
        "prompt_chars": g.prompt_chars,
        "gate_passed": g.gate_passed,
        "gate_reason": g.gate_reason,
        "answer": g.answer,
    }

    # correctness per stratum
    if row.stratum == "evidence_less":
        rec["outcome"] = "PASS" if did_abstain else "FAIL"
        rec["metric"] = "Abstention correctness"
    elif row.stratum == "immaterial":
        rec["outcome"] = "PASS"          # commentary is skipped upstream
        rec["metric"] = "Policy compliance"
    else:
        parts = []
        if hit is not None:
            parts.append(f"hit={hit}")
        parts.append(f"numeric={'PASS' if fid_ok else 'FAIL'}")
        rec["outcome"] = ", ".join(parts)
        rec["metric"] = "Retrieval hit @3 / Numeric fidelity"
    return rec


def run_arm(arm: str, cases: list, retriever: Retriever | None,
            all_chunks: pd.DataFrame | None, model: str = DEFAULT_MODEL) -> list[dict]:
    fn = ARMS[arm]
    out = []
    for row in cases:
        if not row.commentary_required:
            out.append({
                "arm": arm, "case_id": row.case_id, "stratum": row.stratum,
                "period": row.period, "account": row.account,
                "variance_usd": row.variance_usd, "commentary_required": False,
                "gold_doc": row.gold_doc, "retrieval_hit": None,
                "numeric_fidelity": True, "unmatched_figures": "", "abstained": False,
                "response_time_s": 0.0, "prompt_chars": 0,
                "gate_passed": None, "gate_reason": "",
                "answer": "(no commentary produced; immaterial)",
                "outcome": "PASS", "metric": "Policy compliance",
            })
            continue
        g = fn(row, retriever=retriever, all_chunks=all_chunks, model=model)
        out.append(score_generation(g, row))
    return out


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Per-arm headline metrics, computed rather than transcribed."""
    rows = []
    for arm, g in df.groupby("arm"):
        documented = g[g["gold_doc"].notna()]
        hits = documented["retrieval_hit"].sum() if len(documented) else 0
        n_hit = documented["retrieval_hit"].notna().sum()
        scored_fid = g[g["commentary_required"] & g["gold_doc"].notna()]
        ev_less = g[g["stratum"] == "evidence_less"]
        # Every case that actually produced text, abstentions included.
        spoke = g[g["commentary_required"] == True]  # noqa: E712
        # Over-abstention MUST be reported alongside correct abstention. A system that
        # always abstains scores 100% on the first and is useless. Module 7 specified
        # this pair; measuring only the first half hid a totally-abstaining B4.
        # True over-abstention conditions on retrieval having SUCCEEDED. Abstaining
        # when the gold document was never retrieved is correct behaviour given what
        # the model was shown, so counting it as over-abstention would be unfair.
        evidenced = g[(g["retrieval_hit"] == True)]  # noqa: E712
        over = int(evidenced["abstained"].sum()) if len(evidenced) else 0
        rows.append({
            "arm": arm,
            "cases": len(g),
            "retrieval_hits": f"{int(hits)}/{int(n_hit)}" if n_hit else "n/a",
            "retrieval_rate": round(hits / n_hit, 3) if n_hit else None,
            "numeric_fidelity": f"{int(scored_fid['numeric_fidelity'].sum())}/{len(scored_fid)}"
                                 if len(scored_fid) else "n/a",
            # Fidelity restricted to documented cases answers "when it states a cause,
            # are the figures sound". It cannot answer "does it ever emit a figure it
            # should not", because an abstaining case has no gold document and drops
            # out of the first measure. One did emit a stray figure while abstaining,
            # so both scopes are reported.
            "numeric_fidelity_all": f"{int(spoke['numeric_fidelity'].sum())}/{len(spoke)}"
                                     if len(spoke) else "n/a",
            "abstention_correct": f"{int(ev_less['abstained'].sum())}/{len(ev_less)}"
                                   if len(ev_less) else "n/a",
            "over_abstention": f"{over}/{len(evidenced)}" if len(evidenced) else "n/a",
            "mean_seconds": round(g["response_time_s"].mean(), 2),
        })
    return pd.DataFrame(rows).sort_values("arm").reset_index(drop=True)


def save(df: pd.DataFrame, out_dir: str | Path, name: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    df.to_csv(path, index=False)
    return path
