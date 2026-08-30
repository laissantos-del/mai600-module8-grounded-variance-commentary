"""
Module 8 experiments.

Configuration was chosen on the two config periods only, and those periods could not
discriminate between candidates on retrieval. The choice therefore rests on principle:
a precedent document is not a driver memo, so excluding the `precedent` category from
the causes axis is a category correction rather than a fit to observed failures.
Top-k stays at the Module 6 design value of 3. A k=4 variant is reported separately
and labelled as tuned, because its only justification was that it captured a gold
document in a case that had already been inspected.

Everything below reports the HOLD-OUT periods, which the configuration never saw.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

from gvc import harness  # noqa: E402
from gvc.cases import load_cases, periods  # noqa: E402
from gvc.corpus import load_corpus  # noqa: E402
from gvc.generate import DEFAULT_MODEL, ollama_available  # noqa: E402
from gvc.retrieval import Retriever, RetrievalConfig  # noqa: E402

RESULTS = ROOT / "results"

BASELINE = RetrievalConfig(k_causes=3, exclude_precedent=False)          # Module 7
PRINCIPLED = RetrievalConfig(k_causes=3, exclude_precedent=True)          # Module 8
TUNED = RetrievalConfig(k_causes=4, exclude_precedent=True)               # secondary


def retrieval_only(r: Retriever, cases) -> tuple[int, int, list[str]]:
    gold = [c for c in cases if c.gold_doc]
    hits, miss = 0, []
    for c in gold:
        if c.gold_doc in list(r.retrieve(c)[1]["doc_id"]):
            hits += 1
        else:
            miss.append(c.case_id)
    return hits, len(gold), miss


def main() -> int:
    up, msg = ollama_available(DEFAULT_MODEL)
    print(f"  {msg}\n")

    _, chunks = load_corpus(ROOT / "data" / "corpus_documents.json")
    embed = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    hold = load_cases("holdout")

    print(f"HOLD-OUT: {len(hold)} cases across "
          f"{len({c.period for c in hold})} close periods\n")

    # ---------------- retrieval, no model needed ----------------
    print("RETRIEVAL ON HOLD-OUT")
    rows = []
    for name, cfg in (("Module 7 baseline", BASELINE),
                      ("Module 8 principled", PRINCIPLED),
                      ("k=4 variant (tuned)", TUNED)):
        r = Retriever(chunks, embed, cfg)
        h, n, miss = retrieval_only(r, hold)
        print(f"  {name:22} {h}/{n} = {h/n:.0%}   misses: {', '.join(miss) or 'none'}")
        rows.append({"config": name, "hits": h, "n": n,
                     "rate": round(h / n, 3), "misses": ", ".join(miss)})
    pd.DataFrame(rows).to_csv(RESULTS / "retrieval_config_comparison.csv", index=False)

    if not up:
        print("\n  ollama unavailable, stopping before generation")
        return 0

    # ---------------- before and after, arm B3 ----------------
    print("\nB3 BEFORE AND AFTER ON HOLD-OUT")
    frames = {}
    for label, cfg in (("before_module7_config", BASELINE),
                       ("after_module8_config", PRINCIPLED)):
        r = Retriever(chunks, embed, cfg)
        df = pd.DataFrame(harness.run_arm("B3", hold, r, chunks))
        df["config"] = label
        frames[label] = df
        s = harness.summarise(df).iloc[0]
        print(f"  {label:24} retrieval {s['retrieval_hits']}  "
              f"fidelity {s['numeric_fidelity']}  abstention {s['abstention_correct']}  "
              f"{s['mean_seconds']}s")
    ba = pd.concat(frames.values(), ignore_index=True)
    ba.to_csv(RESULTS / "before_after_holdout.csv", index=False)

    # ---------------- full ladder on the chosen config ----------------
    print("\nFULL B0-B4 LADDER ON HOLD-OUT (Module 8 config)")
    r = Retriever(chunks, embed, PRINCIPLED)
    all_arms = []
    for arm in ("B0", "B1", "B2", "B3", "B4"):
        df = pd.DataFrame(harness.run_arm(arm, hold, r, chunks))
        all_arms.append(df)
        print(f"  {arm} done")
    ladder = pd.concat(all_arms, ignore_index=True)
    ladder.to_csv(RESULTS / "evaluation_scores.csv", index=False)

    summary = harness.summarise(ladder)
    summary.to_csv(RESULTS / "benchmark_results.csv", index=False)
    print("\nLADDER SUMMARY")
    print(summary.to_string(index=False))

    ladder[["arm", "case_id", "stratum", "period", "answer",
            "response_time_s"]].to_csv(RESULTS / "generated_outputs.csv", index=False)

    # ---------------- per-period view ----------------
    b3 = ladder[ladder["arm"] == "B3"]
    per = []
    for p in periods():
        g = b3[b3["period"] == p]
        if g.empty:
            continue
        gold = g[g["gold_doc"].notna()]
        ev = g[g["stratum"] == "evidence_less"]
        per.append({
            "period": p, "cases": len(g),
            "retrieval": f"{int(gold['retrieval_hit'].sum())}/{len(gold)}" if len(gold) else "n/a",
            "abstention": f"{int(ev['abstained'].sum())}/{len(ev)}" if len(ev) else "n/a",
            "numeric_fidelity": f"{int(g['numeric_fidelity'].sum())}/{len(g)}",
        })
    pdf = pd.DataFrame(per)
    pdf.to_csv(RESULTS / "per_period_results.csv", index=False)
    print("\nPER-PERIOD (arm B3, hold-out only)")
    print(pdf.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
