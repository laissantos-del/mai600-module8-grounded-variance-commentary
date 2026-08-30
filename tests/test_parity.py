"""
Parity check: does the ported local pipeline reproduce the Module 7 Colab result?

Everything downstream depends on this. If local and Colab disagree, that must be
resolved before any new measurement is trusted.

Module 7 baseline, from mai600_module7/.../results/evaluation_scores.csv:
    retrieval hit      3 of 5 documented cases (C1, C2, C3 hit; C4, C7 miss)
    numeric fidelity   4 of 5 (C3 failed on a self-computed threshold)
    abstention         2 of 2 (C5, C6)

Note the expected numeric-fidelity difference: Module 8 supplies the computed
materiality threshold as a verified figure, which is the Stage 2 fix for exactly
that C3 failure. Therefore fidelity is expected to become 5 of 5, and the check
below treats that as the intended improvement rather than a parity break.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

from gvc import harness  # noqa: E402
from gvc.cases import load_cases  # noqa: E402
from gvc.corpus import load_corpus  # noqa: E402
from gvc.generate import DEFAULT_MODEL, ollama_available  # noqa: E402
from gvc.retrieval import Retriever, RetrievalConfig  # noqa: E402
from gvc.variance_engine import self_check  # noqa: E402

EXPECTED_HITS = {"C1", "C2", "C3"}
EXPECTED_MISSES = {"C4", "C7"}


def main() -> int:
    cases = load_cases("m7")

    # 1. deterministic layer
    ok, checks = self_check(cases)
    print("VARIANCE ENGINE SELF-CHECKS")
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    if not ok:
        print("\nvariance engine self-checks FAILED; stopping.")
        return 1

    # 2. retrieval parity, no model needed
    _, chunks = load_corpus(ROOT / "data" / "corpus_documents.json")
    print(f"\ncorpus: {chunks['doc_id'].nunique()} documents, {len(chunks)} chunks")

    embed = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    baseline = Retriever(chunks, embed, RetrievalConfig())   # Module 7 settings

    hits, misses = set(), set()
    print("\nRETRIEVAL PARITY (Module 7 configuration)")
    for row in cases:
        if row.gold_doc is None:
            continue
        _, causes = baseline.retrieve(row)
        got = list(causes["doc_id"])
        hit = row.gold_doc in got
        (hits if hit else misses).add(row.case_id)
        top = ", ".join(f"{r.doc_id}({r.score:.3f})" for r in causes.itertuples())
        print(f"  {'HIT ' if hit else 'MISS'} {row.case_id}: gold={row.gold_doc:4} {top}")

    parity = (hits == EXPECTED_HITS and misses == EXPECTED_MISSES)
    print(f"\n  expected hits {sorted(EXPECTED_HITS)} misses {sorted(EXPECTED_MISSES)}")
    print(f"  observed hits {sorted(hits)} misses {sorted(misses)}")
    print(f"  -> retrieval parity: {'MATCH' if parity else 'MISMATCH'}")
    if not parity:
        print("\nRetrieval differs from the Colab run. Resolve before proceeding.")
        return 1

    # 3. end-to-end B3 locally, if Ollama is up
    up, msg = ollama_available(DEFAULT_MODEL)
    print(f"\n  {msg}")
    if not up:
        print("  skipping generation parity")
        return 0

    print("\nEND-TO-END B3 (local Ollama)")
    recs = harness.run_arm("B3", cases, baseline, chunks)
    df = pd.DataFrame(recs)
    out = harness.save(df, ROOT / "results", "parity_B3_module7_config.csv")

    documented = df[df["gold_doc"].notna()]
    n_hit = int(documented["retrieval_hit"].sum())
    scored = df[df["commentary_required"] & df["gold_doc"].notna()]
    n_fid = int(scored["numeric_fidelity"].sum())
    ev = df[df["stratum"] == "evidence_less"]
    n_abs = int(ev["abstained"].sum())

    print(f"  retrieval hit    {n_hit}/{len(documented)}   (Module 7: 3/5)")
    print(f"  numeric fidelity {n_fid}/{len(scored)}   (Module 7: 4/5)")
    print(f"  abstention       {n_abs}/{len(ev)}   (Module 7: 2/2)")
    print(f"  mean seconds     {df['response_time_s'].mean():.2f}")
    print(f"  written -> {out.name}")

    verdict = (n_hit == 3 and n_abs == len(ev))
    print(f"\n  -> PARITY {'CONFIRMED' if verdict else 'BROKEN'}")
    if n_fid > 4:
        print("     numeric fidelity improved as intended by the Stage 2 threshold fix")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
