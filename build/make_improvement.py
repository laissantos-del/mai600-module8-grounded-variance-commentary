"""
Build the improvement comparison table and the two charts.

Every "before" value is read from the Module 7 result files, and every "after" value
from the Module 8 result files. Nothing is transcribed by hand, because a
hand-written table slipped into a Module 7 draft once and missed a real failure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import pandas as pd                      # noqa: E402

M7 = (ROOT.parents[1] / "mai600_module7" / "mai600-module7-project-progress" / "results")
RESULTS = ROOT / "results"
IMAGES = ROOT / "images"
IMAGES.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- image guard
# Two of the three figures in images/ are hand-authored and supersede what these
# scripts draw. Regenerating them silently would replace a published figure with an
# older design, so overwriting an existing one requires an explicit flag.
AUTHORED = {"system_architecture.png", "evaluation_chart.png"}


def may_write(path) -> bool:
    import sys
    if path.name in AUTHORED and path.exists() and "--overwrite-images" not in sys.argv:
        print(f"  SKIPPED {path.name}: the shipped figure is hand-authored. "
              f"Pass --overwrite-images to replace it.")
        return False
    return True


BG, GREY, BLUE, GREEN, RED = "#0d2c4d", "#7f9db8", "#2e8bc9", "#1f9d76", "#c0554f"


def frac(n: int, d: int) -> str:
    return f"{n}/{d}" if d else "n/a"


def main() -> int:
    # ---------------- read the evidence ----------------
    m7 = pd.read_csv(M7 / "evaluation_scores.csv")
    m8 = pd.read_csv(RESULTS / "evaluation_scores.csv")
    cfgcmp = pd.read_csv(RESULTS / "retrieval_config_comparison.csv")
    ba = pd.read_csv(RESULTS / "before_after_holdout.csv")
    corpus = json.loads((ROOT / "data" / "corpus_documents.json").read_text())

    # Module 7 headline, recomputed from its own file
    m7_hit_rows = m7[m7["Score / Result"].astype(str).str.contains("hit=")]
    m7_hits = int(m7_hit_rows["Score / Result"].str.contains("hit=True").sum())
    m7_num = m7[m7["Score / Result"].astype(str).str.contains("numeric=")]
    m7_num_pass = int(m7_num["Score / Result"].str.contains("numeric=PASS").sum())
    m7_abst = m7[m7["Metric Used"] == "Abstention correctness"]

    # Module 8, hold-out, arm B3
    b3 = m8[m8.arm == "B3"]
    b3_doc = b3[b3["retrieval_hit"].notna()]
    b3_hits = int(b3_doc["retrieval_hit"].sum())
    b3_fid = b3[b3["commentary_required"] & b3["gold_doc"].notna()]
    b3_ev = b3[b3.stratum == "evidence_less"]
    b3_evidenced = b3[b3["retrieval_hit"] == True]  # noqa: E712

    base_rate = cfgcmp[cfgcmp.config.str.contains("baseline")].iloc[0]
    prin_rate = cfgcmp[cfgcmp.config.str.contains("principled")].iloc[0]

    before_b3 = ba[ba.config == "before_module7_config"]
    # Module 7 stored response times in generated_outputs.csv, not evaluation_scores.csv
    m7_gen = pd.read_csv(M7 / "generated_outputs.csv")
    m7_speed = round(m7_gen[m7_gen.response_time_s > 0].response_time_s.mean(), 2)
    m8_speed = round(b3[b3.response_time_s > 0].response_time_s.mean(), 2)

    rows = [
        {"Area": "Retrieval hit rate (same hold-out cases)",
         "Module 7 prototype": f"{base_rate.hits}/{base_rate.n} ({base_rate.rate:.0%})",
         "Module 8 final": f"{prin_rate.hits}/{prin_rate.n} ({prin_rate.rate:.0%})",
         "Evidence": "results/retrieval_config_comparison.csv; precedent documents "
                     "excluded from the causes axis"},
        {"Area": "Numeric fidelity",
         "Module 7 prototype": f"{frac(m7_num_pass, len(m7_num))} (C3 derived a threshold)",
         "Module 8 final": frac(int(b3_fid['numeric_fidelity'].sum()), len(b3_fid)),
         "Evidence": "materiality thresholds now supplied as verified figures, so the "
                     "model has no reason to compute one"},
        {"Area": "Correct abstention",
         "Module 7 prototype": frac(int((m7_abst['Score / Result'] == 'PASS').sum()), len(m7_abst)),
         "Module 8 final": frac(int(b3_ev['abstained'].sum()), len(b3_ev)),
         "Evidence": f"evidence-less stratum grown from 2 cases to 8, of which "
                     f"{len(b3_ev)} fall in the hold-out periods reported here"},
        {"Area": "Over-abstention (abstained with evidence present)",
         "Module 7 prototype": "not measured",
         "Module 8 final": frac(int(b3_evidenced['abstained'].sum()), len(b3_evidenced)),
         "Evidence": "metric added; measuring only correct abstention had hidden a "
                     "gate that abstained on everything"},
        {"Area": "Test cases",
         "Module 7 prototype": f"{m7['Test ID'].nunique()} hand-picked",
         "Module 8 final": f"{m8.case_id.nunique()} across "
                           f"{m8.period.nunique()} consecutive close periods",
         "Evidence": "data/test_cases.csv"},
        {"Area": "Comparison arms",
         "Module 7 prototype": "1 (B3 only)",
         "Module 8 final": f"{m8.arm.nunique()} (B0 to B4)",
         "Evidence": "results/benchmark_results.csv"},
        {"Area": "Validation design",
         "Module 7 prototype": "none, configured and reported on the same cases",
         "Module 8 final": "temporal split, configured on 2 periods and reported on 8",
         "Evidence": "an absolute threshold scored 4/4 on the config periods and then "
                     "failed on 100% of hold-out cases"},
        {"Area": "Corpus",
         "Module 7 prototype": "12 documents",
         "Module 8 final": f"{len(corpus)} documents",
         "Evidence": "D16 FX, D17 revolver and covenant, D18 stale launch brief added"},
        {"Area": "Materiality rules",
         "Module 7 prototype": "dollar thresholds only",
         "Module 8 final": "dollar thresholds plus the covenant trigger",
         "Evidence": "D01 4.3; a $104K interest movement is immaterial in dollars yet "
                     "reportable on covenant headroom"},
        {"Area": "Serving and speed",
         "Module 7 prototype": f"Colab VM, {m7_speed}s per case",
         "Module 8 final": f"local Ollama, {m8_speed}s per case",
         "Evidence": "the deployment premise is that pre-release figures do not leave "
                     "the organisation, now demonstrated rather than argued"},
    ]
    imp = pd.DataFrame(rows)
    imp.to_csv(RESULTS / "improvement_comparison.csv", index=False)
    print("improvement_comparison.csv")
    print(imp[["Area", "Module 7 prototype", "Module 8 final"]].to_string(index=False))

    # ---------------- chart 1: the ladder ----------------
    summ = pd.read_csv(RESULTS / "benchmark_results.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    fig.patch.set_facecolor("white")

    def as_rate(col):
        out = []
        for v in summ[col]:
            try:
                a, b = str(v).split("/"); out.append(int(a) / int(b) if int(b) else 0)
            except Exception:
                out.append(0)
        return out

    # Panel 1: correct abstention.
    ax = axes[0]
    vals = as_rate("abstention_correct")
    bars = ax.bar(summ["arm"], vals, color=GREEN, edgecolor="#243b53")
    ax.set_title("Correct abstention\n(evidence-less cases)", fontsize=11,
                 fontweight="bold")
    ax.set_ylim(0, 1.18)
    for b, v, lbl in zip(bars, vals, summ["abstention_correct"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.06, f"{v:.0%}", ha="center",
                fontsize=9, fontweight="bold")
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, lbl, ha="center",
                fontsize=8, color="#44607a")

    # Panel 2: numeric fidelity at both scopes. Scoring only the documented cases
    # answers whether a stated cause carries sound figures, and cannot see a figure
    # invented while abstaining. B2 looks perfect on the narrow scope and imports
    # figures from unrelated documents on the wide one, so both are plotted.
    ax = axes[1]
    doc_r, all_r = as_rate("numeric_fidelity"), as_rate("numeric_fidelity_all")
    xs = range(len(summ)); w = 0.33
    b1 = ax.bar([i - w / 2 for i in xs], doc_r, w, label="documented cases only",
                color=GREY, edgecolor="#243b53")
    b2 = ax.bar([i + w / 2 for i in xs], all_r, w, label="every case that produced text",
                color=BLUE, edgecolor="#243b53")
    # The raw fraction only, and the two series sit at different heights. Adjacent
    # bars often reach the same height, and side-by-side labels there run together
    # into one unreadable string.
    for bars_, lbls, dy in ((b1, summ["numeric_fidelity"], 0.115),
                            (b2, summ["numeric_fidelity_all"], 0.035)):
        for b, lbl in zip(bars_, lbls):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy, lbl,
                    ha="center", fontsize=8.4, fontweight="bold", color="#243b53")
    ax.set_xticks(list(xs)); ax.set_xticklabels(summ["arm"])
    ax.set_title("Numeric fidelity\n(two scopes)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.25)
    ax.legend(frameon=False, fontsize=7.6, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)

    # Panel 3: cost.
    ax = axes[2]
    vals = summ["mean_seconds"].tolist()
    bars = ax.bar(summ["arm"], vals, color=GREY, edgecolor="#243b53")
    ax.set_title("Mean response time (s)", fontsize=11, fontweight="bold")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}", ha="center",
                fontsize=9)

    for ax in (axes[0], axes[2]):
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Baseline ladder on the hold-out periods (20 cases, 8 closes)",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, -0.02,
             "B0 is a template and calls no model, so its response time is zero by "
             "construction. B4 abstains before generating on gated cases, which is "
             "why it is faster than B3.",
             ha="center", fontsize=8.5, style="italic", color="#44607a")
    fig.tight_layout()
    if may_write(IMAGES / "evaluation_chart.png"):
        fig.savefig(IMAGES / "evaluation_chart.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\nevaluation_chart.png")

    # ---------------- chart 2: before and after ----------------
    labels = ["Retrieval\nhit rate", "Correct\nabstention", "Numeric\nfidelity"]
    m7_abst_n, m7_abst_d = int((m7_abst["Score / Result"] == "PASS").sum()), max(len(m7_abst), 1)
    m8_abst_n, m8_abst_d = int(b3_ev["abstained"].sum()), max(len(b3_ev), 1)
    m8_fid_n, m8_fid_d = int(b3_fid["numeric_fidelity"].sum()), max(len(b3_fid), 1)

    before = [base_rate.rate, m7_abst_n / m7_abst_d, m7_num_pass / max(len(m7_num), 1)]
    after = [prin_rate.rate, m8_abst_n / m8_abst_d, m8_fid_n / m8_fid_d]
    # Raw fractions matter as much as the rate. Correct abstention is 100% in both
    # columns, but on 2 cases before and 8 after, so the rate alone would suggest
    # nothing changed when the measurement actually became far more trustworthy.
    before_lbl = [f"{int(base_rate.hits)}/{int(base_rate.n)}",
                  f"{m7_abst_n}/{m7_abst_d}", f"{m7_num_pass}/{len(m7_num)}"]
    after_lbl = [f"{int(prin_rate.hits)}/{int(prin_rate.n)}",
                 f"{m8_abst_n}/{m8_abst_d}", f"{m8_fid_n}/{m8_fid_d}"]

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    fig.patch.set_facecolor("white")
    x = range(len(labels)); w = 0.36
    b1 = ax.bar([i - w / 2 for i in x], before, w, label="Module 7 prototype",
                color=GREY, edgecolor="#243b53")
    b2 = ax.bar([i + w / 2 for i in x], after, w, label="Module 8 final",
                color=GREEN, edgecolor="#243b53")
    for bars, lbls in ((b1, before_lbl), (b2, after_lbl)):
        for b, l in zip(bars, lbls):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.055,
                    f"{b.get_height():.0%}", ha="center", fontsize=10, fontweight="bold")
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                    l, ha="center", fontsize=8.5, color="#44607a")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.25); ax.set_ylabel("score")
    ax.set_title("Improvement from the Module 7 prototype\n"
                 "retrieval measured on the same hold-out cases under both configurations",
                 fontsize=12, fontweight="bold")
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    # Spell the counts out of the data. An earlier caption said "8 after", which is the
    # size of the whole evidence-less stratum rather than its hold-out share, and it did
    # not match the fraction printed on the bar directly above it.
    ax.text(0.5, -0.16,
            f"Correct abstention reads 100% in both columns, but on {m7_abst_d} cases "
            f"before and {m8_abst_d} after. Over-abstention was not measured at all in "
            f"Module 7.",
            transform=ax.transAxes, ha="center", fontsize=8.5, style="italic", color="#44607a")
    fig.tight_layout()
    if may_write(IMAGES / "improvement_chart.png"):
        fig.savefig(IMAGES / "improvement_chart.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("improvement_chart.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
