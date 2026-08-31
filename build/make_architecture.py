"""
Draw images/system_architecture.png.

The diagram's job is to make the separation of authority obvious at a glance, because
that separation is the whole argument of the project. Numbers come from a
deterministic engine, rules come from retrieved policy, and causes come from
retrieved driver memos. The language model writes prose over all three and is
permitted to originate none of them.

Connections are drawn as explicit elbows rather than curved arrows, because a curve
that clips through a text box reads as a mistake even when the route is correct.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
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


NAVY, GREY, BLUE, GREEN, RED = "#0d2c4d", "#7f9db8", "#2e8bc9", "#1f9d76", "#c0554f"
INK, FAINT = "#1b2b3a", "#f4f7fa"


def box(ax, x, y, w, h, title, lines, colour, title_size=10, body_size=8.2, lead=2.3):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
        linewidth=1.8, edgecolor=colour, facecolor="white", zorder=3))
    ax.add_patch(Rectangle((x, y + h - 0.32), w, 0.32, linewidth=0,
                           facecolor=colour, zorder=4))
    ax.text(x + w / 2, y + h - 2.0, title, ha="center", va="center", zorder=5,
            fontsize=title_size, fontweight="bold", color=colour, parse_math=False)
    for i, line in enumerate(lines):
        ax.text(x + w / 2, y + h - 4.3 - i * lead, line, ha="center", va="center",
                fontsize=body_size, color=INK, zorder=5, parse_math=False)


def elbow(ax, points, colour=GREY, lw=1.7):
    """Draw an orthogonal route through `points`, arrowhead on the final segment."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.plot(xs[:-1], ys[:-1], color=colour, linewidth=lw, zorder=2,
            solid_capstyle="round", solid_joinstyle="round")
    ax.add_patch(FancyArrowPatch(
        points[-2], points[-1], arrowstyle="-|>", mutation_scale=13,
        linewidth=lw, color=colour, zorder=2, shrinkA=0, shrinkB=0))


def lane(ax, x, y, w, h, label, colour):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.8,rounding_size=1.5",
        linewidth=1.1, edgecolor=colour, facecolor=FAINT, alpha=0.5, zorder=1,
        linestyle=(0, (5, 3))))
    ax.text(x + 1.4, y + h - 1.4, label, ha="left", va="center", zorder=2,
            fontsize=9.5, fontweight="bold", color=colour, parse_math=False)


def main() -> int:
    fig, ax = plt.subplots(figsize=(16.5, 9))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 165); ax.set_ylim(0, 90); ax.axis("off")

    ax.text(82, 88.0, "Grounded Variance Commentary: system architecture",
            ha="center", fontsize=15.5, fontweight="bold", color=NAVY,
            parse_math=False)
    ax.text(82, 85.7,
            "Three separate authorities. The language model writes the prose and "
            "originates none of the content.",
            ha="center", fontsize=10, color="#44607a", style="italic",
            parse_math=False)

    # ---------------------------------------------------------------- lanes
    lane(ax, 3, 47, 72, 31, "NUMBERS AUTHORITY  ·  deterministic, no model involved", BLUE)
    lane(ax, 3, 7, 72, 38, "EVIDENCE AUTHORITY  ·  retrieval over a versioned corpus", GREEN)
    lane(ax, 84, 7, 78, 73, "GENERATION AND CONTROL", NAVY)

    # ------------------------------------------------------------- numbers lane
    box(ax, 6, 49, 28, 24, "Close ledger row", [
        "One account, one cost centre,", "one close period.",
        "Actual and budget only.", "",
        "26 rows across 10 consecutive", "closes."], GREY)

    box(ax, 42, 49, 31, 24, "Deterministic variance engine", [
        "variance $ and %, price / volume / mix",
        "Materiality D01 §4.2:",
        "≥ $250K and ≥ 5%, or ≥ $500K",
        "Covenant trigger D01 §4.3:",
        "headroom ≤ 0.25× EBITDA",
        "Emits verified figures, including",
        "the thresholds themselves."], BLUE, body_size=8.0, lead=2.2)

    elbow(ax, [(34.5, 61), (41.0, 61)], BLUE)

    # ------------------------------------------------------------ evidence lane
    box(ax, 6, 10, 28, 28, "Synthetic corpus", [
        "15 documents, 30 sections.", "",
        "rules 3   ·   causes 9",
        "precedent 1   ·   distractor 2", "",
        "Each carries version, publication",
        "date, effective window, owner,", "and entity scope."], GREY)

    box(ax, 42, 10, 31, 28, "Point-in-time filter", [
        "A document is retrievable only if", "",
        "published_date ≤ close_date",
        "and close_date falls inside",
        "[effective_from, effective_to)", "",
        "The 2025-10 write-down memo does",
        "not exist in the 2023-06 close, so",
        "lookahead leakage is structural,",
        "not a matter of prompt discipline."], GREEN, body_size=8.0, lead=2.2)

    elbow(ax, [(34.5, 24), (41.0, 24)], GREEN)

    # ---------------------------------------------------- feeds into generation
    # Evidence rises on the outside and drops into both axes, so neither axis looks
    # downstream of the other.
    elbow(ax, [(73.8, 31), (78.0, 31), (78.0, 82.5), (142.0, 82.5), (142.0, 77.2)], GREEN)
    elbow(ax, [(102.0, 82.5), (104.0, 82.5), (104.0, 77.2)], GREEN)
    # Verified figures bypass retrieval entirely and land in the prompt.
    elbow(ax, [(73.8, 65), (80.8, 65), (80.8, 28), (86.0, 28)], BLUE)

    # ------------------------------------------------------------ retrieval axes
    box(ax, 87, 57, 34, 19, "Rules axis   k = 2", [
        "category = rules", "",
        "Policy, thresholds, and the",
        "cost-centre owner register.", "",
        "Answers whether commentary is",
        "required, and who owns the line."], BLUE, title_size=9.6, body_size=8.0,
        lead=2.05)

    box(ax, 125, 57, 34, 19, "Causes axis   k = 3", [
        "category = causes, distractor",
        "precedent EXCLUDED", "",
        "Driver memos explaining one",
        "specific variance.", "",
        "A separate budget, so policy",
        "cannot crowd out a memo."], GREEN, title_size=9.6, body_size=8.0,
        lead=2.05)

    # ------------------------------------------------------------------- gate
    box(ax, 87, 40, 72, 13, "Sufficiency gate   ·   decided before generation", [
        "Passes only if a retrieved driver clears the similarity floor τ and matches "
        "the row's entity scope.",
        "Every retrieved driver is scanned, because the supporting memo often sits "
        "behind an irrelevant one.",
        "Deciding here beats instructing the model to abstain, since a generator given "
        "a \"why\" will answer it."], NAVY, title_size=10, body_size=7.9, lead=2.2)

    elbow(ax, [(104, 56.4), (104, 53.4)], NAVY)
    elbow(ax, [(142, 56.4), (142, 53.4)], NAVY)

    # -------------------------------------------------------------------- model
    box(ax, 87, 23, 72, 13, "Local SLM   ·   qwen3:4b-instruct served by Ollama", [
        "The prompt carries the verified figures as fact, and forbids the model from "
        "recomputing any of them.",
        "Inference runs on the analyst's own machine, so pre-release figures never "
        "leave the organisation.",
        "Measured at 1.25 s per generated case locally, against 142 s per case on a "
        "hosted notebook VM."], NAVY, title_size=10, body_size=7.9, lead=2.2)

    elbow(ax, [(123, 39.4), (123, 36.4)], NAVY)

    # ----------------------------------------------------------------- outcomes
    box(ax, 87, 9, 34, 11, "Gate passed", [
        "Commentary citing the driver memo",
        "and the policy section, naming",
        "the accountable owner."], GREEN, title_size=9.5, body_size=8.0, lead=2.2)

    box(ax, 125, 9, 34, 11, "Gate failed", [
        "\"Cause: unsupported\".",
        "No cause is invented. The line is",
        "escalated to the cost-centre owner."], RED, title_size=9.5, body_size=8.0,
        lead=2.2)

    elbow(ax, [(104, 22.4), (104, 20.4)], GREEN)
    elbow(ax, [(142, 22.4), (142, 20.4)], RED)

    # ----------------------------------------------------------------- eval band
    ax.add_patch(FancyBboxPatch(
        (3, 0.8), 159, 4.0, boxstyle="round,pad=0.4,rounding_size=1.0",
        linewidth=1.4, edgecolor=GREY, facecolor="white", zorder=3))
    ax.text(82.5, 2.8,
            "EVALUATION   ·   arms B0 template, B1 ungrounded, B2 long-context, "
            "B3 RAG, B4 RAG with gate   ·   retrieval hit@3, numeric fidelity, "
            "correct abstention reported with over-abstention   ·   "
            "temporal split: configured on 2 closes, reported on 8",
            ha="center", va="center", fontsize=7.2, color=INK, zorder=5,
            parse_math=False)

    out = IMAGES / "system_architecture.png"
    if may_write(out):
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        print(out)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
