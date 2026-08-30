"""
Package the repository to the filenames the Module 8 brief prescribes.

The brief's rule is that a reviewer must locate any artifact within 60 seconds, so
the prescribed names have to exist. The working package is `src/gvc/`, which the
parity test imports, so renaming it would break a passing test for cosmetic reasons.
Thin modules at the prescribed names re-export from `gvc` instead.

Also exports two human-readable views of data that otherwise lives only in code or
in one large JSON file: the evaluation cases as a CSV, and each corpus document as
a Markdown file a finance reviewer can read without a JSON viewer.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from gvc.cases import CLOSE_PERIODS, CONFIG_PERIODS  # noqa: E402

DATA = ROOT / "data"
SRC = ROOT / "src"


# ------------------------------------------------------------------ test cases
def export_test_cases() -> None:
    rows = []
    for c in CLOSE_PERIODS:
        vf = c.verified_figures()
        rows.append({
            "case_id": c.case_id,
            "period": c.period,
            "split": "config" if c.period in CONFIG_PERIODS else "holdout",
            "stratum": c.stratum,
            "account": c.account,
            "cost_centre": c.cost_centre,
            "region": c.region,
            "product_line": c.product_line,
            "actual": c.actual,
            "budget": c.budget,
            "variance_usd": c.variance_usd,
            "variance_pct": c.variance_pct,
            "commentary_required": c.commentary_required,
            "materiality_trigger": c.materiality_trigger,
            "price_effect": c.price_effect,
            "volume_effect": c.volume_effect,
            "mix_effect": c.mix_effect,
            "covenant_headroom": c.covenant_headroom,
            # Ground truth. Held out of every prompt; used only to score.
            "gold_doc": c.gold_doc,
            "true_cause": c.true_cause,
        })
        assert vf  # verified_figures must build for every case
    df = pd.DataFrame(rows)
    df.to_csv(DATA / "test_cases.csv", index=False)
    print(f"data/test_cases.csv  {len(df)} cases, "
          f"{df.split.value_counts().to_dict()}, "
          f"{df.stratum.value_counts().to_dict()}")


# ------------------------------------------------------------ sample documents
def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def export_sample_documents() -> None:
    out = DATA / "sample_documents"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.md"):
        old.unlink()

    docs = json.loads((DATA / "corpus_documents.json").read_text())
    index = []
    for d in docs:
        name = f"{d['doc_id']}_{slug(d['doc_title'])}.md"
        lines = [
            f"# {d['doc_id']} · {d['doc_title']}",
            "",
            f"- Type: {d['doc_type']} (retrieval category `{d['category']}`)",
            f"- Version: {d['version']}"
            + (f", supersedes {d['supersedes']}" if d.get("supersedes") else ""),
            f"- Published: {d['published_date']}",
            f"- Effective: {d['effective_from']} to "
            f"{d['effective_to'] or 'open'}",
            f"- Owner: {d['owner']}",
            f"- Entity scope: {', '.join(d['entity_scope'])}",
            f"- Topics: {', '.join(d['topic'])}",
            f"- Sensitivity: {d['sensitivity']}",
            "",
        ]
        for s in d["sections"]:
            lines += [f"## {s['section']}", "", s["text"], ""]
        (out / name).write_text("\n".join(lines))
        index.append({
            "doc_id": d["doc_id"], "title": d["doc_title"],
            "category": d["category"], "published": d["published_date"],
            "file": name,
        })

    index.sort(key=lambda r: r["doc_id"])
    idx_rows = ["| doc | title | category | published | file |",
                "|---|---|---|---|---|"]
    idx_rows += [f"| {r['doc_id']} | {r['title']} | `{r['category']}` | "
                 f"{r['published']} | [{r['file']}]({r['file']}) |" for r in index]
    readme = [
        "# Synthetic corpus",
        "",
        "Every document here is invented. No real company's figures, policies, or "
        "internal memos appear anywhere in this project. The documents are written "
        "to mirror the artefacts a mid-size manufacturer actually produces at close, "
        "so that retrieval faces the same difficulties it would face in production: "
        "several versions of the same policy, memos that were published after the "
        "close they would explain, and near-duplicate memos about the wrong entity.",
        "",
        "The retrieval `category` is what splits the corpus into the two axes.",
        "",
        "| category | axis | what it is |",
        "|---|---|---|",
        "| `rules` | rules axis | policy, materiality thresholds, ownership |",
        "| `causes` | causes axis | driver memos that explain one variance |",
        "| `precedent` | excluded | prior-period commentary packs |",
        "| `distractor` | causes axis | plausible memos about the wrong scope or period |",
        "",
        "`corpus_documents.json` is the machine-readable source of truth. These "
        "Markdown files are generated from it so a finance reviewer can read them "
        "directly.",
        "",
        "## Index",
        "",
        "\n".join(idx_rows),
        "",
    ]
    (out / "README.md").write_text("\n".join(readme))
    print(f"data/sample_documents/  {len(docs)} documents plus an index")


# ------------------------------------------------------------- src/ alias modules
ALIASES = {
    "rag_pipeline.py": (
        "Retrieval-augmented pipeline: corpus, retrieval, prompting, generation.",
        ["from gvc.corpus import CAUSE_CATS, RULES_CATS, admissible, load_corpus, month_end",
         "from gvc.retrieval import Retriever, RetrievalConfig",
         "from gvc.prompt import (build_grounded_prompt, build_longcontext_prompt,",
         "                        build_ungrounded_prompt, deterministic_template)",
         "from gvc.generate import ARMS, DEFAULT_MODEL, generate, ollama_available"],
        ["CAUSE_CATS", "RULES_CATS", "admissible", "load_corpus", "month_end",
         "Retriever", "RetrievalConfig", "build_grounded_prompt",
         "build_longcontext_prompt", "build_ungrounded_prompt",
         "deterministic_template", "ARMS", "DEFAULT_MODEL", "generate",
         "ollama_available"],
    ),
    "evaluation.py": (
        "Scoring and the evaluation harness: metrics, per-arm runs, summary tables.",
        ["from gvc.metrics import abstained, numeric_fidelity, retrieval_hit",
         "from gvc.harness import run_arm, save, score_generation, summarise",
         "from gvc.cases import CLOSE_PERIODS, CONFIG_PERIODS, load_cases"],
        ["abstained", "numeric_fidelity", "retrieval_hit", "run_arm", "save",
         "score_generation", "summarise", "CLOSE_PERIODS", "CONFIG_PERIODS",
         "load_cases"],
    ),
    "utils.py": (
        "The deterministic variance engine and its materiality rules.",
        ["from gvc.variance_engine import (COVENANT_HEADROOM_TRIGGER, DOLLAR_ALONE_USD,",
         "                                 DOLLAR_AND_PCT_PCT, DOLLAR_AND_PCT_USD,",
         "                                 VarianceRow)"],
        ["COVENANT_HEADROOM_TRIGGER", "DOLLAR_ALONE_USD", "DOLLAR_AND_PCT_PCT",
         "DOLLAR_AND_PCT_USD", "VarianceRow"],
    ),
}

HEADER = '''"""
{doc}

This module exists because the assignment prescribes this filename. The working
implementation lives in `src/gvc/`, which `tests/test_parity.py` imports, so it is
re-exported here rather than moved.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

{imports}

__all__ = {names}
'''


def export_aliases() -> None:
    for fname, (doc, imports, names) in ALIASES.items():
        (SRC / fname).write_text(HEADER.format(
            doc=doc, imports="\n".join(imports),
            names=json.dumps(sorted(names), indent=4).replace('"', '"'),
        ))
        print(f"src/{fname}")

    (SRC / "app.py").write_text('''"""
Entry point for the Streamlit interface.

The assignment prescribes `src/app.py`; the interface itself lives at
`app/streamlit_app.py`, which is where Streamlit convention puts it. Running either
path starts the same application.

    streamlit run src/app.py
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import runpy
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
runpy.run_path(str(APP), run_name="__main__")
''')
    print("src/app.py")


def main() -> int:
    export_test_cases()
    export_sample_documents()
    export_aliases()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
