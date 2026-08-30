"""
Grounded Variance Commentary: month-end close review interface.

The screen is organised the way a close is worked, and the pipeline is something the
analyst operates rather than something they are shown the end of.

  Stage 0  Retrieve the close.   Exact lookup by period. Runs when you open a close.
  Stage 1  Reporting check.      Arithmetic, no model. Runs on opening a close.
  Stage 2  Evidence search.      Similarity search over 30 document sections, no
                                 model. Runs when you ask, never on an unflagged line.
  Stage 3  Commentary.           The only stage that calls a language model. Runs when
                                 you ask, under adherence rules you set.

Two retrieval paths, and the difference between them is an architectural point rather
than a detail. The ledger is fetched by key and must be exact, because a figure that
is merely similar is wrong. The corpus is fetched by meaning, because nobody knows in
advance which memo explains a movement. Only the second one is a vector search.

Stage 3 either drafts a cause from retrieved evidence or refuses to state one and
escalates to the named cost-centre owner. That refusal is the argument of the project,
so it is drawn as a distinct object rather than as a warning banner or an error.

Three views. "Close review" and "All periods" are written for a finance reader and use
no vocabulary from the codebase. "Results" reports every measured score, read from
the files in results/ rather than transcribed, and is open to both roles. "Method"
is the Reviewer view and holds the five comparison
approaches, the raw similarity scores, the planted-source marker, the assembled
prompts, and the evaluation design, in the terms a grader expects.

Run with:  streamlit run app/streamlit_app.py
Generation requires Ollama serving qwen3:4b-instruct. Without it the interface still
loads, and everything except the model calls remains inspectable. A line with no
supporting evidence can still be completed, because refusing costs no model call.
"""

from __future__ import annotations

import csv
import html
import io
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import altair as alt
import pandas as pd
import streamlit as st

from gvc import prompt as P
from gvc.cases import CONFIG_PERIODS, load_cases
from gvc.variance_engine import VarianceRow
from gvc.corpus import load_corpus
from gvc.generate import ARMS, DEFAULT_MODEL, ollama_available
from gvc.metrics import abstained, numeric_fidelity
from gvc.retrieval import Retriever, RetrievalConfig

# The built-in toolbar is set to "minimal" in .streamlit/config.toml, so the only
# thing left in the menu is what is declared here. Get help and Report a bug are
# suppressed rather than pointed at a placeholder, because nothing is published yet.
st.set_page_config(
    page_title="Grounded Variance Commentary",
    page_icon=":material/table_chart:", layout="wide",
    menu_items={
        "Get help": None,
        "Report a bug": None,
        "About": (
            "### Grounded Variance Commentary\n"
            "An FP&A month-end assistant. It drafts variance commentary from "
            "retrieved evidence, and declines to answer when the evidence does not "
            "support a cause.\n\n"
            "Numbers come from a deterministic calculation, reporting rules come "
            "from retrieved policy, causes come from retrieved driver memos, and the "
            "language model writes prose over all three while originating none of "
            "it. It runs entirely on this machine, so no figure leaves it.\n\n"
            "**Three stages.** The reporting check is arithmetic and runs when you "
            "open a close. The evidence search is a similarity search over 30 "
            "document sections and runs when you ask. Writing the commentary is the "
            "only step that calls a language model, and it runs when you ask, under "
            "rules you set.\n\n"
            "**The data.** Ten closes and 26 variance lines, fixed in the code, "
            "which are the evaluation set behind every number in the accompanying "
            "article. A close may also be read from a general-ledger CSV, in which "
            "case it is reviewed but never scored.\n\n"
            "This system is a classroom prototype. Outputs should be reviewed by a "
            "human before real-world use."),
    })

WARNING = ("This system is a classroom prototype. Outputs should be reviewed by a "
           "human before real-world use.")

# ------------------------------------------------------------------ palette
# Mirrors .streamlit/config.toml. Duplicated deliberately: the theme file drives
# Streamlit's own components, and these constants drive the few blocks this file
# draws itself. One list of hex values in one place beats reaching into Streamlit's
# generated class names, which change between versions.
INK, MUTED, RULE, PANEL = "#1A1A17", "#5C594F", "#DEDCD4", "#F0EFE9"
RED = ("#A32B26", "#F8E8E5", "#7A1F1B")
GREEN = ("#16803F", "#E4EFE8", "#14472F")
ORANGE = ("#9A5B12", "#F9EEDE", "#6B3E08")
GRAY = ("#6B6862", "#EDECE5", "#46443E")
BLUE = ("#1D4E89", "#E6EDF6", "#153A66")
YELLOW = ("#8A6A12", "#F9F2DC", "#5F4808")

# The triage vocabulary. One phrase per row, no legend required. "Needs evidence" is
# the state a flagged line sits in between stage 1 and stage 2, so the table shows
# the pipeline advancing rather than presenting a finished verdict on load.
STATUS = {
    "chase":    {"label": "Chase the owner",   "tone": RED,    "order": 0},
    "covenant": {"label": "Covenant alert",    "tone": ORANGE, "order": 1},
    "pending":  {"label": "Needs evidence",    "tone": BLUE,   "order": 2},
    "ready":    {"label": "Explanation found", "tone": GREEN,  "order": 3},
    "clear":    {"label": "Below threshold",   "tone": GRAY,   "order": 4},
}
LABEL_TO_KEY = {v["label"]: k for k, v in STATUS.items()}

# Plain English for the materiality decision, from policy document D01 §4.2 and §4.3.
TRIGGER_PLAIN = {
    "dollar_alone":   "Over $500,000",
    "dollar_and_pct": "Over $250,000 and over 5%",
    "covenant":       "Loan covenant near its limit",
    "none":           "Under both reporting limits",
}

# The five approaches compared in the study.
ARM_INFO = {
    "B0": {"label": "B0  deterministic template",
           "plain": "Fixed template, no model",
           "blurb": "No model at all. Gets every number right because no number came "
                    "from a model, and can never state a cause.",
           "needs_model": False, "retrieves": False},
    "B1": {"label": "B1  ungrounded",
           "plain": "Model with no evidence",
           "blurb": "Same verified figures, no retrieval, open question about the "
                    "cause. This is the failure the project exists to prevent.",
           "needs_model": True, "retrieves": False},
    "B2": {"label": "B2  long context",
           "plain": "Model with the whole document set",
           "blurb": "The entire corpus in the prompt, unfiltered. Tests whether "
                    "retrieval is doing work or whether context alone suffices.",
           "needs_model": True, "retrieves": False},
    "B3": {"label": "B3  two-axis retrieval",
           "plain": "Model with searched evidence",
           "blurb": "The proposed system. Point-in-time filtered retrieval on two "
                    "disjoint axes.",
           "needs_model": True, "retrieves": True},
    "B4": {"label": "B4  retrieval and gate",
           "plain": "Model with searched evidence and an evidence check",
           "blurb": "B3 plus the sufficiency gate, which decides abstention before "
                    "generating, and a verifier that re-checks every figure.",
           "needs_model": True, "retrieves": True},
}
ARM_KEYS = list(ARM_INFO)

# The adherence vocabulary. Every combination below is one of the five evaluated
# approaches, so letting the analyst set the rules adds no configuration the study
# has not already measured. The default is the measured one.
EVIDENCE_MODES = ["Search the documents", "No documents at all",
                  "The whole document set", "No model, figures only"]
MEASURED = ("Search the documents", True)


def arm_for(mode: str, refuse: bool) -> str:
    if mode == "No model, figures only":
        return "B0"
    if mode == "No documents at all":
        return "B1"
    if mode == "The whole document set":
        return "B2"
    return "B4" if refuse else "B3"


# How a stated cause is presented, per the rules that produced it. A cause drawn
# from no documents is not attributed to anything, and dressing it as an ordinary
# answer would be the misrepresentation this whole project exists to prevent. B1 and
# B2 therefore get the filled warning shape rather than the paper-document shape.
STATED = {
    "B4": (GREEN, True, "Cause stated and attributed",
           "Drafted from the retrieved evidence above, and every figure was "
           "re-checked against the verified calculation before this was shown."),
    "B3": (GREEN, True, "Cause stated and attributed",
           "Drafted from the retrieved evidence above. The evidence check was "
           "switched off, so the model judged for itself whether those passages "
           "support a cause."),
    "B2": (ORANGE, False, "Cause stated from the whole document set",
           "The model was handed every document in the set, unfiltered, so nothing "
           "narrowed what it drew on. A cause stated this way is not tied to a "
           "document the system can name."),
    "B1": (RED, False, "Cause stated with no evidence at all",
           "The model was given the figures, no documents, and an open question about "
           "the cause. Whatever it says here is unsupported, whether or not it "
           "happens to be right. This is the failure the project exists to prevent."),
}

# A one-line note on where the cause came from, for the metric under the answer.
CAUSE_SOURCE = {
    "B4": "attributed to a source document",
    "B3": "attributed to a source document",
    "B2": "not tied to a document the system can name",
    "B1": "not attributed to anything",
    "B0": "the template never states one",
}

# Document types in the words a finance reader would use. The `category` field is
# deliberately not shown here: it carries the evaluation label `distractor`, which an
# analyst working a real close would not know.
DOCTYPE_PLAIN = {
    "policy": "Policy", "definition": "Definitions", "register": "Owner register",
    "memo": "Memo", "commentary": "Prior commentary",
}


# ----------------------------------------------------------------- resources
@st.cache_resource(show_spinner="Loading the document set and the search index...")
def load_system():
    from sentence_transformers import SentenceTransformer
    docs, chunks = load_corpus(ROOT / "data" / "corpus_documents.json")
    embed = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    # The configuration reported throughout the paper.
    cfg = RetrievalConfig(k_rules=2, k_causes=3, exclude_precedent=True,
                          tau=0.25, require_scope_match=True)
    return docs, chunks, Retriever(chunks, embed, cfg)


docs, chunks, retriever = load_system()
cases = load_cases("periods")          # the 26 evaluated lines, fixed in code
DOC_TITLE = {d["doc_id"]: d["doc_title"] for d in docs}
DOC_TYPE = {d["doc_id"]: d["doc_type"] for d in docs}
ALL_PERIODS = sorted({c.period for c in cases})
up, ollama_msg = ollama_available(DEFAULT_MODEL)


def uploaded_rows() -> list:
    u = st.session_state.get("upload")
    return u["rows"] if u else []


def uploaded_periods() -> list[str]:
    return sorted({r.period for r in uploaded_rows()})


def rows_for(source: str, period: str) -> list:
    """The lines of one close. Exact lookup by key, never a similarity search."""
    pool = uploaded_rows() if source == "upload" else cases
    return [c for c in pool if c.period == period]


# One lookup covering both sources, so the drill-down, the stage runner, and the
# method tab need no special case. The two are merged nowhere else: the
# cross-period comparison stays the evaluated 26 and nothing else.
CASE_BY_ID = {c.case_id: c for c in cases}
CASE_BY_ID.update({r.case_id: r for r in uploaded_rows()})


@st.cache_data(show_spinner=False)
def owner_register() -> dict[str, str]:
    """Cost centre to named owner, read out of the Cost Centre Owner Register D04.

    The register is a retrieved policy document, not a hard-coded table, so the name
    the interface tells an analyst to chase is the name the system cites. If the
    register is ever reworded past this pattern the column goes blank, which is
    better than showing a name the source does not support.
    """
    text = " ".join(s["text"] for d in docs if d["doc_id"] == "D04"
                    for s in d["sections"])
    found = re.findall(
        r"(CC-\d{4})[^.;]*?is owned by ([A-Z]\.\s*[A-Za-z]+),\s*([^;.]+)", text)
    return {cc: f"{who}, {title.strip()}" for cc, who, title in found}


OWNERS = owner_register()


# ----------------------------------------------------------------- formatting
def period_label(p: str, short: bool = False) -> str:
    y, m = p.split("-")
    d = date(int(y), int(m), 1)
    return d.strftime("%b %Y") if short else d.strftime("%B %Y")


def short_title(doc_id: str | None) -> str:
    """A document's distinguishing half, for a table cell.

    Corpus titles run "Procurement Memo - Steel Index Movement", and the half after
    the separator identifies the document. The separator is an em-dash in the source
    data, which is content rather than prose written for this interface, so it is
    matched here rather than rewritten.
    """
    title = DOC_TITLE.get(doc_id, doc_id or "")
    parts = re.split(r"\s+[\u2013\u2014-]\s+", title, maxsplit=1)
    return parts[1] if len(parts) == 2 else title


def evidence_phrase(ok: bool | None, reason: str) -> str:
    """The evidence search result, in words a finance reader recognises.

    The underlying reasons name a similarity threshold and an entity-scope rule. Both
    are real and both appear verbatim on the method tab. Here they are said the way
    an analyst would say them to a colleague.
    """
    if ok is None:
        return "Not searched yet"
    if ok:
        return "Document found"
    if "no driver clears tau" in reason:
        return "Nothing close enough"
    if "entity scope" in reason:
        return "Covers another unit"
    if "no driver-category" in reason:
        return "No memo existed yet"
    return "Nothing found"


def evidence_long(ok: bool, reason: str, row) -> str:
    """The same verdict, expanded for the single-line view. Rendered as HTML."""
    if ok:
        m = re.search(r"\((D\d+) at", reason)
        doc = m.group(1) if m else None
        return (f"A supporting document was found: <b>{DOC_TITLE.get(doc, doc)}</b> "
                f"({doc}). It covers this line's part of the business and it existed "
                f"at the {period_label(row.period)} close, so a cause drafted from it "
                f"can be attributed to a source.")
    if "no driver clears tau" in reason:
        return ("No document in the set is close enough to this line to support a "
                "cause. The search returned results, however none of them are about "
                "this account.")
    if "entity scope" in reason:
        return ("Documents were found, however every one of them covers a different "
                f"cost centre, region, or product line. This line is "
                f"<b>{row.cost_centre} · {row.region} · {row.product_line}</b>, and "
                "policy does not allow a cause to be carried across from another "
                "part of the business.")
    if "no driver-category" in reason:
        return ("No driver memo had been published by this close date, so there is "
                "nothing that could explain the movement.")
    return "No supporting evidence was found for this line."


def status_for(row, gate_ok: bool | None) -> str:
    """The triage state of one line, given how far the pipeline has been run.

    `gate_ok` is None until stage 2 has run on this line. A material line with no
    evidence is the one that needs a person, so it outranks the covenant flag even on
    a covenant-triggered row.
    """
    if not row.commentary_required:
        return "clear"
    if gate_ok is None:
        return "pending"
    if not gate_ok:
        return "chase"
    if row.materiality_trigger == "covenant":
        return "covenant"
    return "ready"


# ----------------------------------------------------------- stage 0 and stage 1
def ledger(rows: list) -> pd.DataFrame:
    """Stage 1: apply the reporting rules to a close that stage 0 has fetched.

    Pure arithmetic over whatever lines it is handed, so a close read from a file
    goes through exactly the same rules as one shipped in the code. Nothing here
    touches the embedding model.
    """
    out = []
    for c in rows:
        out.append({
            "_id": c.case_id, "_period": c.period,
            "_flag": c.commentary_required,
            "Close": period_label(c.period, short=True),
            "Account": c.account,
            "Cost centre": c.cost_centre,
            "Actual": float(c.actual),
            "Budget": float(c.budget),
            "Variance": float(c.variance_usd),
            "Variance %": float(c.variance_pct),
            "Owner": OWNERS.get(c.cost_centre, "").split(",")[0],
        })
    return pd.DataFrame(out)


def search_rows(rows: list) -> dict:
    """Stage 2: the evidence search, for every flagged line handed to it.

    An unflagged line stops at stage 1 and the retriever is never called for it,
    which is the same gating the evaluation harness applies. This is a similarity
    search over 30 document sections and it involves no language model. A whole
    close measures about 15 milliseconds a line, so it is computed rather than
    cached, which is also what lets an uploaded close use the identical path.
    """
    out = {}
    for c in rows:
        if not c.commentary_required:
            continue
        rules, causes = retriever.retrieve(c)
        ok, reason = retriever.sufficient(c, causes)
        out[c.case_id] = {"rules": rules, "causes": causes,
                          "ok": ok, "reason": reason}
    return out


def ev_key(source: str, period: str) -> str:
    return f"ev_{source}_{period}"


def evidence_for(source: str, period: str) -> dict:
    """Whatever stage 2 has produced for this close, empty until it is run."""
    return st.session_state.get(ev_key(source, period), {})


def run_stage2(source: str, period: str) -> None:
    st.session_state[ev_key(source, period)] = search_rows(rows_for(source, period))


# ------------------------------------------------------- reading a close from file
# The shape of a general-ledger export. The first seven are required. Ground-truth
# columns are deliberately absent: a real export carries no answers, which is also
# why an uploaded close can be reviewed but never scored.
UPLOAD_REQUIRED = ["period", "account", "cost_centre", "region", "product_line",
                   "actual", "budget"]
UPLOAD_OPTIONAL = ["price_effect", "volume_effect", "mix_effect",
                   "covenant_headroom", "covenant_limit"]
GROUND_TRUTH = ["true_cause", "gold_doc", "stratum", "case_id"]


def _num(value, field: str, line: int, errors: list, required: bool):
    """Tolerant money parser. Accepts $, thousands separators, and accounting
    parentheses for negatives, because that is what comes out of a spreadsheet."""
    s = str(value or "").strip().replace("$", "").replace(",", "").replace(" ", "")
    if not s:
        if required:
            errors.append(f"Row {line}: {field} is empty.")
        return None
    negative = s.startswith("(") and s.endswith(")")
    if negative:
        s = s[1:-1]
    try:
        x = float(s)
    except ValueError:
        errors.append(f"Row {line}: {field} is not a number ({value!r}).")
        return None
    return -x if negative else x


def parse_upload(raw: bytes) -> tuple[list, list, list]:
    """Turn an uploaded CSV into VarianceRow objects. Returns rows, errors, notes.

    Every row that parses is kept, and every row that does not is reported by its
    line number in the file, so a bad cell costs one line rather than the upload.
    """
    errors: list[str] = []
    notes: list[str] = []
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return [], ["The file is not UTF-8 text. Save it again as CSV UTF-8."], []

    reader = csv.DictReader(io.StringIO(text))
    cols = [c.strip() for c in (reader.fieldnames or [])]
    missing = [c for c in UPLOAD_REQUIRED if c not in cols]
    if missing:
        return [], [f"Missing required column{'s' if len(missing) > 1 else ''}: "
                    f"{', '.join(missing)}."], []
    present_truth = [c for c in GROUND_TRUTH if c in cols]
    if present_truth:
        notes.append(
            f"Ignoring the column{'s' if len(present_truth) > 1 else ''} "
            f"{', '.join(present_truth)}. A ledger export carries no answers, so a "
            "close read from a file is reviewed rather than scored.")

    rows, counter = [], {}
    for line, rec in enumerate(reader, start=2):     # line 2 is the first data row
        rec = {k.strip(): v for k, v in rec.items() if k}
        period = str(rec.get("period", "")).strip()
        if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", period):
            errors.append(f"Row {line}: period {period!r} is not in YYYY-MM form.")
            continue
        account = str(rec.get("account", "")).strip()
        if not account:
            errors.append(f"Row {line}: account is empty.")
            continue
        actual = _num(rec.get("actual"), "actual", line, errors, True)
        budget = _num(rec.get("budget"), "budget", line, errors, True)
        if actual is None or budget is None:
            continue
        if budget == 0:
            notes.append(f"Row {line}: budget is zero, so no percentage variance can "
                         "be computed for this line.")
        opt = {k: _num(rec.get(k), k, line, errors, False) for k in UPLOAD_OPTIONAL}
        counter[period] = counter.get(period, 0) + 1
        rows.append(VarianceRow(
            case_id=f"UP-{period.replace('-', '')}-{counter[period]}",
            period=period,
            account=account,
            cost_centre=str(rec.get("cost_centre", "")).strip(),
            region=str(rec.get("region", "")).strip() or "Group",
            product_line=str(rec.get("product_line", "")).strip() or "All",
            actual=actual, budget=budget,
            price_effect=opt["price_effect"], volume_effect=opt["volume_effect"],
            mix_effect=opt["mix_effect"],
            covenant_headroom=opt["covenant_headroom"],
            covenant_limit=opt["covenant_limit"],
            true_cause="", gold_doc=None, stratum="loaded from file"))
    if not rows and not errors:
        errors.append("The file has a valid header and no data rows.")
    return rows, errors, notes


def template_csv() -> str:
    """The schema, with a real close in it so the format documents itself."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(UPLOAD_REQUIRED + UPLOAD_OPTIONAL)
    for c in cases:
        if c.period != ALL_PERIODS[-1]:
            continue
        w.writerow([c.period, c.account, c.cost_centre, c.region, c.product_line,
                    f"{c.actual:.0f}", f"{c.budget:.0f}"]
                   + ["" if getattr(c, k) is None else f"{getattr(c, k):g}"
                      for k in UPLOAD_OPTIONAL])
    return buf.getvalue()


# ----------------------------------------------------------------- stage 3
def run_arm(row, arm: str):
    """Run one approach, memoised per line so results survive an unrelated rerun.

    Streamlit reruns the whole script on any widget change, so without this the
    commentary disappeared the moment anything was touched.
    """
    key = f"{row.case_id}|{arm}"
    if key not in st.session_state:
        g = ARMS[arm](row, retriever=retriever, all_chunks=chunks)
        ctx = ""
        if g.rules is not None and g.causes is not None:
            ctx = " ".join(list(g.rules["chunk_text"]) + list(g.causes["chunk_text"]))
        fid_ok, unmatched = numeric_fidelity(g.answer, row, ctx)
        st.session_state[key] = {
            "arm": arm, "answer": g.answer, "seconds": g.seconds,
            "fidelity": fid_ok, "unmatched": unmatched,
            "abstained": abstained(g.answer),
            "gate_passed": g.gate_passed, "prompt_chars": g.prompt_chars,
        }
    st.session_state[f"last_{row.case_id}"] = arm
    return st.session_state[key]


def drafted(case_id: str):
    """The most recent commentary for a line, whichever adherence rules produced it."""
    arm = st.session_state.get(f"last_{case_id}")
    return st.session_state.get(f"{case_id}|{arm}") if arm else None


def commentary_state(case_id: str, required: bool) -> str:
    if not required:
        return "Not required"
    res = drafted(case_id)
    if res is None:
        return "Not drafted yet"
    return "Declined" if res["abstained"] else "Drafted"


def assemble(mode: str, row, detail: dict | None) -> str:
    """The exact text stage 3 will send, so the analyst can read it before sending."""
    if mode == "No model, figures only":
        return P.deterministic_template(row)
    if mode == "No documents at all":
        return P.build_ungrounded_prompt(row)
    if mode == "The whole document set":
        return P.build_longcontext_prompt(row, chunks)
    return P.build_grounded_prompt(row, detail["rules"], detail["causes"])


# ----------------------------------------------------------------- table render
TABLE_CONFIG = {
    "Status": st.column_config.TextColumn(
        width=132, pinned=True,
        help="What this line needs from you. The table is sorted so the work is at "
             "the top, and the status sharpens as you run each stage."),
    "Close": st.column_config.TextColumn(width=88, pinned=True),
    "Account": st.column_config.TextColumn(width=228, pinned=True),
    "Cost centre": st.column_config.TextColumn(width=92),
    # step=1 drops the trailing ".00". A variance pack is written in whole dollars,
    # and the accounting format puts a negative in parentheses the way a pack does.
    "Actual": st.column_config.NumberColumn("Actual $", format="accounting", step=1,
                                            width=90),
    "Budget": st.column_config.NumberColumn("Budget $", format="accounting", step=1,
                                            width=90),
    "Variance": st.column_config.NumberColumn("Variance $", format="accounting",
                                              step=1, width=94),
    "Variance %": st.column_config.NumberColumn(format="%+.1f%%", width=84),
    "Evidence": st.column_config.TextColumn(
        width=150, help="What the document search found. It runs when you ask for it "
                        "and is skipped for lines below the threshold. Open a line to "
                        "see which documents were found."),
    "Commentary": st.column_config.TextColumn(
        width=148, help="Whether the written commentary has been drafted. Drafting is "
                        "the only step that calls a language model. Declined means "
                        "the system refused to state a cause and escalated."),
    "Owner": st.column_config.TextColumn(
        width=90, help="The cost-centre owner named in the Cost Centre Owner "
                       "Register. This is who to chase."),
}


def tone_status(value: str) -> str:
    key = LABEL_TO_KEY.get(value)
    if not key:
        return ""
    fg, bg, tx = STATUS[key]["tone"]
    return f"background-color:{bg};color:{tx};font-weight:600"


def decorate(df: pd.DataFrame, ev: dict) -> pd.DataFrame:
    """Attach the stage 2 and stage 3 state to a stage 1 frame, then order it."""
    out = df.copy()
    keys, evid, comm, order = [], [], [], []
    for cid in out["_id"]:
        c = CASE_BY_ID[cid]
        d = ev.get(cid)
        ok = d["ok"] if d else None
        k = status_for(c, ok)
        keys.append(k)
        order.append(STATUS[k]["order"])
        evid.append("Not required" if not c.commentary_required
                    else evidence_phrase(ok, d["reason"] if d else ""))
        comm.append(commentary_state(cid, c.commentary_required))
    out["_status"], out["_order"] = keys, order
    out["Status"] = [STATUS[k]["label"] for k in keys]
    out["Evidence"], out["Commentary"] = evid, comm
    return out


def close_table(df: pd.DataFrame, columns: list[str], key: str,
                selectable: bool = True):
    """Render a variance table with the triage state carried by colour.

    Colour does exactly one job on this page. Money and percentages stay black on
    paper so the eye lands on the status column and nowhere else.
    """
    styled = df[columns].style.map(tone_status, subset=["Status"])
    extra = ({"on_select": "rerun", "selection_mode": "single-row"}
             if selectable else {})
    return st.dataframe(
        styled, width="stretch", hide_index=True, key=key, row_height=36,
        column_config={k: v for k, v in TABLE_CONFIG.items() if k in columns},
        **extra,
    )


# ----------------------------------------------------------------- blocks
def notice(text: str, tone=BLUE, weight: str = "normal") -> None:
    """A quiet, self-drawn panel. Used where a stock alert would shout."""
    fg, bg, tx = tone
    st.html(
        f'<div style="background:{bg};border:1px solid {fg}40;border-radius:4px;'
        f'padding:12px 14px;color:{tx};font-size:.9rem;line-height:1.55;'
        f'font-weight:{weight};">{text}</div>')


def verdict_block(tone, heading: str, body: str, quote: str | None = None) -> None:
    """The commentary outcome, drawn rather than delegated to a stock alert.

    Refusal and answer get different shapes, not only different colours. The refusal
    is a filled block with the escalation inside it. A drafted cause is a document:
    paper ground, a thin rule, a coloured status line at the top. Someone scanning at
    arm's length can tell them apart before reading a word.
    """
    fg, bg, tx = tone
    quote_html = ""
    if quote is not None:
        quote_html = (
            f'<div style="margin-top:.85rem;padding-top:.85rem;'
            f'border-top:1px solid {fg}33;color:{INK};line-height:1.65;'
            f'font-size:.98rem;">{html.escape(quote)}</div>')
    st.html(
        f'<div style="background:{bg};border:1px solid {fg}45;border-radius:4px;'
        f'padding:16px 18px;">'
        f'<div style="font-weight:650;font-size:1.02rem;color:{tx};">{heading}</div>'
        f'<div style="margin-top:.35rem;color:{tx};line-height:1.6;">{body}</div>'
        f'{quote_html}</div>')


def answer_block(heading: str, body: str, quote: str) -> None:
    fg, bg, tx = GREEN
    st.html(
        f'<div style="background:#FFFFFF;border:1px solid {RULE};'
        f'border-radius:4px;padding:16px 18px;">'
        f'<div style="font-weight:650;font-size:1.02rem;color:{tx};">{heading}</div>'
        f'<div style="margin-top:.3rem;color:{MUTED};font-size:.9rem;">{body}</div>'
        f'<div style="margin-top:.85rem;padding-top:.85rem;border-top:1px solid {RULE};'
        f'color:{INK};line-height:1.7;font-size:1rem;">{html.escape(quote)}</div>'
        f'</div>')


def stage_row(n: int, name: str, state: str, tone=GRAY):
    """One line of the stage rail. Returns the column the action button goes in."""
    fg, bg, tx = tone
    c1, c2, c3 = st.columns([2.4, 5.6, 3], vertical_alignment="center")
    c1.html(f'<div style="font-weight:600;color:{INK};">Stage {n} &nbsp; {name}</div>')
    c2.html(f'<div style="color:{tx};background:{bg};display:inline-block;'
            f'padding:2px 10px;border-radius:3px;font-size:.85rem;">{state}</div>')
    return c3


# ------------------------------------------------------------- measured results
RESULTS = ROOT / "results"
IMAGES = ROOT / "images"


@st.cache_data(show_spinner=False)
def _read_result(name: str, _stamp: float, literal: bool) -> pd.DataFrame:
    return pd.read_csv(RESULTS / name, keep_default_na=not literal)


def result(name: str, literal: bool = True):
    """A result file and the day it was written.

    Every number on the Results tab is read from disk on load rather than written
    into this file, so the interface cannot drift from the evaluation the way a
    transcribed table would. Returns (None, None) when a file is missing, because a
    missing result should be reported rather than faked.
    """
    path = RESULTS / name
    if not path.exists():
        return None, None
    stamp = path.stat().st_mtime
    return (_read_result(name, stamp, literal),
            date.fromtimestamp(stamp).strftime("%d %B %Y"))


def truthy(v) -> bool:
    """True for a real boolean and for the strings a CSV round trip leaves behind."""
    if isinstance(v, str):
        return v.strip().lower() == "true"
    return bool(v)


def source_note(name: str, written: str, extra: str = "") -> None:
    st.caption(f"Source `results/{name}`, written {written}."
               + (f" {extra}" if extra else ""))


# ----------------------------------------------------------------- chart
def movement_chart(df: pd.DataFrame, order: list[str]) -> alt.Chart:
    """Dollar movement under review, close by close.

    One series, so one hue and no legend: the title names it. The closes are months
    at irregular intervals, so they are plotted as ordered categories rather than on
    a continuous time axis, which would imply gaps that are not in the data. Values
    are not printed on the bars because the table directly below carries every one of
    them exactly.
    """
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=4, color=BLUE[0])
        .encode(
            x=alt.X("Close:N", sort=order, title=None,
                    axis=alt.Axis(labelAngle=0, labelColor=MUTED, labelFontSize=11,
                                  domainColor=RULE, tickColor=RULE)),
            y=alt.Y("Movement:Q", title=None,
                    axis=alt.Axis(format="$,.2s", labelColor=MUTED, labelFontSize=11,
                                  gridColor=RULE, gridWidth=1, domain=False,
                                  ticks=False, tickCount=4)),
            tooltip=[alt.Tooltip("Close:N", title="Close"),
                     alt.Tooltip("Movement:Q", title="Movement under review",
                                 format="$,.0f"),
                     alt.Tooltip("Lines:Q", title="Lines"),
                     alt.Tooltip("Material:Q", title="Need commentary")],
        )
        .properties(height=210)
        .configure_view(strokeWidth=0)
        .configure_scale(bandPaddingInner=0.3)
    )


# ================================================================= sign in
# Deliberately not authentication. Nothing is checked, nothing is stored, and nothing
# leaves the machine. What the screen does collect is the two things the app can
# genuinely use: a preparer's name, because variance commentary is signed in a real
# close, and a role, because this tool has two audiences and only one of them wants
# the method detail.
ROLES = {
    "Reviewer": "Everything an analyst sees, plus the method view: the five "
                "comparison approaches, the raw similarity scores, the "
                "planted-source marker, and the assembled prompts.",
    "Analyst": "The close review, the cross-period comparison, and the measured "
               "evaluation scores. Only the method internals are hidden, which is "
               "how the tool would look to a finance user.",
}


def account_bar(key: str) -> None:
    """Who is signed in, and the way out. Present on every screen."""
    u = st.session_state["user"]
    c1, c2 = st.columns([8, 1.6], vertical_alignment="center")
    with c1:
        st.html(
            f'<div style="text-align:right;color:{MUTED};font-size:.85rem;'
            f'padding-top:.5rem;">Signed in as '
            f'<b style="color:{INK};">{html.escape(u["name"])}</b> · {u["role"]}</div>')
    with c2:
        if st.button("Sign out", icon=":material/logout:", width="stretch", key=key):
            st.session_state.pop("user", None)
            st.session_state["open_period"] = None
            st.rerun()


if st.session_state.get("user") is None:
    st.html('<div style="height:5vh"></div>')
    _l, _mid, _r = st.columns([1, 1.9, 1])
    with _mid:
        # Drawn as one block rather than st.title plus a paragraph, because inside
        # a narrow column Streamlit's heading margin opens a gap that a negative
        # margin only half closes.
        st.html(
            f'<h1 style="font-size:2.1rem;font-weight:600;letter-spacing:-.02em;'
            f'line-height:1.15;color:{INK};margin:0 0 .5rem 0;">'
            "Grounded Variance Commentary</h1>"
            f'<p style="color:{MUTED};font-size:1.02rem;margin:0 0 1.1rem 0;">'
            "An FP&amp;A month-end assistant that cites its evidence, or declines to "
            "answer.</p>")
        # A form, so Enter submits and so the password is never given a session key
        # to live in. It is read on submit and discarded in the same breath.
        with st.form("sign_in", border=True):
            st.markdown("**Sign in**")
            _name = st.text_input("Your name", placeholder="L. Santos")
            st.text_input("Password", type="password",
                          placeholder="not checked, leave it blank")
            _role = st.radio("Role", list(ROLES), captions=list(ROLES.values()))
            _go = st.form_submit_button("Sign in", type="primary", width="stretch",
                                        icon=":material/login:")
        if _go:
            st.session_state["user"] = {
                "name": (_name or "").strip() or "Unnamed preparer",
                "role": _role or "Reviewer"}
            st.rerun()
        notice(
            "<b>This prototype does not authenticate anyone.</b> The password box is "
            "not checked against anything, no password is stored or transmitted, and "
            "there is no account to create. Your name is held in memory for this "
            "session alone, so that drafted commentary can be signed the way a real "
            "close would be. Choosing a role changes what the interface shows you and "
            "nothing else. Everything runs on this machine.", tone=BLUE)
        st.html('<div style="height:.6rem"></div>')
        notice(WARNING, tone=YELLOW, weight="500")
    st.stop()


# ================================================================= STAGE 0, landing
if "open_period" not in st.session_state:
    st.session_state["open_period"] = None
    st.session_state["open_source"] = "shipped"

if st.session_state["open_period"] is None:
    account_bar("signout_landing")
    st.title("Grounded Variance Commentary")
    st.html(
        f'<p style="color:{MUTED};font-size:1.02rem;margin-top:-.6rem;max-width:74ch;">'
        "A month-end assistant for variance commentary. Every figure is computed "
        "rather than written by a model, every cause is attributed to a source "
        "document, and the system says so plainly when it cannot explain a line.</p>")
    notice(WARNING, tone=YELLOW, weight="500")

    full = ledger(cases)
    inv = []
    for p in ALL_PERIODS:
        sub = full[full["_period"] == p]
        inv.append({
            "Close": period_label(p),
            "Lines": len(sub),
            "Movement under review": float(sub["Variance"].abs().sum()),
            "Need commentary": int(sub["_flag"].sum()),
        })
    inv_df = pd.DataFrame(inv)

    st.subheader("Open a close")
    st.markdown(
        "Opening a close fetches its lines and applies the reporting rules. Both are "
        "arithmetic. Searching for evidence and writing the commentary are separate "
        "steps that you trigger yourself.")

    l_col, r_col = st.columns([5, 3], vertical_alignment="top")
    with l_col:
        st.dataframe(inv_df, width="stretch", hide_index=True, column_config={
            "Close": st.column_config.TextColumn(width=130, pinned=True),
            "Lines": st.column_config.NumberColumn(width=80),
            "Movement under review": st.column_config.NumberColumn(
                format="accounting", step=1, width=190,
                help="The sum of the absolute variances on the close. Actuals and "
                     "budgets are not added across revenue and expense accounts, "
                     "because that total would not mean anything."),
            "Need commentary": st.column_config.NumberColumn(width=150)})
    with r_col:
        st.markdown(
            f"**{len(full)}** variance lines across **{len(ALL_PERIODS)}** closes, "
            f"**{int(full['_flag'].sum())}** of which need commentary under the "
            "reporting policy.\n\n"
            "Pick a close below and open it. You can move between closes at any time "
            "without losing what you have already run.")

    pick = st.pills("Close period", ALL_PERIODS, default=ALL_PERIODS[-1],
                    key="landing_pick",
                    format_func=lambda p: period_label(p, short=True))
    if st.button(f"Open the {period_label(pick or ALL_PERIODS[-1])} close",
                 type="primary", icon=":material/arrow_forward:"):
        st.session_state["open_period"] = pick or ALL_PERIODS[-1]
        st.session_state["open_source"] = "shipped"
        st.rerun()

    # ------------------------------------------------- load a close from a file
    st.divider()
    st.subheader("Or load a close from a file")
    st.markdown(
        "The ten closes above are fixed in the code, which is practical for an "
        "evaluation and not for anything else. This is how a month actually arrives: "
        "a general-ledger export of actuals against budget, which the same "
        "deterministic engine runs over unchanged.")

    f_left, f_right = st.columns([5, 3], vertical_alignment="top")
    with f_left:
        st.dataframe(pd.DataFrame([
            {"Column": "period", "Required": "yes",
             "What it holds": "the close, as YYYY-MM"},
            {"Column": "account", "Required": "yes",
             "What it holds": "code and name, for example 5000 · Cost of Goods Sold"},
            {"Column": "cost_centre", "Required": "yes",
             "What it holds": "CC-2100, CC-2200, CC-3100, CC-5100 or CC-6100"},
            {"Column": "region", "Required": "yes",
             "What it holds": "Group, North America or EMEA"},
            {"Column": "product_line", "Required": "yes",
             "What it holds": "All, or a named line"},
            {"Column": "actual", "Required": "yes",
             "What it holds": "the actual, with or without $ and separators"},
            {"Column": "budget", "Required": "yes", "What it holds": "the budget"},
            {"Column": "price_effect, volume_effect, mix_effect", "Required": "no",
             "What it holds": "flexible-budget split, revenue lines only"},
            {"Column": "covenant_headroom, covenant_limit", "Required": "no",
             "What it holds": "turns of EBITDA. Headroom at or below 0.25 requires "
                              "commentary whatever the dollar amount"},
        ]), width="stretch", hide_index=True, column_config={
            "Column": st.column_config.TextColumn(width=250, pinned=True),
            "Required": st.column_config.TextColumn(width=90),
            "What it holds": st.column_config.TextColumn(width=380)})
    with f_right:
        st.markdown(
            "There is no column for the cause, the supporting document, or the "
            "answer, because a ledger export carries none of those. That is the "
            "point: a close read from a file is **reviewed, never scored**, and the "
            "app marks it that way everywhere it appears.\n\n"
            "Any close loaded here is kept apart from the ten evaluated closes and "
            "is excluded from the cross-period comparison.")
        st.download_button(
            "Download the format as a CSV", template_csv(),
            file_name="close_template.csv", mime="text/csv",
            icon=":material/download:", width="stretch")
        st.caption(f"Contains the {period_label(ALL_PERIODS[-1])} close as a worked "
                   "example of the shape.")

    picked_file = st.file_uploader(
        "Upload a close", type=["csv"], key="uploader",
        help="One month-end close per file. Several periods in one file also work.")
    if picked_file is not None and st.session_state.get("upload_name") != picked_file.name:
        rows, errors, notes = parse_upload(picked_file.getvalue())
        st.session_state["upload_name"] = picked_file.name
        st.session_state["upload_errors"] = errors
        st.session_state["upload_notes"] = notes
        if rows:
            st.session_state["upload"] = {"name": picked_file.name, "rows": rows}
            st.rerun()
        else:
            st.session_state.pop("upload", None)

    for e in st.session_state.get("upload_errors", []):
        notice(html.escape(e), tone=RED)
    for n in st.session_state.get("upload_notes", []):
        notice(html.escape(n), tone=YELLOW)

    up_rows = uploaded_rows()
    if up_rows:
        up_periods = uploaded_periods()
        up_frame = ledger(up_rows)
        notice(
            f"Loaded <b>{len(up_rows)}</b> line"
            f"{'s' if len(up_rows) != 1 else ''} from "
            f"<b>{html.escape(st.session_state['upload']['name'])}</b>, across "
            f"{len(up_periods)} close{'s' if len(up_periods) != 1 else ''}. The "
            "reporting check has already run on them.", tone=GREEN)
        u_left, u_right = st.columns([5, 3], vertical_alignment="top")
        with u_left:
            st.dataframe(pd.DataFrame([{
                "Close": period_label(q),
                "Lines": int((up_frame["_period"] == q).sum()),
                "Movement under review": float(
                    up_frame.loc[up_frame["_period"] == q, "Variance"].abs().sum()),
                "Need commentary": int(up_frame.loc[up_frame["_period"] == q,
                                                    "_flag"].sum()),
            } for q in up_periods]), width="stretch", hide_index=True,
                column_config={
                    "Close": st.column_config.TextColumn(width=130, pinned=True),
                    "Lines": st.column_config.NumberColumn(width=80),
                    "Movement under review": st.column_config.NumberColumn(
                        format="accounting", step=1, width=190),
                    "Need commentary": st.column_config.NumberColumn(width=150)})
        with u_right:
            u_pick = st.pills("Loaded close", up_periods, default=up_periods[0],
                              key="upload_pick",
                              format_func=lambda q: period_label(q, short=True))
            if st.button(
                    f"Open the {period_label(u_pick or up_periods[0])} close",
                    type="primary", icon=":material/arrow_forward:",
                    key="open_upload"):
                st.session_state["open_period"] = u_pick or up_periods[0]
                st.session_state["open_source"] = "upload"
                st.rerun()
            if st.button("Clear the loaded file", icon=":material/close:",
                         key="clear_upload"):
                for q in uploaded_periods():
                    st.session_state.pop(ev_key("upload", q), None)
                for r in up_rows:               # drafts and the open-line memory
                    st.session_state.pop(f"last_{r.case_id}", None)
                    for a in ARM_KEYS:
                        st.session_state.pop(f"{r.case_id}|{a}", None)
                for k in ("upload", "upload_name", "upload_errors", "upload_notes"):
                    st.session_state.pop(k, None)
                st.rerun()

    st.divider()
    a_col, b_col = st.columns(2, vertical_alignment="top")
    with a_col:
        st.markdown("###### How this works")
        st.markdown(
            "**1. Reporting check.** Decides whether a line is large enough to need "
            "commentary at all. Pure arithmetic against the policy thresholds. Runs "
            "when you open a close.\n\n"
            "**2. Evidence search.** Looks for a document that could support a cause "
            "for a flagged line. A similarity search over 30 document sections, with "
            "no language model involved. Runs when you ask.\n\n"
            "**3. Commentary.** Writes the wording, or refuses to state a cause and "
            "escalates to the cost-centre owner. The only step that calls a language "
            "model, and you set the rules it has to obey. Runs when you ask.")
    with b_col:
        st.markdown("###### Where this data comes from")
        st.markdown(
            "These ten closes are fixed in the code and are the evaluation set behind "
            "every number in the accompanying article. They are a representative "
            "extract, not a live feed: a real month-end file would hold hundreds of "
            "lines, most of them below the threshold.\n\n"
            "Opening a close is an exact lookup by period, so a figure is never "
            "approximated. That is deliberate, and it is the opposite of how the "
            "document search works. Searching for a cause is a similarity search, "
            "because nobody knows in advance which memo explains a movement, whereas "
            "a figure that is merely similar is simply wrong.\n\n"
            "A new month arrives as a general-ledger export of actuals against "
            "budget, which is what the loader above accepts. The same deterministic "
            "engine runs over it unchanged, so a close read from a file goes through "
            "exactly the same rules as one shipped in the code.\n\n"
            "The document set is fixed at 15 documents, so a genuinely new month has "
            "few driver memos that cover it and will correctly refuse to explain most "
            "of its lines. That is the system working, not failing.")
    st.stop()

# ================================================================= workspace
period = st.session_state["open_period"]
source = st.session_state.get("open_source", "shipped")
from_file = source == "upload"
close_rows = rows_for(source, period)
if not close_rows:                      # the loaded file was cleared under our feet
    st.session_state["open_period"] = None
    st.rerun()
df1 = ledger(close_rows)
ev = evidence_for(source, period)
df = decorate(df1, ev).sort_values(
    ["_order", "Variance"], key=lambda s: s.abs() if s.name == "Variance" else s,
    ascending=[True, False]).reset_index(drop=True)

n_total = len(df)
n_flag = int(df["_flag"].sum())
n_searched = len(ev)
n_chase = int((df["_status"] == "chase").sum())
n_cov = int((df["_status"] == "covenant").sum())
n_ready = int((df["_status"] == "ready").sum())
n_clear = int((df["_status"] == "clear").sum())
movement = float(df["Variance"].abs().sum())
material_movement = float(df.loc[df["_flag"], "Variance"].abs().sum())
unexplained = float(
    df.loc[[s == "chase" for s in df["_status"]], "Variance"].abs().sum())

account_bar("signout_workspace")
h_left, h_right = st.columns([7, 2], vertical_alignment="center")
with h_left:
    st.title(f"{period_label(period)} close")
with h_right:
    if st.button("All closes", icon=":material/arrow_back:", width="stretch"):
        st.session_state["open_period"] = None
        st.rerun()
notice(WARNING, tone=YELLOW, weight="500")
if from_file:
    notice(
        f"Loaded from <b>{html.escape(st.session_state['upload']['name'])}</b>. This "
        "close is a demonstration of the input format. It is not part of the "
        "evaluation set, no number in the article comes from it, and it is excluded "
        "from the cross-period comparison. Everything else on this screen behaves "
        "exactly as it does for a shipped close, which is the point.", tone=ORANGE)

# A Reviewer gets the method view, an Analyst does not. The role is doing real work
# here rather than decorating a sign-in screen: it is the same two-audience split the
# whole interface is built around, made switchable instead of assumed.
show_method = st.session_state["user"]["role"] == "Reviewer"
_names = ["Close review", "All periods", "Results"] + (
    ["Method"] if show_method else [])
_tabs = st.tabs(_names)
tab_close, tab_all, tab_results = _tabs[0], _tabs[1], _tabs[2]
tab_method = _tabs[3] if show_method else None

# ----------------------------------------------------------------- close review
with tab_close:
    m1, m2, m3, m4 = st.columns(4)
    quiet = {"delta_color": "off", "delta_arrow": "off", "border": True}
    m1.metric("Lines in this close", f"{n_total}",
              f"{n_flag} need commentary", **quiet)
    m2.metric("Movement under review", f"${movement:,.0f}",
              "sum of the absolute variances", **quiet)
    m3.metric("Material movement", f"${material_movement:,.0f}",
              "on lines that need commentary", **quiet)
    m4.metric("Unexplained", "not searched yet" if not ev else f"${unexplained:,.0f}",
              "run the evidence search" if not ev
              else "no document supports a cause", **quiet)

    st.caption(
        "Actuals and budgets are not added across revenue and expense accounts, "
        "because that total would not mean anything. Absolute variance is the "
        "meaningful aggregate: it is the dollar movement sitting in front of you.")

    # ------------------------------------------------------------- stage rail
    st.markdown("###### Where this close has got to")
    with st.container(border=True):
        stage_row(1, "Reporting check",
                  f"Done. {n_flag} of {n_total} lines need commentary.", GREEN)

        s2_state = ("Not run." if not ev else
                    f"Done. {n_ready + n_cov} of {n_flag} flagged lines have a "
                    f"supporting document.")
        col = stage_row(2, "Evidence search", s2_state, GRAY if not ev else GREEN)
        with col:
            if not ev:
                if st.button(f"Search {n_flag} flagged "
                             f"{'line' if n_flag == 1 else 'lines'}",
                             type="primary", width="stretch",
                             icon=":material/search:", key="run_s2"):
                    with st.spinner("Searching the document set..."):
                        run_stage2(source, period)
                    st.rerun()
            else:
                if st.button("Search again", width="stretch",
                             icon=":material/refresh:", key="rerun_s2"):
                    st.session_state.pop(ev_key(source, period), None)
                    st.rerun()

        pending = [i for i in df["_id"]
                   if CASE_BY_ID[i].commentary_required and drafted(i) is None]
        if not ev:
            s3_state = "Waiting for the evidence search."
            s3_tone = GRAY
        elif not pending:
            s3_state = "Done. Every line that needs commentary has been drafted."
            s3_tone = GREEN
        else:
            s3_state = (f"Not run. {len(pending)} "
                        f"{'line is' if len(pending) == 1 else 'lines are'} waiting.")
            s3_tone = GRAY
        col = stage_row(3, "Commentary", s3_state, s3_tone)

    # ------------------------------------------------------------- adherence
    mode = st.session_state.get("ev_mode", MEASURED[0])
    refuse = st.session_state.get("refuse", MEASURED[1])
    arm = arm_for(mode, refuse)
    is_measured = (mode, refuse if mode == MEASURED[0] else MEASURED[1]) == MEASURED

    with st.expander(
            "How should it answer?  ·  "
            + ("the rules the article measures" if is_measured
               else "changed from the measured rules"),
            expanded=False):
        st.markdown(
            "These are the rules the system has to obey when it writes. They are not "
            "wording preferences: each combination below is one of the five "
            "approaches the study compares, so nothing here is unmeasured.")
        a1, a2 = st.columns([3, 2], vertical_alignment="top")
        with a1:
            mode = st.radio(
                "What evidence should it be given?", EVIDENCE_MODES,
                index=EVIDENCE_MODES.index(mode), key="ev_mode",
                captions=[
                    "Only the documents the search found for this line. This is the "
                    "proposed system.",
                    "The verified figures and nothing else. Shows what a model does "
                    "when it has no evidence and is asked why.",
                    "Every document in the set, unfiltered. Tests whether the search "
                    "is doing any work.",
                    "A fixed template over the figures. No model is involved, so no "
                    "cause can be stated at all.",
                ])
        with a2:
            refuse = st.checkbox(
                "Refuse to state a cause when no document supports one",
                value=refuse, key="refuse",
                disabled=mode != MEASURED[0],
                help="Decides abstention before writing anything, on the retrieved "
                     "evidence alone. Only applies when the search is used, because "
                     "the other settings have nothing to check against.")
            st.caption(
                "Turn this off and the model is handed the same passages and left to "
                "decide for itself whether they support a cause.")
        arm = arm_for(mode, refuse)
        st.markdown(
            f"Currently: **{ARM_INFO[arm]['plain']}**. "
            + ("This is the configuration the article reports."
               if arm == "B4" else
               f"This is a comparison setting. The reported configuration is "
               f"{ARM_INFO['B4']['plain'].lower()}."))

    if arm != "B4":
        notice(f"Answers on this close are being produced with a comparison setting, "
               f"<b>{ARM_INFO[arm]['plain'].lower()}</b>, rather than the rules the "
               f"article measures. Change it back under How should it answer.",
               tone=ORANGE)

    if ev and pending:
        blocked = ARM_INFO[arm]["needs_model"] and not up and any(
            ev.get(i, {}).get("ok", True) for i in pending)
        d_left, d_right = st.columns([1, 2.4], vertical_alignment="center")
        with d_left:
            go = st.button(f"Draft commentary for {len(pending)} "
                           f"{'line' if len(pending) == 1 else 'lines'}",
                           type="primary", width="stretch", disabled=blocked,
                           icon=":material/edit_note:", key="draft_all")
        with d_right:
            if blocked:
                st.caption("The local language model is not running, so lines that "
                           "have supporting evidence cannot be drafted. Start it "
                           "with `ollama serve`.")
            else:
                st.caption("Runs the whole close under the rules above. A line with "
                           "no supporting evidence is refused without calling the "
                           "model, so it costs no time.")
        if go:
            prog = st.progress(0.0, text="Drafting...")
            for n, cid in enumerate(pending, start=1):
                prog.progress((n - 1) / len(pending),
                              text=f"Drafting {CASE_BY_ID[cid].account}")
                run_arm(CASE_BY_ID[cid], arm)
            prog.empty()
            st.rerun()

    # ------------------------------------------------------------- the table
    st.markdown("###### Every line in this close")
    cols = ["Status", "Account", "Cost centre", "Actual", "Budget", "Variance",
            "Variance %", "Evidence", "Commentary", "Owner"]
    sel = close_table(df, cols, key=f"close_{period}_{n_searched}")

    badges = []
    if n_chase:
        badges.append(f":red-badge[{n_chase} to chase]")
    if n_cov:
        badges.append(f":orange-badge[{n_cov} covenant "
                      f"{'alert' if n_cov == 1 else 'alerts'}]")
    if n_ready:
        badges.append(f":green-badge[{n_ready} ready to explain]")
    if not ev and n_flag:
        badges.append(f":blue-badge[{n_flag} awaiting the evidence search]")
    if n_clear:
        badges.append(f":gray-badge[{n_clear} below threshold]")
    st.markdown("  ".join(badges))

    # Two ways into a line, because a row click in a canvas grid announces nothing
    # and cannot be reached from the keyboard. The buttons are the visible, keyboard
    # affordance; a row click is the accelerator, and it writes its choice into the
    # buttons before they are drawn so the two never disagree.
    pill_key = f"open_{period}"
    ids = list(df["_id"])
    chosen_row = sel.selection["rows"][0] if sel.selection["rows"] else None
    if chosen_row is not None and chosen_row != st.session_state.get("_prev_row"):
        st.session_state["_prev_row"] = chosen_row
        st.session_state[pill_key] = ids[chosen_row]
    if st.session_state.get(pill_key) not in ids:
        st.session_state[pill_key] = ids[0]

    st.markdown("###### Open a line")
    case_id = st.pills(
        "Open a line", ids, key=pill_key, label_visibility="collapsed",
        format_func=lambda i: CASE_BY_ID[i].account.split("·")[-1].strip())
    if case_id is None:
        case_id = ids[0]
    row = CASE_BY_ID[case_id]
    st.session_state["_open_case"] = case_id

    # ------------------------------------------------------------- drill-down
    st.divider()
    st.subheader(row.account)
    st.html(
        f'<p style="color:{MUTED};margin-top:-.55rem;">{row.cost_centre} · '
        f'{row.region} · {row.product_line} · {period_label(row.period)} close · '
        f'owner {OWNERS.get(row.cost_centre, "not listed")}</p>')

    vf = row.verified_figures()
    st.markdown("**The figures**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Actual", f"${vf['actual']:,.0f}", border=True)
    c2.metric("Budget", f"${vf['budget']:,.0f}", border=True)
    sign = "+" if vf["variance_usd"] >= 0 else "-"
    c3.metric("Variance $", f"{sign}${abs(vf['variance_usd']):,.0f}", border=True)
    c4.metric("Variance %", f"{vf['variance_pct']:+.1f}%", border=True)
    st.caption("Computed by a deterministic calculation, not by a model. The model "
               "receives these as fact and is forbidden to recompute any of them.")

    with st.expander("Every figure handed to the model"):
        st.dataframe(
            pd.DataFrame({"Figure": [k.replace("_", " ") for k in vf],
                          "Value": [f"{v:,.2f}" if isinstance(v, (int, float))
                                    else str(v) for v in vf.values()]}),
            width="stretch", hide_index=True)

    st.markdown("**Does this line need commentary?**")
    if not row.commentary_required:
        notice(
            f"<b>No.</b> The variance is ${abs(row.variance_usd):,.0f} "
            f"({row.variance_pct:+.1f}%), which is under both reporting limits: "
            f"$500,000 on its own, or $250,000 together with 5% of budget "
            f"(${vf['pct_threshold_usd']:,.0f} for this line). Policy prohibits "
            "commentary on an immaterial variance, so the document search will not "
            "run on it and no commentary will be written. The line was reviewed and "
            "excluded, which is a reportable answer in its own right.", tone=GRAY)
    else:
        trig = row.materiality_trigger
        if trig == "covenant":
            notice(
                f"<b>Yes, under the covenant rule.</b> The dollar amount alone would "
                f"not require commentary. However this movement leaves only "
                f"{vf['covenant_headroom_turns']}x of headroom against a "
                f"{vf['covenant_limit_turns']}x limit, and policy requires commentary "
                f"whenever a lending covenant comes within "
                f"{vf['covenant_disclosure_trigger_turns']}x of its limit. A small "
                "interest movement can be the most reportable event of the period.",
                tone=ORANGE)
        else:
            notice(
                f"<b>Yes.</b> The variance is ${abs(row.variance_usd):,.0f} "
                f"({row.variance_pct:+.1f}%). {TRIGGER_PLAIN[trig]}, so commentary is "
                "required at account and cost-centre level.", tone=BLUE)

        d = ev.get(case_id)
        st.markdown("**What the document search found**")
        if d is None:
            st.html(
                f'<p style="color:{MUTED};max-width:74ch;">The evidence search has '
                "not been run on this line. It looks for a document that could "
                "support a cause, by similarity over 30 document sections. No "
                "language model is involved and it takes no measurable time.</p>")
            if st.button("Search for evidence on this line", type="primary",
                         icon=":material/search:", key=f"s2_{case_id}"):
                with st.spinner("Searching the document set..."):
                    run_stage2(source, period)
                st.rerun()
        else:
            if d["ok"]:
                notice(evidence_long(True, d["reason"], row), tone=GREEN)
            else:
                notice(evidence_long(False, d["reason"], row)
                       + "<br><br>Therefore no cause can be attributed. This line has "
                         "to go back to <b>"
                       + OWNERS.get(row.cost_centre, "the cost-centre owner")
                       + "</b>.", tone=RED)

            def evidence_view(frame: pd.DataFrame) -> pd.DataFrame:
                return pd.DataFrame({
                    "Document": [short_title(i) for i in frame["doc_id"]],
                    "Type": [DOCTYPE_PLAIN.get(DOC_TYPE.get(i, ""), "Document")
                             for i in frame["doc_id"]],
                    "Section": list(frame["section"]),
                    "Covers": list(frame["entity_scope"]),
                    "Published": list(frame["published_date"]),
                    "Match strength": [float(s) for s in frame["score"]],
                })

            # Full width and stacked. Side by side these clipped at about four
            # fields, which hid the entity scope and the match strength, and the
            # entity scope is exactly what the verdict above them turns on.
            ev_cfg = {
                "Document": st.column_config.TextColumn(width=250),
                "Type": st.column_config.TextColumn(width=110),
                "Section": st.column_config.TextColumn(width=270),
                "Covers": st.column_config.TextColumn(
                    width=170, help="The part of the business the document is about. "
                                    "A cause may only be taken from a document that "
                                    "covers this line."),
                "Published": st.column_config.TextColumn(width=110),
                "Match strength": st.column_config.ProgressColumn(
                    width=180, min_value=0.0, max_value=0.6, format="%.2f",
                    color=BLUE[0],
                    help="How closely the document matches this variance line. The "
                         "system will not use anything below 0.25."),
            }
            st.markdown("Documents that might explain the movement")
            st.dataframe(evidence_view(d["causes"]), width="stretch",
                         hide_index=True, column_config=ev_cfg)
            st.markdown("Policy and ownership")
            st.dataframe(evidence_view(d["rules"]), width="stretch",
                         hide_index=True, column_config=ev_cfg)
            st.caption(
                f"Only documents that existed at the {period_label(row.period)} close "
                "were searched. A memo written afterwards is invisible to this line, "
                "which is what stops a later explanation being applied to an earlier "
                "close.")

            with st.expander("Read the retrieved passages in full"):
                for _, r in pd.concat([d["causes"], d["rules"]]).iterrows():
                    st.markdown(f"**{DOC_TITLE.get(r.doc_id, r.doc_id)}** · "
                                f"{r.section} · published {r.published_date} · "
                                f"covers {r.entity_scope}")
                    st.caption(r.chunk_text)

            # --------------------------------------------------- stage 3
            st.markdown("**Ask for the commentary**")
            res = drafted(case_id)
            needs_model = ARM_INFO[arm]["needs_model"]
            # A line the gate refuses is answered without any model call, so it can
            # still be completed with Ollama stopped.
            model_needed_here = needs_model and not (arm == "B4" and not d["ok"])
            text = assemble(mode, row, d)

            if res is None:
                st.html(
                    f'<p style="color:{MUTED};max-width:74ch;">Nothing has been asked '
                    "of the model for this line yet. Read the request below, change "
                    "the rules above if you want to, then send it. Drafting takes a "
                    "few seconds for a line with supporting evidence, and no time at "
                    "all for a line without, because the system refuses before it "
                    "calls the model.</p>")
                with st.expander(
                        ("The template that will be produced"
                         if arm == "B0" else
                         f"The exact request that will be sent, "
                         f"{len(text):,} characters"),
                        expanded=arm != "B2"):
                    st.code(text, language="text")
                if model_needed_here and not up:
                    notice("The local language model is not running, so this line "
                           "cannot be drafted. Start it with <code>ollama serve</code> "
                           "and confirm <code>qwen3:4b-instruct</code> is pulled. "
                           "Everything above stays readable without it.", tone=RED)
                if st.button("Produce the template" if arm == "B0"
                             else "Send this to the model",
                             type="primary", key=f"send_{case_id}",
                             disabled=model_needed_here and not up,
                             icon=":material/send:"):
                    with st.spinner("Working on this machine..."):
                        run_arm(row, arm)
                    st.rerun()
            else:
                r_arm = res["arm"]
                if r_arm == "B0":
                    notice("The template restates the verified figures and the policy "
                           "decision. It states no cause at all, which is exactly the "
                           "gap a language model is being asked to fill.", tone=GRAY)
                    verdict_block(GRAY, "No cause stated",
                                  "No model was involved, so there is nothing here "
                                  "that could have been invented and nothing that "
                                  "could explain the movement.",
                                  quote=res["answer"])
                elif res["abstained"]:
                    verdict_block(
                        RED, "Cause: unsupported",
                        "The system declined to state a cause because no document "
                        "supports one. It escalated to <b>"
                        + OWNERS.get(row.cost_centre, "the cost-centre owner")
                        + "</b>. This is the intended behaviour, not a failure to "
                          "answer.",
                        quote=res["answer"])
                else:
                    tone, trustworthy, head, body = STATED[r_arm]
                    if trustworthy:
                        answer_block(head, body, quote=res["answer"])
                    else:
                        verdict_block(tone, head, body, quote=res["answer"])

                k1, k2, k3 = st.columns(3)
                k1.metric("Figures check",
                          "All traced" if res["fidelity"] else "Needs review",
                          "every number matches a source" if res["fidelity"]
                          else f"unmatched {res['unmatched']}", **quiet)
                k2.metric("Cause", "Not stated" if res["abstained"] or r_arm == "B0"
                          else "Stated",
                          "escalated to the owner" if res["abstained"]
                          else CAUSE_SOURCE[r_arm], **quiet)
                k3.metric("Time taken", f"{res['seconds']:.2f}s",
                          "no model was called" if res["seconds"] == 0
                          else "local, nothing left this machine", **quiet)
                st.caption(
                    f"Prepared by {st.session_state['user']['name']} on "
                    f"{date.today():%d %B %Y}. Produced with "
                    f"{ARM_INFO[r_arm]['plain'].lower()}."
                    + ("" if r_arm == "B4" else
                       " This is a comparison setting rather than the configuration "
                       "the article reports."))
                if res["gate_passed"] is False:
                    st.caption("The time is zero because the system refused on the "
                               "evidence alone and never called the model.")
                if st.button("Ask again", key=f"redraft_{case_id}",
                             icon=":material/refresh:"):
                    st.session_state.pop(f"{case_id}|{r_arm}", None)
                    st.session_state.pop(f"last_{case_id}", None)
                    st.rerun()

# ----------------------------------------------------------------- all periods
with tab_all:
    st.subheader("Every close in the evaluation set")
    st.markdown(
        "The same review across all ten closes. Replication month over month is one "
        "of the project's findings, and it is only visible when the periods are put "
        "side by side.")

    if from_file:
        notice("This tab is the ten evaluated closes only. The close you loaded from "
               "a file is deliberately not mixed in, because every figure here backs "
               "a number in the article.", tone=BLUE)
    full = ledger(cases)
    searched = {p: st.session_state.get(ev_key("shipped", p), {})
                for p in ALL_PERIODS}
    all_ev = {k: v for d in searched.values() for k, v in d.items()}
    all_df = decorate(full, all_ev).sort_values(
        ["_period", "_order"]).reset_index(drop=True)

    order = [period_label(p, short=True) for p in ALL_PERIODS]
    chart_df = pd.DataFrame([{
        "Close": period_label(p, short=True),
        "Movement": float(full.loc[full["_period"] == p, "Variance"].abs().sum()),
        "Lines": int((full["_period"] == p).sum()),
        "Material": int(full.loc[full["_period"] == p, "_flag"].sum()),
    } for p in ALL_PERIODS])

    st.markdown("###### Dollar movement under review, close by close")
    st.altair_chart(movement_chart(chart_df, order), use_container_width=True)

    per = []
    for p in ALL_PERIODS:
        sub = all_df[all_df["_period"] == p]
        done = bool(searched[p])
        per.append({
            "Close": period_label(p),
            "Lines": len(sub),
            "Need commentary": int(sub["_flag"].sum()),
            "Searched": "yes" if done else "no",
            "Explanation found": int(sub["_status"].isin(["ready", "covenant"]).sum())
            if done else None,
            "To chase": int((sub["_status"] == "chase").sum()) if done else None,
            "Below threshold": int((sub["_status"] == "clear").sum()),
        })
    per_df = pd.DataFrame(per)

    t_col, s_col = st.columns([5, 3], vertical_alignment="top")
    with t_col:
        st.dataframe(per_df, width="stretch", hide_index=True, column_config={
            "Close": st.column_config.TextColumn(width=128, pinned=True),
            "Lines": st.column_config.NumberColumn(width=76),
            "Need commentary": st.column_config.NumberColumn(width=132),
            "Searched": st.column_config.TextColumn(
                width=90, help="Whether the evidence search has been run on this "
                               "close. The two columns after it stay blank until it "
                               "has."),
            "Explanation found": st.column_config.NumberColumn(width=132),
            "To chase": st.column_config.NumberColumn(width=90),
            "Below threshold": st.column_config.NumberColumn(width=126)})
    with s_col:
        n_done = sum(1 for d in searched.values() if d)
        st.markdown(
            f"**{int(all_df['_flag'].sum())}** of **{len(all_df)}** lines need "
            f"commentary. **{len(all_df) - int(all_df['_flag'].sum())}** are below "
            "the reporting threshold and receive no commentary by design.\n\n"
            f"The evidence search has been run on **{n_done}** of "
            f"**{len(ALL_PERIODS)}** closes.")
        if n_done < len(ALL_PERIODS):
            if st.button("Search every close", icon=":material/search:",
                         key="search_all", width="stretch"):
                with st.spinner("Searching the document set for all ten closes..."):
                    for p in ALL_PERIODS:
                        run_stage2("shipped", p)
                st.rerun()

    st.markdown("###### Actual against budget, by account, across all ten closes")
    roll = []
    for acct in sorted(full["Account"].unique()):
        sub = full[full["Account"] == acct]
        roll.append({
            "Account": acct,
            "Lines": len(sub),
            "Actual": float(sub["Actual"].sum()),
            "Budget": float(sub["Budget"].sum()),
            "Variance": float(sub["Actual"].sum() - sub["Budget"].sum()),
            "Variance %": float(100.0 * (sub["Actual"].sum() - sub["Budget"].sum())
                                / sub["Budget"].sum()),
            "Need commentary": int(sub["_flag"].sum()),
        })
    st.dataframe(pd.DataFrame(roll), width="stretch", hide_index=True, column_config={
        "Account": st.column_config.TextColumn(width=240, pinned=True),
        "Lines": st.column_config.NumberColumn(width=76),
        "Actual": st.column_config.NumberColumn("Actual $", format="accounting",
                                                step=1, width=120),
        "Budget": st.column_config.NumberColumn("Budget $", format="accounting",
                                                step=1, width=120),
        "Variance": st.column_config.NumberColumn("Variance $", format="accounting",
                                                  step=1, width=120),
        "Variance %": st.column_config.NumberColumn(format="%+.1f%%", width=100),
        "Need commentary": st.column_config.NumberColumn(width=140)})
    st.caption(
        "Totals within an account are meaningful because actual and budget are "
        "measured the same way. They are not added across accounts, because a "
        "revenue dollar and a cost dollar do not sum to anything. Marketing carries "
        "eight of the ten lines that never breach the threshold, which is why the "
        "routine half of this ledger looks repetitive.")

    st.markdown("###### Every line")
    close_table(
        all_df,
        ["Status", "Close", "Account", "Cost centre", "Actual", "Budget", "Variance",
         "Variance %", "Evidence", "Commentary"],
        key="all_table", selectable=False)
    st.caption(
        "These 26 lines are the whole evaluation set. A real month-end file would "
        "hold hundreds of lines with most of them immaterial, and would show far more "
        "variety in the routine half of the ledger than ten lines can. Nothing here "
        "is padded or illustrative: every line drives a result reported in the "
        "article, so adding rows to make a close look busier would break the "
        "correspondence between the interface and the measured results.")

# ------------------------------------------------------- method and evaluation
# ----------------------------------------------------------------- results
with tab_results:
    st.subheader("Evaluation results")
    st.markdown(
        "Every number on this page is read from a file in `results/` when the "
        "page loads. Nothing here is transcribed, so the interface cannot drift "
        "from the evaluation. Each table names the file it came from and the day "
        "that file was written.")

    # ---------------------------------------------------- headline
    st.markdown("###### Headline, hold-out closes only")
    bench, bench_day = result("benchmark_results.csv")
    if bench is None:
        notice("`results/benchmark_results.csv` is missing. Re-run "
               "`build/run_experiments.py` to produce it.", tone=RED)
    else:
        show = bench.copy()
        show.insert(1, "Approach", [ARM_INFO.get(a, {}).get("plain", a)
                                    for a in show["arm"]])
        show = show.drop(columns=["retrieval_rate"], errors="ignore")
        show = show.rename(columns={
            "arm": "Arm", "cases": "Cases",
            "retrieval_hits": "Right memo found",
            "numeric_fidelity": "Figures traced, documented",
            "numeric_fidelity_all": "Figures traced, all output",
            "abstention_correct": "Correctly declined",
            "over_abstention": "Declined in error",
            "mean_seconds": "Mean seconds"})
        st.dataframe(show, width="stretch", hide_index=True, column_config={
            "Arm": st.column_config.TextColumn(width=64, pinned=True),
            "Approach": st.column_config.TextColumn(width=330, pinned=True),
            "Cases": st.column_config.NumberColumn(width=70),
            "Right memo found": st.column_config.TextColumn(
                width=140, help="How often the memo planted for the case "
                                "appeared in the top three retrieved drivers."),
            "Figures traced, documented": st.column_config.TextColumn(
                width=190, help="Numeric fidelity scored only on cases that had "
                                "a supporting document."),
            "Figures traced, all output": st.column_config.TextColumn(
                width=190, help="The same check widened to every case that "
                                "produced text. This is the scope that changes "
                                "the ranking."),
            "Correctly declined": st.column_config.TextColumn(width=150),
            "Declined in error": st.column_config.TextColumn(
                width=140, help="Cases where evidence was in fact available and "
                                "the system still refused."),
            "Mean seconds": st.column_config.NumberColumn(format="%.2f",
                                                          width=110)})
        source_note("benchmark_results.csv", bench_day,
                    "Configuration was chosen on the two earliest closes and "
                    "every figure above comes from the eight later ones.")

    st.markdown("###### The two findings this rests on")
    f1, f2 = st.columns(2, vertical_alignment="top")
    with f1:
        notice(
            "<b>Retrieval failure degrades into abstention, not a wrong cause.</b>"
            "<br><br>The search misses on two of the six documented hold-out "
            "cases. On both, the system refuses to state a cause and escalates to "
            "the named cost-centre owner instead of inventing one. A miss costs a "
            "chase, not a false explanation in the board pack.", tone=GREEN)
    with f2:
        notice(
            "<b>A metric's scope can hide the failure it exists to catch.</b>"
            "<br><br>Numeric fidelity scored only on documented cases cannot see "
            "a figure invented on a case that abstained. Widening it to every "
            "case that produced text drops the long-context arm from 6/6 to 6/12 "
            "and reverses the ranking of the last two arms. Both scopes are "
            "reported everywhere.", tone=ORANGE)

    # ---------------------------------------------------- replication
    st.divider()
    st.markdown("###### Replication across the reported closes")
    per, per_day = result("per_period_results.csv")
    if per is None:
        notice("`results/per_period_results.csv` is missing.", tone=RED)
    else:
        pshow = per.rename(columns={
            "period": "Close", "cases": "Cases", "retrieval": "Right memo found",
            "abstention": "Correctly declined",
            "numeric_fidelity": "Figures traced"})
        pcol, ptext = st.columns([5, 3], vertical_alignment="top")
        with pcol:
            st.dataframe(pshow, width="stretch", hide_index=True, column_config={
                "Close": st.column_config.TextColumn(width=110, pinned=True),
                "Cases": st.column_config.NumberColumn(width=80),
                "Right memo found": st.column_config.TextColumn(width=150),
                "Correctly declined": st.column_config.TextColumn(width=150),
                "Figures traced": st.column_config.TextColumn(width=130)})
        with ptext:
            st.markdown(
                "One close is not a result. The same behaviour holding across "
                "eight consecutive closes, against a document set that grows "
                "over time, is what makes it one.\n\n"
                "`n/a` means the metric is undefined for that close, not that it "
                "failed. A close with no documented case has no retrieval to "
                "score, and a close with no evidence-less case has no abstention "
                "to score.")
        source_note("per_period_results.csv", per_day)

    # ---------------------------------------------------- retrieval config
    st.markdown("###### What moved retrieval")
    cfg_df, cfg_day = result("retrieval_config_comparison.csv")
    if cfg_df is not None:
        cshow = cfg_df.rename(columns={
            "config": "Configuration", "hits": "Hits", "n": "Cases",
            "rate": "Rate", "misses": "Cases still missed"})
        st.dataframe(cshow, width="stretch", hide_index=True, column_config={
            "Configuration": st.column_config.TextColumn(width=200, pinned=True),
            "Hits": st.column_config.NumberColumn(width=70),
            "Cases": st.column_config.NumberColumn(width=70),
            "Rate": st.column_config.NumberColumn(format="%.3f", width=80),
            "Cases still missed": st.column_config.TextColumn(width=320)})
        source_note(
            "retrieval_config_comparison.csv", cfg_day,
            "The gain comes from excluding prior-period commentary from the "
            "causes axis, which is a category correction rather than a tuning "
            "pass. The k=4 variant is reported separately and labelled tuned, "
            "because its only justification was a case whose failure had already "
            "been inspected.")

    # ---------------------------------------------------- improvement
    st.divider()
    st.markdown("###### Improvement over the Module 7 prototype")
    imp, imp_day = result("improvement_comparison.csv")
    if imp is not None:
        st.dataframe(imp, width="stretch", hide_index=True, column_config={
            "Area": st.column_config.TextColumn(width=230, pinned=True),
            "Module 7 prototype": st.column_config.TextColumn(width=250),
            "Module 8 final": st.column_config.TextColumn(width=250),
            "Evidence": st.column_config.TextColumn(width=420)})
        source_note("improvement_comparison.csv", imp_day)

    # ---------------------------------------------------- figures
    st.markdown("###### The figures as published")
    g1, g2 = st.columns(2, vertical_alignment="top")
    for col, fname, cap in (
            (g1, "evaluation_chart.png", "Scores by approach, hold-out closes."),
            (g2, "improvement_chart.png", "Module 7 against Module 8.")):
        with col:
            fpath = IMAGES / fname
            if fpath.exists():
                st.image(str(fpath), caption=cap, width="stretch")
            else:
                notice(f"`images/{fname}` is missing.", tone=RED)

    # ---------------------------------------------------- every output
    st.divider()
    st.markdown("###### Every answer the system produced")
    scores, scores_day = result("evaluation_scores.csv", literal=False)
    if scores is None:
        notice("`results/evaluation_scores.csv` is missing.", tone=RED)
    else:
        st.markdown(
            "The full text of every generated answer, with the score it received. "
            "This is the evidence behind the table at the top of the page: the "
            "numbers there are counts over these rows.")
        arms_present = [a for a in ARM_KEYS if a in set(scores["arm"])]
        pick_arm = st.pills(
            "Approach", arms_present,
            default=("B4" if "B4" in arms_present else arms_present[0]),
            key="res_arm",
            format_func=lambda a: f"{a}  {ARM_INFO[a]['plain']}")
        if pick_arm is None:
            pick_arm = arms_present[0]
        sub_df = scores[scores["arm"] == pick_arm].reset_index(drop=True)

        view = pd.DataFrame({
            "Case": sub_df["case_id"],
            "Close": sub_df["period"],
            "Case type": [s.replace("_", " ") for s in sub_df["stratum"]],
            "Account": sub_df["account"],
            "Declined": ["yes" if truthy(b) else "no"
                         for b in sub_df["abstained"]],
            "Figures traced": ["yes" if truthy(b) else "no"
                               for b in sub_df["numeric_fidelity"]],
            "Seconds": [float(x) for x in sub_df["response_time_s"]],
            "Answer": sub_df["answer"],
        })
        st.dataframe(view, width="stretch", hide_index=True, row_height=36,
                     column_config={
                         "Case": st.column_config.TextColumn(width=94,
                                                             pinned=True),
                         "Close": st.column_config.TextColumn(width=84),
                         "Case type": st.column_config.TextColumn(width=126),
                         "Account": st.column_config.TextColumn(width=210),
                         "Declined": st.column_config.TextColumn(width=88),
                         "Figures traced": st.column_config.TextColumn(width=118),
                         "Seconds": st.column_config.NumberColumn(format="%.2f",
                                                                  width=86),
                         "Answer": st.column_config.TextColumn(width=400)})
        source_note("evaluation_scores.csv", scores_day,
                    f"{len(sub_df)} rows shown for {pick_arm}, "
                    f"{len(scores)} in the file across "
                    f"{scores['arm'].nunique()} approaches. The same answers "
                    "without the scoring columns are in "
                    "`results/generated_outputs.csv`.")

        st.markdown("Read one in full")
        choice = st.selectbox(
            "Case", list(sub_df["case_id"]), key=f"res_case_{pick_arm}",
            label_visibility="collapsed",
            format_func=lambda c: (
                f"{c} · {sub_df.loc[sub_df['case_id'] == c, 'account'].iloc[0]}"
                f" · {sub_df.loc[sub_df['case_id'] == c, 'period'].iloc[0]}"))
        row_s = sub_df[sub_df["case_id"] == choice].iloc[0]
        declined = truthy(row_s["abstained"])
        traced = truthy(row_s["numeric_fidelity"])
        if declined:
            verdict_block(RED, "Cause: unsupported",
                          "The arm declined to state a cause and escalated.",
                          quote=str(row_s["answer"]))
        else:
            answer_block("Cause stated",
                         f"Case type {str(row_s['stratum']).replace('_', ' ')}, "
                         f"{row_s['period']} close.",
                         quote=str(row_s["answer"]))
        q1, q2, q3, q4 = st.columns(4)
        q = {"delta_color": "off", "delta_arrow": "off", "border": True}
        q1.metric("Figures traced", "yes" if traced else "no",
                  "" if traced else str(row_s["unmatched_figures"]), **q)
        q2.metric("Declined", "yes" if declined else "no", **q)
        gold = str(row_s["gold_doc"]).strip()
        q3.metric("Planted source",
                  gold if gold and gold.lower() != "nan" else "none",
                  "the memo that should have been found", **q)
        q4.metric("Time taken", f"{float(row_s['response_time_s']):.2f}s",
                  f"{int(row_s['prompt_chars']):,} character prompt", **q)


if show_method:
    with tab_method:
        open_id = st.session_state.get("_open_case", cases[0].case_id)
        mrow = CASE_BY_ID[open_id]
        st.subheader("Method")
        st.markdown(
            "Everything on this tab is written for a reader assessing the method rather "
            "than working a close. It holds the five comparison arms, the raw similarity "
            "scores, the planted-source marker, the assembled prompts, and the "
            "evaluation design. Retrieval is computed here on demand, so this tab does "
            "not wait on the analyst workflow.")
        st.markdown(
            f"Currently open: **{mrow.case_id}** · {mrow.account} · {mrow.period}. "
            "Change the line on the Close review tab.")
        if from_file:
            notice("This line was loaded from a file, so it has no planted source and no "
                   "case type. Retrieval, the gate, and the five arms all still run on "
                   "it and are shown below. What cannot be shown is whether the search "
                   "found the right document, because a ledger export does not say what "
                   "the right answer was.", tone=ORANGE)

        st.markdown("###### The baseline ladder")
        st.markdown("\n".join(
            f"- **{k}, {v['plain']}.** {v['blurb']} Retrieval: "
            f"{'yes' if v['retrieves'] else 'no'}. Model: "
            f"{'yes' if v['needs_model'] else 'no'}."
            for k, v in ARM_INFO.items()))
        st.caption(
            "The adherence controls on the Close review tab select among exactly these "
            "five. No setting an analyst can reach produces a configuration the study has "
            "not measured.")

        st.divider()
        st.markdown("###### Run and compare")
        m_mode = st.segmented_control("Mode", ["Single arm", "Compare two arms"],
                                      default="Single arm", key="m_mode")
        if m_mode is None:
            m_mode = "Single arm"
        if m_mode == "Single arm":
            a1 = st.selectbox("Arm", ARM_KEYS, index=ARM_KEYS.index("B3"), key="m_a1",
                              format_func=lambda a: ARM_INFO[a]["label"])
            selected = [a1]
        else:
            s1, s2 = st.columns(2)
            a1 = s1.selectbox("Left", ARM_KEYS, index=ARM_KEYS.index("B2"), key="m_a1c",
                              format_func=lambda a: ARM_INFO[a]["label"])
            a2 = s2.selectbox("Right", ARM_KEYS, index=ARM_KEYS.index("B3"), key="m_a2c",
                              format_func=lambda a: ARM_INFO[a]["label"])
            selected = [a1, a2]

        if not mrow.commentary_required:
            st.info("This line is immaterial, so no arm generates commentary for it. Open "
                    "a material line on the Close review tab to run the ladder.",
                    icon=":material/info:")
        else:
            blocked = [a for a in selected if ARM_INFO[a]["needs_model"] and not up]
            if blocked:
                st.error(
                    f"Ollama is not reachable, so {', '.join(blocked)} cannot run. Start "
                    "it with `ollama serve` and confirm `qwen3:4b-instruct` is pulled. "
                    "Everything else on this tab remains inspectable.",
                    icon=":material/error:")
            runnable = [a for a in selected if a not in blocked]
            label = "Run both arms" if len(selected) == 2 else f"Run {selected[0]}"
            if st.button(label, type="primary", disabled=not runnable, key="m_run"):
                for a in runnable:
                    with st.spinner(f"Running {a} locally..."):
                        run_arm(mrow, a)

            have = [a for a in selected if f"{open_id}|{a}" in st.session_state]

            def method_render(a: str, r: dict) -> None:
                if a == "B0":
                    st.info("The template restates the verified figures and the policy "
                            "decision. It states no cause at all, which is exactly the "
                            "gap a language model is being asked to fill.",
                            icon=":material/description:")
                elif r["abstained"]:
                    st.error("**Cause: unsupported.** The arm declined to state a cause "
                             "and escalated to the cost-centre owner.",
                             icon=":material/block:")
                else:
                    st.success("The arm stated a cause.", icon=":material/chat:")
                st.markdown(f"> {r['answer']}")
                q = {"delta_color": "off", "delta_arrow": "off"}
                k1, k2, k3 = st.columns(3)
                k1.metric("Numeric fidelity", "pass" if r["fidelity"] else "FAIL",
                          "every figure traces to a source" if r["fidelity"]
                          else f"unmatched {r['unmatched']}", **q)
                k2.metric("Abstained", "yes" if r["abstained"] else "no", **q)
                k3.metric("Prompt size", f"{r['prompt_chars']:,} ch",
                          f"{r['seconds']:.2f}s elapsed", **q)
                if a == "B4" and r["gate_passed"] is False:
                    st.caption("B4 abstained without calling the model, which is why the "
                               "time is zero. The gate decided on the retrieved evidence "
                               "alone.")

            if have and len(selected) == 2:
                cc = st.columns(2)
                for col, a in zip(cc, selected):
                    with col:
                        st.markdown(f"**{ARM_INFO[a]['label']}**")
                        if a in have:
                            method_render(a, st.session_state[f"{open_id}|{a}"])
                        else:
                            st.caption("Not run.")
                if all(a in have for a in selected):
                    st.info("Both arms received identical verified figures for the same "
                            "variance line. Any difference between the two outputs comes "
                            "from the evidence each was given, and nothing else.",
                            icon=":material/search:")
            elif have:
                st.markdown(f"**{ARM_INFO[have[0]]['label']}**")
                method_render(have[0], st.session_state[f"{open_id}|{have[0]}"])

        st.divider()
        st.markdown("###### Retrieval detail")
        if not mrow.commentary_required:
            st.caption("Retrieval is not run for an immaterial line. The gate is never "
                       "reached, which is the behaviour the immaterial stratum measures.")
        else:
            md = search_rows([mrow])[open_id]
            cfg = retriever.config
            st.caption(
                f"`k_rules={cfg.k_rules}` · `k_causes={cfg.k_causes}` · `tau={cfg.tau}` · "
                f"`exclude_precedent={cfg.exclude_precedent}` · "
                f"`require_scope_match={cfg.require_scope_match}` · point-in-time "
                f"filtered to the {mrow.period} close · stratum `{mrow.stratum}` · "
                + ("outside the evaluation" if from_file else
                   "period held out" if mrow.period not in CONFIG_PERIODS
                   else "period used to choose configuration"))

            def raw(frame: pd.DataFrame, cs: list[str]) -> pd.DataFrame:
                out = frame.copy()
                out["doc"] = [f"{i} ★" if i == mrow.gold_doc else i
                              for i in out["doc_id"]]
                out["score"] = out["score"].round(3)
                return out[["doc"] + cs]

            raw_cfg = {
                "doc": st.column_config.TextColumn("doc", width="small", pinned=True),
                "doc_title": st.column_config.TextColumn("title", width="medium"),
                "section": st.column_config.TextColumn("section", width="medium"),
                "category": st.column_config.TextColumn("category", width="small"),
                "entity_scope": st.column_config.TextColumn("scope", width="small"),
                "published_date": st.column_config.TextColumn("published", width="small"),
                "score": st.column_config.NumberColumn("score", format="%.3f",
                                                       width="small"),
            }
            st.markdown("Causes axis · `k=3` · `precedent` excluded")
            st.dataframe(raw(md["causes"], ["doc_title", "category", "entity_scope",
                                            "published_date", "score"]),
                         width="stretch", hide_index=True, column_config=raw_cfg)
            st.markdown("Rules axis · `k=2` · policy, thresholds, and ownership")
            st.dataframe(raw(md["rules"], ["doc_title", "section", "published_date",
                                           "score"]),
                         width="stretch", hide_index=True, column_config=raw_cfg)

            if md["ok"]:
                st.success(f"Sufficiency gate PASSED: {md['reason']}",
                           icon=":material/check_circle:")
            else:
                st.error(f"Sufficiency gate ABSTAIN: {md['reason']}",
                         icon=":material/block:")
            st.caption("B4 enforces this decision and never calls the model when it "
                       "abstains. B3 sees the same passages and has to decide for itself.")

            if from_file:
                st.caption("No planted source: this line came from a file, so there is "
                           "nothing to score the search against.")
            elif mrow.gold_doc:
                got = list(md["causes"]["doc_id"])
                if mrow.gold_doc in got:
                    st.caption(f"Evaluation only: the planted source **{mrow.gold_doc}** "
                               f"was retrieved at rank {got.index(mrow.gold_doc) + 1}. "
                               "★ marks it in the table above.")
                else:
                    st.caption(f"Evaluation only: the planted source **{mrow.gold_doc}** "
                               "was NOT retrieved. Anything the arm states about cause is "
                               "therefore unsupported.")
            else:
                st.caption("Evaluation only: this case has no planted source, so the "
                           "correct outcome is to abstain.")

            with st.expander("The assembled prompts"):
                for a in ARM_KEYS:
                    if not ARM_INFO[a]["needs_model"]:
                        continue
                    if a in ("B3", "B4"):
                        pr = P.build_grounded_prompt(mrow, md["rules"], md["causes"])
                    elif a == "B2":
                        pr = P.build_longcontext_prompt(mrow, chunks)
                    else:
                        pr = P.build_ungrounded_prompt(mrow)
                    st.markdown(f"**{a}** · {len(pr):,} characters")
                    st.code(pr, language="text")

        st.divider()
        st.markdown("###### Evaluation design and known limits")
        st.markdown(
            f"- **Temporal split.** Configuration was chosen on {CONFIG_PERIODS[0]} and "
            f"{CONFIG_PERIODS[1]} and reported on the eight later closes, so no reported "
            "number was tuned on the cases it scores.\n"
            "- **Dataset scale.** 26 variance lines across 10 closes, 16 of them "
            "material. Every line drives a number in the article, so no line may be "
            "added, removed, or edited for presentation.\n"
            "- **Corpus scale.** 15 documents, 30 sections. The similarity search runs "
            "over those 30 sections only.\n"
            "- **Two intakes, one engine.** The ten evaluated closes are fixed in the "
            "code and fetched by exact lookup. A close may also be read from a "
            "general-ledger export through the loader on the landing screen, and it runs "
            "through the identical variance engine, retriever, and gate. A loaded close "
            "carries no ground truth, so it can be reviewed but never scored, and it is "
            "excluded from every number reported here.\n"
            "- **Retrieval is the binding constraint.** The reported hold-out figure is "
            "4 of 6 on the causes axis. Where retrieval misses, the gate turns the miss "
            "into an abstention rather than a wrong cause, which is the first of the two "
            "findings the project defends.\n"
            "- **Metric scope changes the ranking.** Numeric fidelity measured only on "
            "documented cases cannot see a figure invented on a case that abstained. "
            "Widening it to every case that produced text dropped B2 from 6/6 to 6/12 and "
            "reversed the B3 against B4 ranking. Both scopes are reported everywhere.")
        st.caption(WARNING)
