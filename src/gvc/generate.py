"""
Local generation through Ollama, and the B0 to B4 arms.

Module 8 runs against Ollama on the author's own machine rather than a hosted VM.
That matters beyond speed: the project's premise is that pre-release financial
figures must not leave the organisation, so local execution makes the deployment
claim demonstrated rather than argued.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd
import requests

from . import prompt as P
from .metrics import numeric_fidelity
from .retrieval import Retriever

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
DEFAULT_MODEL = "qwen3:4b-instruct"


@dataclass
class Generation:
    arm: str
    case_id: str
    answer: str
    seconds: float
    prompt_chars: int
    rules: pd.DataFrame | None = None
    causes: pd.DataFrame | None = None
    gate_passed: bool | None = None
    gate_reason: str = ""


def generate(prompt: str, model: str = DEFAULT_MODEL, temperature: float = 0.0,
             seed: int = 42, timeout: int = 300) -> tuple[str, float]:
    t0 = time.time()
    r = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "seed": seed},
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json().get("response", "").strip(), round(time.time() - t0, 2)


# ------------------------------------------------------------------ arms
def run_B0(row, **_) -> Generation:
    """Deterministic template. No model."""
    return Generation("B0", row.case_id, P.deterministic_template(row), 0.0, 0)


def run_B1(row, model: str = DEFAULT_MODEL, **_) -> Generation:
    """Ungrounded: same verified figures, no retrieval, open 'why'."""
    pr = P.build_ungrounded_prompt(row)
    ans, secs = generate(pr, model)
    return Generation("B1", row.case_id, ans, secs, len(pr))


def run_B2(row, all_chunks: pd.DataFrame, model: str = DEFAULT_MODEL, **_) -> Generation:
    """Long-context control: whole corpus in the prompt, no retrieval."""
    pr = P.build_longcontext_prompt(row, all_chunks)
    ans, secs = generate(pr, model)
    return Generation("B2", row.case_id, ans, secs, len(pr))


def run_B3(row, retriever: Retriever, model: str = DEFAULT_MODEL, **_) -> Generation:
    """The proposed system: two-axis filtered retrieval plus the abstention rule."""
    rules, causes = retriever.retrieve(row)
    pr = P.build_grounded_prompt(row, rules, causes)
    ans, secs = generate(pr, model)
    return Generation("B3", row.case_id, ans, secs, len(pr), rules, causes)


def run_B4(row, retriever: Retriever, model: str = DEFAULT_MODEL, **_) -> Generation:
    """B3 plus the pre-generation sufficiency gate and the numeric verifier.

    The gate decides abstention BEFORE generating, because a model handed a 'why'
    question will almost always answer it.
    """
    rules, causes = retriever.retrieve(row)
    ok, reason = retriever.sufficient(row, causes)

    if not ok:
        owner = ""
        for _, r in rules.iterrows():
            if r["doc_id"] == "D04":
                owner = f" Owner per {r['doc_id']} {r['section']}."
                break
        answer = (
            f"{row.account} variance ${abs(row.variance_usd):,.0f} "
            f"({row.variance_pct:+.1f}%) in {row.period} [1]. "
            f"Cause: unsupported, no admissible driver evidence retrieved "
            f"({reason}). Escalate to the cost-centre owner.{owner}"
        )
        return Generation("B4", row.case_id, answer, 0.0, 0, rules, causes, False, reason)

    pr = P.build_grounded_prompt(row, rules, causes)
    ans, secs = generate(pr, model)

    # numeric verifier: one regeneration, then escalate rather than ship a bad figure
    ctx = " ".join(list(rules["chunk_text"]) + list(causes["chunk_text"]))
    passed, unmatched = numeric_fidelity(ans, row, ctx)
    if not passed:
        ans2, secs2 = generate(pr, model, seed=43)
        passed2, _ = numeric_fidelity(ans2, row, ctx)
        if passed2:
            ans, secs = ans2, secs + secs2
        else:
            ans = (ans + "\n[VERIFIER] Unverified figures "
                   f"{unmatched}; held for human review.")
            secs = secs + secs2
    return Generation("B4", row.case_id, ans, secs, len(pr), rules, causes, True, reason)


ARMS = {"B0": run_B0, "B1": run_B1, "B2": run_B2, "B3": run_B3, "B4": run_B4}


def ollama_available(model: str = DEFAULT_MODEL) -> tuple[bool, str]:
    try:
        r = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
        r.raise_for_status()
        names = [m["name"] for m in r.json().get("models", [])]
        if model in names:
            return True, f"ollama up, {model} present"
        return False, f"ollama up but {model} missing; have {names}"
    except Exception as e:  # noqa: BLE001
        return False, f"ollama unreachable: {e}"
