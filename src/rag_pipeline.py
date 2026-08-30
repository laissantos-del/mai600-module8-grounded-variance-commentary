"""
Retrieval-augmented pipeline: corpus, retrieval, prompting, generation.

This module exists because the assignment prescribes this filename. The working
implementation lives in `src/gvc/`, which `tests/test_parity.py` imports, so it is
re-exported here rather than moved.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gvc.corpus import CAUSE_CATS, RULES_CATS, admissible, load_corpus, month_end
from gvc.retrieval import Retriever, RetrievalConfig
from gvc.prompt import (build_grounded_prompt, build_longcontext_prompt,
                        build_ungrounded_prompt, deterministic_template)
from gvc.generate import ARMS, DEFAULT_MODEL, generate, ollama_available

__all__ = [
    "ARMS",
    "CAUSE_CATS",
    "DEFAULT_MODEL",
    "RULES_CATS",
    "RetrievalConfig",
    "Retriever",
    "admissible",
    "build_grounded_prompt",
    "build_longcontext_prompt",
    "build_ungrounded_prompt",
    "deterministic_template",
    "generate",
    "load_corpus",
    "month_end",
    "ollama_available"
]
