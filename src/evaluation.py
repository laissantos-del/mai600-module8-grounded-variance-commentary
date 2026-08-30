"""
Scoring and the evaluation harness: metrics, per-arm runs, summary tables.

This module exists because the assignment prescribes this filename. The working
implementation lives in `src/gvc/`, which `tests/test_parity.py` imports, so it is
re-exported here rather than moved.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gvc.metrics import abstained, numeric_fidelity, retrieval_hit
from gvc.harness import run_arm, save, score_generation, summarise
from gvc.cases import CLOSE_PERIODS, CONFIG_PERIODS, load_cases

__all__ = [
    "CLOSE_PERIODS",
    "CONFIG_PERIODS",
    "abstained",
    "load_cases",
    "numeric_fidelity",
    "retrieval_hit",
    "run_arm",
    "save",
    "score_generation",
    "summarise"
]
