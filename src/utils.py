"""
The deterministic variance engine and its materiality rules.

This module exists because the assignment prescribes this filename. The working
implementation lives in `src/gvc/`, which `tests/test_parity.py` imports, so it is
re-exported here rather than moved.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gvc.variance_engine import (COVENANT_HEADROOM_TRIGGER, DOLLAR_ALONE_USD,
                                 DOLLAR_AND_PCT_PCT, DOLLAR_AND_PCT_USD,
                                 VarianceRow)

__all__ = [
    "COVENANT_HEADROOM_TRIGGER",
    "DOLLAR_ALONE_USD",
    "DOLLAR_AND_PCT_PCT",
    "DOLLAR_AND_PCT_USD",
    "VarianceRow"
]
