"""
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
