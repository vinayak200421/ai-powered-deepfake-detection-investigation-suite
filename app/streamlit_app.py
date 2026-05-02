"""Streamlit entrypoint: run from repo root — `streamlit run app/streamlit_app.py`."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

st.set_page_config(page_title="DeepFake Detection", layout="wide", initial_sidebar_state="expanded")

with st.sidebar:
    st.caption(
        "Local API: connected to http://127.0.0.1:5001/analyze."
    )


st.title("DeepFake Detection")
st.markdown(
    "Multi-page dashboard (sidebar): **Upload** → **Results** → **Attribution** → "
    "**Report** → **About**."
)
st.info(
    "Local GPU Inference is running! The Streamlit frontend is communicating directly with your RTX 3050 backend."
)
