"""
GoldWatch dashboard.

Run from the project root:
    streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "dashboard"))

import core  # noqa: E402

st.set_page_config(page_title="GoldWatch: unusual moves in gold prices",
                   page_icon=str(ROOT / "static" / "goldwatch-icon.svg"), layout="wide")
st.logo(str(ROOT / "static" / "goldwatch-logo.svg"), size="large")
core.inject_css()
core.init_state()

page = st.navigation(
    [st.Page(path, title=title, icon=icon, default=key == "overview")
     for key, (path, title, icon) in core.PAGES.items()],
    position="top",
)
page.run()
core.footer()
