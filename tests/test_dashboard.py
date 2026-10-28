"""
Smoke tests for the Streamlit dashboard: every page renders without an error.
Uses Streamlit's headless test runner, so no browser or network is needed.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent.parent / "app.py"
PAGES = [
    "dashboard/views/overview.py",
    "dashboard/views/anomalies.py",
    "dashboard/views/backtest.py",
    "dashboard/views/data.py",
    "dashboard/views/method.py",
]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_errors(page):
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    at.switch_page(page).run()
    assert not at.exception, at.exception


def test_changing_threshold_updates_results():
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    at.switch_page("dashboard/views/anomalies.py").run()
    at.session_state["threshold"] = 3.5
    at.run()
    assert not at.exception, at.exception
    assert at.session_state["threshold"] == 3.5
