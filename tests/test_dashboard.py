# tests/test_dashboard.py
"""Smoke-test the Streamlit dashboard end to end on the synthetic market."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

import stratbt.dashboard  # noqa: E402

APP = str(Path(stratbt.dashboard.__file__).parent / "app.py")


@pytest.fixture
def app(monkeypatch, tmp_path):
    monkeypatch.setenv("STRATBT_DATA_DIR", str(tmp_path / "missing"))   # force synthetic default
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception
    return at


def test_loads_with_no_data_and_offers_registry_strategies(app):
    assert app.sidebar.radio[0].value == "Synthetic market"
    options = app.sidebar.selectbox[0].options
    assert {"Iron Condor", "Short Straddle", "ATM Straddle (Rolling)"} <= set(options)
    assert not app.sidebar.button[0].disabled


def test_run_short_straddle_and_render_every_tab(app):
    app.sidebar.slider[0].set_value(5)                 # synthetic trading days
    app.sidebar.selectbox[0].select("short_straddle")
    app.run()
    app.sidebar.multiselect[0].set_value(["SPY"])
    app.run()
    app.sidebar.button[0].click()
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    labels = [m.label for m in app.metric]
    assert "Combined PnL" in labels and "Total Trades" in labels
    assert len(app.tabs) == 7
    assert app.session_state["strategy_name"] == "Short Straddle"
    assert app.session_state["unit"] == "USD"
