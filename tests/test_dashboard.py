# tests/test_dashboard.py
"""Smoke-test the Streamlit dashboard end to end on the synthetic market (no network)."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

import stratbt.dashboard  # noqa: E402

APP = str(Path(stratbt.dashboard.__file__).parent / "app.py")


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("STRATBT_SOURCE", "synthetic")
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _underliers_box(at):
    return next(m for m in at.sidebar.multiselect if m.label == "Underliers")


def _run(at, strategy="short_straddle", days=5, underliers=("SPY",)):
    at.sidebar.number_input(key="ndays").set_value(days).run()
    at.sidebar.selectbox(key="strategy").select(strategy).run()
    _underliers_box(at).set_value(list(underliers)).run()
    at.sidebar.button(key="run").click().run()
    assert not at.exception, [e.value for e in at.exception]


def test_landing_page_offers_examples_and_registry_strategies(app):
    labels = [b.label for b in app.main.button]
    assert labels.count("Run this") == 3
    options = app.sidebar.selectbox(key="strategy").options
    assert {"Iron Condor", "Short Straddle", "ATM Straddle (Rolling)"} <= set(options)
    assert not app.sidebar.button(key="run").disabled


def test_run_renders_results_with_every_tab(app):
    _run(app)
    run = app.session_state["runs"][0]
    assert run["data"].strategy_name == "Short Straddle" and run["data"].unit == "USD"
    metrics = [m.label for m in app.metric]
    assert {"Net PnL", "Sharpe", "Max drawdown", "Costs paid"} <= set(metrics)
    assert len(app.tabs) == 6 and not app.error
    assert run["cli"].startswith("stratbt run short_straddle --provider synthetic")


def test_page_is_rerendered_from_the_top_after_a_run(app):
    """No progress block is left above the title (the page used to keep its scroll offset and hide the header)."""
    _run(app)
    assert app.session_state["runs"]
    assert app.main.get("status") == []                       # the progress block is gone
    assert app.main.title[0].value == "Short Straddle"        # and the title is the first thing on the page


def test_second_run_adds_compare_tab_and_history(app):
    _run(app, "short_straddle")
    _run(app, "iron_condor")
    assert len(app.session_state["runs"]) == 2
    assert len(app.tabs) == 7                                  # + Compare


def test_example_button_runs_a_strategy(app):
    app.sidebar.number_input(key="ndays").set_value(5).run()
    next(b for b in app.main.button if b.key == "ex_short_strangle").click().run()
    assert not app.exception, [e.value for e in app.exception]
    assert app.session_state["runs"][0]["data"].strategy_name == "Short Strangle"


def test_underliers_reset_when_the_data_source_changes(app):
    assert _underliers_box(app).value == ["QQQ", "SPY"]
    app.sidebar.number_input(key="seed").set_value(11).run()     # same names, new market: selection kept sensible
    assert set(_underliers_box(app).value) == {"QQQ", "SPY"}
    assert not app.sidebar.button(key="run").disabled


def test_cost_inputs_feed_the_run(app):
    app.sidebar.number_input(key="commission").set_value(1.5).run()
    _run(app, "fixed_window")
    run = app.session_state["runs"][0]
    assert run["result"].meta["execution"]["commission_per_contract"] == 1.5
    assert "--commission 1.5" in run["cli"]
    assert run["data"].fees > 0


def test_missing_yahoo_extra_is_reported_not_crashed(monkeypatch):
    monkeypatch.setenv("STRATBT_SOURCE", "yahoo")
    monkeypatch.setattr("importlib.util.find_spec", lambda name, *a, **k: None if name == "yfinance" else __import__("importlib.machinery").machinery.PathFinder.find_spec(name))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert any("not installed" in e.value for e in at.sidebar.error)
