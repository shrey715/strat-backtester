# tests/test_report.py
#
# PDF tearsheet: structural coverage of every branch (zero-roll strategy,
# single underlier, losses only, empty result, costs, shorts) plus unit
# tests for the formatters and the generated prose.

import re
from datetime import date

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from stratbt import ExecutionModel, SyntheticProvider, run_backtest
from stratbt.engine import BacktestResult, DayResult
from stratbt.report import build_pdf_report, theme as T
from stratbt.report.data import build_report_data
from stratbt.report.pages import build_all_pages


def make_result(pnl_map: dict, with_rolls: bool = False, start_date: date = date(2022, 11, 1),
                unit: str = "pts") -> BacktestResult:
    n_days = max(len(v) for v in pnl_map.values())
    dates = pd.bdate_range(start_date, periods=n_days).date
    day_results = []
    for underlier, pnls in pnl_map.items():
        for i, pnl in enumerate(pnls):
            ts = pd.Timestamp(f"{dates[i]} 09:15:00")
            mtm = pd.Series(np.linspace(0, pnl, 100), index=pd.date_range(ts, periods=100, freq="1s"))
            log = pd.DataFrame()
            if with_rolls:
                idx = pd.date_range(ts, periods=2, freq="1s")
                log = pd.DataFrame(
                    {"strike": [41600, 41700], "ce_entry": [200.0, 210.0], "pe_entry": [400.0, 390.0],
                     "ce_mtm_price": [200.0, 210.0], "pe_mtm_price": [400.0, 390.0]},
                    index=pd.DatetimeIndex(idx, name="timestamp"))
            day_results.append(DayResult(date=dates[i], underlier=underlier, trades=[], mtm=mtm,
                                         final_pnl=float(pnl), holdings_log=log))
    return BacktestResult(day_results, unit=unit)


def _page_count(pdf: bytes) -> int:
    return len(re.findall(rb"/Type\s*/Page[^s]", pdf))


def _text_of(fig) -> str:
    return "\n".join(t.get_text() for t in fig.findobj(matplotlib.text.Text))


class TestBuildPdfReport:

    def test_returns_valid_pdf_bytes(self):
        r = make_result({"NIFTY": [10.0, -5.0, 20.0], "BANKNIFTY": [1.0, -2.0, 3.0]})
        pdf = build_pdf_report(r, "ATM Straddle (Rolling)", "Nov 01 – Nov 03")
        assert pdf[:5] == b"%PDF-" and pdf[-6:].rstrip() == b"%%EOF"

    def test_page_count_matches_composed_pages(self):
        r = make_result({"NIFTY": [10.0, -5.0, 20.0], "BANKNIFTY": [1.0, -2.0, 3.0]})
        pages = build_all_pages(build_report_data(r, "ATM Straddle (Rolling)", "Nov 01 – Nov 03"))
        expected = len(pages)
        for fig, _ in pages:
            plt.close(fig)
        assert _page_count(build_pdf_report(r, "ATM Straddle (Rolling)", "x")) == expected

    @pytest.mark.parametrize("pnl_map,rolls", [
        ({"NIFTY": [10.0, -5.0, 20.0]}, False),                                   # zero-roll
        ({"NIFTY": [10.0, -5.0, 20.0], "BANKNIFTY": [1.0, 2.0, -3.0]}, True),     # with rolls
        ({"NIFTY": [10.0, -5.0, 20.0, -8.0]}, False),                             # single underlier
        ({"NIFTY": [-10.0, -20.0, -5.0]}, False),                                 # loss-making
        ({"NIFTY": [5.0]}, False),                                                # one day: std undefined
        ({"NIFTY": [0.0, 0.0, 0.0]}, False),                                      # flat
    ])
    def test_edge_cases_render(self, pnl_map, rolls):
        pdf = build_pdf_report(make_result(pnl_map, with_rolls=rolls), "S", "")
        assert pdf[:5] == b"%PDF-" and len(pdf) > 10_000

    def test_empty_result_renders(self):
        pdf = build_pdf_report(BacktestResult([]), "Nothing ran", "")
        assert pdf[:5] == b"%PDF-"

    def test_long_runs_paginate_daily_table(self):
        r = make_result({"NIFTY": list(np.random.default_rng(1).normal(0, 10, 70))})
        data = build_report_data(r, "S", "")
        pages = build_all_pages(data)
        sections = [s for _, s in pages]
        assert "Daily results (continued)" in sections
        for fig, _ in pages:
            plt.close(fig)

    def test_synthetic_strategies_with_costs_and_shorts(self):
        prov = SyntheticProvider(n_days=4, seed=3)
        for key in ("short_straddle", "iron_condor"):
            r = run_backtest(key, prov, execution=ExecutionModel.realistic(), underliers=["SPY"])
            assert build_pdf_report(r)[:5] == b"%PDF-"


class TestReportData:

    def test_uses_strategy_label_from_meta(self):
        r = run_backtest("iron_condor", SyntheticProvider(n_days=2), underliers=["SPY"])
        assert build_report_data(r, "ignored key").strategy_name == "Iron Condor"

    def test_falls_back_to_given_name(self):
        r = make_result({"NIFTY": [1.0, 2.0]})
        assert build_report_data(r, "My Strategy").strategy_name == "My Strategy"

    def test_cost_identity_gross_minus_costs_is_net(self):
        r = run_backtest("fixed_window", SyntheticProvider(n_days=3), execution=ExecutionModel.realistic(),
                         underliers=["SPY"])
        d = build_report_data(r)
        assert d.gross_pnl - d.fees - d.slippage == pytest.approx(d.net_pnl)
        assert d.fees > 0 and d.slippage > 0

    def test_observations_flag_small_sample_and_costs(self):
        r = run_backtest("fixed_window", SyntheticProvider(n_days=4), execution=ExecutionModel.realistic(),
                         underliers=["SPY"])
        text = " ".join(build_report_data(r).observations)
        assert "Small sample" in text and "4 days" in text
        assert "costs" in text.lower()

    def test_short_premium_warning_only_for_short_structures(self):
        prov = SyntheticProvider(n_days=2)
        short = " ".join(build_report_data(run_backtest("short_straddle", prov, underliers=["SPY"])).observations)
        long_ = " ".join(build_report_data(run_backtest("long_strangle", prov, underliers=["SPY"])).observations)
        condor = " ".join(build_report_data(run_backtest("iron_condor", prov, underliers=["SPY"])).observations)
        assert "open-ended loss" in short
        assert "open-ended loss" not in long_ and "open-ended loss" not in condor   # defined risk

    def test_combined_column_only_for_multiple_underliers(self):
        one = build_report_data(make_result({"A": [1.0, 2.0, 3.0]}))
        two = build_report_data(make_result({"A": [1.0, 2.0, 3.0], "B": [1.0, 1.0, 1.0]}))
        assert "Combined" not in one.stats_cols and "Combined" in two.stats_cols

    def test_config_rows_describe_the_run(self):
        r = run_backtest("short_strangle", SyntheticProvider(n_days=2), params={"width": 3},
                         execution=ExecutionModel.realistic(), underliers=["SPY"])
        cfg = dict(build_report_data(r).config_rows)
        assert "width = 3" in cfg["Parameters"] and "25 bps" in cfg["Execution"]


class TestPageText:

    def test_dollar_signs_are_not_mathtext(self):
        r = run_backtest("short_strangle", SyntheticProvider(n_days=3), execution=ExecutionModel.realistic(),
                         underliers=["SPY"])
        pages = build_all_pages(build_report_data(r))
        fig = pages[0][0]
        text = _text_of(fig)
        assert "$" in text
        assert all(t.get_text() == t.get_text() and not t.get_parse_math()
                   for t in fig.findobj(matplotlib.text.Text) if "$" in t.get_text())
        for f, _ in pages:
            plt.close(f)


class TestFormatters:

    def test_fnum(self):
        assert T.fnum(1234.5, 1) == "1,234.5"
        assert T.fnum(-2.0, 0) == T.MINUS + "2"
        assert T.fnum(float("nan")) == T.DASH
        assert T.fnum(float("inf")) == "∞"
        assert T.fnum(3.2, 1, signed=True) == "+3.2"
        assert T.fnum(-0.001, 2) == "0.00"                # no "−0.00"

    def test_fmoney_by_unit(self):
        assert T.fmoney(1234, "USD") == "$1,234"
        assert T.fmoney(-1234, "USD") == T.MINUS + "$1,234"
        assert T.fmoney(1234, "pts") == "1,234 pts"
        assert T.fmoney(5, "USD", signed=True) == "+$5"
        assert T.fmoney(float("nan"), "USD") == T.DASH

    def test_fcompact(self):
        assert T.fcompact(27109, "USD") == "$27.1k"
        assert T.fcompact(1_500_000, "USD") == "$1.5M"
        assert T.fcompact(950, "USD") == "$950"

    def test_fduration_and_pct(self):
        assert T.fduration(45) == "45s"
        assert T.fduration(125) == "2m 05s"
        assert T.fduration(22800) == "6h 20m"
        assert T.fpct(0.756) == "76%"

    def test_series_colors_follow_the_name_not_the_rank(self):
        assert T.series_color("SPY") == T.series_color("SPY", ["SPY"]) == T.series_color("SPY", ["QQQ", "SPY"])
        assert T.series_color("SPY") != T.series_color("QQQ")
        assert T.series_color("Combined") == T.COMBINED
        assert T.series_color("NEW1") != T.series_color("NEW2")
