# tests/test_report.py
#
# Smoke + structural coverage for the PDF tearsheet builder. We assert the
# document is a well-formed multi-page PDF and that every branch (zero-roll
# strategy, single underlier, ongoing drawdown) renders without raising.

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pytest
from datetime import date


from stratbt.engine import BacktestResult, DayResult
from stratbt.report import build_pdf_report
from stratbt.report.data import build_report_data
from stratbt.report.pages import build_all_pages
from stratbt.report.widgets import _format_cell, render_table


def make_result(pnl_map: dict, with_rolls: bool = False,
                start_date: date = date(2022, 11, 1)) -> BacktestResult:
    n_days = max(len(v) for v in pnl_map.values())
    dates  = pd.bdate_range(start_date, periods=n_days).date
    day_results = []
    for underlier, pnls in pnl_map.items():
        for i, pnl in enumerate(pnls):
            ts  = pd.Timestamp(f"{dates[i]} 09:15:00")
            mtm = pd.Series(np.linspace(0, pnl, 100),
                            index=pd.date_range(ts, periods=100, freq="1s"))
            log = pd.DataFrame()
            if with_rolls:
                # two strikes over the session -> one roll, gives roll charts data
                idx = pd.date_range(ts, periods=2, freq="1s")
                log = pd.DataFrame(
                    {"strike": [41600, 41700], "ce_entry": [200.0, 210.0],
                     "pe_entry": [400.0, 390.0], "ce_mtm_price": [200.0, 210.0],
                     "pe_mtm_price": [400.0, 390.0]},
                    index=pd.DatetimeIndex(idx, name="timestamp"),
                )
            day_results.append(DayResult(
                date=dates[i], underlier=underlier, trades=[],
                mtm=mtm, final_pnl=float(pnl), holdings_log=log,
            ))
    return BacktestResult(day_results)


def _page_count(pdf: bytes) -> int:
    return len(re.findall(rb"/Type\s*/Page[^s]", pdf))


class TestBuildPdfReport:

    def test_returns_valid_pdf_bytes(self):
        r   = make_result({"NIFTY": [10.0, -5.0, 20.0], "BANKNIFTY": [1.0, -2.0, 3.0]})
        pdf = build_pdf_report(r, "ATM Straddle (Rolling)", "Nov 01 – Nov 03")
        assert isinstance(pdf, bytes)
        assert pdf[:5] == b"%PDF-"
        assert pdf[-6:].rstrip() == b"%%EOF"

    def test_page_count_matches_composed_pages(self):
        # Regression guard for the PDF-serialization step: every figure that
        # build_all_pages() composes must actually make it into the saved
        # PDF (import matplotlib fig, not skipped/closed early, etc).
        # Deliberately NOT a hardcoded page count — pages.py legitimately
        # grows new sections over time (pattern/statistics pages were added
        # after this test was first written), so pinning a magic number
        # here just makes the test go stale, not catch real regressions.
        r    = make_result({"NIFTY": [10.0, -5.0, 20.0], "BANKNIFTY": [1.0, -2.0, 3.0]})
        data = build_report_data(r, "ATM Straddle (Rolling)", "Nov 01 – Nov 03")
        probe_pages = build_all_pages(data)
        expected_pages = len(probe_pages)
        for fig, _ in probe_pages:
            plt.close(fig)

        pdf = build_pdf_report(r, "ATM Straddle (Rolling)", "Nov 01 – Nov 03")
        assert _page_count(pdf) == expected_pages

    def test_zero_roll_strategy_renders(self):
        # No holdings_log -> zero rolls -> the activity page takes its empty branch.
        r    = make_result({"NIFTY": [10.0, -5.0, 20.0]}, with_rolls=False)
        data = build_report_data(r, "Fixed Window Straddle", "Nov 01 – Nov 03")
        probe_pages = build_all_pages(data)
        expected_pages = len(probe_pages)
        for fig, _ in probe_pages:
            plt.close(fig)

        pdf = build_pdf_report(r, "Fixed Window Straddle", "Nov 01 – Nov 03")
        assert pdf[:5] == b"%PDF-"
        assert _page_count(pdf) == expected_pages

    def test_with_rolls_renders(self):
        r   = make_result({"NIFTY": [10.0, -5.0, 20.0], "BANKNIFTY": [1.0, 2.0, -3.0]},
                          with_rolls=True)
        pdf = build_pdf_report(r, "ATM Straddle (Rolling)", "Nov 01 – Nov 03")
        assert pdf[:5] == b"%PDF-"

    def test_single_underlier_renders(self):
        r   = make_result({"NIFTY": [10.0, -5.0, 20.0, -8.0]})
        pdf = build_pdf_report(r, "ATM Straddle (Rolling)", "Nov 01 – Nov 04")
        assert pdf[:5] == b"%PDF-"

    def test_all_negative_pnl_renders(self):
        # Loss-making -> break-even branch takes the "needs subsidy" text.
        r   = make_result({"NIFTY": [-10.0, -20.0, -5.0]})
        pdf = build_pdf_report(r, "ATM Straddle (Rolling)", "Nov 01 – Nov 03")
        assert pdf[:5] == b"%PDF-"

    def test_nonempty_output(self):
        r   = make_result({"NIFTY": [10.0, -5.0, 20.0]})
        pdf = build_pdf_report(r, "ATM Straddle (Rolling)", "Nov 01 – Nov 03")
        assert len(pdf) > 10_000  # a real multi-page figure PDF, not a stub


class TestFormatCell:
    """Unit coverage for widgets._format_cell, the per-value formatter
    render_table calls for every cell it draws."""

    def test_floats_formatted_with_precision(self):
        assert _format_cell(1.23456, precision=2) == "1.23"
        assert _format_cell(2.0, precision=2) == "2.00"

    def test_thousands_separator_on_large_floats(self):
        assert _format_cell(12345.6, precision=1) == "12,345.6"

    def test_integers_get_thousands_separator(self):
        assert _format_cell(1000) == "1,000"
        assert _format_cell(25000) == "25,000"

    def test_nan_float_becomes_dash_by_default(self):
        assert _format_cell(np.nan) == "—"

    def test_none_becomes_dash_by_default(self):
        assert _format_cell(None) == "—"

    def test_none_respects_custom_label(self):
        # drawdown_table's recovery_date is None for an unresolved episode —
        # callers that know the column's meaning can relabel it.
        assert _format_cell(None, none_label="Ongoing") == "Ongoing"

    def test_date_value_stringified_as_is(self):
        assert _format_cell(date(2022, 11, 4)) == "2022-11-04"


class TestRenderTableNoneLabels:
    """Integration check that render_table actually threads none_labels
    through to the cells it draws, not just that _format_cell supports it."""

    def test_recovery_date_none_renders_as_ongoing(self):
        df = pd.DataFrame({
            "underlier": ["NIFTY", "BANKNIFTY"],
            "recovery_date": [date(2022, 11, 4), None],
        })
        fig = plt.figure(figsize=(6, 2))
        render_table(fig, (0.05, 0.1, 0.9, 0.8), df,
                     none_labels={"recovery_date": "Ongoing"})
        texts = [c.get_text().get_text() for c in fig.axes[0].tables[0].get_celld().values()]
        assert "Ongoing" in texts
        assert "2022-11-04" in texts
        plt.close(fig)

    def test_unlisted_column_falls_back_to_dash(self):
        df  = pd.DataFrame({"depth": [None]})
        fig = plt.figure(figsize=(6, 2))
        render_table(fig, (0.05, 0.1, 0.9, 0.8), df)
        texts = [c.get_text().get_text() for c in fig.axes[0].tables[0].get_celld().values()]
        assert "—" in texts
        plt.close(fig)
