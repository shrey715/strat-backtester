# tests/test_engine.py
#
# BacktestResult / DayResult containers on hand-built data (fast, no market data needed).
# Engine behaviour on a full market is covered in test_engine_synthetic.py.

import pytest
import numpy as np
import pandas as pd
from datetime import date

from stratbt.engine import BacktestResult, DayResult
from stratbt.position import Trade
from stratbt.instruments import Instrument

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_fake_day_result(
    trading_date: date  = date(2022, 11, 1),
    underlier:    str   = "BANKNIFTY",
    n_trades:     int   = 4,
    final_pnl:    float = 100.0,
) -> DayResult:
    """Build a DayResult with fake data for unit testing BacktestResult."""
    ts  = pd.Timestamp(f"{trading_date} 09:15:00")
    idx = pd.date_range(ts, periods=22_500, freq="1s")
    mtm = pd.Series(np.linspace(0, final_pnl, 22_500), index=idx)

    inst   = Instrument(underlier, date(2022, 11, 3), 41600, "CE")
    trades = [
        Trade(
            timestamp=ts + pd.Timedelta(seconds=i),
            instrument=inst,
            side="BUY" if i % 2 == 0 else "SELL",
            price=242.0 + i,
            qty=1,
        )
        for i in range(n_trades)
    ]
    # minimal holdings_log: two rows at the same strike, 1 second apart
    holdings_log = pd.DataFrame(
        [
            {"strike": 41600, "ce_instrument": inst.filename,
             "pe_instrument": inst.filename,
             "ce_entry": 242.0, "pe_entry": 414.0,
             "ce_mtm_price": 242.0, "pe_mtm_price": 414.0},
        ],
        index=pd.Index([ts], name="timestamp"),
    )
    return DayResult(
        date=trading_date,
        underlier=underlier,
        trades=trades,
        mtm=mtm,
        final_pnl=final_pnl,
        holdings_log=holdings_log,
    )


# ===========================================================================
# BacktestResult unit tests (fake data — fast)
# ===========================================================================

class TestBacktestResultEmpty:
    def test_empty_trades_df_has_columns(self):
        result = BacktestResult([])
        assert list(result.trades_df.columns) == [
            "timestamp", "date", "underlier", "instrument",
            "expiry", "strike", "option_type", "side", "price", "qty", "fee", "slippage"
        ]

    def test_empty_daily_pnl_is_empty(self):
        result = BacktestResult([])
        assert result.daily_pnl.empty

    def test_empty_daily_pnl_has_no_columns(self):
        # Regression test: daily_pnl is documented as "pivoted by underlier",
        # so its columns should be underlier names (none, when there are no
        # results) — not the long-format ["date","underlier","pnl"] schema
        # of the intermediate rows used to build the pivot. Metrics functions
        # iterate `daily_pnl.columns` expecting underlier tickers; the old
        # placeholder columns made them silently treat "date"/"underlier"/
        # "pnl" as if they were tickers (e.g. risk_table would emit a
        # spurious row indexed "pnl").
        result = BacktestResult([])
        assert list(result.daily_pnl.columns) == []
        assert result.daily_pnl.index.name == "date"

    def test_empty_mtm_df_is_empty(self):
        result = BacktestResult([])
        assert result.mtm_df.empty

    def test_empty_day_results(self):
        result = BacktestResult([])
        assert result.day_results == []

    def test_empty_holdings_log_df_is_empty(self):
        result = BacktestResult([])
        assert result.holdings_log_df.empty


class TestBacktestResultSingleDay:
    def setup_method(self):
        self.result = BacktestResult([
            make_fake_day_result(
                trading_date=date(2022, 11, 1),
                underlier="BANKNIFTY",
                n_trades=4,
                final_pnl=200.0,
            )
        ])

    def test_day_results_count(self):
        assert len(self.result.day_results) == 1

    def test_trades_df_row_count(self):
        assert len(self.result.trades_df) == 4

    def test_trades_df_columns_present(self):
        expected = {
            "timestamp", "date", "underlier", "instrument",
            "expiry", "strike", "option_type", "side", "price", "qty"
        }
        assert expected.issubset(set(self.result.trades_df.columns))

    def test_trades_df_sorted_by_timestamp(self):
        df = self.result.trades_df
        assert list(df["timestamp"]) == sorted(df["timestamp"])

    def test_daily_pnl_shape(self):
        assert self.result.daily_pnl.shape == (1, 1)

    def test_daily_pnl_value(self):
        assert self.result.daily_pnl.at[date(2022, 11, 1), "BANKNIFTY"] == pytest.approx(200.0)

    def test_mtm_df_has_underlier_column(self):
        assert "BANKNIFTY" in self.result.mtm_df.columns

    def test_mtm_df_length(self):
        assert len(self.result.mtm_df) == 22_500

    def test_holdings_log_df_not_empty(self):
        assert not self.result.holdings_log_df.empty

    def test_holdings_log_df_has_underlier(self):
        assert "underlier" in self.result.holdings_log_df.columns


class TestBacktestResultMultipleDays:
    def setup_method(self):
        days = [date(2022, 11, 1), date(2022, 11, 2), date(2022, 11, 3)]
        self.result = BacktestResult([
            make_fake_day_result(
                trading_date=d,
                underlier="BANKNIFTY",
                final_pnl=float(i * 100),
            )
            for i, d in enumerate(days)
        ])

    def test_day_results_count(self):
        assert len(self.result.day_results) == 3

    def test_daily_pnl_rows(self):
        assert len(self.result.daily_pnl) == 3

    def test_mtm_df_length(self):
        assert len(self.result.mtm_df) == 3 * 22_500


class TestBacktestResultTwoUnderliers:
    def setup_method(self):
        self.result = BacktestResult([
            make_fake_day_result(underlier="BANKNIFTY", final_pnl=100.0),
            make_fake_day_result(underlier="NIFTY",     final_pnl=200.0),
        ])

    def test_daily_pnl_has_both_underliers(self):
        assert "BANKNIFTY" in self.result.daily_pnl.columns
        assert "NIFTY" in self.result.daily_pnl.columns

    def test_mtm_df_has_both_underliers(self):
        assert "BANKNIFTY" in self.result.mtm_df.columns
        assert "NIFTY" in self.result.mtm_df.columns

    def test_trades_df_sorted_across_underliers(self):
        df = self.result.trades_df
        assert list(df["timestamp"]) == sorted(df["timestamp"])

    def test_trades_df_contains_both_underliers(self):
        underliers = set(self.result.trades_df["underlier"])
        assert "BANKNIFTY" in underliers
        assert "NIFTY" in underliers

    def test_holdings_log_df_contains_both_underliers(self):
        underliers = set(self.result.holdings_log_df["underlier"])
        assert "BANKNIFTY" in underliers
        assert "NIFTY" in underliers


class TestHoldingsLogMixedDays:

    def test_empty_days_skipped_in_holdings_log_df(self):
        # One day with a holdings log, one without -> the empty one is skipped.
        ts  = pd.Timestamp("2022-11-01 09:15:00")
        log = pd.DataFrame(
            {"strike": [41600]},
            index=pd.DatetimeIndex([ts], name="timestamp"),
        )
        with_log    = make_fake_day_result(date(2022, 11, 1), "BANKNIFTY", n_trades=0)
        with_log.holdings_log = log
        without_log = make_fake_day_result(date(2022, 11, 2), "BANKNIFTY", n_trades=0)
        without_log.holdings_log = pd.DataFrame()
        combined = BacktestResult([with_log, without_log]).holdings_log_df
        assert len(combined) == 1
        assert combined["date"].iloc[0] == date(2022, 11, 1)

