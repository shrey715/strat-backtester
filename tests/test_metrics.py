# tests/test_metrics.py

import pytest
import numpy as np
import pandas as pd
from datetime import date
from pathlib import Path
from matplotlib.figure import Figure

from stratbt.instruments import Instrument
from stratbt.position import Trade
from stratbt.engine import BacktestResult, DayResult
from stratbt.metrics import (
    _round_trips,
    total_pnl,
    mean_daily_pnl,
    std_daily_pnl,
    sharpe_ratio,
    rolling_sharpe,
    sortino_ratio,
    calmar_ratio,
    max_drawdown,
    max_drawdown_duration,
    win_rate,
    profit_factor,
    avg_holding_time,
    rolls_per_day,
    cost_sensitivity,
    summary_table,
    plot_cumulative_pnl,
    plot_daily_pnl,
    plot_drawdown,
    plot_intraday_mtm,
    plot_rolls_per_day,
    plot_holdings,
    plot_rolling_sharpe,
    plot_cost_sensitivity,
)
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Builder helpers — all fake data, no disk access
# ---------------------------------------------------------------------------

def make_day_result(
    trading_date: date,
    underlier: str,
    pnl: float,
    trades: list | None = None,
    holdings_log: pd.DataFrame | None = None,
) -> DayResult:
    ts  = pd.Timestamp(f"{trading_date} 09:15:00")
    idx = pd.date_range(ts, periods=22_500, freq="1s")
    mtm = pd.Series(np.linspace(0, pnl, 22_500), index=idx)
    return DayResult(
        date=trading_date,
        underlier=underlier,
        trades=trades or [],
        mtm=mtm,
        final_pnl=pnl,
        holdings_log=holdings_log if holdings_log is not None else pd.DataFrame(),
    )


def make_result(
    pnl_map: dict,
    start_date: date = date(2022, 11, 1),
) -> BacktestResult:
    """
    Build a BacktestResult with precisely controlled daily PnLs.
    pnl_map: { underlier: [pnl_day1, pnl_day2, ...] }
    Uses business dates starting from start_date.
    """
    n_days = max(len(v) for v in pnl_map.values())
    dates  = pd.bdate_range(start_date, periods=n_days).date
    day_results = []
    for underlier, pnls in pnl_map.items():
        for i, pnl in enumerate(pnls):
            day_results.append(make_day_result(dates[i], underlier, pnl))
    return BacktestResult(day_results)


def make_holdings_log(timestamps: list, strikes: list) -> pd.DataFrame:
    """Build a fake holdings_log DataFrame keyed by DatetimeIndex."""
    records = [
        {
            "strike":          strike,
            "ce_instrument":   f"BANKNIFTY221103{strike}CE.csv",
            "pe_instrument":   f"BANKNIFTY221103{strike}PE.csv",
            "ce_entry":        200.0,
            "pe_entry":        400.0,
            "ce_mtm_price":    200.0,
            "pe_mtm_price":    400.0,
        }
        for strike in strikes
    ]
    df = pd.DataFrame(records)
    df.index = pd.DatetimeIndex(timestamps, name="timestamp")
    return df


# Shared instruments for trade-based tests
CE = Instrument("BANKNIFTY", date(2022, 11, 3), 41600, "CE")
PE = Instrument("BANKNIFTY", date(2022, 11, 3), 41600, "PE")

# ===========================================================================
# 1. _round_trips (private helper)
# ===========================================================================

class TestRoundTrips:

    def _df(self, rows):
        return pd.DataFrame(rows)

    def _row(self, side, price, qty=1, ts="2022-11-01 09:15:00",
             instrument="A.csv", date_=date(2022, 11, 1), underlier="BANKNIFTY"):
        return {
            "date": date_, "underlier": underlier, "instrument": instrument,
            "side": side, "timestamp": pd.Timestamp(ts), "price": price, "qty": qty,
        }

    def test_empty_df_returns_empty(self):
        df = pd.DataFrame(columns=["date", "underlier", "instrument",
                                    "side", "timestamp", "price", "qty"])
        assert _round_trips(df).empty

    def test_single_buy_no_sell_returns_empty(self):
        df = self._df([self._row("BUY", 100.0)])
        assert _round_trips(df).empty

    def test_single_sell_no_buy_returns_empty(self):
        df = self._df([self._row("SELL", 100.0)])
        assert _round_trips(df).empty

    def test_one_round_trip_profit(self):
        df = self._df([
            self._row("BUY",  100.0, ts="2022-11-01 09:15:00"),
            self._row("SELL", 120.0, ts="2022-11-01 09:16:00"),
        ])
        rt = _round_trips(df)
        assert len(rt) == 1
        assert rt.iloc[0]["pnl"] == pytest.approx(20.0)

    def test_one_round_trip_loss(self):
        df = self._df([
            self._row("BUY",  100.0, ts="2022-11-01 09:15:00"),
            self._row("SELL",  80.0, ts="2022-11-01 09:16:00"),
        ])
        rt = _round_trips(df)
        assert rt.iloc[0]["pnl"] == pytest.approx(-20.0)

    def test_breakeven_pnl_zero(self):
        df = self._df([
            self._row("BUY",  100.0, ts="2022-11-01 09:15:00"),
            self._row("SELL", 100.0, ts="2022-11-01 09:16:00"),
        ])
        rt = _round_trips(df)
        assert rt.iloc[0]["pnl"] == pytest.approx(0.0)

    def test_holding_time_exact_seconds(self):
        """BUY at 09:15:00, SELL at 09:15:45 → 45 seconds."""
        df = self._df([
            self._row("BUY",  100.0, ts="2022-11-01 09:15:00"),
            self._row("SELL", 110.0, ts="2022-11-01 09:15:45"),
        ])
        rt = _round_trips(df)
        assert rt.iloc[0]["holding_seconds"] == pytest.approx(45.0)

    def test_holding_time_across_minutes(self):
        """BUY at 09:15:00, SELL at 09:30:00 → 900 seconds."""
        df = self._df([
            self._row("BUY",  100.0, ts="2022-11-01 09:15:00"),
            self._row("SELL", 110.0, ts="2022-11-01 09:30:00"),
        ])
        rt = _round_trips(df)
        assert rt.iloc[0]["holding_seconds"] == pytest.approx(900.0)

    def test_qty_scaling(self):
        """pnl = (sell - buy) * qty = (120 - 100) * 3 = 60."""
        df = self._df([
            self._row("BUY",  100.0, qty=3, ts="2022-11-01 09:15:00"),
            self._row("SELL", 120.0, qty=3, ts="2022-11-01 09:16:00"),
        ])
        rt = _round_trips(df)
        assert rt.iloc[0]["pnl"] == pytest.approx(60.0)
        assert rt.iloc[0]["qty"] == 3

    def test_multiple_round_trips_same_instrument(self):
        """Three sequential BUY-SELL pairs → three round trips, each +10."""
        rows = []
        for i in range(3):
            rows.append(self._row("BUY",  100.0 + i, ts=f"2022-11-01 09:{15+i*2}:00"))
            rows.append(self._row("SELL", 110.0 + i, ts=f"2022-11-01 09:{15+i*2+1}:00"))
        rt = _round_trips(self._df(rows))
        assert len(rt) == 3
        for _, row in rt.iterrows():
            assert row["pnl"] == pytest.approx(10.0)

    def test_sequential_matching_order(self):
        """
        BUY@100, SELL@110, BUY@115, SELL@130 →
        round trip 1: pnl=10, round trip 2: pnl=15
        """
        rows = [
            self._row("BUY",  100.0, ts="2022-11-01 09:15:00"),
            self._row("SELL", 110.0, ts="2022-11-01 09:16:00"),
            self._row("BUY",  115.0, ts="2022-11-01 09:17:00"),
            self._row("SELL", 130.0, ts="2022-11-01 09:18:00"),
        ]
        rt = _round_trips(self._df(rows))
        assert len(rt) == 2
        assert rt.iloc[0]["pnl"] == pytest.approx(10.0)
        assert rt.iloc[1]["pnl"] == pytest.approx(15.0)

    def test_two_instruments_grouped_separately(self):
        """CE and PE are different instruments → 2 separate round trips."""
        rows = [
            self._row("BUY",  242.0, instrument="CE.csv", ts="2022-11-01 09:15:00"),
            self._row("BUY",  414.0, instrument="PE.csv", ts="2022-11-01 09:15:00"),
            self._row("SELL", 260.0, instrument="CE.csv", ts="2022-11-01 09:30:00"),
            self._row("SELL", 400.0, instrument="PE.csv", ts="2022-11-01 09:30:00"),
        ]
        rt = _round_trips(self._df(rows))
        assert len(rt) == 2
        ce_rt = rt[rt["instrument"] == "CE.csv"].iloc[0]
        pe_rt = rt[rt["instrument"] == "PE.csv"].iloc[0]
        assert ce_rt["pnl"] == pytest.approx(18.0)
        assert pe_rt["pnl"] == pytest.approx(-14.0)

    def test_unbalanced_more_buys_than_sells(self):
        """2 buys, 1 sell → only 1 round trip; second buy stays open."""
        rows = [
            self._row("BUY",  100.0, ts="2022-11-01 09:15:00"),
            self._row("BUY",  110.0, ts="2022-11-01 09:16:00"),
            self._row("SELL", 120.0, ts="2022-11-01 09:17:00"),
        ]
        rt = _round_trips(self._df(rows))
        assert len(rt) == 1

    def test_unbalanced_more_sells_than_buys(self):
        """1 buy, 2 sells → only 1 round trip."""
        rows = [
            self._row("BUY",  100.0, ts="2022-11-01 09:15:00"),
            self._row("SELL", 110.0, ts="2022-11-01 09:16:00"),
            self._row("SELL", 120.0, ts="2022-11-01 09:17:00"),
        ]
        rt = _round_trips(self._df(rows))
        assert len(rt) == 1

    def test_result_has_expected_columns(self):
        rows = [
            self._row("BUY",  100.0, ts="2022-11-01 09:15:00"),
            self._row("SELL", 110.0, ts="2022-11-01 09:16:00"),
        ]
        rt = _round_trips(self._df(rows))
        expected = {"date", "underlier", "instrument", "buy_time", "sell_time",
                    "buy_price", "sell_price", "qty", "pnl", "holding_seconds"}
        assert expected.issubset(set(rt.columns))

    def test_two_underliers_grouped_separately(self):
        bn_rows = [
            self._row("BUY",  100.0, underlier="BANKNIFTY", ts="2022-11-01 09:15:00"),
            self._row("SELL", 110.0, underlier="BANKNIFTY", ts="2022-11-01 09:16:00"),
        ]
        ni_rows = [
            self._row("BUY",  50.0, underlier="NIFTY", ts="2022-11-01 09:15:00"),
            self._row("SELL", 55.0, underlier="NIFTY", ts="2022-11-01 09:16:00"),
        ]
        rt = _round_trips(pd.DataFrame(bn_rows + ni_rows))
        assert set(rt["underlier"]) == {"BANKNIFTY", "NIFTY"}
        assert rt[rt["underlier"] == "BANKNIFTY"].iloc[0]["pnl"] == pytest.approx(10.0)
        assert rt[rt["underlier"] == "NIFTY"].iloc[0]["pnl"]     == pytest.approx(5.0)


# ===========================================================================
# 2. total_pnl
# ===========================================================================

class TestTotalPnl:

    def test_all_positive(self):
        result = make_result({"BANKNIFTY": [100.0, 200.0, 300.0]})
        assert total_pnl(result)["BANKNIFTY"] == pytest.approx(600.0)

    def test_all_negative(self):
        result = make_result({"BANKNIFTY": [-50.0, -100.0, -150.0]})
        assert total_pnl(result)["BANKNIFTY"] == pytest.approx(-300.0)

    def test_mixed_cancels(self):
        result = make_result({"BANKNIFTY": [100.0, -100.0]})
        assert total_pnl(result)["BANKNIFTY"] == pytest.approx(0.0)

    def test_mixed_net_positive(self):
        result = make_result({"BANKNIFTY": [100.0, -50.0, 200.0, -75.0]})
        assert total_pnl(result)["BANKNIFTY"] == pytest.approx(175.0)

    def test_two_underliers_independent(self):
        result = make_result({"BANKNIFTY": [100.0, 200.0], "NIFTY": [50.0, -30.0]})
        assert total_pnl(result)["BANKNIFTY"] == pytest.approx(300.0)
        assert total_pnl(result)["NIFTY"]     == pytest.approx(20.0)

    def test_single_day(self):
        result = make_result({"BANKNIFTY": [42.0]})
        assert total_pnl(result)["BANKNIFTY"] == pytest.approx(42.0)


# ===========================================================================
# 3. mean_daily_pnl
# ===========================================================================

class TestMeanDailyPnl:

    def test_uniform_days(self):
        result = make_result({"BANKNIFTY": [50.0, 50.0, 50.0]})
        assert mean_daily_pnl(result)["BANKNIFTY"] == pytest.approx(50.0)

    def test_known_mean(self):
        # mean([100, -50, 200, -25]) = 225/4 = 56.25
        result = make_result({"BANKNIFTY": [100.0, -50.0, 200.0, -25.0]})
        assert mean_daily_pnl(result)["BANKNIFTY"] == pytest.approx(56.25)

    def test_zero_mean(self):
        result = make_result({"BANKNIFTY": [100.0, -100.0]})
        assert mean_daily_pnl(result)["BANKNIFTY"] == pytest.approx(0.0)

    def test_single_day_mean_equals_value(self):
        result = make_result({"BANKNIFTY": [77.0]})
        assert mean_daily_pnl(result)["BANKNIFTY"] == pytest.approx(77.0)


# ===========================================================================
# 4. std_daily_pnl
# ===========================================================================

class TestStdDailyPnl:

    def test_uniform_days_std_is_zero(self):
        result = make_result({"BANKNIFTY": [50.0, 50.0, 50.0]})
        assert std_daily_pnl(result)["BANKNIFTY"] == pytest.approx(0.0)

    def test_known_std_two_values(self):
        # sample std([10, 20]) = sqrt(50) = 5*sqrt(2)
        result   = make_result({"BANKNIFTY": [10.0, 20.0]})
        expected = np.std([10.0, 20.0], ddof=1)
        assert std_daily_pnl(result)["BANKNIFTY"] == pytest.approx(expected)

    def test_known_std_four_values(self):
        vals     = [100.0, -50.0, 200.0, -25.0]
        result   = make_result({"BANKNIFTY": vals})
        expected = np.std(vals, ddof=1)
        assert std_daily_pnl(result)["BANKNIFTY"] == pytest.approx(expected)


# ===========================================================================
# 5. sharpe_ratio
# ===========================================================================

class TestSharpeRatio:

    def test_formula_matches_hand_calculation(self):
        pnls     = [10.0, -5.0, 20.0, 8.0, -3.0]
        result   = make_result({"BANKNIFTY": pnls})
        expected = (np.mean(pnls) / np.std(pnls, ddof=1)) * np.sqrt(252)
        assert sharpe_ratio(result)["BANKNIFTY"] == pytest.approx(expected)

    def test_negative_mean_gives_negative_sharpe(self):
        result = make_result({"BANKNIFTY": [-10.0, -20.0, -15.0]})
        assert sharpe_ratio(result)["BANKNIFTY"] < 0

    def test_positive_mean_gives_positive_sharpe(self):
        result = make_result({"BANKNIFTY": [10.0, 20.0, 15.0]})
        assert sharpe_ratio(result)["BANKNIFTY"] > 0

    def test_zero_mean_gives_zero_sharpe(self):
        result = make_result({"BANKNIFTY": [50.0, -50.0, 50.0, -50.0]})
        assert sharpe_ratio(result)["BANKNIFTY"] == pytest.approx(0.0)

    def test_two_underliers_opposite_sign(self):
        result = make_result({
            "BANKNIFTY": [100.0, 100.0, 100.0],
            "NIFTY":     [-50.0, -50.0, -50.0],
        })
        assert sharpe_ratio(result)["BANKNIFTY"] > 0
        assert sharpe_ratio(result)["NIFTY"]     < 0

    def test_annualisation_factor_is_sqrt252(self):
        """Verify sqrt(252) is the annualisation factor, not sqrt(21) or 252."""
        pnls   = [10.0, 20.0, 15.0]
        result = make_result({"BANKNIFTY": pnls})
        mean   = np.mean(pnls)
        std    = np.std(pnls, ddof=1)
        expected_with_252    = (mean / std) * np.sqrt(252)
        expected_with_21     = (mean / std) * np.sqrt(21)
        actual = sharpe_ratio(result)["BANKNIFTY"]
        assert actual == pytest.approx(expected_with_252)
        assert actual != pytest.approx(expected_with_21)


class TestRollingSharpe:
    """
    Regression coverage gap: rolling_sharpe() itself had zero direct tests —
    only plot_rolling_sharpe() (a thin matplotlib wrapper around it) was
    tested, and only for "returns a Figure without crashing", which would
    not catch a wrong formula, wrong window, or misaligned index.
    """

    def test_first_window_minus_one_rows_are_nan(self):
        # window=3 needs 3 data points; the first 2 rows can't have a full
        # window yet and must be NaN, not zero or some partial estimate.
        result = make_result({"BANKNIFTY": [10.0, -5.0, 20.0, 8.0, -3.0]})
        rs = rolling_sharpe(result, window=3)["BANKNIFTY"]
        assert rs.iloc[:2].isna().all()
        assert rs.iloc[2:].notna().all()

    def test_full_window_matches_hand_calculation(self):
        pnls   = [10.0, -5.0, 20.0, 8.0, -3.0]
        result = make_result({"BANKNIFTY": pnls})
        window = 3
        rs = rolling_sharpe(result, window=window)["BANKNIFTY"]

        # third row (index 2): window covers pnls[0:3]
        expected = (np.mean(pnls[0:3]) / np.std(pnls[0:3], ddof=1)) * np.sqrt(252)
        assert rs.iloc[2] == pytest.approx(expected)

        # last row (index 4): window covers pnls[2:5]
        expected_last = (np.mean(pnls[2:5]) / np.std(pnls[2:5], ddof=1)) * np.sqrt(252)
        assert rs.iloc[4] == pytest.approx(expected_last)

    def test_default_window_is_five(self):
        pnls   = [10.0, -5.0, 20.0, 8.0, -3.0, 15.0]
        result = make_result({"BANKNIFTY": pnls})
        rs = rolling_sharpe(result)["BANKNIFTY"]  # no window arg -> default 5
        # row index 3 (4th row) doesn't have a full 5-day window yet -> NaN
        assert np.isnan(rs.iloc[3])
        # row index 4 (5th row) is the first with a full 5-day window
        expected = (np.mean(pnls[0:5]) / np.std(pnls[0:5], ddof=1)) * np.sqrt(252)
        assert rs.iloc[4] == pytest.approx(expected)

    def test_index_matches_daily_pnl_index(self):
        result = make_result({"BANKNIFTY": [10.0, -5.0, 20.0, 8.0, -3.0]})
        rs = rolling_sharpe(result, window=3)
        assert list(rs.index) == list(result.daily_pnl.index)

    def test_two_underliers_independent(self):
        result = make_result({
            "BANKNIFTY": [10.0, -5.0, 20.0, 8.0, -3.0],
            "NIFTY":     [1.0, 1.0, 1.0, 1.0, 1.0],   # zero variance after window fills
        })
        rs = rolling_sharpe(result, window=3)
        # NIFTY has zero std within any full window -> Sharpe is NaN/inf, not
        # a crash, and must not be contaminated by BANKNIFTY's column.
        assert rs["BANKNIFTY"].iloc[2:].notna().all()


# ===========================================================================
# 6. sortino_ratio
# ===========================================================================

class TestSortinoRatio:

    def test_formula_uses_only_negative_days(self):
        """
        pnls = [100, -50, 200, -25]
        downside_std = std([-50, -25]) = sqrt(312.5) ≈ 17.678
        sortino = (56.25 / 17.678) * sqrt(252)
        """
        pnls   = [100.0, -50.0, 200.0, -25.0]
        result = make_result({"BANKNIFTY": pnls})
        mean   = np.mean(pnls)
        neg    = [p for p in pnls if p < 0]
        dstd   = np.std(neg, ddof=1)
        expected = (mean / dstd) * np.sqrt(252)
        assert sortino_ratio(result)["BANKNIFTY"] == pytest.approx(expected)

    def test_sortino_higher_than_sharpe_with_positive_outlier(self):
        """
        A large positive outlier inflates total_std but not downside_std.
        Sortino denominator < Sharpe denominator → sortino > sharpe.
        """
        pnls = [500.0, -10.0, -20.0, -15.0, -12.0]
        result = make_result({"BANKNIFTY": pnls})
        assert sortino_ratio(result)["BANKNIFTY"] > sharpe_ratio(result)["BANKNIFTY"]

    def test_negative_mean_gives_negative_sortino(self):
        result = make_result({"BANKNIFTY": [-10.0, -20.0, -5.0]})
        assert sortino_ratio(result)["BANKNIFTY"] < 0


# ===========================================================================
# 7. calmar_ratio
# ===========================================================================

class TestCalmarRatio:

    def test_formula(self):
        """
        pnls = [100, -50, 150]
        cum = [100, 50, 200], peak = [100, 100, 200]
        MDD = max([0, 50, 0]) = 50
        calmar = mean([100,-50,150]) * 252 / 50
        """
        pnls   = [100.0, -50.0, 150.0]
        result = make_result({"BANKNIFTY": pnls})
        cum    = np.cumsum(pnls)
        peak   = np.maximum.accumulate(cum)
        mdd    = max(peak - cum)
        expected = np.mean(pnls) * 252 / mdd
        assert calmar_ratio(result)["BANKNIFTY"] == pytest.approx(expected)

    def test_zero_drawdown_returns_nan_or_inf(self):
        """All-positive PnL → MDD=0 → Calmar is undefined."""
        result = make_result({"BANKNIFTY": [100.0, 200.0, 300.0]})
        calmar = calmar_ratio(result)["BANKNIFTY"]
        assert np.isnan(calmar) or np.isinf(calmar)

    def test_negative_mean_negative_calmar(self):
        result = make_result({"BANKNIFTY": [-100.0, -200.0]})
        assert calmar_ratio(result)["BANKNIFTY"] < 0


# ===========================================================================
# 8. max_drawdown
# ===========================================================================

class TestMaxDrawdown:

    def test_all_positive_zero_drawdown(self):
        """Monotonically increasing equity → MDD = 0."""
        result = make_result({"BANKNIFTY": [100.0, 200.0, 300.0]})
        assert max_drawdown(result)["BANKNIFTY"] == pytest.approx(0.0)

    def test_all_negative_mdd(self):
        """
        pnls = [-100, -100, -100]
        cum = [-100, -200, -300]
        cummax = [-100, -100, -100]  ← only tracks from first element, not from 0
        (peak - cum) = [0, 100, 200] → MDD = 200
        Note: implementation does not include the implicit starting 0 in cummax.
        """
        result = make_result({"BANKNIFTY": [-100.0, -100.0, -100.0]})
        assert max_drawdown(result)["BANKNIFTY"] == pytest.approx(200.0)

    def test_peak_then_trough(self):
        """
        pnls = [100, -50]
        cum = [100, 50], peak = [100, 100]
        dd = [0, 50] → MDD = 50
        """
        result = make_result({"BANKNIFTY": [100.0, -50.0]})
        assert max_drawdown(result)["BANKNIFTY"] == pytest.approx(50.0)

    def test_drawdown_then_recovery(self):
        """
        pnls = [100, -50, 200]
        cum = [100, 50, 250], peak = [100, 100, 250]
        dd = [0, 50, 0] → MDD = 50
        """
        result = make_result({"BANKNIFTY": [100.0, -50.0, 200.0]})
        assert max_drawdown(result)["BANKNIFTY"] == pytest.approx(50.0)

    def test_two_drawdowns_takes_larger(self):
        """
        pnls = [100, -30, 100, -80, 100]
        cum  = [100, 70, 170, 90, 190]
        peak = [100, 100, 170, 170, 190]
        dd   = [0, 30, 0, 80, 0] → MDD = 80
        """
        result = make_result({"BANKNIFTY": [100.0, -30.0, 100.0, -80.0, 100.0]})
        assert max_drawdown(result)["BANKNIFTY"] == pytest.approx(80.0)

    def test_zero_pnl_zero_drawdown(self):
        result = make_result({"BANKNIFTY": [0.0, 0.0, 0.0]})
        assert max_drawdown(result)["BANKNIFTY"] == pytest.approx(0.0)

    def test_returns_series_with_underlier_index(self):
        result = make_result({"BANKNIFTY": [100.0, -50.0], "NIFTY": [50.0, -20.0]})
        mdd = max_drawdown(result)
        assert "BANKNIFTY" in mdd.index
        assert "NIFTY" in mdd.index

    def test_two_underliers_computed_independently(self):
        """Each underlier's MDD uses its own PnL series."""
        result = make_result({
            "BANKNIFTY": [100.0, -80.0],   # MDD = 80
            "NIFTY":     [100.0, -20.0],   # MDD = 20
        })
        assert max_drawdown(result)["BANKNIFTY"] == pytest.approx(80.0)
        assert max_drawdown(result)["NIFTY"]     == pytest.approx(20.0)


# ===========================================================================
# 9. max_drawdown_duration
# ===========================================================================

class TestMaxDrawdownDuration:

    def test_all_positive_zero_duration(self):
        """Never below peak → 0 days in drawdown."""
        result = make_result({"BANKNIFTY": [100.0, 200.0, 300.0]})
        assert max_drawdown_duration(result)["BANKNIFTY"] == 0

    def test_two_consecutive_down_days(self):
        """
        pnls = [100, -10, -10, 100]
        cum  = [100, 90, 80, 180]
        peak = [100, 100, 100, 180]
        in_dd = [F, T, T, F] → max consecutive = 2
        """
        result = make_result({"BANKNIFTY": [100.0, -10.0, -10.0, 100.0]})
        assert max_drawdown_duration(result)["BANKNIFTY"] == 2

    def test_never_recovers_full_duration(self):
        """
        pnls = [100, -10, -10, -10]
        in_dd = [F, T, T, T] → max consecutive = 3
        """
        result = make_result({"BANKNIFTY": [100.0, -10.0, -10.0, -10.0]})
        assert max_drawdown_duration(result)["BANKNIFTY"] == 3

    def test_two_drawdown_periods_takes_longer(self):
        """
        pnls = [100, -10, 50, -10, -10, -10, 200]
        in_dd = [F, T, F, T, T, T, F] → durations [1, 3] → max = 3
        """
        result = make_result({"BANKNIFTY": [100.0, -10.0, 50.0, -10.0, -10.0, -10.0, 200.0]})
        assert max_drawdown_duration(result)["BANKNIFTY"] == 3

    def test_returns_series_with_int_values(self):
        result = make_result({"BANKNIFTY": [100.0, -10.0]})
        dur = max_drawdown_duration(result)
        assert isinstance(dur["BANKNIFTY"], (int, np.integer))


# ===========================================================================
# 10. win_rate
# ===========================================================================

class TestWinRate:

    def test_all_positive(self):
        result = make_result({"BANKNIFTY": [10.0, 20.0, 30.0]})
        assert win_rate(result)["BANKNIFTY"] == pytest.approx(1.0)

    def test_all_negative(self):
        result = make_result({"BANKNIFTY": [-10.0, -20.0, -30.0]})
        assert win_rate(result)["BANKNIFTY"] == pytest.approx(0.0)

    def test_exactly_half(self):
        result = make_result({"BANKNIFTY": [10.0, -10.0, 10.0, -10.0]})
        assert win_rate(result)["BANKNIFTY"] == pytest.approx(0.5)

    def test_zero_pnl_not_a_win(self):
        """A breakeven day (PnL = 0) must not count as a win."""
        result = make_result({"BANKNIFTY": [10.0, 0.0, 0.0, 0.0]})
        assert win_rate(result)["BANKNIFTY"] == pytest.approx(0.25)

    def test_three_of_five(self):
        result = make_result({"BANKNIFTY": [10.0, -5.0, 20.0, -3.0, 15.0]})
        assert win_rate(result)["BANKNIFTY"] == pytest.approx(3.0 / 5.0)

    def test_single_positive_day(self):
        result = make_result({"BANKNIFTY": [100.0]})
        assert win_rate(result)["BANKNIFTY"] == pytest.approx(1.0)

    def test_single_negative_day(self):
        result = make_result({"BANKNIFTY": [-100.0]})
        assert win_rate(result)["BANKNIFTY"] == pytest.approx(0.0)


# ===========================================================================
# 11. profit_factor
# ===========================================================================

class TestProfitFactor:

    def test_all_gains_returns_inf(self):
        result = make_result({"BANKNIFTY": [10.0, 20.0, 30.0]})
        assert profit_factor(result)["BANKNIFTY"] == float("inf")

    def test_all_losses_returns_zero(self):
        result = make_result({"BANKNIFTY": [-10.0, -20.0]})
        assert profit_factor(result)["BANKNIFTY"] == pytest.approx(0.0)

    def test_exact_two_to_one(self):
        """gains = 100+200 = 300, losses = |-50|+|-100| = 150 → PF = 2.0"""
        result = make_result({"BANKNIFTY": [100.0, -50.0, 200.0, -100.0]})
        assert profit_factor(result)["BANKNIFTY"] == pytest.approx(2.0)

    def test_pf_above_one_is_profitable(self):
        result = make_result({"BANKNIFTY": [100.0, -40.0]})
        assert profit_factor(result)["BANKNIFTY"] > 1.0

    def test_pf_below_one_is_losing(self):
        result = make_result({"BANKNIFTY": [20.0, -100.0]})
        assert profit_factor(result)["BANKNIFTY"] < 1.0

    def test_pf_equal_one_is_breakeven(self):
        """gains = 50, losses = 50 → PF = 1.0"""
        result = make_result({"BANKNIFTY": [50.0, -50.0]})
        assert profit_factor(result)["BANKNIFTY"] == pytest.approx(1.0)

    def test_zero_days_not_counted(self):
        """Zero days are neither gains nor losses."""
        result = make_result({"BANKNIFTY": [100.0, 0.0, -50.0]})
        assert profit_factor(result)["BANKNIFTY"] == pytest.approx(2.0)


# ===========================================================================
# 12. avg_holding_time
# ===========================================================================

class TestAvgHoldingTime:

    def test_no_trades_returns_empty_series(self):
        result = make_result({"BANKNIFTY": [0.0]})
        assert avg_holding_time(result).empty

    def test_single_round_trip_45_seconds(self):
        trades = [
            Trade(pd.Timestamp("2022-11-01 09:15:00"), CE, "BUY",  100.0, 1),
            Trade(pd.Timestamp("2022-11-01 09:15:45"), CE, "SELL", 110.0, 1),
        ]
        dr     = make_day_result(date(2022, 11, 1), "BANKNIFTY", 10.0, trades=trades)
        result = BacktestResult([dr])
        assert avg_holding_time(result)["BANKNIFTY"] == pytest.approx(45.0)

    def test_two_round_trips_average(self):
        """Trips of 30s and 90s → avg = 60s."""
        trades = [
            Trade(pd.Timestamp("2022-11-01 09:15:00"), CE, "BUY",  100.0, 1),
            Trade(pd.Timestamp("2022-11-01 09:15:30"), CE, "SELL", 110.0, 1),
            Trade(pd.Timestamp("2022-11-01 09:16:00"), CE, "BUY",  110.0, 1),
            Trade(pd.Timestamp("2022-11-01 09:17:30"), CE, "SELL", 120.0, 1),
        ]
        dr     = make_day_result(date(2022, 11, 1), "BANKNIFTY", 20.0, trades=trades)
        result = BacktestResult([dr])
        assert avg_holding_time(result)["BANKNIFTY"] == pytest.approx(60.0)

    def test_across_two_days_averaged_together(self):
        """Day1: 30s, Day2: 90s → avg = 60s across the whole month."""
        t = pd.Timestamp
        trades1 = [
            Trade(t("2022-11-01 09:15:00"), CE, "BUY",  100.0, 1),
            Trade(t("2022-11-01 09:15:30"), CE, "SELL", 110.0, 1),
        ]
        trades2 = [
            Trade(t("2022-11-02 09:15:00"), CE, "BUY",  100.0, 1),
            Trade(t("2022-11-02 09:16:30"), CE, "SELL", 110.0, 1),
        ]
        dr1 = make_day_result(date(2022, 11, 1), "BANKNIFTY", 10.0, trades=trades1)
        dr2 = make_day_result(date(2022, 11, 2), "BANKNIFTY", 10.0, trades=trades2)
        result = BacktestResult([dr1, dr2])
        assert avg_holding_time(result)["BANKNIFTY"] == pytest.approx(60.0)


# ===========================================================================
# 13. rolls_per_day
# ===========================================================================
# NOTE: tests below assume the updated implementation with time-gap checking:
#   is_roll = (strike changed) AND (time_gap == 1 second)
# A re-entry after going flat (gap > 1 second) must NOT count as a roll.

class TestRollsPerDay:

    def test_empty_holdings_log_zero_rolls(self):
        dr     = make_day_result(date(2022, 11, 1), "BANKNIFTY", 0.0)
        result = BacktestResult([dr])
        assert rolls_per_day(result).iloc[0]["rolls"] == 0

    def test_single_row_zero_rolls(self):
        log    = make_holdings_log([pd.Timestamp("2022-11-01 09:15:00")], [41600])
        dr     = make_day_result(date(2022, 11, 1), "BANKNIFTY", 0.0, holdings_log=log)
        result = BacktestResult([dr])
        assert rolls_per_day(result).iloc[0]["rolls"] == 0

    def test_two_consecutive_same_strike_zero_rolls(self):
        t   = pd.Timestamp("2022-11-01 09:15:00")
        log = make_holdings_log([t, t + pd.Timedelta(seconds=1)], [41600, 41600])
        dr  = make_day_result(date(2022, 11, 1), "BANKNIFTY", 0.0, holdings_log=log)
        assert rolls_per_day(BacktestResult([dr])).iloc[0]["rolls"] == 0

    def test_two_consecutive_different_strike_one_roll(self):
        t   = pd.Timestamp("2022-11-01 09:15:00")
        log = make_holdings_log([t, t + pd.Timedelta(seconds=1)], [41600, 41700])
        dr  = make_day_result(date(2022, 11, 1), "BANKNIFTY", 0.0, holdings_log=log)
        assert rolls_per_day(BacktestResult([dr])).iloc[0]["rolls"] == 1

    def test_gap_more_than_one_second_not_a_roll(self):
        """Position went flat then re-entered at a new strike — not a roll."""
        t   = pd.Timestamp("2022-11-01 09:15:00")
        log = make_holdings_log([t, t + pd.Timedelta(seconds=2)], [41600, 41700])
        dr  = make_day_result(date(2022, 11, 1), "BANKNIFTY", 0.0, holdings_log=log)
        assert rolls_per_day(BacktestResult([dr])).iloc[0]["rolls"] == 0

    def test_large_gap_not_a_roll(self):
        """30-second gap → definitely a re-entry."""
        t   = pd.Timestamp("2022-11-01 09:15:00")
        log = make_holdings_log([t, t + pd.Timedelta(seconds=30)], [41600, 41700])
        dr  = make_day_result(date(2022, 11, 1), "BANKNIFTY", 0.0, holdings_log=log)
        assert rolls_per_day(BacktestResult([dr])).iloc[0]["rolls"] == 0

    def test_four_consecutive_alternating_three_rolls(self):
        t          = pd.Timestamp("2022-11-01 09:15:00")
        timestamps = [t + pd.Timedelta(seconds=i) for i in range(4)]
        strikes    = [41600, 41700, 41600, 41700]
        log = make_holdings_log(timestamps, strikes)
        dr  = make_day_result(date(2022, 11, 1), "BANKNIFTY", 0.0, holdings_log=log)
        assert rolls_per_day(BacktestResult([dr])).iloc[0]["rolls"] == 3

    def test_two_rolls_then_reentry_still_two_rolls(self):
        """Continuous: 2 rolls. Then flat for 30s. Re-entry: not a roll."""
        t = pd.Timestamp("2022-11-01 09:15:00")
        timestamps = [
            t,
            t + pd.Timedelta(seconds=1),   # roll 1
            t + pd.Timedelta(seconds=2),   # roll 2
            t + pd.Timedelta(seconds=32),  # re-entry (gap = 30s)
        ]
        strikes = [41600, 41700, 41600, 41800]
        log = make_holdings_log(timestamps, strikes)
        dr  = make_day_result(date(2022, 11, 1), "BANKNIFTY", 0.0, holdings_log=log)
        assert rolls_per_day(BacktestResult([dr])).iloc[0]["rolls"] == 2

    def test_returns_correct_columns(self):
        dr     = make_day_result(date(2022, 11, 1), "BANKNIFTY", 0.0)
        result = BacktestResult([dr])
        df     = rolls_per_day(result)
        assert set(df.columns) == {"date", "underlier", "trades", "rolls"}

    def test_trades_column_matches_trade_count(self):
        trades = [
            Trade(pd.Timestamp("2022-11-01 09:15:00"), CE, "BUY",  100.0, 1),
            Trade(pd.Timestamp("2022-11-01 09:16:00"), CE, "SELL", 110.0, 1),
        ]
        dr     = make_day_result(date(2022, 11, 1), "BANKNIFTY", 10.0, trades=trades)
        result = BacktestResult([dr])
        assert rolls_per_day(result).iloc[0]["trades"] == 2

    def test_multiple_days(self):
        t = pd.Timestamp("2022-11-01 09:15:00")
        # Day 1: 1 roll
        log1 = make_holdings_log([t, t + pd.Timedelta(seconds=1)], [41600, 41700])
        dr1  = make_day_result(date(2022, 11, 1), "BANKNIFTY", 0.0, holdings_log=log1)
        # Day 2: 0 rolls
        dr2  = make_day_result(date(2022, 11, 2), "BANKNIFTY", 0.0)
        result = BacktestResult([dr1, dr2])
        df     = rolls_per_day(result)
        assert df[df["date"] == date(2022, 11, 1)].iloc[0]["rolls"] == 1
        assert df[df["date"] == date(2022, 11, 2)].iloc[0]["rolls"] == 0


# ===========================================================================
# 14. cost_sensitivity
# ===========================================================================

class TestCostSensitivity:

    def _result_with_trades(self, prices: list) -> BacktestResult:
        trades = [
            Trade(pd.Timestamp(f"2022-11-01 09:{15+i}:00"), CE,
                  "BUY" if i % 2 == 0 else "SELL", p, 1)
            for i, p in enumerate(prices)
        ]
        dr = make_day_result(date(2022, 11, 1), "BANKNIFTY", 10.0, trades=trades)
        return BacktestResult([dr])

    def test_zero_bps_net_equals_gross(self):
        result = self._result_with_trades([100.0, 110.0])
        df     = cost_sensitivity(result, bps_range=[0])
        assert df.iloc[0]["net_pnl"] == pytest.approx(df.iloc[0]["gross_pnl"])

    def test_positive_bps_reduces_net(self):
        result = self._result_with_trades([100.0, 110.0])
        df     = cost_sensitivity(result, bps_range=[0, 10])
        assert df.iloc[1]["net_pnl"] < df.iloc[0]["net_pnl"]

    def test_total_notional_is_sum_of_price_times_qty(self):
        """Two trades at 100 and 200 → notional = 300."""
        result = self._result_with_trades([100.0, 200.0])
        df     = cost_sensitivity(result, bps_range=[0])
        assert df.iloc[0]["total_notional"] == pytest.approx(300.0)

    def test_cost_drag_formula(self):
        """drag = total_notional * bps / 10_000"""
        result = self._result_with_trades([1000.0])  # 1 trade, notional = 1000
        df     = cost_sensitivity(result, bps_range=[5])
        assert df.iloc[0]["cost_drag"] == pytest.approx(1000.0 * 5 / 10_000)

    def test_net_pnl_formula(self):
        result = self._result_with_trades([500.0, 600.0])  # notional = 1100
        df     = cost_sensitivity(result, bps_range=[10])
        drag   = 1100.0 * 10 / 10_000
        assert df.iloc[0]["net_pnl"] == pytest.approx(df.iloc[0]["gross_pnl"] - drag, abs=0.01)

    def test_default_bps_range(self):
        result = self._result_with_trades([100.0])
        df     = cost_sensitivity(result)
        assert list(df["cost_bps"]) == [0, 1, 2, 5, 10]

    def test_custom_bps_range(self):
        result = self._result_with_trades([100.0])
        df     = cost_sensitivity(result, bps_range=[0, 1, 2, 3])
        assert list(df["cost_bps"]) == [0, 1, 2, 3]
        assert len(df) == 4

    def test_net_pnl_monotonically_decreasing_with_cost(self):
        result   = self._result_with_trades([100.0, 200.0, 300.0])
        df       = cost_sensitivity(result, bps_range=[0, 1, 2, 5, 10])
        net_pnls = df["net_pnl"].tolist()
        for i in range(1, len(net_pnls)):
            assert net_pnls[i] <= net_pnls[i-1]

    def test_expected_columns(self):
        result = self._result_with_trades([100.0])
        df     = cost_sensitivity(result)
        assert set(df.columns) == {"cost_bps", "total_notional", "cost_drag", "gross_pnl", "net_pnl"}

    def test_gross_pnl_is_constant_across_rows(self):
        """Gross PnL is fixed — only net_pnl changes as bps increases."""
        result = self._result_with_trades([100.0, 110.0])
        df     = cost_sensitivity(result, bps_range=[0, 5, 10])
        assert df["gross_pnl"].nunique() == 1


# ===========================================================================
# 15. summary_table
# ===========================================================================

class TestSummaryTable:

    def setup_method(self):
        self.pnls   = [100.0, -50.0, 200.0, -25.0, 80.0]
        self.result = make_result({"BANKNIFTY": self.pnls})
        self.table  = summary_table(self.result)

    def test_index_name_is_underlier(self):
        assert self.table.index.name == "underlier"

    def test_banknifty_in_index(self):
        assert "BANKNIFTY" in self.table.index

    def test_expected_columns_present(self):
        expected = {
            "total_pnl", "mean_daily_pnl", "std_daily_pnl",
            "best_day", "worst_day", "sharpe", "sortino", "calmar",
            "max_drawdown", "max_dd_duration", "win_rate",
            "profit_factor", "avg_holding_secs", "total_trades",
        }
        assert expected.issubset(set(self.table.columns))

    def test_total_pnl_matches(self):
        assert self.table.loc["BANKNIFTY", "total_pnl"] == pytest.approx(sum(self.pnls), rel=1e-4)

    def test_mean_daily_pnl_matches(self):
        assert self.table.loc["BANKNIFTY", "mean_daily_pnl"] == pytest.approx(np.mean(self.pnls), rel=1e-4)

    def test_best_day_matches(self):
        assert self.table.loc["BANKNIFTY", "best_day"] == pytest.approx(max(self.pnls))

    def test_worst_day_matches(self):
        assert self.table.loc["BANKNIFTY", "worst_day"] == pytest.approx(min(self.pnls))

    def test_win_rate_is_string_with_percent(self):
        wr = self.table.loc["BANKNIFTY", "win_rate"]
        assert isinstance(wr, str)
        assert "%" in wr

    def test_win_rate_value_correct(self):
        # 3 positive out of 5 days → 60.0%
        assert self.table.loc["BANKNIFTY", "win_rate"] == "60.0%"

    def test_sharpe_matches_scalar_function(self):
        expected = sharpe_ratio(self.result)["BANKNIFTY"]
        assert self.table.loc["BANKNIFTY", "sharpe"] == pytest.approx(expected, rel=1e-3)

    def test_max_drawdown_matches_scalar_function(self):
        expected = max_drawdown(self.result)["BANKNIFTY"]
        assert self.table.loc["BANKNIFTY", "max_drawdown"] == pytest.approx(expected)

    def test_two_underliers_two_rows(self):
        result = make_result({"BANKNIFTY": [100.0, -50.0], "NIFTY": [20.0, -10.0]})
        table  = summary_table(result)
        assert len(table) == 2
        assert "BANKNIFTY" in table.index
        assert "NIFTY" in table.index

    def test_no_trades_avg_holding_zero(self):
        """No trades in the result → avg_holding_secs should be 0.0."""
        assert self.table.loc["BANKNIFTY", "avg_holding_secs"] == pytest.approx(0.0)


# ===========================================================================
# 16. Plot functions
# ===========================================================================

class TestPlots:
    """
    Verify plot functions return Figure objects and don't raise.
    We test structural correctness — not pixel output.
    """

    def setup_method(self):
        self.result = make_result({
            "BANKNIFTY": [100.0, -50.0, 200.0, -25.0, 80.0],
            "NIFTY":     [50.0,  -30.0, 100.0, -10.0, 60.0],
        })

    def teardown_method(self):
        plt.close("all")

    def test_plot_cumulative_pnl_returns_figure(self):
        assert isinstance(plot_cumulative_pnl(self.result), Figure)

    def test_plot_daily_pnl_returns_figure(self):
        assert isinstance(plot_daily_pnl(self.result), Figure)

    def test_plot_drawdown_returns_figure(self):
        assert isinstance(plot_drawdown(self.result), Figure)

    def test_plot_rolls_per_day_returns_figure(self):
        assert isinstance(plot_rolls_per_day(self.result), Figure)

    def test_plot_cost_sensitivity_returns_figure(self):
        assert isinstance(plot_cost_sensitivity(self.result), Figure)

    def test_plot_rolling_sharpe_returns_figure(self):
        assert isinstance(plot_rolling_sharpe(self.result), Figure)

    def test_plot_intraday_mtm_returns_figure(self):
        fig = plot_intraday_mtm(self.result, "2022-11-01", "BANKNIFTY")
        assert isinstance(fig, Figure)

    def test_plot_intraday_mtm_wrong_date_raises(self):
        with pytest.raises(ValueError):
            plot_intraday_mtm(self.result, "2099-01-01", "BANKNIFTY")

    def test_plot_intraday_mtm_wrong_underlier_raises(self):
        with pytest.raises(ValueError):
            plot_intraday_mtm(self.result, "2022-11-01", "FAKENDEX")

    def test_plot_holdings_returns_figure(self):
        t   = pd.Timestamp("2022-11-01 09:15:00")
        log = make_holdings_log(
            [t + pd.Timedelta(seconds=i) for i in range(5)],
            [41600, 41600, 41700, 41700, 41600],
        )
        dr     = make_day_result(date(2022, 11, 1), "BANKNIFTY", 0.0, holdings_log=log)
        result = BacktestResult([dr])
        assert isinstance(plot_holdings(result, "2022-11-01", "BANKNIFTY"), Figure)

    def test_plot_holdings_empty_log_raises(self):
        dr     = make_day_result(date(2022, 11, 1), "BANKNIFTY", 0.0)
        result = BacktestResult([dr])
        with pytest.raises(ValueError):
            plot_holdings(result, "2022-11-01", "BANKNIFTY")

    def test_plot_holdings_wrong_date_raises(self):
        t      = pd.Timestamp("2022-11-01 09:15:00")
        log    = make_holdings_log([t], [41600])
        dr     = make_day_result(date(2022, 11, 1), "BANKNIFTY", 0.0, holdings_log=log)
        result = BacktestResult([dr])
        with pytest.raises(ValueError):
            plot_holdings(result, "2099-01-01", "BANKNIFTY")

    def test_plot_holdings_wrong_underlier_raises(self):
        t      = pd.Timestamp("2022-11-01 09:15:00")
        log    = make_holdings_log([t], [41600])
        dr     = make_day_result(date(2022, 11, 1), "BANKNIFTY", 0.0, holdings_log=log)
        result = BacktestResult([dr])
        with pytest.raises(ValueError):
            plot_holdings(result, "2022-11-01", "FAKENDEX")

    def test_each_plot_produces_fresh_figure(self):
        """Two calls must not return the same Figure object."""
        fig1 = plot_cumulative_pnl(self.result)
        fig2 = plot_cumulative_pnl(self.result)
        assert fig1 is not fig2