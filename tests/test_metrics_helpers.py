# tests/test_metrics_helpers.py
#
# Coverage for the "Streamlit helper" metrics (round_trips_by_leg,
# morning_afternoon_split, pnl_by_weekday, expiry_day_pnl, avg_intraday_mtm,
# notional_per_day, breakeven_bps). These back the dashboard and the PDF report
# but previously had no direct unit tests.

import numpy as np
import pandas as pd
import pytest
from datetime import date

from stratbt.instruments import Instrument
from stratbt.position import Trade
from stratbt.engine import BacktestResult, DayResult
from stratbt.metrics import (
    round_trips_by_leg,
    morning_afternoon_split,
    pnl_by_weekday,
    expiry_day_pnl,
    avg_intraday_mtm,
    notional_per_day,
    breakeven_bps,
)


FULL_GRID = pd.date_range("2022-11-01 09:15:00", periods=22_500, freq="1s")
NOON_POS  = 9_900  # 09:15:00 + 9900s == 12:00:00


def day_with_mtm(trading_date, underlier, mtm, final_pnl, trades=None):
    return DayResult(
        date=trading_date, underlier=underlier, trades=trades or [],
        mtm=mtm, final_pnl=final_pnl, holdings_log=pd.DataFrame(),
    )


def ramp_day(trading_date, underlier, noon_val, close_val, trades=None):
    """Full-session day whose MTM is 0 at open, noon_val at 12:00, close_val at close."""
    vals = np.zeros(22_500)
    vals[NOON_POS] = noon_val
    vals[-1]       = close_val
    idx = pd.date_range(f"{trading_date} 09:15:00", periods=22_500, freq="1s")
    return day_with_mtm(trading_date, underlier, pd.Series(vals, index=idx), close_val, trades)


# ===========================================================================
# round_trips_by_leg
# ===========================================================================

class TestRoundTripsByLeg:

    def _trades_df(self, instrument):
        return pd.DataFrame([
            {"date": date(2022, 11, 1), "underlier": "NIFTY", "instrument": instrument,
             "side": "BUY", "timestamp": pd.Timestamp("2022-11-01 09:15:00"), "price": 100.0, "qty": 1},
            {"date": date(2022, 11, 1), "underlier": "NIFTY", "instrument": instrument,
             "side": "SELL", "timestamp": pd.Timestamp("2022-11-01 09:16:00"), "price": 120.0, "qty": 1},
        ])

    def test_extracts_ce_leg(self):
        rt = round_trips_by_leg(self._trades_df("NIFTY22110318000CE.csv"))
        assert rt["leg"].iloc[0] == "CE"

    def test_extracts_pe_leg(self):
        rt = round_trips_by_leg(self._trades_df("NIFTY22110318000PE.csv"))
        assert rt["leg"].iloc[0] == "PE"

    def test_empty_trades_returns_empty(self):
        empty = pd.DataFrame(columns=["date", "underlier", "instrument", "side",
                                      "timestamp", "price", "qty"])
        assert round_trips_by_leg(empty).empty


# ===========================================================================
# morning_afternoon_split
# ===========================================================================

class TestMorningAfternoonSplit:

    def test_split_at_exact_noon(self):
        # open 0, noon 100, close 150 -> morning 100, afternoon 50.
        dr = ramp_day(date(2022, 11, 1), "NIFTY", noon_val=100.0, close_val=150.0)
        out = morning_afternoon_split([dr]).iloc[0]
        assert out["morning_pnl"] == pytest.approx(100.0)
        assert out["afternoon_pnl"] == pytest.approx(50.0)

    def test_halves_sum_to_final_pnl(self):
        dr = ramp_day(date(2022, 11, 1), "NIFTY", noon_val=42.0, close_val=170.0)
        out = morning_afternoon_split([dr]).iloc[0]
        assert out["morning_pnl"] + out["afternoon_pnl"] == pytest.approx(170.0)

    def test_noon_not_in_index_uses_prior_tick(self):
        # Sparse index without exactly 12:00:00 -> falls back to the last tick before noon.
        idx  = pd.to_datetime(["2022-11-01 09:15:00", "2022-11-01 11:59:58", "2022-11-01 15:29:59"])
        mtm  = pd.Series([0.0, 80.0, 120.0], index=idx)
        dr   = day_with_mtm(date(2022, 11, 1), "NIFTY", mtm, 120.0)
        out  = morning_afternoon_split([dr]).iloc[0]
        assert out["morning_pnl"] == pytest.approx(80.0)    # prior-tick value 80 - open 0
        assert out["afternoon_pnl"] == pytest.approx(40.0)  # close 120 - 80

    def test_one_row_per_day(self):
        days = [ramp_day(date(2022, 11, 1), "NIFTY", 10.0, 20.0),
                ramp_day(date(2022, 11, 2), "NIFTY", 5.0, 8.0)]
        assert len(morning_afternoon_split(days)) == 2


# ===========================================================================
# pnl_by_weekday
# ===========================================================================

class TestPnlByWeekday:

    def _result(self):
        # Nov 2022 business days: 1=Tue, 2=Wed, 3=Thu, 4=Fri, 7=Mon, 8=Tue
        dates = [date(2022, 11, d) for d in (1, 2, 3, 4, 7, 8)]
        pnls  = [10.0, 20.0, 30.0, 40.0, 50.0, 30.0]  # two Tuesdays: 10 and 30
        drs = []
        for d, p in zip(dates, pnls):
            ts  = pd.Timestamp(f"{d} 09:15:00")
            mtm = pd.Series([0.0, p], index=pd.date_range(ts, periods=2, freq="1s"))
            drs.append(day_with_mtm(d, "NIFTY", mtm, p))
        return BacktestResult(drs)

    def test_tuesday_averaged(self):
        out = pnl_by_weekday(self._result())
        tue = out[(out["weekday"] == "Tuesday") & (out["underlier"] == "NIFTY")]
        assert tue["avg_pnl"].iloc[0] == pytest.approx(20.0)  # mean(10, 30)
        assert tue["count"].iloc[0] == 2

    def test_single_occurrence_weekday(self):
        out = pnl_by_weekday(self._result())
        wed = out[(out["weekday"] == "Wednesday") & (out["underlier"] == "NIFTY")]
        assert wed["avg_pnl"].iloc[0] == pytest.approx(20.0)
        assert wed["count"].iloc[0] == 1

    def test_weekday_ordering_monday_first(self):
        out = pnl_by_weekday(self._result())
        present = [w for w in out["weekday"].tolist()]
        # Monday must precede Tuesday which must precede Wednesday, etc.
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        ranks = [order.index(w) for w in present]
        assert ranks == sorted(ranks)


# ===========================================================================
# expiry_day_pnl
# ===========================================================================

class TestExpiryDayPnl:

    def _day(self, trading_date, expiry, pnl):
        inst = Instrument("NIFTY", expiry, 18000, "CE")
        trade = Trade(pd.Timestamp(f"{trading_date} 09:15:00"), inst, "BUY", 100.0, 1)
        ts  = pd.Timestamp(f"{trading_date} 09:15:00")
        mtm = pd.Series([0.0, pnl], index=pd.date_range(ts, periods=2, freq="1s"))
        return day_with_mtm(trading_date, "NIFTY", mtm, pnl, trades=[trade])

    def test_classifies_expiry_vs_non_expiry(self):
        # Nov 3 trades the Nov 3 expiry (expiry day); Nov 1 trades the Nov 3 expiry (not).
        r = BacktestResult([
            self._day(date(2022, 11, 3), date(2022, 11, 3), 50.0),
            self._day(date(2022, 11, 1), date(2022, 11, 3), 20.0),
        ])
        out = expiry_day_pnl(r).set_index("day_type")
        assert out.loc["Expiry day", "avg_pnl"] == pytest.approx(50.0)
        assert out.loc["Non-expiry", "avg_pnl"] == pytest.approx(20.0)

    def test_averages_multiple_expiry_days(self):
        r = BacktestResult([
            self._day(date(2022, 11, 3), date(2022, 11, 3), 40.0),
            self._day(date(2022, 11, 10), date(2022, 11, 10), 60.0),
        ])
        out = expiry_day_pnl(r).set_index("day_type")
        assert out.loc["Expiry day", "avg_pnl"] == pytest.approx(50.0)
        assert out.loc["Expiry day", "count"] == 2

    def test_days_without_trades_skipped(self):
        # A day with no trades contributes no row.
        no_trades = day_with_mtm(
            date(2022, 11, 2), "NIFTY",
            pd.Series([0.0, 0.0], index=pd.date_range("2022-11-02 09:15:00", periods=2, freq="1s")),
            0.0,
        )
        r = BacktestResult([self._day(date(2022, 11, 3), date(2022, 11, 3), 50.0), no_trades])
        out = expiry_day_pnl(r)
        assert out["count"].sum() == 1

    def test_no_day_has_trades_returns_empty_not_a_crash(self):
        # Regression test: when EVERY day is trade-less, the intermediate
        # `rows` list is empty, so pd.DataFrame(rows) has no columns at all —
        # groupby(["underlier", "day_type"]) used to raise KeyError instead
        # of returning an empty result. Zero-trade days happen for real when
        # a strategy is configured with a window that never triggers an entry.
        no_trades = day_with_mtm(
            date(2022, 11, 2), "NIFTY",
            pd.Series([0.0, 0.0], index=pd.date_range("2022-11-02 09:15:00", periods=2, freq="1s")),
            0.0,
        )
        r   = BacktestResult([no_trades])
        out = expiry_day_pnl(r)
        assert out.empty
        assert list(out.columns) == ["underlier", "day_type", "avg_pnl", "count"]


# ===========================================================================
# avg_intraday_mtm
# ===========================================================================

class TestAvgIntradayMtm:

    def test_elementwise_mean_by_time_of_day(self):
        idx = pd.date_range("2022-11-01 09:15:00", periods=2, freq="1s")
        d1  = day_with_mtm(date(2022, 11, 1), "NIFTY", pd.Series([0.0, 10.0], index=idx), 10.0)
        idx2 = pd.date_range("2022-11-02 09:15:00", periods=2, freq="1s")
        d2  = day_with_mtm(date(2022, 11, 2), "NIFTY", pd.Series([0.0, 20.0], index=idx2), 20.0)
        out = avg_intraday_mtm(BacktestResult([d1, d2]))
        assert out["NIFTY"].tolist() == pytest.approx([0.0, 15.0])  # mean of (10, 20) = 15

    def test_separate_columns_per_underlier(self):
        idx = pd.date_range("2022-11-01 09:15:00", periods=2, freq="1s")
        dn  = day_with_mtm(date(2022, 11, 1), "NIFTY", pd.Series([0.0, 10.0], index=idx), 10.0)
        db  = day_with_mtm(date(2022, 11, 1), "BANKNIFTY", pd.Series([0.0, 30.0], index=idx), 30.0)
        out = avg_intraday_mtm(BacktestResult([dn, db]))
        assert set(out.columns) == {"NIFTY", "BANKNIFTY"}
        assert out["BANKNIFTY"].iloc[-1] == pytest.approx(30.0)


# ===========================================================================
# notional_per_day
# ===========================================================================

class TestNotionalPerDay:

    def test_sums_price_times_qty(self):
        inst = Instrument("NIFTY", date(2022, 11, 3), 18000, "CE")
        trades = [
            Trade(pd.Timestamp("2022-11-01 09:15:00"), inst, "BUY", 100.0, 2),
            Trade(pd.Timestamp("2022-11-01 09:16:00"), inst, "SELL", 150.0, 2),
        ]
        idx = pd.date_range("2022-11-01 09:15:00", periods=2, freq="1s")
        dr  = day_with_mtm(date(2022, 11, 1), "NIFTY", pd.Series([0.0, 100.0], index=idx), 100.0, trades)
        out = notional_per_day(BacktestResult([dr]))
        # 100*2 + 150*2 = 500
        assert out["notional"].iloc[0] == pytest.approx(500.0)

    def test_grouped_by_day_and_underlier(self):
        inst = Instrument("NIFTY", date(2022, 11, 3), 18000, "CE")
        def d(day, price):
            tr  = [Trade(pd.Timestamp(f"{day} 09:15:00"), inst, "BUY", price, 1)]
            idx = pd.date_range(f"{day} 09:15:00", periods=2, freq="1s")
            return day_with_mtm(day, "NIFTY", pd.Series([0.0, 1.0], index=idx), 1.0, tr)
        out = notional_per_day(BacktestResult([d(date(2022, 11, 1), 100.0),
                                               d(date(2022, 11, 2), 200.0)]))
        assert len(out) == 2
        assert set(out["notional"]) == {100.0, 200.0}


# ===========================================================================
# breakeven_bps
# ===========================================================================

class TestBreakevenBps:

    def test_known_value(self):
        # gross = 100, total_notional = 400 -> bps = 100*10000/400 = 2500.
        inst = Instrument("NIFTY", date(2022, 11, 3), 18000, "CE")
        trades = [
            Trade(pd.Timestamp("2022-11-01 09:15:00"), inst, "BUY", 100.0, 2),
            Trade(pd.Timestamp("2022-11-01 09:16:00"), inst, "SELL", 100.0, 2),
        ]
        idx = pd.date_range("2022-11-01 09:15:00", periods=2, freq="1s")
        dr  = day_with_mtm(date(2022, 11, 1), "NIFTY", pd.Series([0.0, 100.0], index=idx), 100.0, trades)
        assert breakeven_bps(BacktestResult([dr])) == pytest.approx(2500.0)

    def test_no_trades_returns_zero(self):
        idx = pd.date_range("2022-11-01 09:15:00", periods=2, freq="1s")
        dr  = day_with_mtm(date(2022, 11, 1), "NIFTY", pd.Series([0.0, 0.0], index=idx), 0.0)
        assert breakeven_bps(BacktestResult([dr])) == 0.0

    def test_negative_gross_negative_bps(self):
        # Loss-making at zero cost -> break-even bps is negative (needs subsidy).
        inst = Instrument("NIFTY", date(2022, 11, 3), 18000, "CE")
        trades = [
            Trade(pd.Timestamp("2022-11-01 09:15:00"), inst, "BUY", 100.0, 1),
            Trade(pd.Timestamp("2022-11-01 09:16:00"), inst, "SELL", 100.0, 1),
        ]
        idx = pd.date_range("2022-11-01 09:15:00", periods=2, freq="1s")
        dr  = day_with_mtm(date(2022, 11, 1), "NIFTY", pd.Series([0.0, -50.0], index=idx), -50.0, trades)
        assert breakeven_bps(BacktestResult([dr])) < 0
