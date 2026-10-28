# tests/test_engine.py

import pytest
import numpy as np
import pandas as pd
from datetime import date
from pathlib import Path

from stratbt.universe import Universe
from stratbt.providers import NSETickProvider
from stratbt.execution import ExecutionModel
from stratbt.engine import BacktestEngine, BacktestResult, DayResult
from stratbt.position import Trade
from stratbt.instruments import Instrument
from stratbt.strategy.base import BaseStrategy, Order
from stratbt.strategy.atm_straddle import ATMStraddleStrategy

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / "data" / "allData"

# ---------------------------------------------------------------------------
# Fake strategies for agnosticism tests
# ---------------------------------------------------------------------------

class DoNothingStrategy(BaseStrategy):
    """Never places any orders. Should produce zero trades and zero PnL."""
    def __init__(self, underlier, expiry, strikes):
        pass
    def on_second(self, ts, futures_price, option_price, holdings):
        return []


class BuyOnceStrategy(BaseStrategy):
    """
    Buys the ATM straddle at the very first second and never rolls.
    Used to verify the engine correctly executes a single entry + forced close.
    """
    def __init__(self, underlier, expiry, strikes):
        self.underlier = underlier
        self.expiry    = expiry
        self.strikes   = strikes
        self._entered  = False

    def on_second(self, ts, futures_price, option_price, holdings):
        from datetime import time
        if ts.time() >= time(15, 29, 59) and holdings:
            return [Order(inst, "SELL") for inst in holdings]
        if self._entered or holdings:
            return []
        idx    = np.searchsorted(self.strikes, futures_price)
        idx    = np.clip(idx, 1, len(self.strikes) - 1)
        strike = int(self.strikes[idx - 1])
        ce = Instrument(self.underlier, self.expiry, strike, "CE")
        pe = Instrument(self.underlier, self.expiry, strike, "PE")
        if option_price(ce) is None or option_price(pe) is None:
            return []
        self._entered = True
        return [Order(ce, "BUY"), Order(pe, "BUY")]


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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Real-data tests pin the original optimistic model (same-second fills, no
# costs) so their numbers stay comparable with the published results.
INSTANT = ExecutionModel.instant()

@pytest.fixture(scope="module")
def universe():
    return Universe(DATA_DIR)

@pytest.fixture(scope="module")
def provider():
    return NSETickProvider(DATA_DIR)

@pytest.fixture(scope="module")
def engine_atm(provider):
    return BacktestEngine(provider, ATMStraddleStrategy, execution=INSTANT)

@pytest.fixture(scope="module")
def engine_nothing(provider):
    return BacktestEngine(provider, DoNothingStrategy, execution=INSTANT)

@pytest.fixture(scope="module")
def engine_buy_once(provider):
    return BacktestEngine(provider, BuyOnceStrategy, execution=INSTANT)

@pytest.fixture(scope="module")
def day_result_banknifty(engine_atm):
    return engine_atm.run_day(date(2022, 11, 1), "BANKNIFTY")

@pytest.fixture(scope="module")
def full_result_banknifty(engine_atm):
    """Full 21-day run. Expensive — runs once per session."""
    return engine_atm.run("BANKNIFTY")


# ===========================================================================
# BacktestResult unit tests (fake data — fast)
# ===========================================================================

class TestBacktestResultEmpty:
    def test_empty_trades_df_has_columns(self):
        result = BacktestResult([])
        assert list(result.trades_df.columns) == [
            "timestamp", "date", "underlier", "instrument",
            "expiry", "strike", "option_type", "side", "price", "qty", "fee"
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


# ===========================================================================
# DayResult integration tests — one real day, BANKNIFTY
# ===========================================================================

class TestDayResultStructure:
    def test_returns_day_result(self, day_result_banknifty):
        assert isinstance(day_result_banknifty, DayResult)

    def test_correct_date(self, day_result_banknifty):
        assert day_result_banknifty.date == date(2022, 11, 1)

    def test_correct_underlier(self, day_result_banknifty):
        assert day_result_banknifty.underlier == "BANKNIFTY"

    def test_mtm_length(self, day_result_banknifty):
        assert len(day_result_banknifty.mtm) == 22_500

    def test_mtm_is_datetimeindex(self, day_result_banknifty):
        assert isinstance(day_result_banknifty.mtm.index, pd.DatetimeIndex)

    def test_mtm_starts_at_session_open(self, day_result_banknifty):
        assert str(day_result_banknifty.mtm.index[0]) == "2022-11-01 09:15:00"

    def test_mtm_ends_at_session_close(self, day_result_banknifty):
        assert str(day_result_banknifty.mtm.index[-1]) == "2022-11-01 15:29:59"

    def test_mtm_has_no_nans(self, day_result_banknifty):
        assert day_result_banknifty.mtm.isna().sum() == 0

    def test_mtm_starts_at_zero(self, day_result_banknifty):
        assert day_result_banknifty.mtm.iloc[0] == pytest.approx(0.0)

    def test_final_pnl_is_float(self, day_result_banknifty):
        assert isinstance(day_result_banknifty.final_pnl, float)

    def test_final_pnl_matches_mtm(self, day_result_banknifty):
        assert day_result_banknifty.mtm.iloc[-1] == pytest.approx(
            day_result_banknifty.final_pnl, rel=1e-6
        )

    def test_holdings_log_not_empty(self, day_result_banknifty):
        assert not day_result_banknifty.holdings_log.empty

    def test_holdings_log_has_strike_column(self, day_result_banknifty):
        assert "strike" in day_result_banknifty.holdings_log.columns

    def test_holdings_log_index_is_datetimeindex(self, day_result_banknifty):
        assert isinstance(day_result_banknifty.holdings_log.index, pd.DatetimeIndex)

    def test_holdings_log_strikes_are_integers(self, day_result_banknifty):
        strikes = day_result_banknifty.holdings_log["strike"].dropna().unique()
        for s in strikes:
            assert isinstance(s, (int, np.integer))


class TestDayResultTrades:
    def test_trades_non_empty(self, day_result_banknifty):
        assert len(day_result_banknifty.trades) > 0

    def test_trade_count_in_expected_range(self, day_result_banknifty):
        n = len(day_result_banknifty.trades)
        assert 100 < n < 5_000

    def test_all_trades_are_buy_or_sell(self, day_result_banknifty):
        sides = {t.side for t in day_result_banknifty.trades}
        assert sides.issubset({"BUY", "SELL"})

    def test_all_trade_prices_positive(self, day_result_banknifty):
        assert all(t.price > 0 for t in day_result_banknifty.trades)

    def test_all_trade_qty_positive(self, day_result_banknifty):
        assert all(t.qty > 0 for t in day_result_banknifty.trades)

    def test_trades_are_chronological(self, day_result_banknifty):
        timestamps = [t.timestamp for t in day_result_banknifty.trades]
        assert timestamps == sorted(timestamps)

    def test_trades_all_within_session(self, day_result_banknifty):
        session_start = pd.Timestamp("2022-11-01 09:15:00")
        session_end   = pd.Timestamp("2022-11-01 15:29:59")
        for t in day_result_banknifty.trades:
            assert session_start <= t.timestamp <= session_end

    def test_all_instruments_are_banknifty(self, day_result_banknifty):
        for t in day_result_banknifty.trades:
            assert t.instrument.underlier == "BANKNIFTY"

    def test_buys_and_sells_balanced(self, day_result_banknifty):
        bought = sum(t.qty for t in day_result_banknifty.trades if t.side == "BUY")
        sold   = sum(t.qty for t in day_result_banknifty.trades if t.side == "SELL")
        assert bought == sold


class TestMTMContinuity:
    def test_no_large_jumps(self, day_result_banknifty):
        diffs = day_result_banknifty.mtm.diff().dropna().abs()
        assert diffs.max() < 500, \
            f"Large MTM jump at {diffs.idxmax()}: {diffs.max():.2f}"

    def test_ends_at_realized(self, day_result_banknifty):
        assert day_result_banknifty.mtm.iloc[-1] == pytest.approx(
            day_result_banknifty.final_pnl, rel=1e-6
        )


# ===========================================================================
# Strategy agnosticism tests
# ===========================================================================

class TestDoNothingStrategy:
    def test_zero_trades(self, engine_nothing):
        result = engine_nothing.run_day(date(2022, 11, 1), "BANKNIFTY")
        assert len(result.trades) == 0

    def test_zero_pnl(self, engine_nothing):
        result = engine_nothing.run_day(date(2022, 11, 1), "BANKNIFTY")
        assert result.final_pnl == pytest.approx(0.0)

    def test_mtm_all_zeros(self, engine_nothing):
        result = engine_nothing.run_day(date(2022, 11, 1), "BANKNIFTY")
        assert (result.mtm == 0.0).all()

    def test_holdings_log_empty(self, engine_nothing):
        result = engine_nothing.run_day(date(2022, 11, 1), "BANKNIFTY")
        assert result.holdings_log.empty

    def test_engine_accepts_any_factory(self, engine_nothing):
        result = engine_nothing.run("BANKNIFTY")
        assert len(result.day_results) == 21


class TestBuyOnceStrategy:
    def test_exactly_four_trades(self, engine_buy_once):
        result = engine_buy_once.run_day(date(2022, 11, 1), "BANKNIFTY")
        assert len(result.trades) == 4

    def test_first_two_are_buys(self, engine_buy_once):
        result = engine_buy_once.run_day(date(2022, 11, 1), "BANKNIFTY")
        assert result.trades[0].side == "BUY"
        assert result.trades[1].side == "BUY"

    def test_last_two_are_sells(self, engine_buy_once):
        result = engine_buy_once.run_day(date(2022, 11, 1), "BANKNIFTY")
        assert result.trades[2].side == "SELL"
        assert result.trades[3].side == "SELL"

    def test_day_ends_flat(self, engine_buy_once):
        result = engine_buy_once.run_day(date(2022, 11, 1), "BANKNIFTY")
        assert result.mtm.iloc[-1] == pytest.approx(result.final_pnl, rel=1e-6)

    def test_holdings_log_has_one_strike(self, engine_buy_once):
        """BuyOnce never rolls so holdings_log should have exactly one unique strike."""
        result = engine_buy_once.run_day(date(2022, 11, 1), "BANKNIFTY")
        assert result.holdings_log["strike"].nunique() == 1


# ===========================================================================
# Full run — 21 days, BANKNIFTY
# ===========================================================================

class TestFullRunBankNifty:
    def test_day_results_count(self, full_result_banknifty):
        assert len(full_result_banknifty.day_results) == 21

    def test_all_dates_covered(self, full_result_banknifty, universe):
        result_dates = {dr.date for dr in full_result_banknifty.day_results}
        assert result_dates == set(universe.trading_days)

    def test_daily_pnl_shape(self, full_result_banknifty):
        assert full_result_banknifty.daily_pnl.shape == (21, 1)

    def test_daily_pnl_no_nans(self, full_result_banknifty):
        assert full_result_banknifty.daily_pnl.isna().sum().sum() == 0

    def test_mtm_df_length(self, full_result_banknifty):
        assert len(full_result_banknifty.mtm_df) == 21 * 22_500

    def test_trades_df_sorted(self, full_result_banknifty):
        df = full_result_banknifty.trades_df
        assert list(df["timestamp"]) == sorted(df["timestamp"])

    def test_trades_df_all_banknifty(self, full_result_banknifty):
        assert (full_result_banknifty.trades_df["underlier"] == "BANKNIFTY").all()

    def test_all_days_end_flat(self, full_result_banknifty):
        for dr in full_result_banknifty.day_results:
            assert dr.mtm.iloc[-1] == pytest.approx(dr.final_pnl, rel=1e-6), \
                f"{dr.date}: mtm={dr.mtm.iloc[-1]:.4f} realized={dr.final_pnl:.4f}"

    def test_holdings_log_df_not_empty(self, full_result_banknifty):
        assert not full_result_banknifty.holdings_log_df.empty

    def test_holdings_log_df_has_date_and_underlier(self, full_result_banknifty):
        cols = set(full_result_banknifty.holdings_log_df.columns)
        assert "date" in cols
        assert "underlier" in cols

# ===========================================================================
# run_all — orchestration and date-range filtering
# ===========================================================================

class BuyUnpricedStrategy(BaseStrategy):
    """Issues a BUY for a strike that has no data file (option_price -> None)."""
    def __init__(self, underlier, expiry, strikes):
        self.underlier = underlier
        self.expiry    = expiry
    def on_second(self, ts, futures_price, option_price, holdings):
        if ts.time().hour == 9 and ts.time().minute == 15 and ts.time().second == 0:
            return [Order(Instrument(self.underlier, self.expiry, 99999, "CE"), "BUY")]
        return []


class TestRunAll:

    def test_single_day_single_underlier(self, engine_atm):
        d = date(2022, 11, 1)
        result = engine_atm.run_all(underliers=["NIFTY"], start_date=d, end_date=d)
        assert len(result.day_results) == 1
        assert result.day_results[0].date == d
        assert result.day_results[0].underlier == "NIFTY"

    def test_default_underliers_are_both(self, engine_atm):
        d = date(2022, 11, 1)
        result = engine_atm.run_all(start_date=d, end_date=d)
        assert {dr.underlier for dr in result.day_results} == {"NIFTY", "BANKNIFTY"}

    def test_start_date_filters_earlier_days(self, engine_atm, universe):
        cutoff = universe.trading_days[2]
        result = engine_atm.run_all(underliers=["NIFTY"], start_date=cutoff,
                                    end_date=universe.trading_days[2])
        assert all(dr.date >= cutoff for dr in result.day_results)
        assert len(result.day_results) == 1

    def test_end_date_filters_later_days(self, engine_atm, universe):
        first = universe.trading_days[0]
        result = engine_atm.run_all(underliers=["NIFTY"], start_date=first, end_date=first)
        assert all(dr.date <= first for dr in result.day_results)

    def test_date_range_spans_multiple_days(self, engine_atm, universe):
        lo, hi = universe.trading_days[0], universe.trading_days[1]
        result = engine_atm.run_all(underliers=["NIFTY"], start_date=lo, end_date=hi)
        assert {dr.date for dr in result.day_results} == {lo, hi}


class TestRunSingleUnderlier:

    def test_run_covers_all_days(self, engine_nothing, universe):
        # DoNothing is cheap (no option loads) -> exercises run() over the calendar.
        result = engine_nothing.run("NIFTY")
        assert len(result.day_results) == len(universe.trading_days)
        assert all(len(dr.trades) == 0 for dr in result.day_results)


class TestBuyUnpricedInstrument:

    def test_unpriced_buy_is_skipped_with_warning(self, provider, caplog):
        engine = BacktestEngine(provider, BuyUnpricedStrategy, execution=INSTANT)
        dr = engine.run_day(date(2022, 11, 1), "NIFTY")
        # the BUY targets a nonexistent strike -> skipped, no trade recorded
        assert len(dr.trades) == 0
        assert dr.final_pnl == pytest.approx(0.0)
        assert [r.reason for r in dr.rejections] == ["no price available"]
        assert "Rejected BUY" in caplog.text


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
