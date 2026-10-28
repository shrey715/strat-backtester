# tests/test_fixed_window.py
#
# Behavioural coverage for FixedWindowStraddleStrategy: enter once after a delay,
# hold without rolling, force-close before the bell.

from typing import Callable

import numpy as np
import pandas as pd
import pytest
from datetime import date

from stratbt.instruments import Instrument
from stratbt.strategy.base import Order
from stratbt.strategy.fixed_window import FixedWindowStraddleStrategy

UNDERLIER = "BANKNIFTY"
EXPIRY    = date(2022, 11, 3)
STRIKES   = np.array([41400, 41500, 41600, 41700, 41800, 41900, 42000])

# entry_delay 300s -> entry deadline 09:20:00; exit_buffer 300s -> exit deadline 15:24:59
T_BEFORE_ENTRY = pd.Timestamp("2022-11-01 09:17:00")
T_AT_ENTRY     = pd.Timestamp("2022-11-01 09:20:00")
T_HOLD         = pd.Timestamp("2022-11-01 11:00:00")
T_AT_EXIT      = pd.Timestamp("2022-11-01 15:24:59")
T_AFTER_EXIT   = pd.Timestamp("2022-11-01 15:25:30")


def make_inst(strike, opt):
    return Instrument(UNDERLIER, EXPIRY, strike, opt)


def make_strategy(entry_delay_s=300, exit_buffer_s=300):
    return FixedWindowStraddleStrategy(UNDERLIER, EXPIRY, STRIKES,
                                       entry_delay_s=entry_delay_s, exit_buffer_s=exit_buffer_s)


def price_lookup(prices: dict) -> Callable:
    return lambda inst: prices.get(inst, None)


BOTH_LEGS = price_lookup({make_inst(41600, "CE"): 200.0, make_inst(41600, "PE"): 180.0})


# ---------------------------------------------------------------------------
# _nearest_strike (mirror of the ATM logic, retested independently)
# ---------------------------------------------------------------------------

class TestNearestStrike:

    def test_exact_hit(self):
        assert make_strategy()._nearest_strike(41600.0) == 41600

    def test_ties_go_lower(self):
        assert make_strategy()._nearest_strike(41650.0) == 41600

    def test_rounds_to_upper(self):
        assert make_strategy()._nearest_strike(41680.0) == 41700

    def test_clamps_below_range(self):
        assert make_strategy()._nearest_strike(40000.0) == 41400

    def test_clamps_above_range(self):
        assert make_strategy()._nearest_strike(43000.0) == 42000

    def test_returns_python_int(self):
        assert isinstance(make_strategy()._nearest_strike(41620.0), int)


# ---------------------------------------------------------------------------
# Entry timing
# ---------------------------------------------------------------------------

class TestEntry:

    def test_no_entry_before_deadline(self):
        strat = make_strategy()
        assert strat.on_second(T_BEFORE_ENTRY, 41600.0, BOTH_LEGS, {}) == []
        assert strat._entered is False

    def test_enters_at_deadline(self):
        strat  = make_strategy()
        orders = strat.on_second(T_AT_ENTRY, 41600.0, BOTH_LEGS, {})
        assert len(orders) == 2
        assert all(o.side == "BUY" for o in orders)
        strikes = {o.instrument.strike for o in orders}
        types   = {o.instrument.option_type for o in orders}
        assert strikes == {41600}
        assert types == {"CE", "PE"}
        assert strat._entered is True
        assert strat._entry_ts == T_AT_ENTRY

    def test_enters_at_atm_strike(self):
        # futures at 41680 -> nearest strike 41700; both legs must be priced.
        legs  = price_lookup({make_inst(41700, "CE"): 100.0, make_inst(41700, "PE"): 90.0})
        strat = make_strategy()
        orders = strat.on_second(T_AT_ENTRY, 41680.0, legs, {})
        assert {o.instrument.strike for o in orders} == {41700}

    def test_no_entry_when_ce_leg_unpriced(self):
        legs  = price_lookup({make_inst(41600, "PE"): 180.0})  # CE missing
        strat = make_strategy()
        assert strat.on_second(T_AT_ENTRY, 41600.0, legs, {}) == []
        assert strat._entered is False  # can retry next second

    def test_no_entry_when_pe_leg_unpriced(self):
        legs  = price_lookup({make_inst(41600, "CE"): 200.0})  # PE missing
        strat = make_strategy()
        assert strat.on_second(T_AT_ENTRY, 41600.0, legs, {}) == []
        assert strat._entered is False

    def test_retry_after_unpriced_then_priced(self):
        strat = make_strategy()
        assert strat.on_second(T_AT_ENTRY, 41600.0, price_lookup({}), {}) == []
        assert strat._entered is False
        orders = strat.on_second(T_HOLD, 41600.0, BOTH_LEGS, {})
        assert len(orders) == 2 and strat._entered is True


# ---------------------------------------------------------------------------
# Holding — never rolls
# ---------------------------------------------------------------------------

class TestHold:

    def test_holds_without_rolling_when_price_moves(self):
        # Holding 41600, futures drifts to 41900 (a different ATM). Must NOT roll.
        strat = make_strategy()
        strat.on_second(T_AT_ENTRY, 41600.0, BOTH_LEGS, {})
        holdings = {make_inst(41600, "CE"): (200.0, 1), make_inst(41600, "PE"): (180.0, 1)}
        assert strat.on_second(T_HOLD, 41900.0, BOTH_LEGS, holdings) == []

    def test_no_reentry_after_flat_once_entered(self):
        # Once entered, an empty book must not trigger a second entry.
        strat = make_strategy()
        strat.on_second(T_AT_ENTRY, 41600.0, BOTH_LEGS, {})
        assert strat._entered is True
        assert strat.on_second(T_HOLD, 41600.0, BOTH_LEGS, {}) == []


# ---------------------------------------------------------------------------
# Exit
# ---------------------------------------------------------------------------

class TestExit:

    def test_force_close_at_exit_deadline(self):
        strat    = make_strategy()
        holdings = {make_inst(41600, "CE"): (200.0, 1), make_inst(41600, "PE"): (180.0, 1)}
        orders   = strat.on_second(T_AT_EXIT, 41600.0, BOTH_LEGS, holdings)
        assert len(orders) == 2
        assert all(o.side == "SELL" for o in orders)
        assert {o.instrument for o in orders} == set(holdings)

    def test_force_close_after_exit_deadline(self):
        strat    = make_strategy()
        holdings = {make_inst(41600, "CE"): (200.0, 1)}
        orders   = strat.on_second(T_AFTER_EXIT, 41600.0, BOTH_LEGS, holdings)
        assert len(orders) == 1 and orders[0].side == "SELL"

    def test_no_close_when_flat_at_exit(self):
        strat = make_strategy()
        assert strat.on_second(T_AT_EXIT, 41600.0, BOTH_LEGS, {}) == []

    def test_no_new_entry_after_exit_window(self):
        # Past the exit deadline and flat -> no entry even if never entered.
        strat = make_strategy()
        assert strat.on_second(T_AFTER_EXIT, 41600.0, BOTH_LEGS, {}) == []


# ---------------------------------------------------------------------------
# Configurable window
# ---------------------------------------------------------------------------

class TestWindowParameters:

    def test_zero_entry_delay_enters_at_open(self):
        strat  = make_strategy(entry_delay_s=0)
        orders = strat.on_second(pd.Timestamp("2022-11-01 09:15:00"), 41600.0, BOTH_LEGS, {})
        assert len(orders) == 2

    def test_larger_entry_delay_defers_entry(self):
        # 600s delay -> entry deadline 09:25:00; at 09:20:00 still too early.
        strat = make_strategy(entry_delay_s=600)
        assert strat.on_second(T_AT_ENTRY, 41600.0, BOTH_LEGS, {}) == []

    def test_larger_exit_buffer_closes_earlier(self):
        # 900s buffer -> exit deadline 15:14:59; holding at 15:20:00 must close.
        strat    = make_strategy(exit_buffer_s=900)
        holdings = {make_inst(41600, "CE"): (200.0, 1)}
        orders   = strat.on_second(pd.Timestamp("2022-11-01 15:20:00"), 41600.0, BOTH_LEGS, holdings)
        assert orders and orders[0].side == "SELL"

    def test_default_parameters(self):
        strat = FixedWindowStraddleStrategy(UNDERLIER, EXPIRY, STRIKES)
        assert strat.entry_delay_s == 300
        assert strat.exit_buffer_s == 300


# ---------------------------------------------------------------------------
# Full-session lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:

    def test_enter_hold_close_sequence(self):
        strat = make_strategy()

        # before entry -> nothing
        assert strat.on_second(T_BEFORE_ENTRY, 41600.0, BOTH_LEGS, {}) == []

        # entry
        entry = strat.on_second(T_AT_ENTRY, 41600.0, BOTH_LEGS, {})
        assert all(o.side == "BUY" for o in entry)
        holdings = {o.instrument: (0.0, 1) for o in entry}

        # hold through the session, futures wandering -> no orders
        assert strat.on_second(T_HOLD, 41850.0, BOTH_LEGS, holdings) == []

        # force close at the buffer
        close = strat.on_second(T_AT_EXIT, 41850.0, BOTH_LEGS, holdings)
        assert all(o.side == "SELL" for o in close)
        assert {o.instrument for o in close} == set(holdings)

    def test_returns_order_objects(self):
        strat  = make_strategy()
        orders = strat.on_second(T_AT_ENTRY, 41600.0, BOTH_LEGS, {})
        assert all(isinstance(o, Order) for o in orders)
