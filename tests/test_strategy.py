# tests/test_strategy.py
from typing import Callable
import numpy as np
import pandas as pd
from datetime import date
from stratbt.instruments import Instrument
from stratbt.strategy.atm_straddle import ATMStraddleStrategy

UNDERLIER = "BANKNIFTY"
EXPIRY    = date(2022, 11, 3)
STRIKES   = np.array([41400, 41500, 41600, 41700, 41800, 41900, 42000])

T_OPEN    = pd.Timestamp("2022-11-01 09:15:00")
T_MID     = pd.Timestamp("2022-11-01 11:00:00")
T_CLOSE   = pd.Timestamp("2022-11-01 15:29:59")
T_BEFORE  = pd.Timestamp("2022-11-01 15:29:58")

def make_inst(strike: int, option_type: str) -> Instrument:
    return Instrument(UNDERLIER, EXPIRY, strike, option_type)

def make_strategy() -> ATMStraddleStrategy:
    return ATMStraddleStrategy(UNDERLIER, EXPIRY, STRIKES)

def price_lookup(prices: dict) -> Callable:
    """Build a fake option_price callable from a dict."""
    return lambda inst: prices.get(inst, None)

def test_nearest_strike_exact_hit():
    strat = make_strategy()
    assert strat._nearest_strike(41600.0) == 41600

def test_nearest_strike_closer_to_lower():
    strat = make_strategy()
    # midpoint between 41600 and 41700 is 41650
    # 41620 is closer to 41600
    assert strat._nearest_strike(41620.0) == 41600

def test_nearest_strike_closer_to_upper():
    strat = make_strategy()
    # 41680 is closer to 41700
    assert strat._nearest_strike(41680.0) == 41700

def test_nearest_strike_ties_go_to_lower():
    strat = make_strategy()
    # exactly at midpoint 41650 → lower strike wins
    assert strat._nearest_strike(41650.0) == 41600

def test_nearest_strike_below_all_strikes():
    strat = make_strategy()
    # below 41400 → clamp to lowest
    assert strat._nearest_strike(40000.0) == 41400

def test_nearest_strike_above_all_strikes():
    strat = make_strategy()
    # above 42000 → clamp to highest
    assert strat._nearest_strike(43000.0) == 42000

def test_nearest_strike_returns_python_int():
    strat = make_strategy()
    result = strat._nearest_strike(41620.0)
    assert isinstance(result, int), \
        f"Expected Python int, got {type(result)} — numpy int will cause hashing issues"

def test_nearest_strike_consistent_with_instrument_equality():
    """
    The strike returned by _nearest_strike must produce an Instrument
    that compares equal to one created directly with a Python int.
    This is the critical property for roll correctness.
    """
    strat  = make_strategy()
    strike = strat._nearest_strike(41620.0)
    ce     = Instrument(UNDERLIER, EXPIRY, strike, "CE")
    ce_ref = Instrument(UNDERLIER, EXPIRY, 41600, "CE")
    assert ce == ce_ref
    assert hash(ce) == hash(ce_ref)

def test_make_legs_returns_ce_and_pe():
    strat = make_strategy()
    ce, pe = strat._make_legs(41600)
    assert ce.option_type == "CE"
    assert pe.option_type == "PE"

def test_make_legs_correct_strike():
    strat = make_strategy()
    ce, pe = strat._make_legs(41600)
    assert ce.strike == 41600
    assert pe.strike == 41600

def test_make_legs_correct_underlier():
    strat = make_strategy()
    ce, pe = strat._make_legs(41600)
    assert ce.underlier == UNDERLIER
    assert pe.underlier == UNDERLIER

def test_make_legs_correct_expiry():
    strat = make_strategy()
    ce, pe = strat._make_legs(41600)
    assert ce.expiry == EXPIRY
    assert pe.expiry == EXPIRY

def test_make_legs_matches_direct_construction():
    strat  = make_strategy()
    ce, pe = strat._make_legs(41600)
    assert ce == make_inst(41600, "CE")
    assert pe == make_inst(41600, "PE")

def test_held_strike_empty_holdings():
    strat = make_strategy()
    assert strat._held_strike({}) is None

def test_held_strike_with_one_instrument():
    strat    = make_strategy()
    holdings = {make_inst(41600, "CE"): (242.0, 1)}
    assert strat._held_strike(holdings) == 41600

def test_held_strike_with_full_straddle():
    strat    = make_strategy()
    holdings = {
        make_inst(41600, "CE"): (242.0, 1),
        make_inst(41600, "PE"): (414.0, 1),
    }
    assert strat._held_strike(holdings) == 41600

def test_held_strike_different_strike():
    strat    = make_strategy()
    holdings = {
        make_inst(41700, "CE"): (200.0, 1),
        make_inst(41700, "PE"): (380.0, 1),
    }
    assert strat._held_strike(holdings) == 41700

def test_flat_both_prices_available_returns_two_buys():
    strat  = make_strategy()
    prices = {
        make_inst(41600, "CE"): 242.0,
        make_inst(41600, "PE"): 414.0,
    }
    orders = strat.on_second(T_MID, 41620.0, price_lookup(prices), {})
    assert len(orders) == 2
    assert all(o.side == "BUY" for o in orders)

def test_flat_buys_correct_instruments():
    strat  = make_strategy()
    prices = {
        make_inst(41600, "CE"): 242.0,
        make_inst(41600, "PE"): 414.0,
    }
    orders = strat.on_second(T_MID, 41620.0, price_lookup(prices), {})
    instruments = {o.instrument for o in orders}
    assert make_inst(41600, "CE") in instruments
    assert make_inst(41600, "PE") in instruments

def test_flat_ce_price_none_returns_empty():
    strat  = make_strategy()
    prices = {make_inst(41600, "PE"): 414.0}   # CE missing
    orders = strat.on_second(T_MID, 41620.0, price_lookup(prices), {})
    assert orders == []

def test_flat_pe_price_none_returns_empty():
    strat  = make_strategy()
    prices = {make_inst(41600, "CE"): 242.0}   # PE missing
    orders = strat.on_second(T_MID, 41620.0, price_lookup(prices), {})
    assert orders == []

def test_flat_both_prices_none_returns_empty():
    strat  = make_strategy()
    orders = strat.on_second(T_MID, 41620.0, price_lookup({}), {})
    assert orders == []

def test_flat_buy_qty_is_one():
    strat  = make_strategy()
    prices = {
        make_inst(41600, "CE"): 242.0,
        make_inst(41600, "PE"): 414.0,
    }
    orders = strat.on_second(T_MID, 41620.0, price_lookup(prices), {})
    assert all(o.qty == 1 for o in orders)

def test_holding_correct_strike_returns_empty():
    strat    = make_strategy()
    holdings = {
        make_inst(41600, "CE"): (242.0, 1),
        make_inst(41600, "PE"): (414.0, 1),
    }
    prices = {
        make_inst(41600, "CE"): 250.0,
        make_inst(41600, "PE"): 410.0,
    }
    # futures still near 41600
    orders = strat.on_second(T_MID, 41620.0, price_lookup(prices), holdings)
    assert orders == []

def test_holding_correct_strike_returns_empty_even_at_open():
    strat    = make_strategy()
    holdings = {
        make_inst(41600, "CE"): (242.0, 1),
        make_inst(41600, "PE"): (414.0, 1),
    }
    prices = {
        make_inst(41600, "CE"): 242.0,
        make_inst(41600, "PE"): 414.0,
    }
    orders = strat.on_second(T_OPEN, 41620.0, price_lookup(prices), holdings)
    assert orders == []

def test_roll_returns_four_orders():
    strat    = make_strategy()
    holdings = {
        make_inst(41600, "CE"): (242.0, 1),
        make_inst(41600, "PE"): (414.0, 1),
    }
    prices = {
        make_inst(41700, "CE"): 200.0,
        make_inst(41700, "PE"): 380.0,
    }
    # futures moved to 41700 zone
    orders = strat.on_second(T_MID, 41720.0, price_lookup(prices), holdings)
    assert len(orders) == 4

def test_roll_order_sequence():
    """
    Roll must be: SELL old CE, SELL old PE, BUY new CE, BUY new PE.
    The sells must come before the buys — you cannot buy before closing.
    """
    strat    = make_strategy()
    holdings = {
        make_inst(41600, "CE"): (242.0, 1),
        make_inst(41600, "PE"): (414.0, 1),
    }
    prices = {
        make_inst(41700, "CE"): 200.0,
        make_inst(41700, "PE"): 380.0,
    }
    orders = strat.on_second(T_MID, 41720.0, price_lookup(prices), holdings)
    assert orders[0].side == "SELL"
    assert orders[1].side == "SELL"
    assert orders[2].side == "BUY"
    assert orders[3].side == "BUY"

def test_roll_sells_old_instruments():
    strat    = make_strategy()
    holdings = {
        make_inst(41600, "CE"): (242.0, 1),
        make_inst(41600, "PE"): (414.0, 1),
    }
    prices = {
        make_inst(41700, "CE"): 200.0,
        make_inst(41700, "PE"): 380.0,
    }
    orders = strat.on_second(T_MID, 41720.0, price_lookup(prices), holdings)
    sold = {o.instrument for o in orders if o.side == "SELL"}
    assert make_inst(41600, "CE") in sold
    assert make_inst(41600, "PE") in sold

def test_roll_buys_new_instruments():
    strat    = make_strategy()
    holdings = {
        make_inst(41600, "CE"): (242.0, 1),
        make_inst(41600, "PE"): (414.0, 1),
    }
    prices = {
        make_inst(41700, "CE"): 200.0,
        make_inst(41700, "PE"): 380.0,
    }
    orders = strat.on_second(T_MID, 41720.0, price_lookup(prices), holdings)
    bought = {o.instrument for o in orders if o.side == "BUY"}
    assert make_inst(41700, "CE") in bought
    assert make_inst(41700, "PE") in bought

def test_roll_new_ce_price_none_waits():
    """If new CE has no price, do not roll — hold current straddle."""
    strat    = make_strategy()
    holdings = {
        make_inst(41600, "CE"): (242.0, 1),
        make_inst(41600, "PE"): (414.0, 1),
    }
    prices = {make_inst(41700, "PE"): 380.0}   # CE missing
    orders = strat.on_second(T_MID, 41720.0, price_lookup(prices), holdings)
    assert orders == []

def test_roll_new_pe_price_none_waits():
    """If new PE has no price, do not roll — hold current straddle."""
    strat    = make_strategy()
    holdings = {
        make_inst(41600, "CE"): (242.0, 1),
        make_inst(41600, "PE"): (414.0, 1),
    }
    prices = {make_inst(41700, "CE"): 200.0}   # PE missing
    orders = strat.on_second(T_MID, 41720.0, price_lookup(prices), holdings)
    assert orders == []

def test_roll_old_instruments_match_holdings():
    """
    The SELL orders must reference the exact same Instrument objects
    (by equality + hash) as what is in holdings. If they don't match,
    position.sell() will raise ValueError.
    """
    strat    = make_strategy()
    held_ce  = make_inst(41600, "CE")
    held_pe  = make_inst(41600, "PE")
    holdings = {held_ce: (242.0, 1), held_pe: (414.0, 1)}
    prices   = {
        make_inst(41700, "CE"): 200.0,
        make_inst(41700, "PE"): 380.0,
    }
    orders   = strat.on_second(T_MID, 41720.0, price_lookup(prices), holdings)
    sold_insts = {o.instrument for o in orders if o.side == "SELL"}
    # reconstructed instruments must match held ones exactly
    assert held_ce in sold_insts
    assert held_pe in sold_insts

def test_roll_back_and_forth():
    """
    Simulate the ping-pong: roll to 41700, then roll back to 41600.
    Both rolls must produce correct order sequences.
    """
    strat = make_strategy()

    # first roll: 41600 → 41700
    holdings_1 = {
        make_inst(41600, "CE"): (242.0, 1),
        make_inst(41600, "PE"): (414.0, 1),
    }
    prices_1 = {
        make_inst(41700, "CE"): 200.0,
        make_inst(41700, "PE"): 380.0,
    }
    orders_1 = strat.on_second(T_MID, 41720.0, price_lookup(prices_1), holdings_1)
    assert len(orders_1) == 4
    bought_1 = {o.instrument for o in orders_1 if o.side == "BUY"}
    assert make_inst(41700, "CE") in bought_1

    # second roll: 41700 → 41600
    holdings_2 = {
        make_inst(41700, "CE"): (200.0, 1),
        make_inst(41700, "PE"): (380.0, 1),
    }
    prices_2 = {
        make_inst(41600, "CE"): 245.0,
        make_inst(41600, "PE"): 420.0,
    }
    orders_2 = strat.on_second(T_MID, 41620.0, price_lookup(prices_2), holdings_2)
    assert len(orders_2) == 4
    sold_2 = {o.instrument for o in orders_2 if o.side == "SELL"}
    assert make_inst(41700, "CE") in sold_2


# ---------------------------------------------------------------------------
# 7. on_second — day close
# ---------------------------------------------------------------------------

def test_close_with_holdings_returns_sells():
    strat    = make_strategy()
    holdings = {
        make_inst(41600, "CE"): (242.0, 1),
        make_inst(41600, "PE"): (414.0, 1),
    }
    prices = {
        make_inst(41600, "CE"): 250.0,
        make_inst(41600, "PE"): 410.0,
    }
    orders = strat.on_second(T_CLOSE, 41620.0, price_lookup(prices), holdings)
    assert len(orders) == 2
    assert all(o.side == "SELL" for o in orders)

def test_close_sells_all_held_instruments():
    strat    = make_strategy()
    holdings = {
        make_inst(41600, "CE"): (242.0, 1),
        make_inst(41600, "PE"): (414.0, 1),
    }
    orders = strat.on_second(T_CLOSE, 41620.0, price_lookup({}), holdings)
    sold = {o.instrument for o in orders}
    assert make_inst(41600, "CE") in sold
    assert make_inst(41600, "PE") in sold

def test_close_with_no_holdings_returns_empty():
    strat  = make_strategy()
    orders = strat.on_second(T_CLOSE, 41620.0, price_lookup({}), {})
    assert orders == []

def test_one_second_before_close_normal_logic():
    """One second before close must behave like normal trading, not force-close."""
    strat    = make_strategy()
    holdings = {
        make_inst(41600, "CE"): (242.0, 1),
        make_inst(41600, "PE"): (414.0, 1),
    }
    prices = {
        make_inst(41600, "CE"): 250.0,
        make_inst(41600, "PE"): 410.0,
    }
    # T_BEFORE = 15:29:58, one second before DAY_CLOSE_TIME
    orders = strat.on_second(T_BEFORE, 41620.0, price_lookup(prices), holdings)
    # holding correct strike → normal logic says hold
    assert orders == []

def test_close_does_not_open_new_positions():
    """At close time with flat position, must not try to enter."""
    strat  = make_strategy()
    prices = {
        make_inst(41600, "CE"): 242.0,
        make_inst(41600, "PE"): 414.0,
    }
    orders = strat.on_second(T_CLOSE, 41620.0, price_lookup(prices), {})
    assert orders == []

def test_order_fields_on_entry():
    strat  = make_strategy()
    prices = {
        make_inst(41600, "CE"): 242.0,
        make_inst(41600, "PE"): 414.0,
    }
    orders = strat.on_second(T_MID, 41620.0, price_lookup(prices), {})
    for o in orders:
        assert o.qty == 1
        assert o.side in ("BUY", "SELL")
        assert isinstance(o.instrument, Instrument)

def test_order_fields_on_roll():
    strat    = make_strategy()
    holdings = {
        make_inst(41600, "CE"): (242.0, 1),
        make_inst(41600, "PE"): (414.0, 1),
    }
    prices = {
        make_inst(41700, "CE"): 200.0,
        make_inst(41700, "PE"): 380.0,
    }
    orders = strat.on_second(T_MID, 41720.0, price_lookup(prices), holdings)
    for o in orders:
        assert o.qty == 1
        assert o.side in ("BUY", "SELL")
        assert isinstance(o.instrument, Instrument)

def test_mini_simulation_entry_then_hold_then_roll():
    """
    Run three consecutive seconds:
      t=0: flat, prices available → enter at 41600
      t=1: holding 41600, futures still at 41600 → hold
      t=2: futures moved to 41700 zone → roll
    Verify orders at each step are correct.
    """
    strat = make_strategy()

    prices_600 = {
        make_inst(41600, "CE"): 242.0,
        make_inst(41600, "PE"): 414.0,
    }
    prices_700 = {
        make_inst(41700, "CE"): 200.0,
        make_inst(41700, "PE"): 380.0,
    }

    ts0 = pd.Timestamp("2022-11-01 09:15:00")
    ts1 = pd.Timestamp("2022-11-01 09:15:01")
    ts2 = pd.Timestamp("2022-11-01 09:15:02")

    # t=0: entry
    orders_0 = strat.on_second(ts0, 41620.0, price_lookup(prices_600), {})
    assert len(orders_0) == 2
    assert all(o.side == "BUY" for o in orders_0)

    # simulate the engine executing: now holding 41600 straddle
    holdings = {
        make_inst(41600, "CE"): (242.0, 1),
        make_inst(41600, "PE"): (414.0, 1),
    }

    # t=1: hold
    orders_1 = strat.on_second(ts1, 41620.0, price_lookup(prices_600), holdings)
    assert orders_1 == []

    # t=2: roll
    orders_2 = strat.on_second(ts2, 41720.0, price_lookup(prices_700), holdings)
    assert len(orders_2) == 4
    assert orders_2[0].side == "SELL"
    assert orders_2[1].side == "SELL"
    assert orders_2[2].side == "BUY"
    assert orders_2[3].side == "BUY"