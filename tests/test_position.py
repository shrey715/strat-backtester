# tests/test_position.py

import pytest
import pandas as pd
from datetime import date
from stratbt.instruments import Instrument
from stratbt.position import Position, Trade

T0 = pd.Timestamp("2022-11-01 09:15:00")
T1 = pd.Timestamp("2022-11-01 09:15:01")
T2 = pd.Timestamp("2022-11-01 09:15:02")
T3 = pd.Timestamp("2022-11-01 09:15:03")

CE = Instrument("BANKNIFTY", date(2022, 11, 3), 41600, "CE")
PE = Instrument("BANKNIFTY", date(2022, 11, 3), 41600, "PE")
CE2 = Instrument("BANKNIFTY", date(2022, 11, 3), 41700, "CE")
PE2 = Instrument("BANKNIFTY", date(2022, 11, 3), 41700, "PE")

@pytest.fixture
def pos():
    return Position()

def test_fresh_realized_is_zero(pos):
    assert pos.realized == 0.0

def test_fresh_is_empty(pos):
    assert pos.is_empty is True

def test_fresh_holdings_empty(pos):
    assert pos.holdings == {}

def test_fresh_trades_empty(pos):
    assert pos.trades == []

def test_fresh_unrealized_zero(pos):
    assert pos.unrealized({}) == 0.0

def test_buy_holdings_has_instrument(pos):
    pos.buy(T0, CE, price=242.0)
    assert CE in pos.holdings

def test_buy_holdings_entry_price(pos):
    pos.buy(T0, CE, price=242.0)
    entry_price, qty = pos.holdings[CE]
    assert entry_price == 242.0

def test_buy_holdings_default_qty(pos):
    pos.buy(T0, CE, price=242.0)
    _, qty = pos.holdings[CE]
    assert qty == 1

def test_buy_holdings_explicit_qty(pos):
    pos.buy(T0, CE, price=242.0, qty=3)
    _, qty = pos.holdings[CE]
    assert qty == 3

def test_buy_realized_still_zero(pos):
    pos.buy(T0, CE, price=242.0)
    assert pos.realized == 0.0

def test_buy_is_not_empty(pos):
    pos.buy(T0, CE, price=242.0)
    assert pos.is_empty is False

def test_buy_returns_trade(pos):
    trade = pos.buy(T0, CE, price=242.0)
    assert isinstance(trade, Trade)
    assert trade.side == "BUY"
    assert trade.price == 242.0
    assert trade.instrument == CE
    assert trade.timestamp == T0
    assert trade.qty == 1

def test_buy_appended_to_trade_log(pos):
    pos.buy(T0, CE, price=242.0)
    assert len(pos.trades) == 1
    assert pos.trades[0].side == "BUY"

def test_sell_at_profit_realized(pos):
    pos.buy(T0, CE, price=100.0)
    pos.sell(T1, CE, price=130.0)
    assert pos.realized == pytest.approx(30.0)

def test_sell_at_loss_realized(pos):
    pos.buy(T0, CE, price=100.0)
    pos.sell(T1, CE, price=80.0)
    assert pos.realized == pytest.approx(-20.0)

def test_sell_at_breakeven_realized(pos):
    pos.buy(T0, CE, price=100.0)
    pos.sell(T1, CE, price=100.0)
    assert pos.realized == pytest.approx(0.0)

def test_sell_removes_from_holdings(pos):
    pos.buy(T0, CE, price=100.0)
    pos.sell(T1, CE, price=130.0)
    assert CE not in pos.holdings

def test_sell_is_empty_after_full_close(pos):
    pos.buy(T0, CE, price=100.0)
    pos.sell(T1, CE, price=130.0)
    assert pos.is_empty is True

def test_sell_returns_trade(pos):
    pos.buy(T0, CE, price=100.0)
    trade = pos.sell(T1, CE, price=130.0)
    assert isinstance(trade, Trade)
    assert trade.side == "SELL"
    assert trade.price == 130.0
    assert trade.qty == 1

def test_sell_with_qty_realized(pos):
    pos.buy(T0, CE, price=100.0, qty=3)
    pos.sell(T1, CE, price=120.0, qty=3)
    assert pos.realized == pytest.approx(60.0)

def test_partial_sell_remaining_qty(pos):
    pos.buy(T0, CE, price=100.0, qty=5)
    pos.sell(T1, CE, price=120.0, qty=2)
    _, remaining = pos.holdings[CE]
    assert remaining == 3

def test_partial_sell_realized(pos):
    pos.buy(T0, CE, price=100.0, qty=5)
    pos.sell(T1, CE, price=120.0, qty=2)
    assert pos.realized == pytest.approx(40.0)

def test_partial_sell_entry_price_unchanged(pos):
    pos.buy(T0, CE, price=100.0, qty=5)
    pos.sell(T1, CE, price=120.0, qty=2)
    entry_price, _ = pos.holdings[CE]
    assert entry_price == pytest.approx(100.0)

def test_partial_sell_not_empty(pos):
    pos.buy(T0, CE, price=100.0, qty=5)
    pos.sell(T1, CE, price=120.0, qty=2)
    assert pos.is_empty is False

def test_scale_in_average_cost(pos):
    pos.buy(T0, CE, price=100.0, qty=1)
    pos.buy(T1, CE, price=120.0, qty=1)
    entry_price, qty = pos.holdings[CE]
    assert qty == 2
    assert entry_price == pytest.approx(110.0)

def test_scale_in_average_cost_unequal_qty(pos):
    # 1 lot at 100, 2 lots at 130 → avg = (100 + 260) / 3 = 120
    pos.buy(T0, CE, price=100.0, qty=1)
    pos.buy(T1, CE, price=130.0, qty=2)
    entry_price, qty = pos.holdings[CE]
    assert qty == 3
    assert entry_price == pytest.approx(120.0)

def test_scale_in_realized_unchanged(pos):
    pos.buy(T0, CE, price=100.0, qty=1)
    pos.buy(T1, CE, price=120.0, qty=1)
    assert pos.realized == 0.0

def test_scale_in_then_sell_uses_avg_cost(pos):
    pos.buy(T0, CE, price=100.0, qty=1)
    pos.buy(T1, CE, price=120.0, qty=1)
    pos.sell(T2, CE, price=130.0, qty=2)
    # avg entry = 110, sell at 130, qty=2 → realized = (130-110)*2 = 40
    assert pos.realized == pytest.approx(40.0)

def test_unrealized_mid_hold(pos):
    pos.buy(T0, CE, price=100.0)
    assert pos.unrealized({CE: 115.0}) == pytest.approx(15.0)

def test_unrealized_mid_hold_loss(pos):
    pos.buy(T0, CE, price=100.0)
    assert pos.unrealized({CE: 85.0}) == pytest.approx(-15.0)

def test_unrealized_two_holdings(pos):
    pos.buy(T0, CE, price=242.0)
    pos.buy(T0, PE, price=414.0)
    prices = {CE: 260.0, PE: 400.0}
    # CE: +18, PE: -14 → total +4
    assert pos.unrealized(prices) == pytest.approx(4.0)

def test_unrealized_with_qty(pos):
    pos.buy(T0, CE, price=100.0, qty=3)
    assert pos.unrealized({CE: 110.0}) == pytest.approx(30.0)

def test_unrealized_empty_position(pos):
    assert pos.unrealized({}) == 0.0

def test_mtm_mid_hold(pos):
    pos.buy(T0, CE, price=100.0)
    pos.buy(T0, PE, price=200.0)
    prices = {CE: 110.0, PE: 190.0}
    # realized=0, unrealized = +10 -10 = 0
    assert pos.mtm(prices) == pytest.approx(0.0)

def test_mtm_after_partial_close(pos):
    pos.buy(T0, CE, price=100.0)
    pos.sell(T1, CE, price=120.0)   # realized = +20
    pos.buy(T2, PE, price=200.0)
    prices = {PE: 210.0}
    # realized=20, unrealized=10 → mtm=30
    assert pos.mtm(prices) == pytest.approx(30.0)

def test_invariant_1_mtm_equals_realized_after_close(pos):
    pos.buy(T0, CE, price=242.0)
    pos.buy(T0, PE, price=414.0)
    pos.sell(T1, CE, price=260.0)
    pos.sell(T1, PE, price=400.0)
    # unrealized must be 0 (nothing held)
    assert pos.unrealized({}) == 0.0
    # mtm must equal realized
    assert pos.mtm({}) == pytest.approx(pos.realized)

def test_invariant_1_realized_correct(pos):
    pos.buy(T0, CE, price=242.0)
    pos.buy(T0, PE, price=414.0)
    pos.sell(T1, CE, price=260.0)
    pos.sell(T1, PE, price=400.0)
    # CE: +18, PE: -14 → total +4
    assert pos.realized == pytest.approx(4.0)

def test_invariant_2_roll_continuity(pos):
    """
    Buy straddle. Price moves. Roll at current market price.
    MTM just before roll must equal MTM just after roll.
    """
    # open straddle 1
    pos.buy(T0, CE,  price=242.0)
    pos.buy(T0, PE,  price=414.0)

    # price moves
    prices_before = {CE: 260.0, PE: 400.0}
    mtm_before = pos.mtm(prices_before)

    # roll: sell straddle 1 at current prices, buy straddle 2 at same prices
    pos.sell(T1, CE,  price=260.0)
    pos.sell(T1, PE,  price=400.0)
    pos.buy(T1,  CE2, price=260.0)
    pos.buy(T1,  PE2, price=400.0)

    prices_after = {CE2: 260.0, PE2: 400.0}
    mtm_after = pos.mtm(prices_after)

    assert mtm_after == pytest.approx(mtm_before, abs=1e-9)

def test_sell_without_holding_raises(pos):
    with pytest.raises(ValueError):
        pos.sell(T0, CE, price=100.0)

def test_sell_more_than_held_raises(pos):
    pos.buy(T0, CE, price=100.0, qty=2)
    with pytest.raises(ValueError):
        pos.sell(T1, CE, price=120.0, qty=5)

def test_buy_zero_qty_raises(pos):
    with pytest.raises(ValueError):
        pos.buy(T0, CE, price=100.0, qty=0)

def test_sell_zero_qty_raises(pos):
    pos.buy(T0, CE, price=100.0)
    with pytest.raises(ValueError):
        pos.sell(T1, CE, price=120.0, qty=0)

def test_buy_negative_qty_raises(pos):
    with pytest.raises(ValueError):
        pos.buy(T0, CE, price=100.0, qty=-1)

def test_sell_negative_qty_raises(pos):
    pos.buy(T0, CE, price=100.0)
    with pytest.raises(ValueError):
        pos.sell(T1, CE, price=120.0, qty=-1)

def test_sell_wrong_instrument_raises(pos):
    pos.buy(T0, CE, price=100.0)
    with pytest.raises(ValueError):
        pos.sell(T1, PE, price=120.0)   # PE not held, CE is

def test_holdings_returns_copy(pos):
    pos.buy(T0, CE, price=100.0)
    h = pos.holdings
    h[CE] = (999.0, 99)     # mutate the copy
    # internal state must be unchanged
    entry_price, qty = pos.holdings[CE]
    assert entry_price == 100.0
    assert qty == 1

def test_trades_returns_copy(pos):
    pos.buy(T0, CE, price=100.0)
    t = pos.trades
    t.clear()               # mutate the copy
    assert len(pos.trades) == 1

def test_trade_log_length_after_buy_sell(pos):
    pos.buy(T0, CE, price=100.0)
    pos.sell(T1, CE, price=120.0)
    assert len(pos.trades) == 2

def test_trade_log_sides(pos):
    pos.buy(T0, CE, price=100.0)
    pos.sell(T1, CE, price=120.0)
    assert pos.trades[0].side == "BUY"
    assert pos.trades[1].side == "SELL"

def test_trade_log_prices(pos):
    pos.buy(T0, CE, price=100.0)
    pos.sell(T1, CE, price=120.0)
    assert pos.trades[0].price == 100.0
    assert pos.trades[1].price == 120.0

def test_trade_log_timestamps(pos):
    pos.buy(T0, CE, price=100.0)
    pos.sell(T1, CE, price=120.0)
    assert pos.trades[0].timestamp == T0
    assert pos.trades[1].timestamp == T1

def test_trade_log_multiple_instruments(pos):
    pos.buy(T0, CE, price=242.0)
    pos.buy(T0, PE, price=414.0)
    pos.sell(T1, CE, price=260.0)
    pos.sell(T1, PE, price=400.0)
    assert len(pos.trades) == 4
    sides = [t.side for t in pos.trades]
    assert sides == ["BUY", "BUY", "SELL", "SELL"]

# ---------------------------------------------------------------------------
# Shorts, fees, multiplier
# ---------------------------------------------------------------------------

class TestShortsFeesMultiplier:
    TS = pd.Timestamp("2024-01-02 10:00:00")
    INST = Instrument("SPY", date(2024, 1, 5), 450, "CE")

    def test_short_then_cover_realizes_gain_when_price_falls(self):
        p = Position(allow_short=True)
        p.sell(self.TS, self.INST, 5.0)
        assert p.holdings[self.INST] == (5.0, -1)
        p.buy(self.TS, self.INST, 3.0)
        assert p.realized == pytest.approx(2.0) and p.is_empty

    def test_short_unrealized_sign(self):
        p = Position(allow_short=True)
        p.sell(self.TS, self.INST, 5.0)
        assert p.unrealized({self.INST: 7.0}) == pytest.approx(-2.0)

    def test_scale_into_short_averages_cost(self):
        p = Position(allow_short=True)
        p.sell(self.TS, self.INST, 4.0)
        p.sell(self.TS, self.INST, 6.0)
        assert p.holdings[self.INST] == (pytest.approx(5.0), -2)

    def test_selling_through_zero_flips_to_short(self):
        p = Position(allow_short=True)
        p.buy(self.TS, self.INST, 4.0)
        p.sell(self.TS, self.INST, 5.0, qty=3)
        assert p.realized == pytest.approx(1.0)
        assert p.holdings[self.INST] == (5.0, -2)

    def test_multiplier_scales_pnl(self):
        p = Position(multiplier=100)
        p.buy(self.TS, self.INST, 2.0, qty=2)
        assert p.unrealized({self.INST: 2.5}) == pytest.approx(100.0)
        p.sell(self.TS, self.INST, 2.5, qty=2)
        assert p.realized == pytest.approx(100.0)

    def test_fees_tracked_separately(self):
        p = Position()
        p.buy(self.TS, self.INST, 2.0, fee=0.5)
        p.sell(self.TS, self.INST, 3.0, fee=0.5)
        assert p.realized == pytest.approx(1.0)
        assert p.fees == pytest.approx(1.0)
        assert p.net_realized == pytest.approx(0.0)
        assert [t.fee for t in p.trades] == [0.5, 0.5]

    def test_long_only_default_still_rejects_naked_sell(self):
        with pytest.raises(ValueError, match="not held"):
            Position().sell(self.TS, self.INST, 1.0)

    def test_bad_multiplier(self):
        with pytest.raises(ValueError):
            Position(multiplier=0)
