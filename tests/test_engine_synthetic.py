# tests/test_engine_synthetic.py
"""Engine behaviour on the synthetic market — runs anywhere, no data files needed."""
from datetime import date

import numpy as np
import pytest

from stratbt import BacktestEngine, ExecutionModel, RiskLimits, SyntheticProvider, run_backtest
from stratbt.instruments import Instrument
from stratbt.strategy.base import BaseStrategy, Order


@pytest.fixture(scope="module")
def prov():
    return SyntheticProvider(n_days=2, seed=11)


class EnterAt(BaseStrategy):
    """Buys the ATM call at a given second-of-session; never exits (engine flattens)."""
    def __init__(self, underlier, expiry, strikes, at=100):
        self.u, self.e, self.k, self.at, self.done = underlier, expiry, strikes, at, False
        self.seen_holdings = {}

    def on_second(self, ts, futures_price, option_price, holdings):
        self.seen_holdings[ts] = dict(holdings)
        if self.done or ts.second + 60 * ts.minute < 0:
            return []
        if (ts - ts.normalize()).total_seconds() < 9.5 * 3600 + self.at:
            return []
        strike = int(self.k[np.abs(self.k - futures_price).argmin()])
        self.done = True
        return [Order(Instrument(self.u, self.e, strike, "CE"), "BUY")]


def _run(prov, execution, factory=EnterAt, **kw):
    return BacktestEngine(prov, factory, execution=execution, **kw).run_day(prov.trading_days[0], "SPY")


def test_instant_fill_happens_in_decision_second(prov):
    dr = _run(prov, ExecutionModel.instant())
    buy = dr.trades[0]
    assert buy.timestamp == prov.session_grid(prov.trading_days[0])[100]


def test_default_fill_is_one_second_after_decision(prov):
    dr = _run(prov, ExecutionModel())
    assert dr.trades[0].timestamp == prov.session_grid(prov.trading_days[0])[101]


def test_delayed_fill_uses_later_price_not_decision_price(prov):
    day = prov.trading_days[0]
    dr0, dr1 = _run(prov, ExecutionModel.instant()), _run(prov, ExecutionModel())
    inst = dr1.trades[0].instrument
    series = prov.option_prices(day, inst)
    assert dr1.trades[0].price == pytest.approx(series.iloc[101])
    assert dr0.trades[0].price == pytest.approx(prov.option_prices(day, dr0.trades[0].instrument).iloc[100])


def test_strategy_not_consulted_while_orders_in_flight(prov):
    seen = []

    class Spy(EnterAt):
        def on_second(self, ts, *a):
            seen.append(ts)
            return super().on_second(ts, *a)

    dr = _run(prov, ExecutionModel(delay_s=5), Spy)
    grid = prov.session_grid(prov.trading_days[0])
    assert not {grid[i] for i in range(101, 105)} & set(seen)    # in flight: not consulted
    assert grid[105] in seen                                     # filled, consulted again
    assert dr.trades[0].timestamp == grid[105]


def test_open_positions_flattened_at_session_close(prov):
    dr = _run(prov, ExecutionModel())
    assert [t.side for t in dr.trades] == ["BUY", "SELL"]
    assert dr.trades[-1].timestamp == prov.session_grid(prov.trading_days[0])[-1]


def test_flatten_at_close_can_be_disabled(prov):
    dr = _run(prov, ExecutionModel(), flatten_at_close=False)
    assert [t.side for t in dr.trades] == ["BUY"]
    assert dr.final_pnl == 0.0                       # nothing realized


def test_fees_are_deducted_from_pnl_and_mtm(prov):
    free = _run(prov, ExecutionModel())
    paid = _run(prov, ExecutionModel(commission_per_contract=2.0))
    assert paid.fees == pytest.approx(4.0)
    assert paid.final_pnl == pytest.approx(free.final_pnl - 4.0)
    assert paid.mtm.iloc[-1] == pytest.approx(paid.final_pnl)


def test_pnl_scales_with_contract_multiplier(prov):
    dr = _run(prov, ExecutionModel())
    t_buy, t_sell = dr.trades
    assert dr.final_pnl == pytest.approx((t_sell.price - t_buy.price) * 100)


def test_trades_df_carries_fee_column(prov):
    r = run_backtest("fixed_window", prov, execution=ExecutionModel(commission_per_contract=1.0),
                     underliers=["SPY"])
    assert set(r.trades_df["fee"]) == {1.0}
    assert r.unit == "USD"


def test_daily_loss_limit_halts_and_flattens(prov):
    from stratbt.strategy import registry
    ex = ExecutionModel(half_spread_bps=200)           # rolling pays the spread over and over
    factory = registry.get("atm_straddle").factory(prov)
    free = BacktestEngine(prov, factory, execution=ex).run_day(prov.trading_days[0], "SPY")
    capped = BacktestEngine(prov, factory, execution=ex,
                            risk=RiskLimits(max_daily_loss=100)).run_day(prov.trading_days[0], "SPY")
    grid = prov.session_grid(prov.trading_days[0])
    assert len(capped.trades) < len(free.trades)
    assert capped.trades[-1].timestamp < grid[-1]       # stopped trading long before the close
    assert capped.mtm.iloc[-1] == pytest.approx(capped.final_pnl)   # flat: nothing left unrealized
    assert capped.final_pnl > free.final_pnl            # the cap saved money here


def test_max_open_contracts_enforced(prov):
    r = run_backtest("iron_condor", prov, risk=RiskLimits(max_open_contracts=2), underliers=["SPY"])
    assert all(dr.rejections for dr in r.day_results)     # 4-leg entry cannot fit


def test_progress_callback(prov):
    calls = []
    run_backtest("fixed_window", prov, underliers=["SPY"], progress=lambda u, d, n, t: calls.append((n, t)))
    assert calls == [(1, 2), (2, 2)]


def test_unknown_day_raises_value_error(prov):
    with pytest.raises(ValueError):
        BacktestEngine(prov, EnterAt).run_day(date(1999, 1, 4), "SPY")
