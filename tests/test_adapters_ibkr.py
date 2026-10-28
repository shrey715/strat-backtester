# tests/test_adapters_ibkr.py
"""Interactive Brokers adapter against a fake ib_async session: no TWS, no network."""
from types import SimpleNamespace

import pytest

pytest.importorskip("ib_async")
from stratbt import ExecutionModel  # noqa: E402
from stratbt.adapters import LiveSafety, REAL_MONEY_TOKEN  # noqa: E402
from stratbt.adapters.ibkr import IBKRBroker, IBKRFeed  # noqa: E402
from stratbt.instruments import Instrument  # noqa: E402
from stratbt.live import LiveRunner  # noqa: E402
from stratbt.risk import RiskLimits  # noqa: E402
from tests.venue_fakes import FakeIB, MarketSim, feed_kwargs  # noqa: E402

LOSS = RiskLimits(max_daily_loss=500)


def make(mode="paper", accounts=("DU1234567",), fill="full", **fake):
    sim = MarketSim()
    ib = FakeIB(sim, accounts=accounts, fill=fill, **fake)
    safety = LiveSafety(mode=mode, confirm_real_money=REAL_MONEY_TOKEN)
    broker = IBKRBroker(ib, risk=LOSS, safety=safety, clock=sim.clock)
    feed = IBKRFeed.create("SPY", ib, mode=mode, today=sim.day, **feed_kwargs(sim))
    return sim, ib, feed, broker.attach(feed)


def run(sim, feed, broker):
    strat = sim.condor()
    broker.position.allow_short = strat.allow_short
    return LiveRunner(strat, feed, broker, ExecutionModel.instant()).run()


def test_paper_session_end_to_end():
    sim, ib, feed, broker = make("paper")
    summary = run(sim, feed, broker)
    assert len(ib.orders) == 8 and summary.n_fills == 8 and summary.open_positions == 0
    for (contract, order), rec in zip(ib.orders, broker.sent):
        assert contract.secType == "OPT" and contract.symbol == "SPY" and contract.currency == "USD" and contract.multiplier == "100"
        assert contract.right in ("C", "P") and len(contract.lastTradeDateOrContractMonth) == 8
        assert order.orderType == "LMT" and order.tif == "DAY" and int(order.totalQuantity) == 1
        assert order.action == rec["side"] and order.lmtPrice == pytest.approx(rec["ask"] if rec["side"] == "BUY" else rec["bid"])
        assert order.orderRef == rec["client_id"]


def test_dry_run_sends_nothing():
    sim, ib, feed, broker = make("dry-run")
    summary = run(sim, feed, broker)
    assert ib.orders == [] and summary.n_fills == 8


def test_unfilled_orders_are_cancelled():
    sim, ib, feed, broker = make("paper", fill="never")
    broker.safety = LiveSafety(mode="paper", fill_timeout_s=2.0)
    summary = run(sim, feed, broker)
    assert ib.orders and len(ib.cancelled) == len(ib.orders) and summary.n_fills == 0


class TestModeMustMatchTheAccount:
    """A wrong port must never silently turn a rehearsal into real trading, or the reverse."""

    @pytest.mark.parametrize("mode", ["dry-run", "paper"])
    def test_rehearsal_modes_refuse_a_real_account(self, mode):
        with pytest.raises(PermissionError, match="non-paper account"):
            make(mode, accounts=("U7654321",))

    def test_live_refuses_a_paper_account(self):
        with pytest.raises(PermissionError, match="include a paper account"):
            make("live", accounts=("DU1234567",))

    def test_live_with_a_real_account_is_allowed_with_confirmation(self):
        make("live", accounts=("U7654321",))

    def test_default_ports_follow_the_mode(self):
        from stratbt.adapters.ibkr import PAPER_PORTS
        assert PAPER_PORTS["dry-run"] == PAPER_PORTS["paper"] == 7497 and PAPER_PORTS["live"] == 7496


def test_positions_convert_cost_per_contract_to_a_price():
    sim = MarketSim()
    k = sim.strikes[10]
    contract = SimpleNamespace(secType="OPT", symbol="SPY", lastTradeDateOrContractMonth=sim.expiry.strftime("%Y%m%d"),
                               strike=float(k), right="C", multiplier="100")
    stock = SimpleNamespace(secType="STK", symbol="SPY", lastTradeDateOrContractMonth="", strike=0.0, right="", multiplier="")
    _, _, _, broker = make("paper", positions=[SimpleNamespace(contract=contract, position=-2, avgCost=150.0),
                                               SimpleNamespace(contract=stock, position=100, avgCost=450.0)])
    assert broker.positions() == {Instrument("SPY", sim.expiry, k, "CE"): (1.5, -2)}      # 150 per contract / 100 = $1.50; stock ignored


def test_bad_ibkr_data_values_are_treated_as_missing():
    from stratbt.adapters.ibkr import _px
    assert _px(-1) is None and _px(float("nan")) is None and _px(None) is None and _px(0) is None and _px(1.25) == 1.25


def test_feed_streams_quotes_and_discovers_the_chain():
    sim, ib, feed, _ = make("paper")
    assert feed.expiry == sim.expiry and list(feed.strikes) == sim.strikes
    inst = Instrument("SPY", sim.expiry, sim.strikes[len(sim.strikes) // 2], "CE")
    it = feed.snapshots()
    assert next(it).option_price(inst) is None
    assert next(it).option_price(inst) is not None and ib.subscriptions
