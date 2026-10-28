# tests/test_adapters_public.py
"""Public.com adapter against a fake SDK client: nothing here touches the network or an account."""
from decimal import Decimal
from types import SimpleNamespace

import pytest

pytest.importorskip("public_api_sdk")
from public_api_sdk import InstrumentType, OpenCloseIndicator, OrderSide, OrderType, TimeInForce  # noqa: E402

from stratbt import ExecutionModel  # noqa: E402
from stratbt.adapters import LiveSafety, LiveTradingNotConfirmed, REAL_MONEY_TOKEN  # noqa: E402
from stratbt.adapters.base import osi_symbol  # noqa: E402
from stratbt.adapters.public import PublicBroker, PublicFeed  # noqa: E402
from stratbt.instruments import Instrument  # noqa: E402
from stratbt.live import LiveRunner  # noqa: E402
from stratbt.risk import RiskLimits  # noqa: E402
from stratbt.strategy.base import Order  # noqa: E402
from tests.venue_fakes import FakePublic, MarketSim, feed_kwargs  # noqa: E402

LOSS = RiskLimits(max_daily_loss=500)


def make(mode="dry-run", fill="full", **fake):
    sim = MarketSim()
    client = FakePublic(sim, fill=fill, **fake)
    feed = PublicFeed.create("SPY", client=client, today=sim.day, **feed_kwargs(sim))
    broker = PublicBroker(client=client, risk=LOSS, safety=LiveSafety(mode=mode, confirm_real_money=REAL_MONEY_TOKEN),
                          clock=sim.clock, sleep=sim.sleep).attach(feed)
    return sim, client, feed, broker


def run(sim, feed, broker, **kw):
    strat = sim.condor()
    broker.position.allow_short = strat.allow_short
    return LiveRunner(strat, feed, broker, ExecutionModel.instant(), **kw).run()


def test_public_has_no_paper_environment():
    with pytest.raises(ValueError, match="no paper-trading environment"):
        PublicBroker(client=object(), risk=LOSS, safety=LiveSafety(mode="paper"))


def test_live_needs_the_confirmation_token():
    with pytest.raises(LiveTradingNotConfirmed):
        PublicBroker(client=object(), risk=LOSS, safety=LiveSafety(mode="live"))


def test_dry_run_preflights_every_order_and_never_places_one():
    sim, client, feed, broker = make("dry-run")
    summary = run(sim, feed, broker)
    assert client.placed == [] and client.cancelled == []                    # nothing sent
    assert len(client.preflights) == 8 and summary.n_fills == 8 and broker.holdings == {}


def test_preflight_rejection_blocks_the_order():
    sim, client, feed, broker = make("dry-run", preflight_error=RuntimeError("insufficient options approval level"))
    summary = run(sim, feed, broker)
    assert summary.n_fills == 0 and client.placed == []
    assert any("insufficient options approval" in r.reason for r in broker.rejections)


def test_live_orders_are_correct_single_leg_option_limit_orders():
    sim, client, feed, broker = make("live")
    summary = run(sim, feed, broker)
    assert len(client.placed) == 8 and summary.n_fills == 8 and summary.open_positions == 0
    for req, rec in zip(client.placed, broker.sent):
        assert req.instrument.type == InstrumentType.OPTION
        assert req.instrument.symbol.startswith("SPY") and len(req.instrument.symbol) == 18      # OSI
        assert req.order_type == OrderType.LIMIT and req.expiration.time_in_force == TimeInForce.DAY
        assert req.quantity == 1 and isinstance(req.limit_price, Decimal)
        assert req.limit_price == Decimal(str(rec["ask"] if rec["side"] == "BUY" else rec["bid"]))
        assert req.order_side == (OrderSide.BUY if rec["side"] == "BUY" else OrderSide.SELL)
    opening, closing = client.placed[:4], client.placed[4:]
    assert {r.open_close_indicator for r in opening} == {OpenCloseIndicator.OPEN}
    assert {r.open_close_indicator for r in closing} == {OpenCloseIndicator.CLOSE}
    assert len({r.order_id for r in client.placed}) == 8                    # distinct UUIDs: safe to retry without double-sending


def test_unfilled_live_orders_are_cancelled():
    sim, client, feed, broker = make("live", fill="never")
    broker.safety = LiveSafety(mode="live", confirm_real_money=REAL_MONEY_TOKEN, fill_timeout_s=2.0)
    summary = run(sim, feed, broker)
    assert client.placed and len(client.cancelled) == len(client.placed) and summary.n_fills == 0


def test_positions_with_short_quantities_and_unknown_cost_basis():
    sim = MarketSim()
    k = sim.strikes[10]
    sym = lambda t: osi_symbol(Instrument("SPY", sim.expiry, k, t))
    pos = [SimpleNamespace(instrument=SimpleNamespace(symbol=sym("CE"), type=InstrumentType.OPTION), quantity=Decimal("-2"),
                           cost_basis=SimpleNamespace(unit_cost=Decimal("1.5")), last_price=Decimal("1.4")),
           SimpleNamespace(instrument=SimpleNamespace(symbol=sym("PE"), type=InstrumentType.OPTION), quantity=Decimal("1"),
                           cost_basis=None, last_price=Decimal("0.9")),                      # no cost basis: must not become 0
           SimpleNamespace(instrument=SimpleNamespace(symbol="AAPL", type=InstrumentType.EQUITY), quantity=Decimal("10"),
                           cost_basis=None, last_price=Decimal("190"))]
    _, _, _, broker = make("dry-run", positions=pos)
    got = broker.positions()
    assert got == {Instrument("SPY", sim.expiry, k, "CE"): (1.5, -2), Instrument("SPY", sim.expiry, k, "PE"): (0.9, 1)}


def test_chain_discovery():
    sim, _, feed, _ = make()
    assert feed.expiry == sim.expiry and list(feed.strikes) == sim.strikes


def test_missing_credentials_are_explained(monkeypatch):
    monkeypatch.delenv("PUBLIC_API_SECRET_KEY", raising=False)
    monkeypatch.delenv("PUBLIC_ACCOUNT_NUMBER", raising=False)
    with pytest.raises(ValueError, match="PUBLIC_API_SECRET_KEY"):
        PublicBroker(risk=LOSS, safety=LiveSafety(mode="dry-run"))
