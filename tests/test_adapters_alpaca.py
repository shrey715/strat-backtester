# tests/test_adapters_alpaca.py
"""Alpaca adapter against fake SDK clients: nothing here touches a network or an account."""
import pytest

pytest.importorskip("alpaca")
from alpaca.trading.enums import PositionIntent  # noqa: E402

import stratbt.adapters as adapters  # noqa: E402
from stratbt import ExecutionModel  # noqa: E402
from stratbt.adapters import LiveSafety, LiveTradingNotConfirmed, REAL_MONEY_ENV, REAL_MONEY_TOKEN  # noqa: E402
from stratbt.adapters.alpaca import AlpacaBroker, AlpacaFeed  # noqa: E402
from stratbt.adapters.base import FeedError, osi_symbol  # noqa: E402
from stratbt.instruments import Instrument  # noqa: E402
from stratbt.live import AuditLog, LiveRunner, ReconciliationError  # noqa: E402
from stratbt.risk import RiskLimits  # noqa: E402
from stratbt.strategy.base import Order  # noqa: E402
from tests.venue_fakes import (FakeAlpacaOptions, FakeAlpacaStock, FakeAlpacaTrading, MarketSim, feed_kwargs)  # noqa: E402

LOSS = RiskLimits(max_daily_loss=500)


def make(mode="paper", fill="full", sim=None, safety=None, risk=LOSS, positions=None, **fake):
    sim = sim or MarketSim()
    trading = FakeAlpacaTrading(sim, fill=fill, positions=positions, **fake)
    opts = FakeAlpacaOptions(sim)
    feed = AlpacaFeed.create("SPY", trading_client=trading, stock_client=FakeAlpacaStock(sim), option_client=opts,
                             today=sim.day, **feed_kwargs(sim))
    safety = safety or LiveSafety(mode=mode)
    broker = AlpacaBroker(risk=risk, safety=safety, trading_client=trading, clock=sim.clock, sleep=sim.sleep).attach(feed)
    return sim, trading, opts, feed, broker


def run(sim, feed, broker, **kw):
    strat = sim.condor()
    broker.position.allow_short = strat.allow_short
    return LiveRunner(strat, feed, broker, ExecutionModel.instant(), **kw).run()


class TestEndToEnd:

    def test_paper_session_sends_a_correct_iron_condor_and_closes_it(self):
        sim, trading, _, feed, broker = make("paper")
        summary = run(sim, feed, broker)
        reqs = trading.submitted
        assert len(reqs) == 8 and summary.n_fills == 8 and summary.open_positions == 0 and broker.holdings == {}
        opening, closing = reqs[:4], reqs[4:]
        by_leg = {(r.symbol[-9:], str(r.side.value)): r for r in opening}
        # short body: sell to open; protective wings: buy to open
        assert sorted(r.position_intent.value for r in opening) == ["buy_to_open", "buy_to_open", "sell_to_open", "sell_to_open"]
        assert sorted(r.position_intent.value for r in closing) == ["buy_to_close", "buy_to_close", "sell_to_close", "sell_to_close"]
        for r in reqs:
            assert r.symbol.startswith("SPY") and len(r.symbol) == 18            # OSI: SPY + YYMMDD + C/P + 8 digits
            assert int(r.qty) == 1 and r.time_in_force.value == "day" and r.client_order_id
            assert round(r.limit_price, 2) == r.limit_price                       # on the penny tick
        assert len({r.client_order_id for r in reqs}) == 8                         # idempotency keys are unique

    def test_orders_are_marketable_limits_priced_exactly_off_the_quote_they_saw(self):
        sim, trading, _, feed, broker = make("paper")
        run(sim, feed, broker)
        assert {type(r).__name__ for r in trading.submitted} == {"LimitOrderRequest"}      # never a bare market order
        assert len(broker.sent) == len(trading.submitted) == 8
        for rec, req in zip(broker.sent, trading.submitted):
            want = rec["ask"] if rec["side"] == "BUY" else rec["bid"]                       # pay the ask / hit the bid
            assert req.limit_price == pytest.approx(want), rec
            assert req.side.value == rec["side"].lower() and req.client_order_id == rec["client_id"]

    def test_limit_price_uses_the_right_side_of_the_book(self):
        sim, trading, _, feed, broker = make("paper")
        inst = Instrument("SPY", sim.expiry, sim.strikes[len(sim.strikes) // 2], "CE")
        feed._price(inst); sim.sleep(1.0); feed._refresh()
        q = feed.quote(inst)
        assert broker._limit_price(Order(inst, "BUY"))[0] == pytest.approx(q.ask)
        assert broker._limit_price(Order(inst, "SELL"))[0] == pytest.approx(q.bid)
        mid = make("paper", safety=LiveSafety(mode="paper", price_at="mid"))
        mid[3]._price(inst); mid[0].sleep(1.0); mid[3]._refresh()
        assert mid[4]._limit_price(Order(inst, "BUY"))[0] == pytest.approx((mid[3].quote(inst).bid + mid[3].quote(inst).ask) / 2, abs=0.01)

    def test_dry_run_sends_nothing_but_behaves_like_a_session(self):
        sim, trading, _, feed, broker = make("dry-run")
        summary = run(sim, feed, broker)
        assert trading.submitted == [] and trading.cancelled == []
        assert summary.n_fills == 8 and broker.holdings == {}
        assert broker.sent == []                                                   # not even recorded as sent

    def test_quotes_are_fetched_in_one_batched_call_per_tick(self):
        sim, _, opts, feed, broker = make("dry-run")
        run(sim, feed, broker)
        assert opts.requests and max(len(r) for r in opts.requests) <= 8           # only watched contracts, batched
        assert len(opts.requests) <= 1210                                          # at most one call per second

    def test_audit_log_records_fills(self, tmp_path):
        sim, _, _, feed, broker = make("paper")
        summary = run(sim, feed, broker, audit=AuditLog(tmp_path / "a.jsonl"))
        assert sum(e["event"] == "fill" for e in summary.audit.events) == 8


def _inst(symbol):
    from stratbt.adapters.base import parse_osi
    return parse_osi(symbol)


class TestSafetyGates:

    def test_real_money_requires_the_confirmation_token(self):
        with pytest.raises(LiveTradingNotConfirmed, match="Real-money trading is off"):
            make("live", safety=LiveSafety(mode="live"))
        make("live", safety=LiveSafety(mode="live", confirm_real_money=REAL_MONEY_TOKEN))          # explicit: allowed

    def test_token_may_come_from_the_environment(self, monkeypatch):
        monkeypatch.setenv(REAL_MONEY_ENV, REAL_MONEY_TOKEN)
        make("live", safety=LiveSafety(mode="live"))
        monkeypatch.setenv(REAL_MONEY_ENV, "yes")
        with pytest.raises(LiveTradingNotConfirmed):
            make("live", safety=LiveSafety(mode="live"))

    @pytest.mark.parametrize("mode", ["paper", "live"])
    def test_paper_and_live_require_a_daily_loss_limit(self, mode):
        with pytest.raises(ValueError, match="daily loss limit"):
            make(mode, safety=LiveSafety(mode=mode, confirm_real_money=REAL_MONEY_TOKEN), risk=RiskLimits())

    def test_dry_run_needs_no_loss_limit(self):
        make("dry-run", risk=RiskLimits())

    def test_unknown_mode(self):
        with pytest.raises(ValueError, match="mode must be"):
            LiveSafety(mode="yolo")

    def test_live_mode_cannot_be_pointed_at_the_paper_endpoint(self):
        with pytest.raises(ValueError, match="would not trade real money"):
            AlpacaBroker(safety=LiveSafety(mode="live", confirm_real_money=REAL_MONEY_TOKEN), risk=LOSS, paper=True,
                         trading_client=object())

    def test_endpoint_selection_follows_mode(self, monkeypatch):
        import alpaca.trading.client as tc
        seen = []
        monkeypatch.setattr(tc, "TradingClient", lambda key, secret, paper: seen.append(paper) or object())
        monkeypatch.setenv("ALPACA_API_KEY", "k")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
        AlpacaBroker(safety=LiveSafety(mode="paper"), risk=LOSS)
        AlpacaBroker(safety=LiveSafety(mode="dry-run"), risk=LOSS)
        AlpacaBroker(safety=LiveSafety(mode="live", confirm_real_money=REAL_MONEY_TOKEN), risk=LOSS)
        assert seen == [True, True, False]

    def test_missing_credentials_are_explained(self, monkeypatch):
        monkeypatch.delenv("ALPACA_API_KEY", raising=False)
        monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
        with pytest.raises(ValueError, match="ALPACA_API_KEY"):
            AlpacaBroker(safety=LiveSafety(mode="paper"), risk=LOSS)

    def test_size_caps_default_to_one_contract(self):
        sim, trading, _, feed, broker = make("paper")
        assert broker.risk.max_order_qty == 1
        inst = Instrument("SPY", sim.expiry, sim.strikes[len(sim.strikes) // 2], "CE")
        feed._price(inst); sim.sleep(2); feed._refresh()
        assert broker.submit(Order(inst, "BUY", 2), None, 1.0) is None
        assert "max_order_qty" in broker.rejections[-1].reason and trading.submitted == []

    def test_caller_cannot_loosen_the_safety_caps(self):
        sim, _, _, _, broker = make("paper", risk=RiskLimits(max_daily_loss=100, max_order_qty=50, max_open_contracts=99))
        assert broker.risk.max_order_qty == 1 and broker.risk.max_open_contracts == 4

    def test_caps_can_be_raised_deliberately(self):
        _, _, _, _, broker = make("paper", safety=LiveSafety(mode="paper", max_order_qty=5, max_open_contracts=20),
                                  risk=RiskLimits(max_daily_loss=100, max_order_qty=5, max_open_contracts=20))
        assert broker.risk.max_order_qty == 5 and broker.risk.max_open_contracts == 20

    def test_wide_markets_are_refused(self):
        sim, trading, _, feed, broker = make("paper")
        sim.wide_spread = True
        summary = run(sim, feed, broker)
        assert trading.submitted == [] and summary.n_fills == 0
        assert any("spread too wide" in r.reason for r in broker.rejections)

    def test_stale_quotes_are_refused(self):
        sim, trading, _, feed, broker = make("paper", safety=LiveSafety(mode="paper", quote_max_age_s=2.0))
        inst = Instrument("SPY", sim.expiry, sim.strikes[len(sim.strikes) // 2], "CE")
        feed._price(inst); sim.sleep(1.0); feed._refresh()
        sim.sleep(10)                                                              # the feed stops updating
        assert broker.submit(Order(inst, "BUY"), None, 1.0) is None
        assert "stale quote" in broker.rejections[-1].reason and trading.submitted == []

    def test_no_quote_no_trade(self):
        sim, trading, _, feed, broker = make("paper")
        inst = Instrument("SPY", sim.expiry, sim.strikes[10], "CE")
        assert broker.submit(Order(inst, "BUY"), None, 1.0) is None and "no quote" in broker.rejections[-1].reason
        assert broker.submit(Order(inst, "BUY"), None, None) is None                # a missing price is a retry, not a crash


class TestFills:

    def test_unfilled_orders_are_cancelled_after_the_timeout(self):
        sim, trading, _, feed, broker = make("paper", fill="never", safety=LiveSafety(mode="paper", fill_timeout_s=3.0, poll_s=0.5))
        summary = run(sim, feed, broker)
        assert trading.submitted and len(trading.cancelled) == len(trading.submitted)     # every order cancelled, none left working
        assert summary.n_fills == 0 and broker.holdings == {}
        assert any("not filled" in r.reason for r in broker.rejections)

    def test_partial_fills_are_booked_as_filled(self):
        sim, trading, _, feed, broker = make("paper", fill="partial", fill_qty=2,
                                             safety=LiveSafety(mode="paper", max_order_qty=3, fill_timeout_s=2.0),
                                             risk=RiskLimits(max_daily_loss=500, max_order_qty=3))
        inst = Instrument("SPY", sim.expiry, sim.strikes[len(sim.strikes) // 2], "CE")
        feed._price(inst); sim.sleep(2); feed._refresh()
        fill = broker.submit(Order(inst, "BUY", 3), None, 1.0)
        assert fill is not None and fill.qty == 2 and broker.holdings[inst][1] == 2        # 2 of 3 filled; the rest cancelled
        assert len(trading.cancelled) == 1

    def test_venue_errors_are_a_rejection_not_a_crash(self):
        sim, trading, _, feed, broker = make("paper", fill="error")
        summary = run(sim, feed, broker)
        assert summary.n_fills == 0
        assert any("venue error" in r.reason and "503" in r.reason for r in broker.rejections)


class TestPositionsAndReconciliation:

    def _position(self, sim, k, t, qty, side, avg="1.25"):
        from types import SimpleNamespace
        from alpaca.trading.enums import PositionSide
        return SimpleNamespace(symbol=osi_symbol(Instrument("SPY", sim.expiry, k, t)), qty=str(qty),
                               side=PositionSide.SHORT if side == "short" else PositionSide.LONG, avg_entry_price=avg)

    def test_positions_are_parsed_with_sign_and_stocks_are_ignored(self):
        from types import SimpleNamespace
        sim = MarketSim()
        k = sim.strikes[10]
        pos = [self._position(sim, k, "CE", 2, "short"), self._position(sim, k, "PE", 1, "long"),
               SimpleNamespace(symbol="AAPL", qty="10", side="long", avg_entry_price="190")]
        _, _, _, _, broker = make("paper", sim=sim, positions=pos)
        got = broker.positions()
        assert got == {Instrument("SPY", sim.expiry, k, "CE"): (1.25, -2), Instrument("SPY", sim.expiry, k, "PE"): (1.25, 1)}

    def test_runner_syncs_the_venue_and_refuses_unknown_positions(self):
        sim = MarketSim()
        sim_, trading, _, feed, broker = make("paper", sim=sim, positions=[self._position(sim, sim.strikes[10], "CE", 1, "long")])
        with pytest.raises(ReconciliationError, match="Broker holds 1 position"):
            run(sim, feed, broker)
        assert trading.submitted == []                                              # nothing traded before the check passed

    def test_adopting_existing_positions_is_explicit(self):
        sim = MarketSim()
        _, trading, _, feed, broker = make("paper", sim=sim, positions=[self._position(sim, sim.strikes[10], "CE", 1, "long")])
        summary = run(sim, feed, broker, adopt_existing=True)
        assert any(e["event"] == "adopted_positions" for e in summary.audit.events)


class TestFeed:

    def test_chain_discovery_picks_the_front_expiry_and_whole_strikes(self):
        sim, _, _, feed, _ = make("paper")
        assert feed.expiry == sim.expiry and list(feed.strikes) == sim.strikes

    def test_session_ends_with_a_final_snapshot(self):
        sim, _, _, feed, _ = make("paper")
        snaps = list(feed.snapshots())
        assert snaps[-1].is_last and not any(s.is_last for s in snaps[:-1])
        assert snaps[0].ts.time().strftime("%H:%M") == "09:30" and len(snaps) in range(1195, 1210)

    def test_first_request_for_a_contract_is_none_then_it_streams(self):
        sim, _, _, feed, _ = make("paper")
        inst = Instrument("SPY", sim.expiry, sim.strikes[len(sim.strikes) // 2], "CE")
        it = feed.snapshots()
        first = next(it)
        assert first.option_price(inst) is None                                     # registers interest
        assert next(it).option_price(inst) is not None                              # then it is fetched

    def test_repeated_data_failures_stop_the_session_loudly(self):
        sim, _, opts, feed, _ = make("paper")
        feed.max_failures = 3
        it = feed.snapshots()
        next(it).option_price(Instrument("SPY", sim.expiry, sim.strikes[5], "CE"))
        opts.get_option_latest_quote = lambda req: (_ for _ in ()).throw(ConnectionError("down"))
        with pytest.raises(FeedError, match="3 times"):
            for _ in range(10):
                next(it)

    def test_a_feed_failure_with_positions_open_is_recorded_loudly(self):
        sim, trading, opts, feed, broker = make("paper")
        feed.max_failures = 2
        calls = {"n": 0}
        real = opts.get_option_latest_quote

        def flaky(req):
            calls["n"] += 1
            if calls["n"] > 120:
                raise ConnectionError("down")
            return real(req)
        opts.get_option_latest_quote = flaky
        runner = LiveRunner(sim.condor(), feed, broker, ExecutionModel.instant())
        broker.position.allow_short = True
        with pytest.raises(FeedError):
            runner.run()
        aborted = [e for e in runner.audit.events if e["event"] == "aborted"]
        assert aborted and aborted[0]["open_positions"]                               # says which positions are still open


def test_factory_wires_feed_and_broker(monkeypatch):
    sim = MarketSim()
    trading = FakeAlpacaTrading(sim)
    import stratbt.adapters.alpaca as mod
    real = mod.AlpacaFeed.create
    monkeypatch.setattr(mod.AlpacaFeed, "create", classmethod(lambda cls, ticker, **kw: real(
        ticker, trading_client=trading, stock_client=FakeAlpacaStock(sim), option_client=FakeAlpacaOptions(sim), today=sim.day,
        **feed_kwargs(sim))))
    monkeypatch.setattr(mod, "AlpacaBroker", lambda **kw: AlpacaBroker(trading_client=trading, clock=sim.clock, sleep=sim.sleep, **kw))
    feed, broker = adapters.build("alpaca", "SPY", mode="paper", max_daily_loss=300)
    assert broker.quote_source == feed.quote and broker.safety.mode == "paper" and broker.risk.max_daily_loss == 300
    with pytest.raises(ValueError, match="Unknown broker"):
        adapters.build("nope", "SPY")
