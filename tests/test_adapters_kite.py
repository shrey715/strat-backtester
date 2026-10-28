# tests/test_adapters_kite.py
"""Zerodha Kite adapter against a fake KiteConnect client on a NIFTY-like market: no network, no account."""
from types import SimpleNamespace

import pytest

pytest.importorskip("kiteconnect")
import stratbt.adapters as adapters  # noqa: E402
from stratbt import ExecutionModel  # noqa: E402
from stratbt.adapters import LiveSafety, LiveTradingNotConfirmed, REAL_MONEY_TOKEN  # noqa: E402
from stratbt.adapters import kite as kmod  # noqa: E402
from stratbt.adapters.kite import KiteBroker, KiteFeed, KiteInstruments  # noqa: E402
from stratbt.instruments import Instrument  # noqa: E402
from stratbt.live import LiveRunner  # noqa: E402
from stratbt.risk import RiskLimits  # noqa: E402
from stratbt.strategy.base import Order  # noqa: E402
from tests.venue_fakes import FakeKite, feed_kwargs, nse_sim  # noqa: E402

LOSS = RiskLimits(max_daily_loss=5000)
LOT = 75


def make(mode="dry-run", fill="full", product="NRML", risk=LOSS, safety=None, **fake):
    sim = nse_sim()
    kite = FakeKite(sim, lot_size=LOT, fill=fill, **fake)
    feed = KiteFeed.create("NIFTY", kite, today=sim.day, **feed_kwargs(sim))
    safety = safety or LiveSafety(mode=mode, confirm_real_money=REAL_MONEY_TOKEN, tick=0.05)
    broker = KiteBroker(kite, feed.table, lot_size=feed.lot_size, product=product, risk=risk, safety=safety,
                        clock=sim.clock, sleep=sim.sleep).attach(feed)
    return sim, kite, feed, broker


def run(sim, feed, broker, **kw):
    strat = sim.condor()
    broker.position.allow_short = strat.allow_short
    return LiveRunner(strat, feed, broker, ExecutionModel.instant(), **kw).run()


class TestInstrumentMaster:

    def test_chain_uses_only_option_rows_for_the_named_underlying(self):
        sim, kite, feed, _ = make()
        assert feed.expiry == sim.expiry and list(feed.strikes) == sim.strikes
        assert feed.lot_size == LOT and feed.tick_size == 0.05
        assert set(i.underlier for i in feed.table._by_inst) == {"NIFTY", "BANKNIFTY"}      # futures row ignored, others kept apart
        assert feed.table.row(Instrument("BANKNIFTY", sim.expiry, 1, "CE"))["lot_size"] == 15      # each underlying keeps its own lot size
        with pytest.raises(ValueError, match="Too few strikes"):                              # and a degenerate chain is refused, not traded
            feed.table.chain("BANKNIFTY", sim.day)

    def test_symbols_are_looked_up_never_constructed(self):
        sim, kite, feed, _ = make()
        inst = Instrument("NIFTY", sim.expiry, sim.strikes[10], "CE")
        assert feed.table.row(inst)["tradingsymbol"] == f"NIFTY{sim.expiry:%y%b}{sim.strikes[10]}CE".upper()
        with pytest.raises(ValueError, match="not in Kite's NFO instruments list"):
            feed.table.row(Instrument("NIFTY", sim.expiry, 1, "CE"))

    def test_unknown_underlying_is_explained(self):
        sim = nse_sim()
        with pytest.raises(ValueError, match="no upcoming NFO options for 'NOPE'"):
            KiteInstruments(FakeKite(sim), "NFO").chain("NOPE", sim.day)

    def test_wrong_index_key_is_caught_at_startup(self):
        sim = nse_sim()
        with pytest.raises(ValueError, match="no quote for 'NSE:NIFTY BANKK'"):
            KiteFeed.create("NIFTY", FakeKite(sim), index_key="NSE:NIFTY BANKK", today=sim.day, **feed_kwargs(sim))


class TestOrders:

    def test_live_iron_condor_end_to_end_with_correct_kite_parameters(self):
        sim, kite, feed, broker = make("live")
        summary = run(sim, feed, broker)
        assert len(kite.placed) == 8 and summary.n_fills == 8 and summary.open_positions == 0 and broker.holdings == {}
        for kw, rec in zip(kite.placed, broker.sent):
            assert kw["variety"] == "regular" and kw["exchange"] == "NFO" and kw["order_type"] == "LIMIT"
            assert kw["product"] == "NRML" and kw["validity"] == "DAY"
            assert kw["quantity"] == LOT                                              # 1 contract = 1 lot = 75 UNITS
            assert kw["tradingsymbol"] in kite.symbols and kw["transaction_type"] == rec["side"]
            assert kw["price"] == pytest.approx(rec["ask"] if rec["side"] == "BUY" else rec["bid"])
            assert round(kw["price"] / 0.05) * 0.05 == pytest.approx(kw["price"])      # on the 5-paise tick
            assert len(kw["tag"]) <= 20 and kw["tag"] == rec["client_id"][:20]
        assert len({kw["tag"] for kw in kite.placed}) == 8                            # unique idempotency tags

    def test_pnl_is_in_rupees_per_lot_with_brokerage_per_order(self):
        sim, kite, feed, broker = make("live")
        summary = run(sim, feed, broker)
        assert broker.position.multiplier == LOT
        assert broker.position.fees == pytest.approx(8 * 20.0)                         # flat Rs 20 per executed order
        gross = sum((t.price if t.side == "SELL" else -t.price) * LOT for t in broker.position.trades)
        assert summary.net_pnl == pytest.approx(gross - 160.0)

    def test_quantity_scales_with_contracts(self):
        sim, kite, feed, broker = make("live", safety=LiveSafety(mode="live", confirm_real_money=REAL_MONEY_TOKEN, tick=0.05, max_order_qty=3),
                                       risk=RiskLimits(max_daily_loss=5000, max_order_qty=3))
        inst = Instrument("NIFTY", sim.expiry, sim.strikes[len(sim.strikes) // 2], "CE")
        feed._price(inst); sim.sleep(1.0); feed._refresh()
        fill = broker.submit(Order(inst, "BUY", 3), None, 1.0)
        assert kite.placed[0]["quantity"] == 3 * LOT and fill.qty == 3

    def test_product_is_configurable(self):
        sim, kite, feed, broker = make("live", product="MIS")
        run(sim, feed, broker)
        assert {kw["product"] for kw in kite.placed} == {"MIS"}

    def test_dry_run_checks_margins_with_kite_but_places_nothing(self):
        sim, kite, feed, broker = make("dry-run")
        summary = run(sim, feed, broker)
        assert kite.placed == [] and kite.cancelled == []
        assert len(kite.margin_checks) == 8 and summary.n_fills == 8
        row = kite.margin_checks[0][0]
        assert set(row) >= {"exchange", "tradingsymbol", "transaction_type", "product", "order_type", "quantity", "price", "variety"}

    def test_a_margin_rejection_blocks_the_order(self):
        sim, kite, feed, broker = make("dry-run", margin_error=RuntimeError("Insufficient funds"))
        summary = run(sim, feed, broker)
        assert summary.n_fills == 0 and any("Insufficient funds" in r.reason for r in broker.rejections)


class TestSafety:

    def test_no_paper_environment(self):
        with pytest.raises(ValueError, match="no paper-trading environment"):
            make("paper")

    def test_live_requires_confirmation_and_a_loss_limit(self):
        with pytest.raises(LiveTradingNotConfirmed):
            make("live", safety=LiveSafety(mode="live", tick=0.05))
        with pytest.raises(ValueError, match="daily loss limit"):
            make("live", risk=RiskLimits())

    def test_size_caps_default_to_one_lot(self):
        sim, kite, feed, broker = make("live")
        inst = Instrument("NIFTY", sim.expiry, sim.strikes[len(sim.strikes) // 2], "CE")
        feed._price(inst); sim.sleep(1.0); feed._refresh()
        assert broker.submit(Order(inst, "BUY", 2), None, 1.0) is None and kite.placed == []

    def test_unfilled_orders_are_cancelled_with_the_right_variety_and_id(self):
        sim, kite, feed, broker = make("live", fill="never", safety=LiveSafety(mode="live", confirm_real_money=REAL_MONEY_TOKEN, tick=0.05, fill_timeout_s=2.0))
        summary = run(sim, feed, broker)
        assert kite.placed and len(kite.cancelled) == len(kite.placed) and summary.n_fills == 0
        assert all(v == "regular" for v, _ in kite.cancelled)

    def test_partial_fills_are_booked_in_whole_lots(self):
        sim, kite, feed, broker = make("live", fill="partial", fill_lots=2,
                                       safety=LiveSafety(mode="live", confirm_real_money=REAL_MONEY_TOKEN, tick=0.05, max_order_qty=3, fill_timeout_s=2.0),
                                       risk=RiskLimits(max_daily_loss=5000, max_order_qty=3))
        inst = Instrument("NIFTY", sim.expiry, sim.strikes[len(sim.strikes) // 2], "CE")
        feed._price(inst); sim.sleep(1.0); feed._refresh()
        fill = broker.submit(Order(inst, "BUY", 3), None, 1.0)
        assert fill.qty == 2 and broker.holdings[inst][1] == 2 and len(kite.cancelled) == 1         # 2 of 3 lots (150 of 225 units)

    @pytest.mark.parametrize("units,lots", [(100, 1), (149, 1), (150, 2), (40, 0)])
    def test_fractional_lot_fills_round_down_never_up(self, units, lots):
        """Booking a lot that wasn't fully filled would record a position we don't own."""
        sim, kite, feed, broker = make("live", fill="partial", fill_units=units,
                                       safety=LiveSafety(mode="live", confirm_real_money=REAL_MONEY_TOKEN, tick=0.05, max_order_qty=3, fill_timeout_s=2.0),
                                       risk=RiskLimits(max_daily_loss=5000, max_order_qty=3))
        inst = Instrument("NIFTY", sim.expiry, sim.strikes[len(sim.strikes) // 2], "CE")
        feed._price(inst); sim.sleep(1.0); feed._refresh()
        fill = broker.submit(Order(inst, "BUY", 3), None, 1.0)
        assert (fill.qty if fill else 0) == lots and (broker.holdings.get(inst, (0, 0))[1]) == lots

    def test_a_fill_smaller_than_one_lot_is_not_booked(self):
        sim, kite, feed, broker = make("live", fill="partial", fill_lots=0,
                                       safety=LiveSafety(mode="live", confirm_real_money=REAL_MONEY_TOKEN, tick=0.05, fill_timeout_s=2.0))
        kite.fill_lots = 0
        inst = Instrument("NIFTY", sim.expiry, sim.strikes[len(sim.strikes) // 2], "CE")
        feed._price(inst); sim.sleep(1.0); feed._refresh()
        assert broker.submit(Order(inst, "BUY"), None, 1.0) is None and broker.holdings == {}

    def test_venue_errors_are_rejections(self):
        sim, kite, feed, broker = make("live", fill="error")
        assert run(sim, feed, broker).n_fills == 0
        assert any("503" in r.reason for r in broker.rejections)


class TestFeed:

    def test_one_quote_call_per_second_with_underlying_and_watched_options_together(self):
        sim, kite, feed, broker = make("dry-run")
        run(sim, feed, broker)
        calls = kite.quote_calls[1:]                                  # [0] is the start-up key check
        assert 1190 <= len(calls) <= 1210                             # one per tick, never two
        assert all(c[0] == "NSE:NIFTY 50" and len(c) <= 61 and all(k.startswith("NFO:") for k in c[1:]) for c in calls)

    def test_quotes_use_best_bid_and_ask_not_last(self):
        sim, kite, feed, _ = make()
        inst = Instrument("NIFTY", sim.expiry, sim.strikes[len(sim.strikes) // 2], "CE")
        feed._price(inst); sim.sleep(1.0); feed._refresh()
        q = feed.quote(inst)
        assert q.bid and q.ask and q.ask > q.bid and q.mid == pytest.approx((q.bid + q.ask) / 2)

    def test_defaults_are_india_hours_with_a_flatten_buffer(self):
        sim = nse_sim()
        kw = {k: v for k, v in feed_kwargs(sim).items() if k not in ("session_open", "session_close", "tz")}
        f = KiteFeed.create("NIFTY", FakeKite(sim), today=sim.day, **kw)
        assert f.tz.key == "Asia/Kolkata" and f.session_open.strftime("%H:%M") == "09:15" and f.session_close.strftime("%H:%M") == "15:25"

    def test_missing_underlying_quote_is_a_feed_failure_not_a_silent_gap(self):
        sim, kite, feed, _ = make()
        kite.quote = lambda *keys: {}
        with pytest.raises(RuntimeError, match="no quote for the underlying"):
            feed._fetch([])


class TestPositions:

    def _rows(self, sim, kite):
        k = sim.strikes[10]
        sym = lambda t: f"NIFTY{sim.expiry:%y%b}{k}{t}".upper()
        return k, [dict(tradingsymbol=sym("CE"), exchange="NFO", product="NRML", quantity=-2 * LOT, average_price=61.5, last_price=60.0),
                   dict(tradingsymbol=sym("PE"), exchange="NFO", product="NRML", quantity=LOT, average_price=0, last_price=44.0),     # no avg: fall back
                   dict(tradingsymbol=sym("CE").replace("CE", "PE"), exchange="NFO", product="MIS", quantity=LOT, average_price=1.0, last_price=1.0),   # other product
                   dict(tradingsymbol="RELIANCE", exchange="NSE", product="CNC", quantity=10, average_price=1400, last_price=1410),                    # a stock
                   dict(tradingsymbol=sym("CE"), exchange="NFO", product="NRML", quantity=0, average_price=0, last_price=0)]

    def test_signed_unit_quantities_become_lots_and_unknown_cost_never_zero(self):
        sim = nse_sim()
        kite = FakeKite(sim, lot_size=LOT)
        kite.instruments()
        k, rows = self._rows(sim, kite)
        kite._positions = rows[:2] + rows[3:4]
        _, _, _, broker = make("live", positions=kite._positions)
        got = broker.positions()
        assert got == {Instrument("NIFTY", sim.expiry, k, "CE"): (61.5, -2), Instrument("NIFTY", sim.expiry, k, "PE"): (44.0, 1)}

    def test_only_the_configured_product_is_ours(self):
        sim = nse_sim()
        k = sim.strikes[10]
        sym = f"NIFTY{sim.expiry:%y%b}{k}PE".upper()
        _, _, _, broker = make("live", positions=[dict(tradingsymbol=sym, exchange="NFO", product="MIS", quantity=LOT, average_price=1.0, last_price=1.0)])
        assert broker.positions() == {}                                                  # an MIS position is not an NRML-run position


class TestCredentialsAndLogin:

    def test_missing_credentials_point_at_the_login_helper(self, monkeypatch):
        monkeypatch.delenv("KITE_API_KEY", raising=False)
        monkeypatch.delenv("KITE_ACCESS_TOKEN", raising=False)
        with pytest.raises(ValueError, match="stratbt kite-login"):
            kmod.make_client()

    def test_an_expired_token_is_explained(self, monkeypatch):
        import kiteconnect
        from kiteconnect.exceptions import TokenException

        class Expired:
            def __init__(self, **kw): pass
            def profile(self): raise TokenException("Incorrect `api_key` or `access_token`.")
        monkeypatch.setattr(kiteconnect, "KiteConnect", Expired)
        with pytest.raises(ValueError, match="expires every day"):
            kmod.make_client("key", "stale")

    def test_request_token_is_extracted_from_the_redirect_url(self):
        url = "https://example.com/callback?type=login&status=success&request_token=Zt9XyAbC&action=login"
        assert kmod.request_token_from(url) == "Zt9XyAbC" and kmod.request_token_from("  Zt9XyAbC \n") == "Zt9XyAbC"

    def test_token_exchange(self, monkeypatch):
        kite = FakeKite(nse_sim())
        monkeypatch.setenv("KITE_API_SECRET", "shh")
        tok = kmod.exchange_request_token("https://x/cb?request_token=RT1&status=success", kite=kite)
        assert tok == "TOKEN_FOR_TODAY" and kite.session_args == ("RT1", "shh")
        monkeypatch.delenv("KITE_API_SECRET")
        with pytest.raises(ValueError, match="KITE_API_SECRET"):
            kmod.exchange_request_token("RT1", kite=kite)

    def test_cli_login_flow_prints_the_export_line(self, monkeypatch, capsys):
        from stratbt import cli
        monkeypatch.setenv("KITE_API_KEY", "key")
        monkeypatch.setenv("KITE_API_SECRET", "shh")
        monkeypatch.setattr("builtins.input", lambda prompt="": "https://x/cb?request_token=RT9&status=success")
        fake = FakeKite(nse_sim())
        monkeypatch.setattr(kmod, "login_url", lambda *a, **k: fake.login_url())
        monkeypatch.setattr(kmod, "exchange_request_token", lambda raw, **k: kmod_token(raw))
        assert cli.main(["kite-login"]) == 0
        out = capsys.readouterr().out
        assert "kite.zerodha.com" in out and "export KITE_ACCESS_TOKEN=TOKEN-RT9" in out


def kmod_token(raw):
    return "TOKEN-" + kmod.request_token_from(raw)


def test_factory_wires_kite(monkeypatch):
    sim = nse_sim()
    kite = FakeKite(sim, lot_size=LOT)
    real_create = KiteFeed.create
    monkeypatch.setattr(KiteFeed, "create", classmethod(lambda cls, ticker, k=None, **kw: real_create(ticker, kite, today=sim.day, **feed_kwargs(sim), **kw)))
    feed, broker = adapters.build("kite", "NIFTY", mode="dry-run", kite=kite, product="NRML")
    assert broker.quote_source == feed.quote and broker.safety.tick == 0.05 and broker.position.multiplier == LOT
    with pytest.raises(ValueError, match="no paper-trading environment"):
        adapters.build("kite", "NIFTY", mode="paper", max_daily_loss=100, kite=kite)
