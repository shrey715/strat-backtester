# tests/test_cli_live.py
"""`stratbt live`: the confirmation flow and guard-rails, with the venue replaced by fakes."""
import signal

import pytest

pytest.importorskip("alpaca")
import stratbt.adapters as adapters  # noqa: E402
from stratbt import cli  # noqa: E402
from stratbt.adapters import LiveSafety, REAL_MONEY_ENV, REAL_MONEY_TOKEN  # noqa: E402
from stratbt.adapters.alpaca import AlpacaBroker, AlpacaFeed  # noqa: E402
from stratbt.risk import RiskLimits  # noqa: E402
from tests.venue_fakes import FakeAlpacaOptions, FakeAlpacaStock, FakeAlpacaTrading, MarketSim, feed_kwargs  # noqa: E402


@pytest.fixture
def venue(monkeypatch):
    """Patch adapters.build to return fake-backed adapters; record what it was asked for."""
    calls, state = [], {}
    old = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}

    def fake_build(broker, ticker, mode="dry-run", max_daily_loss=None, max_open_contracts=None, max_order_qty=None,
                   confirm_real_money=None, allow_short=True, **venue_kw):
        calls.append(dict(broker=broker, ticker=ticker, mode=mode, max_daily_loss=max_daily_loss,
                          max_open_contracts=max_open_contracts, max_order_qty=max_order_qty,
                          confirm_real_money=confirm_real_money, venue=venue_kw))
        sim = MarketSim()
        trading = FakeAlpacaTrading(sim)
        feed = AlpacaFeed.create(ticker, trading_client=trading, stock_client=FakeAlpacaStock(sim),
                                 option_client=FakeAlpacaOptions(sim), today=sim.day, **feed_kwargs(sim))
        kw = {k: v for k, v in dict(max_order_qty=max_order_qty, max_open_contracts=max_open_contracts).items() if v}
        b = AlpacaBroker(risk=RiskLimits(max_daily_loss=max_daily_loss),
                         safety=LiveSafety(mode=mode, confirm_real_money=confirm_real_money, **kw),
                         trading_client=trading, clock=sim.clock, sleep=sim.sleep)
        state.update(trading=trading, broker=b)
        return feed, b.attach(feed)

    monkeypatch.setattr(adapters, "build", fake_build)
    monkeypatch.delenv(REAL_MONEY_ENV, raising=False)
    yield SimpleCalls(calls, state)
    for s, h in old.items():
        signal.signal(s, h)


class SimpleCalls:
    def __init__(self, calls, state):
        self.calls, self.state = calls, state


def live(args, tmp_path, capsys):
    rc = cli.main(["live", *args, "--log-dir", str(tmp_path)])
    return rc, capsys.readouterr().out


def test_defaults_to_dry_run_and_says_nothing_is_sent(venue, tmp_path, capsys):
    rc, out = live(["iron_condor", "--broker", "alpaca", "--ticker", "SPY"], tmp_path, capsys)
    assert rc == 0 and "NO ORDERS ARE SENT" in out and venue.calls[0]["mode"] == "dry-run"
    assert venue.state["trading"].submitted == []
    assert any(p.name.endswith(".jsonl") for p in tmp_path.iterdir())                  # an audit trail is always written


def test_paper_and_live_require_a_loss_limit(venue, tmp_path, capsys):
    for mode in ("paper", "live"):
        with pytest.raises(SystemExit, match="--max-daily-loss is required"):
            cli.main(["live", "iron_condor", "--broker", "alpaca", "--ticker", "SPY", "--mode", mode, "--log-dir", str(tmp_path)])
    assert venue.calls == []                                                           # refused before touching any venue


def test_live_without_confirmation_is_refused_when_not_interactive(venue, tmp_path, monkeypatch):
    monkeypatch.setattr("sys.stdin", type("NoTTY", (), {"isatty": lambda self: False})())
    with pytest.raises(SystemExit, match=REAL_MONEY_ENV):
        cli.main(["live", "iron_condor", "--broker", "alpaca", "--ticker", "SPY", "--mode", "live", "--max-daily-loss", "100",
                  "--log-dir", str(tmp_path)])
    assert venue.calls == []


def test_live_with_the_environment_token_passes_it_through(venue, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv(REAL_MONEY_ENV, REAL_MONEY_TOKEN)
    rc, out = live(["iron_condor", "--broker", "alpaca", "--ticker", "SPY", "--mode", "live", "--max-daily-loss", "150"], tmp_path, capsys)
    assert rc == 0 and "LIVE — REAL MONEY" in out
    assert venue.calls[0]["confirm_real_money"] == REAL_MONEY_TOKEN and venue.calls[0]["max_daily_loss"] == 150


def test_interactive_confirmation_must_be_typed_exactly(venue, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", type("TTY", (), {"isatty": lambda self: True})())
    monkeypatch.setattr("builtins.input", lambda prompt="": "yes")
    with pytest.raises(adapters.LiveTradingNotConfirmed):
        cli.main(["live", "iron_condor", "--broker", "alpaca", "--ticker", "SPY", "--mode", "live", "--max-daily-loss", "100",
                  "--log-dir", str(tmp_path)])
    monkeypatch.setattr("builtins.input", lambda prompt="": REAL_MONEY_TOKEN)
    assert cli.main(["live", "iron_condor", "--broker", "alpaca", "--ticker", "SPY", "--mode", "live", "--max-daily-loss", "100",
                     "--log-dir", str(tmp_path)]) == 0


def test_size_caps_default_small_and_are_passed_through(venue, tmp_path, capsys):
    live(["iron_condor", "--broker", "alpaca", "--ticker", "SPY"], tmp_path, capsys)
    assert venue.calls[0]["max_order_qty"] == 1 and venue.calls[0]["max_open_contracts"] == 4
    live(["iron_condor", "--broker", "alpaca", "--ticker", "SPY", "--max-order-qty", "3", "--max-open-contracts", "12"], tmp_path, capsys)
    assert venue.calls[1]["max_order_qty"] == 3 and venue.calls[1]["max_open_contracts"] == 12


def test_long_only_strategies_cannot_short(venue, tmp_path, capsys):
    live(["fixed_window", "--broker", "alpaca", "--ticker", "SPY"], tmp_path, capsys)
    assert venue.state["broker"].position.allow_short is False
    live(["short_straddle", "--broker", "alpaca", "--ticker", "SPY"], tmp_path, capsys)
    assert venue.state["broker"].position.allow_short is True


def test_kill_file_stops_and_flattens(venue, tmp_path, capsys):
    stop = tmp_path / "STOP"
    stop.touch()
    rc, out = live(["iron_condor", "--broker", "alpaca", "--ticker", "SPY", "--kill-file", str(stop)], tmp_path, capsys)
    assert "KILLED" in out and rc == 0 and venue.state["broker"].holdings == {}


def test_strategy_parameters_reach_the_strategy(venue, tmp_path, capsys):
    rc, out = live(["short_strangle", "--broker", "alpaca", "--ticker", "SPY", "--param", "width=3"], tmp_path, capsys)
    assert rc == 0 and "Short Strangle" in out


def test_ibkr_options_are_forwarded(venue, tmp_path, capsys):
    live(["iron_condor", "--broker", "ibkr", "--ticker", "SPY", "--ib-port", "4002", "--ib-exchange", "SMART", "--ib-delayed"], tmp_path, capsys)
    assert venue.calls[0]["venue"] == {"port": 4002, "exchange": "SMART", "delayed": True}
