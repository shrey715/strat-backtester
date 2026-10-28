# tests/test_live.py
import json

import pandas as pd
import pytest

from stratbt import BacktestEngine, ExecutionModel, PaperBroker, RiskLimits, SyntheticProvider
from stratbt.broker import Broker, Fill
from stratbt.live import (AuditLog, KillSwitch, LiveRunner, ReconciliationError, ReplayFeed)
from stratbt.strategy import registry


@pytest.fixture(scope="module")
def prov():
    return SyntheticProvider(n_days=2, seed=5)


def _runner(prov, key="iron_condor", execution=None, risk=None, **kw):
    day = prov.trading_days[0]
    feed = ReplayFeed(prov, day, "SPY")
    spec = registry.get(key)
    strat = spec.factory(prov)("SPY", feed.expiry, feed.strikes)
    ex = execution or ExecutionModel.realistic()
    broker = PaperBroker(prov.multiplier("SPY"), ex, risk, allow_short=strat.allow_short)
    return LiveRunner(strat, feed, broker, ex, **kw), broker, day


def _fills(trades):
    return [(t.timestamp, t.instrument, t.side, round(t.price, 6), t.qty) for t in trades]


@pytest.mark.parametrize("key", ["iron_condor", "short_strangle", "fixed_window", "atm_straddle"])
@pytest.mark.parametrize("ex", [ExecutionModel.realistic(), ExecutionModel.instant()])
def test_runner_matches_backtester_fill_for_fill(prov, key, ex):
    """The live loop and the backtest loop must be the same machine."""
    runner, broker, day = _runner(prov, key, execution=ex)
    summary = runner.run()
    bt = BacktestEngine(prov, registry.get(key).factory(prov), execution=ex).run_day(day, "SPY")
    assert _fills(broker.position.trades) == _fills(bt.trades)
    assert summary.net_pnl == pytest.approx(bt.final_pnl)
    assert broker.holdings == {}                         # flat at the close


def test_matches_backtester_under_a_risk_halt(prov):
    ex, risk = ExecutionModel(half_spread_bps=200), RiskLimits(max_daily_loss=100)
    runner, broker, day = _runner(prov, "atm_straddle", execution=ex, risk=risk)
    summary = runner.run()
    bt = BacktestEngine(prov, registry.get("atm_straddle").factory(prov), execution=ex, risk=risk).run_day(day, "SPY")
    assert _fills(broker.position.trades) == _fills(bt.trades) and summary.halted
    assert any(e["event"] == "risk_halt" for e in summary.audit.events)


def test_audit_log_records_decisions_fills_and_end(prov, tmp_path):
    log = tmp_path / "audit.jsonl"
    runner, broker, _ = _runner(prov, audit=AuditLog(log))
    summary = runner.run()
    lines = [json.loads(l) for l in log.read_text().splitlines()]
    kinds = [l["event"] for l in lines]
    assert kinds[0] == "start" and kinds[-1] == "end" and "reconciled" in kinds
    assert kinds.count("fill") == summary.n_fills == len(broker.position.trades)
    assert "decision" in kinds
    assert all("at" in l for l in lines)                  # wall-clock stamped


def test_kill_switch_flattens_and_stops_early(prov):
    day = prov.trading_days[0]
    feed = ReplayFeed(prov, day, "SPY")
    kill = KillSwitch()

    class Feed(ReplayFeed):
        def snapshots(self):
            for i, snap in enumerate(super().snapshots()):
                if i == 2_000:                           # ~33 minutes in, the condor is on
                    kill.kill()
                yield snap

    f = Feed(prov, day, "SPY")
    strat = registry.get("iron_condor").factory(prov)("SPY", f.expiry, f.strikes)
    broker = PaperBroker(100, ExecutionModel(), allow_short=True)
    summary = LiveRunner(strat, f, broker, ExecutionModel(), kill=kill).run()
    assert summary.killed and broker.holdings == {}
    assert broker.position.trades[-1].timestamp < prov.session_grid(day)[3_000]      # stopped long before the close
    assert any(e["event"] == "kill_switch" for e in summary.audit.events)


def test_kill_switch_file(tmp_path):
    k = KillSwitch(tmp_path / "STOP")
    assert not k.triggered
    (tmp_path / "STOP").touch()
    assert k.triggered


class TestReconciliation:

    def test_state_is_saved_after_fills_and_roundtrips(self, prov, tmp_path):
        state = tmp_path / "state.json"
        runner, _, _ = _runner(prov, state_path=state)
        runner.run()
        assert json.loads(state.read_text())["holdings"] == []        # flat at the end

    def test_unknown_broker_positions_block_trading(self, prov, tmp_path):
        runner, broker, _ = _runner(prov, state_path=tmp_path / "s.json")
        from stratbt.instruments import Instrument
        broker.position.buy(pd.Timestamp("2024-01-02 09:30"), Instrument("SPY", runner.feed.expiry, int(runner.feed.strikes[10]), "CE"), 1.0)
        with pytest.raises(ReconciliationError, match="Broker holds 1 position"):
            runner.run()
        assert any(e["event"] == "reconcile_mismatch" for e in runner.audit.events)

    def test_adopt_existing_allows_start(self, prov):
        from stratbt.instruments import Instrument
        runner, broker, _ = _runner(prov, adopt_existing=True)
        broker.position.buy(pd.Timestamp("2024-01-02 09:30"), Instrument("SPY", runner.feed.expiry, int(runner.feed.strikes[10]), "CE"), 1.0)
        summary = runner.run()
        assert any(e["event"] == "adopted_positions" for e in summary.audit.events)


def test_any_broker_implementation_plugs_in(prov):
    """A live adapter only has to implement submit / holdings / equity."""
    class RecordingBroker(Broker):
        def __init__(self):
            self.sent, self._h = [], {}

        def submit(self, order, ts, ref_price):
            self.sent.append((order.instrument.filename, order.side))
            qty = order.qty if order.side == "BUY" else -order.qty
            e, q = self._h.get(order.instrument, (0.0, 0))
            if q + qty:
                self._h[order.instrument] = (ref_price, q + qty)
            else:
                self._h.pop(order.instrument, None)
            return Fill(ts, order.instrument, order.side, order.qty, ref_price, 0.0)

        @property
        def holdings(self):
            return dict(self._h)

        def equity(self, prices):
            return 0.0

    day = prov.trading_days[0]
    feed = ReplayFeed(prov, day, "SPY")
    strat = registry.get("long_strangle").factory(prov)("SPY", feed.expiry, feed.strikes)
    broker = RecordingBroker()
    summary = LiveRunner(strat, feed, broker, ExecutionModel()).run()
    assert len(broker.sent) == 4 and broker.holdings == {} and summary.n_fills == 4
