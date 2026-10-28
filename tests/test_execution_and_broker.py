# tests/test_execution_and_broker.py
from datetime import date

import pandas as pd
import pytest

from stratbt.broker import PaperBroker
from stratbt.execution import ExecutionModel
from stratbt.instruments import Instrument
from stratbt.risk import RiskLimits
from stratbt.strategy.base import Order

INST = Instrument("SPY", date(2024, 1, 5), 450, "CE")
OTHER = Instrument("SPY", date(2024, 1, 5), 451, "CE")
TS = pd.Timestamp("2024-01-02 10:00:00")


class TestExecutionModel:
    def test_defaults_are_next_second_and_free(self):
        m = ExecutionModel()
        assert m.delay_s == 1
        assert m.fill_price("BUY", 2.0) == 2.0
        assert m.fee(2.0, 3, 100) == 0.0

    def test_instant_preset(self):
        assert ExecutionModel.instant().delay_s == 0

    def test_buy_pays_and_sell_gives_up_spread(self):
        m = ExecutionModel(half_spread_bps=100)           # 1%
        assert m.fill_price("BUY", 2.0) == pytest.approx(2.02)
        assert m.fill_price("SELL", 2.0) == pytest.approx(1.98)

    def test_min_half_spread_floor_and_zero_clamp(self):
        m = ExecutionModel(min_half_spread=0.05)
        assert m.fill_price("BUY", 1.0) == pytest.approx(1.05)
        assert m.fill_price("SELL", 0.02) == 0.0

    def test_fee_combines_per_contract_and_bps(self):
        m = ExecutionModel(commission_per_contract=0.65, commission_bps=10)
        # 0.65*2 + 10bps of (3.0*2*100)
        assert m.fee(3.0, 2, 100) == pytest.approx(1.3 + 0.6)

    @pytest.mark.parametrize("kw", [{"delay_s": -1}, {"half_spread_bps": -1}, {"commission_bps": -0.1}])
    def test_negative_inputs_rejected(self, kw):
        with pytest.raises(ValueError):
            ExecutionModel(**kw)


class TestPaperBroker:
    def test_fill_applies_slippage_and_fee(self):
        b = PaperBroker(multiplier=100, execution=ExecutionModel(0, 100, 0, 1.0, 0))
        fill = b.submit(Order(INST, "BUY"), TS, 2.0)
        assert fill.price == pytest.approx(2.02) and fill.fee == 1.0
        assert b.holdings[INST] == (pytest.approx(2.02), 1)

    def test_equity_is_net_of_fees(self):
        b = PaperBroker(multiplier=100, execution=ExecutionModel(0, 0, 0, 1.0, 0))
        b.submit(Order(INST, "BUY"), TS, 2.0)
        b.submit(Order(INST, "SELL"), TS, 3.0)
        assert b.equity({}) == pytest.approx(100.0 - 2.0)      # +$100 gross, two $1 fees

    def test_buy_without_price_is_rejected_not_raised(self):
        b = PaperBroker()
        assert b.submit(Order(INST, "BUY"), TS, None) is None
        assert b.rejections[0].reason == "no price available"

    def test_closing_without_price_raises(self):
        b = PaperBroker()
        b.submit(Order(INST, "BUY"), TS, 1.0)
        with pytest.raises(RuntimeError, match="No fill price for SELL"):
            b.submit(Order(INST, "SELL"), TS, None)

    def test_short_requires_permission(self):
        with pytest.raises(ValueError):
            PaperBroker(allow_short=False).submit(Order(INST, "SELL"), TS, 1.0)
        b = PaperBroker(allow_short=True)
        b.submit(Order(INST, "SELL"), TS, 1.0)
        assert b.holdings[INST][1] == -1

    def test_flatten_orders_cover_shorts_and_sell_longs(self):
        b = PaperBroker(allow_short=True)
        b.submit(Order(INST, "SELL", 2), TS, 1.0)
        b.submit(Order(OTHER, "BUY"), TS, 1.0)
        got = {(o.instrument, o.side, o.qty) for o in b.flatten_orders()}
        assert got == {(INST, "BUY", 2), (OTHER, "SELL", 1)}


class TestRiskLimits:
    def test_max_open_contracts_rejects_openers_but_not_closers(self):
        b = PaperBroker(risk=RiskLimits(max_open_contracts=1))
        assert b.submit(Order(INST, "BUY"), TS, 1.0) is not None
        assert b.submit(Order(OTHER, "BUY"), TS, 1.0) is None
        assert "max_open_contracts" in b.rejections[-1].reason
        assert b.submit(Order(INST, "SELL"), TS, 1.0) is not None

    def test_max_order_qty(self):
        b = PaperBroker(risk=RiskLimits(max_order_qty=2))
        assert b.submit(Order(INST, "BUY", 3), TS, 1.0) is None
        assert b.submit(Order(INST, "BUY", 2), TS, 1.0) is not None

    def test_daily_loss_latches_halt_and_blocks_new_risk(self):
        b = PaperBroker(risk=RiskLimits(max_daily_loss=50))
        b.check_risk(-49.0)
        assert not b.halted
        b.check_risk(-50.0)
        assert b.halted
        b.check_risk(+500.0)
        assert b.halted                                   # latched for the day
        assert b.submit(Order(INST, "BUY"), TS, 1.0) is None
        assert "halted" in b.rejections[-1].reason

    @pytest.mark.parametrize("kw", [{"max_daily_loss": 0}, {"max_open_contracts": -1}])
    def test_limits_must_be_positive(self, kw):
        with pytest.raises(ValueError):
            RiskLimits(**kw)
