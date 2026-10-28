# tests/test_structures.py
from datetime import date

import numpy as np
import pandas as pd
import pytest

from stratbt.instruments import Instrument
from stratbt.strategy import registry
from stratbt.strategy.structures import IronCondor, LongStrangle, ShortStraddle, ShortStrangle

EXP = date(2024, 1, 5)
STRIKES = np.arange(440, 461)
TS0 = pd.Timestamp("2024-01-02 09:30:00")


def at(sec):
    return TS0 + pd.Timedelta(seconds=sec)


def px(_inst):
    return 1.0


def legs(orders):
    return sorted((o.instrument.option_type, o.instrument.strike, o.side) for o in orders)


def mk(cls, **kw):
    return cls("SPY", EXP, STRIKES, session_open=TS0.time(), session_close=pd.Timestamp("15:59:59").time(), **kw)


def test_no_entry_before_delay_or_after_exit():
    s = mk(ShortStraddle, entry_delay_s=300, exit_buffer_s=300)
    assert s.on_second(at(299), 450.0, px, {}) == []
    assert s.on_second(at(23_400 - 300), 450.0, px, {}) == []


def test_short_straddle_sells_atm_legs():
    s = mk(ShortStraddle)
    assert legs(s.on_second(at(300), 450.2, px, {})) == [("CE", 450, "SELL"), ("PE", 450, "SELL")]
    assert s.allow_short


def test_long_strangle_buys_wings():
    s = mk(LongStrangle, width=3)
    assert legs(s.on_second(at(300), 450.0, px, {})) == [("CE", 453, "BUY"), ("PE", 447, "BUY")]


def test_short_strangle_sells_wings():
    s = mk(ShortStrangle, width=2)
    assert legs(s.on_second(at(300), 450.0, px, {})) == [("CE", 452, "SELL"), ("PE", 448, "SELL")]


def test_iron_condor_has_four_legs_and_defined_risk():
    s = mk(IronCondor, width=2, wing=3)
    got = legs(s.on_second(at(300), 450.0, px, {}))
    assert got == [("CE", 452, "SELL"), ("CE", 455, "BUY"), ("PE", 445, "BUY"), ("PE", 448, "SELL")]


def test_enters_only_once():
    s = mk(ShortStraddle)
    assert s.on_second(at(300), 450.0, px, {})
    assert s.on_second(at(301), 450.0, px, {}) == []        # even though still flat in this call


def test_waits_if_any_leg_unpriced():
    s = mk(IronCondor)
    unpriced = lambda inst: None if inst.strike == 454 else 1.0
    assert s.on_second(at(300), 450.0, unpriced, {}) == []
    assert s.on_second(at(301), 450.0, px, {})              # retries once priced


def test_skips_entry_if_strikes_not_listed():
    s = mk(IronCondor, width=10, wing=10)                   # wings fall off the 440..460 chain
    assert s.on_second(at(300), 450.0, px, {}) == []


def test_exit_orders_close_longs_and_cover_shorts():
    s = mk(IronCondor)
    held = {Instrument("SPY", EXP, 452, "CE"): (1.0, -1), Instrument("SPY", EXP, 455, "CE"): (0.5, 1)}
    out = s.on_second(at(23_400 - 300), 450.0, px, held)
    assert legs(out) == [("CE", 452, "BUY"), ("CE", 455, "SELL")]


def test_stop_loss_and_take_profit_close_everything():
    inst = Instrument("SPY", EXP, 450, "CE")
    short = {inst: (2.0, -1)}
    s = mk(ShortStraddle, stop_loss=1.0)
    assert s.on_second(at(600), 450.0, lambda _: 3.5, short)[0].side == "BUY"        # lost 1.5 >= 1.0
    assert s.on_second(at(601), 450.0, lambda _: 2.5, short) == []                   # lost 0.5
    s = mk(ShortStraddle, take_profit=1.0)
    assert s.on_second(at(600), 450.0, lambda _: 0.9, short)[0].side == "BUY"        # won 1.1


def test_no_reentry_after_stop():
    s = mk(ShortStraddle, stop_loss=1.0)
    s.on_second(at(300), 450.0, px, {})
    assert s.on_second(at(700), 450.0, px, {}) == []


class TestRegistry:
    def test_builtins_present(self):
        assert {"atm_straddle", "fixed_window", "short_straddle", "long_strangle",
                "short_strangle", "iron_condor"} <= set(registry.available())

    def test_unknown_key(self):
        with pytest.raises(KeyError, match="Unknown strategy"):
            registry.get("nope")

    def test_param_validation(self):
        spec = registry.get("iron_condor")
        with pytest.raises(ValueError, match="unknown parameter"):
            spec.resolve(bogus=1)
        with pytest.raises(ValueError, match="outside"):
            spec.resolve(width=0)
        assert spec.resolve(width="3")["width"] == 3

    def test_factory_injects_provider_session(self):
        from stratbt import SyntheticProvider
        prov = SyntheticProvider(n_days=1)
        strat = registry.get("fixed_window").factory(prov)("SPY", EXP, STRIKES)
        assert strat.SESSION_OPEN == prov.session_open and strat.SESSION_CLOSE == prov.session_close
        atm = registry.get("atm_straddle").factory(prov)("SPY", EXP, STRIKES)
        assert atm.DAY_CLOSE_TIME == prov.session_close

    def test_register_custom_and_duplicate(self):
        from stratbt import BaseStrategy
        class Noop(BaseStrategy):
            def __init__(self, underlier, expiry, strikes): pass
            def on_second(self, *a): return []
        spec = registry.StrategySpec("noop_test", "Noop", Noop, "does nothing")
        registry.register(spec)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(spec)
        assert registry.get("noop_test") is spec
