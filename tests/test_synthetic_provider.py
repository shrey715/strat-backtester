# tests/test_synthetic_provider.py
from datetime import date

import numpy as np
import pytest

from stratbt.instruments import Instrument
from stratbt.providers import SyntheticProvider
from stratbt.providers.synthetic import bs_price


@pytest.fixture(scope="module")
def prov():
    return SyntheticProvider(n_days=6, start=date(2024, 1, 3), seed=1)


def test_trading_days_skip_weekends(prov):
    assert len(prov.trading_days) == 6
    assert all(d.weekday() < 5 for d in prov.trading_days)
    assert prov.trading_days[0] == date(2024, 1, 3)


def test_same_seed_same_market():
    a = SyntheticProvider(n_days=2, seed=3).underlying_prices(date(2024, 1, 2), "SPY")
    b = SyntheticProvider(n_days=2, seed=3).underlying_prices(date(2024, 1, 2), "SPY")
    c = SyntheticProvider(n_days=2, seed=4).underlying_prices(date(2024, 1, 2), "SPY")
    assert a.equals(b)
    assert not a.equals(c)


def test_session_grid_and_no_nans(prov):
    day = prov.trading_days[0]
    s = prov.underlying_prices(day, "SPY")
    assert s.index.equals(prov.session_grid(day))
    assert len(s) == 23_400                      # 09:30:00 -> 15:59:59
    assert not s.isna().any()
    assert (s > 0).all()


def test_front_expiry_is_next_friday_on_or_after(prov):
    assert prov.front_expiry(date(2024, 1, 3), "SPY") == date(2024, 1, 5)   # Wed -> Fri
    assert prov.front_expiry(date(2024, 1, 5), "SPY") == date(2024, 1, 5)   # Fri -> same day


def test_strikes_bracket_the_day(prov):
    day = prov.trading_days[0]
    k = prov.strikes(day, "SPY", prov.front_expiry(day, "SPY"))
    s = prov.underlying_prices(day, "SPY")
    assert k.dtype.kind == "i"
    assert k[0] < s.min() and k[-1] > s.max()
    assert (np.diff(k) == 1).all()


def test_unknown_inputs_raise(prov):
    day = prov.trading_days[0]
    with pytest.raises(ValueError):
        prov.front_expiry(day, "NOPE")
    with pytest.raises(ValueError):
        prov.strikes(day, "SPY", date(2030, 1, 4))


def test_option_prices_valid_and_ordered(prov):
    day = prov.trading_days[0]
    exp = prov.front_expiry(day, "SPY")
    atm = int(round(prov.underlying_prices(day, "SPY").iloc[0]))
    call = prov.option_prices(day, Instrument("SPY", exp, atm, "CE"))
    put = prov.option_prices(day, Instrument("SPY", exp, atm, "PE"))
    assert len(call) == 23_400 and not call.isna().any()
    assert (call >= 0.01).all() and (put >= 0.01).all()
    # a deep OTM call is cheaper than the ATM call
    otm = prov.option_prices(day, Instrument("SPY", exp, atm + 8, "CE"))
    assert otm.iloc[0] < call.iloc[0]


def test_missing_contracts_return_none(prov):
    day = prov.trading_days[0]
    exp = prov.front_expiry(day, "SPY")
    assert prov.option_prices(day, Instrument("SPY", exp, 1, "CE")) is None
    assert prov.option_prices(day, Instrument("SPY", date(2030, 1, 4), 450, "CE")) is None


def test_black_scholes_put_call_parity():
    spot, k, t, vol, r = np.array([100.0]), 105.0, np.array([0.25]), 0.3, 0.03
    c = bs_price(spot, k, t, vol, r, True)
    p = bs_price(spot, k, t, vol, r, False)
    assert (c - p)[0] == pytest.approx(spot[0] - k * np.exp(-r * t[0]), abs=1e-9)


def test_contract_multiplier_is_100(prov):
    assert prov.multiplier("SPY") == 100.0
