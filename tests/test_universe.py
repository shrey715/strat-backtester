# tests/test_universe.py

import pytest
import numpy as np
from pathlib import Path
from datetime import date

from stratbt.universe import Universe

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / "data" / "allData"

@pytest.fixture(scope="module")
def universe():
    """Build the universe once for all tests in this module."""
    return Universe(DATA_DIR)

def test_trading_days_count(universe):
    """Exactly 21 trading days in November 2022."""
    assert len(universe.trading_days) == 21

def test_trading_days_first(universe):
    assert universe.trading_days[0] == date(2022, 11, 1)

def test_trading_days_last(universe):
    assert universe.trading_days[-1] == date(2022, 11, 30)

def test_trading_days_sorted(universe):
    days = universe.trading_days
    assert days == sorted(days)

def test_nov_8_missing(universe):
    """Nov 8 was Guru Nanak Jayanti — market holiday."""
    assert date(2022, 11, 8) not in universe.trading_days

def test_front_expiry_banknifty_nov1(universe):
    """On Nov 1, front expiry is Nov 3 (nearest Thursday)."""
    assert universe.front_expiry(date(2022, 11, 1), "BANKNIFTY") == date(2022, 11, 3)

def test_front_expiry_nifty_nov1(universe):
    assert universe.front_expiry(date(2022, 11, 1), "NIFTY") == date(2022, 11, 3)

def test_front_expiry_rolls_after_expiry(universe):
    """
    On Nov 4 (Friday, day after expiry), front expiry must roll to Nov 10.
    Nov 3 has passed — it must not be returned.
    """
    exp = universe.front_expiry(date(2022, 11, 4), "BANKNIFTY")
    assert exp == date(2022, 11, 10)

def test_front_expiry_on_expiry_day_itself(universe):
    """
    On Nov 3 (expiry day), the expiring contract still trades until 15:30.
    front_expiry must return Nov 3, not Nov 10.
    """
    assert universe.front_expiry(date(2022, 11, 3), "BANKNIFTY") == date(2022, 11, 3)

def test_front_expiry_unknown_underlier_raises(universe):
    with pytest.raises(ValueError):
        universe.front_expiry(date(2022, 11, 1), "FAKENDEX")

def test_strikes_returns_numpy_array(universe):
    exp = universe.front_expiry(date(2022, 11, 1), "BANKNIFTY")
    strikes = universe.strikes(date(2022, 11, 1), "BANKNIFTY", exp)
    assert isinstance(strikes, np.ndarray)

def test_strikes_are_sorted(universe):
    exp = universe.front_expiry(date(2022, 11, 1), "BANKNIFTY")
    strikes = universe.strikes(date(2022, 11, 1), "BANKNIFTY", exp)
    assert list(strikes) == sorted(strikes)

def test_strikes_contains_atm(universe):
    """From Phase 1: ATM strike on Nov 1 was 41600."""
    exp = universe.front_expiry(date(2022, 11, 1), "BANKNIFTY")
    strikes = universe.strikes(date(2022, 11, 1), "BANKNIFTY", exp)
    assert 41600 in strikes

def test_strikes_are_positive(universe):
    exp = universe.front_expiry(date(2022, 11, 1), "BANKNIFTY")
    strikes = universe.strikes(date(2022, 11, 1), "BANKNIFTY", exp)
    assert (strikes > 0).all()

def test_strikes_bad_combination_raises(universe):
    with pytest.raises(ValueError):
        universe.strikes(date(2022, 11, 1), "BANKNIFTY", date(1999, 1, 1))

def test_midcpnifty_excluded(universe):
    """MIDCPNIFTY must not appear in the default universe."""
    for day in universe.trading_days:
        with pytest.raises(ValueError):
            universe.front_expiry(day, "MIDCPNIFTY")

def test_finnifty_excluded(universe):
    """FINNIFTY must not appear in the default universe."""
    for day in universe.trading_days:
        with pytest.raises(ValueError):
            universe.front_expiry(day, "FINNIFTY")

def test_custom_underlier_list():
    """Universe can be constructed with a custom underlier list."""
    u = Universe(DATA_DIR, underliers=["FINNIFTY"])
    # FINNIFTY should now be present
    exp = u.front_expiry(date(2022, 11, 1), "FINNIFTY")
    assert exp is not None
    # BANKNIFTY should now be absent
    with pytest.raises(ValueError):
        u.front_expiry(date(2022, 11, 1), "BANKNIFTY")

def test_day_dir_returns_path(universe):
    p = universe.day_dir(date(2022, 11, 1))
    assert isinstance(p, Path)

def test_day_dir_exists_on_disk(universe):
    p = universe.day_dir(date(2022, 11, 1))
    assert p.exists()

def test_day_dir_correct_name(universe):
    p = universe.day_dir(date(2022, 11, 1))
    assert p.name == "NSE_20221101"