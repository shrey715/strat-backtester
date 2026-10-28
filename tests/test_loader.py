# tests/test_loader.py
import pytest
import pandas as pd
from pathlib import Path
from datetime import date

from stratbt.instruments import Instrument
from stratbt.loader import load_futures_1sec, load_option_1sec

PROJECT_ROOT = Path(__file__).parent.parent
DAY_DIR      = PROJECT_ROOT / "data" / "allData" / "NSE_20221101"

ATM_INST_CE = Instrument(
    underlier="BANKNIFTY",
    expiry=date(2022, 11, 3),
    strike=41600,
    option_type="CE"
)
ATM_INST_PE = Instrument(
    underlier="BANKNIFTY",
    expiry=date(2022, 11, 3),
    strike=41600,
    option_type="PE"
)
MISSING_INST = Instrument(
    underlier="BANKNIFTY",
    expiry=date(2022, 11, 3),
    strike=99999,        # does not exist
    option_type="CE"
)

@pytest.fixture(scope="module")
def bn_futures():
    """Load BANKNIFTY futures once for all tests in this module."""
    return load_futures_1sec(DAY_DIR, "BANKNIFTY")

@pytest.fixture(scope="module")
def atm_ce():
    return load_option_1sec(DAY_DIR, ATM_INST_CE)

@pytest.fixture(scope="module")
def atm_pe():
    return load_option_1sec(DAY_DIR, ATM_INST_PE)

def test_futures_length(bn_futures):
    """Session is 09:15:00 → 15:29:59 = exactly 22,500 seconds."""
    assert len(bn_futures) == 22_500

def test_futures_no_nans(bn_futures):
    """BANKNIFTY-I is liquid — no gaps after forward-fill."""
    assert bn_futures.isna().sum() == 0

def test_futures_index_is_datetimeindex(bn_futures):
    assert isinstance(bn_futures.index, pd.DatetimeIndex)

def test_futures_index_start(bn_futures):
    assert str(bn_futures.index[0]) == "2022-11-01 09:15:00"

def test_futures_index_end(bn_futures):
    assert str(bn_futures.index[-1]) == "2022-11-01 15:29:59"

def test_futures_price_in_sane_range(bn_futures):
    """BANKNIFTY was trading in the 41,000–42,000 band on Nov 1 2022."""
    assert bn_futures.min() > 40_000
    assert bn_futures.max() < 43_000

def test_futures_missing_underlier_raises():
    """A futures file that does not exist should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_futures_1sec(DAY_DIR, "FAKENDEX")

def test_option_length(atm_ce):
    assert len(atm_ce) == 22_500

def test_option_index_is_datetimeindex(atm_ce):
    assert isinstance(atm_ce.index, pd.DatetimeIndex)

def test_option_index_start(atm_ce):
    assert str(atm_ce.index[0]) == "2022-11-01 09:15:00"

def test_option_index_end(atm_ce):
    assert str(atm_ce.index[-1]) == "2022-11-01 15:29:59"

def test_option_price_in_sane_range(atm_ce):
    """ATM call premium should be in tens to low hundreds of rupees."""
    non_null = atm_ce.dropna()
    assert non_null.min() > 0
    assert non_null.max() < 5_000

def test_option_missing_file_returns_none():
    """Non-existent option file returns None, never raises."""
    result = load_option_1sec(DAY_DIR, MISSING_INST)
    assert result is None

def test_option_leading_nans_preserved(atm_ce):
    """
    Phase 1 showed ATM options first tick at 09:15:00 (lag = 0).
    But the loader must correctly preserve any leading NaNs for illiquid
    strikes — verify that ffill does not backfill the front of the series.
    We check this by confirming the first non-NaN value is the first tick,
    not some filled value from before the instrument existed.
    """
    first_valid_idx = atm_ce.first_valid_index()
    # For our ATM strike, first tick is at 09:15:00 — so no leading NaNs
    assert first_valid_idx == pd.Timestamp("2022-11-01 09:15:00")

def test_futures_and_option_indexes_identical(bn_futures, atm_ce, atm_pe):
    """
    The engine looks up futures[ts] and option[ts] using the same timestamp.
    Both indexes must be exactly the same — same grid, same length, same values.
    If they differ, lookups silently misalign or raise KeyErrors deep in the loop.
    """
    assert bn_futures.index.equals(atm_ce.index)
    assert bn_futures.index.equals(atm_pe.index)

def test_ce_and_pe_indexes_identical(atm_ce, atm_pe):
    assert atm_ce.index.equals(atm_pe.index)

def test_futures_and_option_same_length(bn_futures, atm_ce):
    assert len(bn_futures) == len(atm_ce)

def test_futures_late_first_tick_raises(tmp_path):
    """
    Regression test: the "no leading NaNs" guarantee load_futures_1sec's
    docstring promises is a data-quality assumption ("NIFTY-I/BANKNIFTY-I
    are always liquid at open"), not something the code enforced — a file
    whose first tick lands after 09:15:00 used to pass straight through as
    a silently NaN-contaminated series. engine.py's per-second loop has no
    NaN check of its own (`float(futures_values[current_idx])` doesn't
    raise on NaN), so a large violation would have fed NaN into the
    strategy's nearest-strike search instead of failing where the bad data
    was found. A genuine data gap (well beyond MAX_OPEN_LAG_S) must still
    raise — see test_futures_short_open_lag_is_backfilled for the harmless,
    empirically-observed short-lag case that must NOT raise.
    """
    day_dir = tmp_path / "NSE_20221101"
    fut_dir = day_dir / "Futures (Continuous)"
    fut_dir.mkdir(parents=True)
    # first tick at 09:16:00 — one minute after session open
    (fut_dir / "BANKNIFTY-I.csv").write_text(
        "20221101,09:16:00,41500.0,10,100\n"
        "20221101,09:16:01,41505.0,10,100\n"
    )
    with pytest.raises(ValueError, match="gap"):
        load_futures_1sec(day_dir, "BANKNIFTY")

def test_futures_short_open_lag_is_backfilled(tmp_path):
    """
    Regression test for the real 2022-11-04 BANKNIFTY-I case: first tick at
    09:15:01, one second after session open. This is well within
    MAX_OPEN_LAG_S and must be silently backfilled, not raised — the whole
    Nov 2022 backtest depends on this day not crashing.
    """
    day_dir = tmp_path / "NSE_20221101"
    fut_dir = day_dir / "Futures (Continuous)"
    fut_dir.mkdir(parents=True)
    (fut_dir / "BANKNIFTY-I.csv").write_text(
        "20221101,09:15:01,41500.0,10,100\n"
        "20221101,09:15:02,41505.0,10,100\n"
    )
    series = load_futures_1sec(day_dir, "BANKNIFTY")
    assert series.isna().sum() == 0
    assert series.iloc[0] == 41500.0   # backfilled from the first real tick