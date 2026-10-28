# src/stratbt/loader.py
import pandas as pd
from pathlib import Path
from datetime import date, datetime

from stratbt.instruments import Instrument

COLUMNS = ['date', 'time', 'price', 'volume', 'open_interest']

# Tolerance for how late the continuous futures' first tick of the day can
# land before it's treated as a genuine data problem rather than normal
# market-open settling. Empirically, real NSE data has hit up to 1s of lag
# (BANKNIFTY-I on 2022-11-04); this is set generously above that.
MAX_OPEN_LAG_S = 5

def _dir_to_date(day_dir: Path) -> date:
    """
    Parameters
    ----------
    day_dir : Path
        A folder named "NSE_YYYYMMDD".

    Returns
    -------
    date
        The trading date encoded in the folder name.
    """
    date_str = day_dir.name.replace("NSE_", "")
    return datetime.strptime(date_str, "%Y%m%d").date()

def _session_grid(trading_date: date) -> pd.DatetimeIndex:
    """
    The full 1-second timestamp grid for one NSE trading session.

    Parameters
    ----------
    trading_date : date

    Returns
    -------
    pd.DatetimeIndex
        09:15:00 -> 15:29:59 inclusive, exactly 22,500 seconds. Every price
        series this module produces is reindexed onto this same grid, so
        futures/option/day comparisons always line up second-for-second.
    """
    start = datetime.combine(trading_date, datetime.strptime("09:15:00", "%H:%M:%S").time())
    end = datetime.combine(trading_date, datetime.strptime("15:29:59", "%H:%M:%S").time())
    return pd.date_range(start=start, end=end, freq="1s")

def _parse_ticks(path: Path) -> pd.Series:
    """
    Read a raw tick CSV and resample to one price per second.

    Parameters
    ----------
    path : Path
        CSV with no header, columns [date, time, price, volume, open_interest].

    Returns
    -------
    pd.Series
        DatetimeIndex at 1-second resolution, value = last trade price within
        that second (ties broken by keeping the most recent tick). Seconds
        with no trade are NaN — callers reindex onto the full session grid
        and forward-fill; this function does neither.
    """
    # usecols skips parsing volume/open_interest entirely — neither is used
    # anywhere in this codebase, only 'price' survives past this function.
    df = pd.read_csv(
        path, header=None, names=COLUMNS,
        usecols=['date', 'time', 'price'],
        dtype={'date': 'int64', 'price': 'float64'},
    )
    df['datetime'] = pd.to_datetime(
        df['date'].astype(str) + ' ' + df['time'],
        format='%Y%m%d %H:%M:%S'
    )
    df = df.set_index('datetime').sort_index()
    return df['price'].resample('1s').last()

def load_futures_1sec(day_dir: Path, underlier: str) -> pd.Series:
    """
    Front-month continuous futures price for one underlier/day.

    Parameters
    ----------
    day_dir : Path
        A folder named "NSE_YYYYMMDD" containing a "Futures (Continuous)"
        subfolder.
    underlier : str
        e.g. "NIFTY" or "BANKNIFTY".

    Returns
    -------
    pd.Series
        DatetimeIndex covering the full 22,500-second session; last-traded
        futures price at each second, forward-filled from the first tick.

    Raises
    ------
    FileNotFoundError
        If the futures file for this underlier/day doesn't exist.
    ValueError
        If the first tick lands more than MAX_OPEN_LAG_S seconds after
        session open. A short lag (e.g. BANKNIFTY-I's first tick landed at
        09:15:01, not 09:15:00, on 2022-11-04 in the real dataset) is a
        harmless market-open settling quirk and is backfilled instead of
        raised — every downstream consumer (the per-second engine loop, the
        strategy's nearest-strike search) trusts a fully NaN-free series
        with no check of its own, so a longer gap must fail loudly here
        rather than silently corrupt a strike lookup later.
    """
    path = day_dir / "Futures (Continuous)" / f"{underlier}-I.csv"
    grid = _session_grid(_dir_to_date(day_dir))
    price_1s = _parse_ticks(path)
    series = price_1s.reindex(grid).ffill()
    if series.isna().any():
        first_valid = series.first_valid_index()
        if first_valid is None:
            raise ValueError(
                f"{underlier} futures on {_dir_to_date(day_dir)} has no valid ticks at all."
            )
        lag_s = (first_valid - grid[0]).total_seconds()
        if lag_s > MAX_OPEN_LAG_S:
            raise ValueError(
                f"{underlier} futures on {_dir_to_date(day_dir)} has a {lag_s:.0f}s gap "
                f"before the first tick (first tick at {first_valid}) — likely a genuine "
                "data problem, not normal market-open settling."
            )
        series = series.bfill()
    series.name = f"{underlier}_futures"
    return series

def load_option_1sec(day_dir: Path, inst: Instrument) -> pd.Series | None:
    """
    One option instrument's price for one day.

    Parameters
    ----------
    day_dir : Path
        A folder named "NSE_YYYYMMDD" containing an "Options" subfolder.
    inst : Instrument
        The specific contract to load.

    Returns
    -------
    pd.Series or None
        DatetimeIndex covering the full 22,500-second session; last-traded
        option price, forward-filled. Leading NaNs are preserved (unlike
        load_futures_1sec) — seconds before the instrument's first tick have
        no known price and none is invented; Phase 1 measured this lag as
        <= 2 seconds for ATM strikes. Returns None, rather than raising, if
        the instrument's file doesn't exist at all (illiquid/unlisted
        strikes are an expected, not exceptional, occurrence).
    """
    path = day_dir / "Options" / inst.filename
    if not path.exists():
        return None
    grid = _session_grid(_dir_to_date(day_dir))
    price_1s = _parse_ticks(path)
    series = price_1s.reindex(grid).ffill()
    series.name = inst.filename
    return series
