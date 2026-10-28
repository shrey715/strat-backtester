# src/stratbt/providers/yahoo.py
"""
Real market data from Yahoo Finance (via the optional `yfinance` package),
for any market it covers: US equities/ETFs, NSE/BSE (`^NSEI`, `RELIANCE.NS`),
and so on.

What is real and what is modelled — read this before trusting a result:

* Underlying prices are **real** intraday bars. Within a bar the price is held
  at the bar's open and steps to its close in the bar's final second, so a
  strategy never sees a bar's close before the bar has ended.
* Option prices are **modelled** (Black-Scholes at every second). Yahoo only
  publishes the *current* option chain, never historical option quotes, so
  there is nothing real to replay. Implied vol is the trailing realised vol of
  the underlying (daily returns strictly *before* each day, so no look-ahead)
  times (1 + iv_premium).

Yahoo's intraday history is short: ~29 days of 1m, ~59 days of 5m–30m bars,
~2 years of hourly bars.
"""
from __future__ import annotations

import hashlib
import os
import pickle
import time as _time
from collections import Counter
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from stratbt.providers.modelled import ModelPricedProvider, nice_strike_step

INTERVAL_SECONDS = {"1m": 60, "2m": 120, "5m": 300, "15m": 900, "30m": 1800, "60m": 3600, "90m": 5400}
INTERVAL_MAX_DAYS = {"1m": 29, "2m": 59, "5m": 59, "15m": 59, "30m": 59, "60m": 729, "90m": 59}
_CHUNK_DAYS = {"1m": 7}   # Yahoo serves 1m bars at most 7 days per request

_CURRENCY_BY_SUFFIX = {".NS": "INR", ".BO": "INR", ".L": "GBP", ".T": "JPY", ".HK": "HKD", ".TO": "CAD",
                       ".AX": "AUD", ".DE": "EUR", ".PA": "EUR", ".MI": "EUR", ".AS": "EUR"}
_CURRENCY_BY_TICKER = {"^NSEI": "INR", "^NSEBANK": "INR", "^BSESN": "INR", "^FTSE": "GBP", "^N225": "JPY"}
_RATE_BY_CURRENCY = {"USD": 0.04, "INR": 0.065, "EUR": 0.025, "GBP": 0.04, "JPY": 0.005}


class Downloader(Protocol):
    """Where bars come from. Swap in your own (a different vendor, a database, a test fake)."""

    def history(self, ticker: str, start: date, end: date, interval: str) -> pd.DataFrame:
        """OHLC bars in [start, end), indexed by bar start (tz-aware for intraday); columns Open/High/Low/Close."""

    def currency(self, ticker: str) -> str | None:
        """ISO currency code if known."""


class YahooDownloader:
    """yfinance with an on-disk cache so repeated runs don't re-download."""

    def __init__(self, cache_dir: str | os.PathLike | None = None, ttl_hours: float = 6.0):
        self.cache_dir = Path(cache_dir or os.environ.get("STRATBT_CACHE_DIR")
                              or Path.home() / ".cache" / "stratbt" / "yahoo")
        self.ttl = ttl_hours * 3600

    @staticmethod
    def _yf():
        try:
            import yfinance
        except ImportError as e:
            raise ImportError("Yahoo Finance data needs the optional dependency:  pip install 'strat-backtester[data]'") from e
        return yfinance

    def _cached(self, key: str, fresh_for: float, fetch):
        path = self.cache_dir / (hashlib.sha1(key.encode()).hexdigest() + ".pkl")
        if path.exists() and _time.time() - path.stat().st_mtime < fresh_for:
            try:
                return pickle.loads(path.read_bytes())
            except Exception:
                pass
        value = fetch()
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            path.write_bytes(pickle.dumps(value))
        except OSError:
            pass
        return value

    def history(self, ticker: str, start: date, end: date, interval: str) -> pd.DataFrame:
        settled = end < date.today() - timedelta(days=1)     # history that can no longer change
        key = f"hist|{ticker}|{start}|{end}|{interval}"
        return self._cached(key, 90 * 86400 if settled else self.ttl,
                            lambda: self._fetch(ticker, start, end, interval))

    def _fetch(self, ticker: str, start: date, end: date, interval: str) -> pd.DataFrame:
        yf = self._yf()
        chunk = _CHUNK_DAYS.get(interval)
        windows, a = [], start
        while a < end:
            b = min(a + timedelta(days=chunk), end) if chunk else end
            windows.append((a, b))
            a = b
        frames = []
        for a, b in windows:
            df = yf.Ticker(ticker).history(start=a.isoformat(), end=b.isoformat(), interval=interval,
                                           auto_adjust=False, actions=False)
            if df is not None and not df.empty:
                frames.append(df[["Open", "High", "Low", "Close"]])
        if not frames:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close"])
        out = pd.concat(frames)
        return out[~out.index.duplicated(keep="last")].sort_index()

    def currency(self, ticker: str) -> str | None:
        def fetch():
            try:
                return self._yf().Ticker(ticker).fast_info.get("currency")
            except Exception:
                return None
        return self._cached(f"ccy|{ticker}", 30 * 86400, fetch)


def _guess_currency(ticker: str) -> str:
    if ticker in _CURRENCY_BY_TICKER:
        return _CURRENCY_BY_TICKER[ticker]
    for suffix, ccy in _CURRENCY_BY_SUFFIX.items():
        if ticker.endswith(suffix):
            return ccy
    return "USD"


class YahooFinanceProvider(ModelPricedProvider):
    """
    Parameters
    ----------
    tickers : str or list[str]
        Yahoo symbols, e.g. ["SPY", "QQQ"], ["^NSEI", "^NSEBANK"], ["RELIANCE.NS"].
        All must trade in the same session hours (one market per run).
    interval : str
        Bar size: "1m", "2m", "5m", "15m", "30m", "60m" or "90m".
    lookback_days : int, optional
        Calendar days of history ending at `end`. Defaults to 30 (clamped to what Yahoo keeps for `interval`).
    end : date, optional
        Last day to include (default: today; an unfinished session is dropped automatically).
    vol_window : int
        Trading days of daily returns behind the realised-vol estimate.
    iv_premium : float
        Implied vol = trailing realised vol * (1 + iv_premium). Equity options usually trade a little
        above realised vol; 0.10 is a reasonable default. 0 prices options at realised vol.
    vol : float, optional
        Fixed annualised implied vol instead of the trailing estimate.
    rate : float, optional
        Risk-free rate; defaults by currency (USD 4%, INR 6.5%, ...).
    expiry_weekday : int
        0=Mon … 4=Fri weekly expiry (Yahoo can't tell us the historical expiry calendar).
    strike_step : int or dict, optional
        Distance between strikes; default is a conventional step near 0.25% of the price.
    multiplier : float or dict, optional
        Contract multiplier. Defaults: 100 for USD underlyings (US equity options), 1 otherwise
        (PnL in index/price points; pass e.g. 75 for a NIFTY lot).
    min_coverage : float
        A day with fewer than this fraction of its expected bars is dropped (half-days, today's partial session).
    downloader : Downloader, optional
        Data source; default is YahooDownloader (disk-cached).
    """

    name = "yahoo"

    def __init__(
        self,
        tickers: str | list[str],
        interval: str = "5m",
        lookback_days: int | None = None,
        end: date | None = None,
        vol_window: int = 20,
        iv_premium: float = 0.10,
        vol: float | None = None,
        rate: float | None = None,
        expiry_weekday: int = 4,
        strike_step: int | dict | None = None,
        multiplier: float | dict | None = None,
        min_coverage: float = 0.6,
        downloader: Downloader | None = None,
    ):
        if interval not in INTERVAL_SECONDS:
            raise ValueError(f"interval must be one of {sorted(INTERVAL_SECONDS)}")
        tickers = [tickers] if isinstance(tickers, str) else list(tickers)
        self._tickers = list(dict.fromkeys(t.strip().upper() for t in tickers if t.strip()))
        if not self._tickers:
            raise ValueError("Give at least one ticker symbol")
        self.interval, self._bar_s = interval, INTERVAL_SECONDS[interval]
        self.vol_window, self.iv_premium, self._fixed_vol = vol_window, iv_premium, vol
        self.expiry_weekday = expiry_weekday
        self._dl = downloader or YahooDownloader()

        max_days = INTERVAL_MAX_DAYS[interval]
        self.lookback_days = min(lookback_days or 30, max_days)
        self.end = end or date.today()
        start = self.end - timedelta(days=self.lookback_days)

        self._bars: dict[tuple[date, str], pd.DataFrame] = {}
        self._daily: dict[str, pd.Series] = {}
        self._load(start)
        self._detect_session()
        self._drop_incomplete(min_coverage)
        if not self._bars:
            raise ValueError("No complete trading sessions found in the requested window; widen the lookback "
                             f"(Yahoo keeps about {max_days} days of {interval} bars).")
        self._days = sorted({d for d, _ in self._bars})

        ccys = {t: (self._dl.currency(t) or _guess_currency(t)) for t in self._tickers}
        self._currency = ccys
        if len(set(ccys.values())) > 1:
            raise ValueError(f"Tickers trade in different currencies {ccys}; backtest one market at a time.")
        ccy = next(iter(ccys.values()))
        self.rate = rate if rate is not None else _RATE_BY_CURRENCY.get(ccy, 0.04)
        self.price_tick = 0.05 if ccy == "INR" else 0.01
        self._mult = {t: self._pick(multiplier, t, 100.0 if ccy == "USD" else 1.0) for t in self._tickers}
        self.unit = ccy if any(m != 1.0 for m in self._mult.values()) else "pts"
        first_spot = {t: float(self._bars[next(k for k in sorted(self._bars) if k[1] == t)]["Close"].iloc[0])
                      for t in self._tickers if any(k[1] == t for k in self._bars)}
        self._step = {t: int(self._pick(strike_step, t, nice_strike_step(first_spot.get(t, 100.0)))) for t in self._tickers}
        self._spot_cache: dict[tuple[date, str], pd.Series] = {}

    # ---- construction ---------------------------------------------------
    @staticmethod
    def _pick(value, ticker, default):
        if value is None:
            return default
        return value.get(ticker, default) if isinstance(value, dict) else value

    def _load(self, start: date) -> None:
        end_excl = self.end + timedelta(days=1)
        daily_start = start - timedelta(days=int(self.vol_window * 1.7) + 10)
        for t in self._tickers:
            df = self._dl.history(t, start, end_excl, self.interval)
            if df is None or df.empty:
                raise ValueError(f"No {self.interval} data for '{t}' between {start} and {self.end}. Check the symbol; "
                                 f"Yahoo keeps only ~{INTERVAL_MAX_DAYS[self.interval]} days of {self.interval} history.")
            idx = pd.DatetimeIndex(df.index)
            if idx.tz is not None:
                idx = idx.tz_localize(None)            # keep exchange-local wall-clock time
            df = df.set_axis(idx)[["Open", "Close"]].dropna().sort_index()
            for d, g in df.groupby(df.index.date):
                self._bars[(d, t)] = g
            dd = self._dl.history(t, daily_start, end_excl, "1d")
            if dd is not None and not dd.empty:
                di = pd.DatetimeIndex(dd.index)
                di = di.tz_localize(None) if di.tz is not None else di
                self._daily[t] = pd.Series(dd["Close"].to_numpy(), index=di.normalize()).dropna()

    def _detect_session(self) -> None:
        per_ticker = {}
        for t in self._tickers:
            opens, closes = Counter(), Counter()
            for (d, tk), g in self._bars.items():
                if tk == t:
                    opens[g.index.min().time()] += 1
                    closes[g.index.max().time()] += 1
            per_ticker[t] = (opens.most_common(1)[0][0], closes.most_common(1)[0][0])
        if len(set(per_ticker.values())) > 1:
            raise ValueError("Tickers trade in different sessions " + str({t: f"{o:%H:%M}-{c:%H:%M}" for t, (o, c) in per_ticker.items()})
                             + "; backtest one market at a time.")
        open_t, last_start = next(iter(per_ticker.values()))
        self.session_open = open_t
        end_dt = datetime.combine(date(2000, 1, 3), last_start) + timedelta(seconds=self._bar_s - 1)
        self.session_close = end_dt.time()

    def _drop_incomplete(self, min_coverage: float) -> None:
        span = (datetime.combine(date(2000, 1, 3), self.session_close)
                - datetime.combine(date(2000, 1, 3), self.session_open)).total_seconds() + 1
        expected = max(1, int(span // self._bar_s))
        self._bars = {k: g for k, g in self._bars.items() if len(g) >= min_coverage * expected}

    # ---- ModelPricedProvider hooks -------------------------------------
    @property
    def underliers(self) -> list[str]:
        return list(self._tickers)

    @property
    def trading_days(self) -> list[date]:
        return list(self._days)

    def multiplier(self, underlier: str) -> float:
        return float(self._mult.get(underlier, 1.0))

    def _has(self, day: date, underlier: str) -> bool:
        return (day, underlier) in self._bars

    def _spot(self, day: date, underlier: str) -> pd.Series:
        key = (day, underlier)
        if key not in self._spot_cache:
            g = self._bars[key]
            grid = self.session_grid(day)
            step = pd.Timedelta(seconds=self._bar_s - 1)
            pts = pd.concat([pd.Series(g["Open"].to_numpy(), index=g.index),
                             pd.Series(g["Close"].to_numpy(), index=g.index + step)])
            pts = pts[~pts.index.duplicated(keep="last")].sort_index()
            s = pts.reindex(pts.index.union(grid)).ffill().bfill().reindex(grid)
            self._spot_cache[key] = s.round(2).rename(f"{underlier}_underlying")
        return self._spot_cache[key]

    def _iv(self, day: date, underlier: str) -> float:
        if self._fixed_vol is not None:
            return float(self._fixed_vol)
        return float(self._realised_vol(day, underlier) * (1.0 + self.iv_premium))

    def _realised_vol(self, day: date, underlier: str) -> float:
        """Annualised close-to-close vol from daily bars strictly before `day`."""
        closes = self._daily.get(underlier)
        if closes is not None:
            prior = closes[closes.index < pd.Timestamp(day)].tail(self.vol_window + 1)
            if len(prior) >= 6:
                v = float(np.log(prior).diff().dropna().std(ddof=1) * np.sqrt(252))
                if np.isfinite(v):
                    return min(max(v, 0.05), 2.0)
        return 0.25     # not enough history before this day

    def _strike_step(self, underlier: str) -> int:
        return self._step[underlier]
