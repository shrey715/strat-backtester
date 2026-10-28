# tests/test_yahoo_provider.py
"""YahooFinanceProvider on an offline fake downloader (no network)."""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from stratbt import BacktestEngine, ExecutionModel, run_backtest
from stratbt.providers import YahooFinanceProvider
from stratbt.providers.yahoo import _guess_currency


class FakeDownloader:
    """Deterministic bars shaped like Yahoo's: tz-aware exchange-local index, bar-start stamped."""

    def __init__(self, tz="America/New_York", open_="09:30", n_bars=78, bar="5min", ccy="USD", start_px=450.0,
                 n_days=12, end=date(2024, 3, 15), drop_days=(), short_days=()):
        self.tz, self.ccy, self.calls = tz, ccy, []
        self.frames: dict[str, dict[str, pd.DataFrame]] = {}
        self.params = dict(open_=open_, n_bars=n_bars, bar=bar, start_px=start_px, n_days=n_days, end=end,
                           drop_days=set(drop_days), short_days=set(short_days))

    def _days(self):
        p = self.params
        days, d = [], p["end"]
        while len(days) < p["n_days"]:
            if d.weekday() < 5 and d not in p["drop_days"]:
                days.append(d)
            d -= timedelta(days=1)
        return sorted(days)

    def _build(self, ticker: str):
        p, rng = self.params, np.random.default_rng(abs(hash(ticker)) % 2**32)
        px, rows = p["start_px"], []
        for d in self._days():
            n = p["n_bars"] // 3 if d in p["short_days"] else p["n_bars"]
            idx = pd.date_range(f"{d} {p['open_']}", periods=n, freq=p["bar"], tz=self.tz)
            closes = px * np.exp(np.cumsum(rng.normal(0, 0.0006, n)))
            opens = np.r_[px, closes[:-1]]
            rows.append(pd.DataFrame({"Open": opens, "High": np.maximum(opens, closes), "Low": np.minimum(opens, closes),
                                      "Close": closes}, index=idx))
            px = closes[-1]
        intraday = pd.concat(rows)
        daily = intraday.groupby(intraday.index.date).agg(Open=("Open", "first"), High=("High", "max"),
                                                           Low=("Low", "min"), Close=("Close", "last"))
        daily.index = pd.DatetimeIndex(pd.to_datetime(daily.index)).tz_localize(self.tz)
        # extra history before the intraday window for the vol estimate
        pre = pd.DataFrame({"Open": 440.0, "High": 441.0, "Low": 439.0,
                            "Close": 440.0 + np.cumsum(rng.normal(0, 2.0, 40))},
                           index=pd.bdate_range(end=min(daily.index.date) - timedelta(days=1), periods=40, tz=self.tz))
        return intraday, pd.concat([pre, daily])

    def history(self, ticker, start, end, interval):
        self.calls.append((ticker, interval))
        if ticker not in self.frames:
            if ticker.startswith("BAD"):
                return pd.DataFrame(columns=["Open", "High", "Low", "Close"])
            i, d = self._build(ticker)
            self.frames[ticker] = {"intraday": i, "daily": d}
        return self.frames[ticker]["daily" if interval == "1d" else "intraday"]

    def currency(self, ticker):
        return self.ccy


@pytest.fixture(scope="module")
def prov():
    return YahooFinanceProvider(["SPY", "QQQ"], interval="5m", end=date(2024, 3, 15), lookback_days=20,
                                downloader=FakeDownloader())


def test_session_is_detected_from_the_bars(prov):
    assert prov.session_open.strftime("%H:%M:%S") == "09:30:00"
    assert prov.session_close.strftime("%H:%M:%S") == "15:59:59"       # last 5m bar starts 15:55
    assert len(prov.session_grid(prov.trading_days[0])) == 23_400


def test_days_and_underliers(prov):
    assert prov.underliers == ["SPY", "QQQ"]
    assert len(prov.trading_days) == 12 and all(d.weekday() < 5 for d in prov.trading_days)


def test_underlying_on_full_second_grid_no_nans(prov):
    day = prov.trading_days[3]
    s = prov.underlying_prices(day, "SPY")
    assert s.index.equals(prov.session_grid(day)) and not s.isna().any()


def test_bar_close_is_not_visible_before_the_bar_ends(prov):
    """Within a bar the price is its open; the close only appears in the bar's last second."""
    dl = prov._dl
    day = prov.trading_days[3]
    bars = prov._bars[(day, "SPY")]
    s = prov.underlying_prices(day, "SPY")
    t0 = bars.index[10]
    assert s[t0] == pytest.approx(round(bars["Open"].iloc[10], 2))
    assert s[t0 + pd.Timedelta(seconds=150)] == pytest.approx(round(bars["Open"].iloc[10], 2))
    assert s[t0 + pd.Timedelta(seconds=299)] == pytest.approx(round(bars["Close"].iloc[10], 2))
    assert s[t0 + pd.Timedelta(seconds=300)] == pytest.approx(round(bars["Open"].iloc[11], 2))


def test_vol_uses_only_prior_days_no_lookahead():
    a, b = FakeDownloader(), FakeDownloader()
    pa = YahooFinanceProvider("SPY", end=date(2024, 3, 15), lookback_days=20, downloader=a)
    day = pa.trading_days[5]
    vol_before = pa._iv(day, "SPY")
    # corrupt every daily close on/after `day`: the estimate for `day` must not move
    pb = YahooFinanceProvider("SPY", end=date(2024, 3, 15), lookback_days=20, downloader=b)
    d = b.frames["SPY"]["daily"]
    d.loc[d.index >= pd.Timestamp(day).tz_localize(d.index.tz), "Close"] *= 3.7
    pb._daily["SPY"] = pd.Series(d["Close"].to_numpy(), index=pd.DatetimeIndex(d.index).tz_localize(None).normalize())
    assert pb._iv(day, "SPY") == pytest.approx(vol_before)
    assert pb._iv(pb.trading_days[-1], "SPY") != pytest.approx(pa._iv(pa.trading_days[-1], "SPY"))


def test_iv_premium_and_fixed_vol():
    base = YahooFinanceProvider("SPY", iv_premium=0.0, end=date(2024, 3, 15), lookback_days=20, downloader=FakeDownloader())
    rich = YahooFinanceProvider("SPY", iv_premium=0.5, end=date(2024, 3, 15), lookback_days=20, downloader=FakeDownloader())
    day = base.trading_days[6]
    assert rich._iv(day, "SPY") == pytest.approx(base._iv(day, "SPY") * 1.5)
    fixed = YahooFinanceProvider("SPY", vol=0.33, end=date(2024, 3, 15), lookback_days=20, downloader=FakeDownloader())
    assert fixed._iv(day, "SPY") == 0.33


def test_incomplete_sessions_are_dropped():
    dl = FakeDownloader(short_days={date(2024, 3, 12)})
    p = YahooFinanceProvider("SPY", end=date(2024, 3, 15), lookback_days=20, downloader=dl)
    assert date(2024, 3, 12) not in p.trading_days and len(p.trading_days) == 11


def test_multiplier_and_unit_defaults_by_currency(prov):
    assert prov.multiplier("SPY") == 100 and prov.unit == "USD"
    inr = YahooFinanceProvider("^NSEI", end=date(2024, 3, 15), lookback_days=20,
                               downloader=FakeDownloader(tz="Asia/Kolkata", open_="09:15", n_bars=75, ccy="INR", start_px=22000))
    assert inr.multiplier("^NSEI") == 1 and inr.unit == "pts"
    assert inr.session_open.strftime("%H:%M") == "09:15" and inr.session_close.strftime("%H:%M:%S") == "15:29:59"
    assert inr.price_tick == 0.05 and inr.rate == pytest.approx(0.065)
    lot = YahooFinanceProvider("^NSEI", multiplier=75, end=date(2024, 3, 15), lookback_days=20,
                               downloader=FakeDownloader(tz="Asia/Kolkata", open_="09:15", n_bars=75, ccy="INR", start_px=22000))
    assert lot.multiplier("^NSEI") == 75 and lot.unit == "INR"


def test_strike_step_scales_with_price():
    nifty = YahooFinanceProvider("^NSEI", end=date(2024, 3, 15), lookback_days=20,
                                 downloader=FakeDownloader(tz="Asia/Kolkata", open_="09:15", n_bars=75, ccy="INR", start_px=22000))
    spy = YahooFinanceProvider("SPY", end=date(2024, 3, 15), lookback_days=20, downloader=FakeDownloader())
    assert nifty._strike_step("^NSEI") == 50 and spy._strike_step("SPY") == 1
    day = nifty.trading_days[2]
    k = nifty.strikes(day, "^NSEI", nifty.front_expiry(day, "^NSEI"))
    assert (np.diff(k) == 50).all() and k[0] < nifty.underlying_prices(day, "^NSEI").min() < k[-1]


def test_errors_are_clear():
    with pytest.raises(ValueError, match="No 5m data for 'BADX'"):
        YahooFinanceProvider("BADX", downloader=FakeDownloader())
    with pytest.raises(ValueError, match="interval"):
        YahooFinanceProvider("SPY", interval="3m", downloader=FakeDownloader())
    with pytest.raises(ValueError, match="at least one ticker"):
        YahooFinanceProvider([" "], downloader=FakeDownloader())


def test_mixed_sessions_rejected():
    class Mixed(FakeDownloader):
        def history(self, ticker, start, end, interval):
            if ticker == "^NSEI":
                self.params["open_"], self.tz = "09:15", "Asia/Kolkata"
            else:
                self.params["open_"], self.tz = "09:30", "America/New_York"
            self.frames.pop(ticker, None)
            return super().history(ticker, start, end, interval)
    with pytest.raises(ValueError, match="different sessions"):
        YahooFinanceProvider(["SPY", "^NSEI"], end=date(2024, 3, 15), lookback_days=20, downloader=Mixed())


def test_lookback_is_capped_to_what_yahoo_keeps():
    p = YahooFinanceProvider("SPY", interval="1m", lookback_days=500, end=date(2024, 3, 15),
                             downloader=FakeDownloader(bar="1min", n_bars=390))
    assert p.lookback_days == 29


def test_ticker_symbols_are_normalised_and_deduplicated():
    p = YahooFinanceProvider([" spy", "SPY", "qqq "], end=date(2024, 3, 15), lookback_days=20, downloader=FakeDownloader())
    assert p.underliers == ["SPY", "QQQ"]


def test_currency_guess_from_symbol():
    assert _guess_currency("^NSEI") == "INR" and _guess_currency("RELIANCE.NS") == "INR"
    assert _guess_currency("SPY") == "USD" and _guess_currency("VOD.L") == "GBP"


def test_full_backtest_on_yahoo_data_with_costs(prov):
    r = run_backtest("iron_condor", prov, execution=ExecutionModel.realistic(), underliers=["SPY"])
    assert len(r.day_results) == 12 and r.unit == "USD"
    assert r.trades_df["fee"].sum() > 0
    assert r.meta["provider"] == "yahoo"


def test_rolling_strategy_runs_on_stepwise_bar_data(prov):
    engine = BacktestEngine(prov, __import__("stratbt.strategy.registry", fromlist=["x"]).get("atm_straddle").factory(prov))
    dr = engine.run_day(prov.trading_days[2], "QQQ")
    assert len(dr.mtm) == 23_400


def test_missing_day_underlier_raises_value_error(prov):
    with pytest.raises(ValueError):
        prov.front_expiry(date(1999, 1, 4), "SPY")
