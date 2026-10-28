"""
Venue adapters: connect strategies to real quote feeds and brokers.

    from stratbt.adapters import build

    feed, broker = build("alpaca", "SPY", mode="paper", max_daily_loss=200)

Modes: "dry-run" (decide on real quotes, send nothing), "paper" (the venue's paper account), "live" (real money,
behind an explicit confirmation). See `stratbt.adapters.base` for the safety model.
"""
from stratbt.adapters.base import (REAL_MONEY_ENV, REAL_MONEY_TOKEN, FeedError, FillReport, LiveBroker, LiveSafety,
                                   LiveTradingNotConfirmed, PollingFeed, Quote, osi_symbol, parse_osi)

BROKERS = ("alpaca", "public", "ibkr")


def build(broker: str, ticker: str, mode: str = "dry-run", max_daily_loss: float | None = None,
          max_open_contracts: int | None = None, max_order_qty: int | None = None, confirm_real_money: str | None = None,
          allow_short: bool = True, **venue_kwargs):
    """
    Create `(feed, broker)` for a venue, wired together. Credentials come from the environment
    (see each adapter module). `venue_kwargs` go to the adapter (e.g. port/exchange for IBKR).
    """
    from stratbt.risk import RiskLimits
    kw = {k: v for k, v in dict(max_order_qty=max_order_qty, max_open_contracts=max_open_contracts).items() if v}
    safety = LiveSafety(mode=mode, confirm_real_money=confirm_real_money, **kw)
    risk = RiskLimits(max_daily_loss=max_daily_loss)
    broker = broker.lower()
    if broker == "alpaca":
        from stratbt.adapters.alpaca import AlpacaBroker, AlpacaFeed
        b = AlpacaBroker(risk=risk, safety=safety, allow_short=allow_short, **venue_kwargs.get("broker", {}))
        feed = AlpacaFeed.create(ticker, **venue_kwargs.get("feed", {}))
    elif broker == "public":
        from stratbt.adapters.public import PublicBroker, PublicFeed, make_client
        client = venue_kwargs.get("client") or make_client()
        b = PublicBroker(client=client, risk=risk, safety=safety, allow_short=allow_short)
        feed = PublicFeed.create(ticker, client=client, **venue_kwargs.get("feed", {}))
    elif broker == "ibkr":
        from stratbt.adapters.ibkr import IBKRBroker, IBKRFeed, connect
        opts = {k: venue_kwargs[k] for k in ("host", "port", "exchange", "currency", "multiplier", "underlying_kind", "delayed") if k in venue_kwargs}
        ib = venue_kwargs.get("ib") or connect(opts.get("host", "127.0.0.1"), opts.get("port"), 7, mode)
        b = IBKRBroker(ib, risk=risk, safety=safety, allow_short=allow_short,
                       **{k: opts[k] for k in ("exchange", "currency", "multiplier") if k in opts})
        feed = IBKRFeed.create(ticker, ib, mode=mode, **{k: opts[k] for k in ("exchange", "currency", "multiplier", "underlying_kind", "delayed") if k in opts})
    else:
        raise ValueError(f"Unknown broker '{broker}'. Available: {', '.join(BROKERS)}")
    return feed, b.attach(feed)


__all__ = ["build", "BROKERS", "LiveBroker", "LiveSafety", "PollingFeed", "Quote", "FillReport", "FeedError",
           "LiveTradingNotConfirmed", "REAL_MONEY_TOKEN", "REAL_MONEY_ENV", "osi_symbol", "parse_osi"]
