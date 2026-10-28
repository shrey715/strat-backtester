# src/stratbt/adapters/alpaca.py
"""
Alpaca (US equities and options). Uses the official `alpaca-py` SDK.

Alpaca has a paper-trading environment with its own API keys, so `paper` and `dry-run` talk to
paper.alpaca.markets and only `live` touches real money. Install: pip install 'strat-backtester[alpaca]'.
Credentials come from the environment (ALPACA_API_KEY, ALPACA_SECRET_KEY) — never from the command line.
"""
from __future__ import annotations

import os
from datetime import date

from stratbt.adapters.base import FillReport, LiveBroker, LiveSafety, PollingFeed, Quote, osi_symbol, parse_osi
from stratbt.instruments import Instrument
from stratbt.risk import RiskLimits
from stratbt.strategy.base import Order

TERMINAL = frozenset({"filled", "canceled", "expired", "rejected", "done_for_day", "replaced"})


def _creds(api_key, secret_key):
    key, secret = api_key or os.environ.get("ALPACA_API_KEY"), secret_key or os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise ValueError("Alpaca credentials missing: set ALPACA_API_KEY and ALPACA_SECRET_KEY.")
    return key, secret


def _status(o) -> str:
    return str(getattr(o.status, "value", o.status)).lower()


class AlpacaBroker(LiveBroker):
    name = "alpaca"

    def __init__(self, api_key: str | None = None, secret_key: str | None = None, *, risk: RiskLimits | None = None,
                 safety: LiveSafety | None = None, paper: bool | None = None, trading_client=None, **kw):
        super().__init__(risk=risk, safety=safety, **kw)
        # `live` is the only mode that uses the real-money endpoint, unless `paper` is forced.
        self.paper = (self.safety.mode != "live") if paper is None else paper
        if self.safety.mode == "live" and self.paper:
            raise ValueError("mode='live' with paper=True would not trade real money; choose one.")
        if trading_client is None:
            from alpaca.trading.client import TradingClient
            key, secret = _creds(api_key, secret_key)
            trading_client = TradingClient(key, secret, paper=self.paper)
        self.client = trading_client

    def _place(self, order: Order, limit_price: float, client_id: str) -> FillReport:
        from alpaca.trading.enums import OrderSide, PositionIntent, TimeInForce
        from alpaca.trading.requests import LimitOrderRequest
        closing = self.is_closing(order)
        buy = order.side == "BUY"
        intent = {(True, False): PositionIntent.BUY_TO_OPEN, (True, True): PositionIntent.BUY_TO_CLOSE,
                  (False, False): PositionIntent.SELL_TO_OPEN, (False, True): PositionIntent.SELL_TO_CLOSE}[(buy, closing)]
        req = LimitOrderRequest(symbol=osi_symbol(order.instrument), qty=order.qty,
                                side=OrderSide.BUY if buy else OrderSide.SELL, time_in_force=TimeInForce.DAY,
                                limit_price=limit_price, client_order_id=client_id, position_intent=intent)
        placed = self.client.submit_order(req)

        def fetch() -> FillReport:
            o = self.client.get_order_by_id(placed.id)
            return FillReport(float(o.filled_qty or 0), float(o.filled_avg_price or 0), _status(o), order_id=str(o.id))

        return self._track(fetch, lambda: self.client.cancel_order_by_id(placed.id), TERMINAL)

    def positions(self) -> dict[Instrument, tuple[float, int]]:
        out = {}
        for p in self.client.get_all_positions():
            inst = parse_osi(p.symbol)
            if inst is None:
                continue                                 # a stock position: not ours to manage
            qty = abs(float(p.qty))
            short = str(getattr(p.side, "value", p.side)).lower() == "short"
            out[inst] = (float(p.avg_entry_price), int(-qty if short else qty))
        return out


class AlpacaFeed(PollingFeed):
    """Quotes from Alpaca's market-data API (`indicative` is free; `opra` needs a paid plan)."""

    def __init__(self, underlier: str, expiry: date, strikes, *, stock_client, option_client, options_feed=None, **kw):
        super().__init__(underlier, expiry, strikes, **kw)
        self.stock, self.option, self.options_feed = stock_client, option_client, options_feed

    @classmethod
    def create(cls, ticker: str, api_key: str | None = None, secret_key: str | None = None, *, options_feed: str = "indicative",
               trading_client=None, stock_client=None, option_client=None, today: date | None = None, **kw) -> "AlpacaFeed":
        from alpaca.data.enums import OptionsFeed
        from alpaca.trading.enums import AssetStatus
        from alpaca.trading.requests import GetOptionContractsRequest
        if trading_client is None or stock_client is None or option_client is None:
            from alpaca.data.historical.option import OptionHistoricalDataClient
            from alpaca.data.historical.stock import StockHistoricalDataClient
            from alpaca.trading.client import TradingClient
            key, secret = _creds(api_key, secret_key)
            trading_client = trading_client or TradingClient(key, secret, paper=True)
            stock_client = stock_client or StockHistoricalDataClient(key, secret)
            option_client = option_client or OptionHistoricalDataClient(key, secret)
        ticker = ticker.upper()
        res = trading_client.get_option_contracts(GetOptionContractsRequest(
            underlying_symbols=[ticker], status=AssetStatus.ACTIVE, expiration_date_gte=today or date.today(), limit=1000))
        contracts = [c for c in res.option_contracts if getattr(c, "tradable", True)]
        if not contracts:
            raise ValueError(f"Alpaca lists no tradable options for {ticker}.")
        expiry = min(c.expiration_date for c in contracts)
        strikes = sorted({int(c.strike_price) for c in contracts if c.expiration_date == expiry and float(c.strike_price).is_integer()})
        if len(strikes) < 4:
            raise ValueError(f"Too few whole-dollar strikes for {ticker} {expiry}: {strikes}")
        return cls(ticker, expiry, strikes, stock_client=stock_client, option_client=option_client,
                   options_feed=OptionsFeed(options_feed), **kw)

    def fetch_underlying(self) -> Quote:
        from alpaca.data.requests import StockLatestQuoteRequest
        q = self.stock.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=self.underlier))[self.underlier]
        return Quote(float(q.bid_price or 0) or None, float(q.ask_price or 0) or None, ts=self.clock())

    def fetch_options(self, instruments: list[Instrument]) -> dict[Instrument, Quote]:
        from alpaca.data.requests import OptionLatestQuoteRequest
        by_symbol = {osi_symbol(i): i for i in instruments}
        kwargs = {"feed": self.options_feed} if self.options_feed else {}
        res = self.option.get_option_latest_quote(OptionLatestQuoteRequest(symbol_or_symbols=list(by_symbol), **kwargs))
        now = self.clock()
        return {by_symbol[s]: Quote(float(q.bid_price or 0) or None, float(q.ask_price or 0) or None, ts=now)
                for s, q in res.items() if s in by_symbol}
