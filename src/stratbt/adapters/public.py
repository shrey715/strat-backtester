# src/stratbt/adapters/public.py
"""
Public.com, via its official `publicdotcom-py` SDK.

Public has NO paper-trading environment, so this adapter offers two modes only:

* dry-run — real quotes and decisions; each order is validated with Public's pre-flight endpoint and then
            **not sent** (the fill is simulated locally);
* live    — real money, behind the confirmation token and loss limit.

Credentials: PUBLIC_API_SECRET_KEY and PUBLIC_ACCOUNT_NUMBER in the environment.
Install: pip install 'strat-backtester[public]'.

Verify with the smallest possible order before relying on it: this adapter was written against the SDK's
documented models and has not been exercised against a live Public account.
"""
from __future__ import annotations

import os
import uuid
from datetime import date, datetime
from decimal import Decimal

from stratbt.adapters.base import FillReport, LiveBroker, LiveSafety, PollingFeed, Quote, osi_symbol, parse_osi
from stratbt.instruments import Instrument
from stratbt.risk import RiskLimits
from stratbt.strategy.base import Order

TERMINAL = frozenset({"FILLED", "CANCELLED", "REJECTED", "EXPIRED", "REPLACED"})


def make_client(api_secret_key: str | None = None, account_number: str | None = None):
    from public_api_sdk import ApiKeyAuthConfig, PublicApiClient, PublicApiClientConfiguration
    key = api_secret_key or os.environ.get("PUBLIC_API_SECRET_KEY")
    acct = account_number or os.environ.get("PUBLIC_ACCOUNT_NUMBER")
    if not key or not acct:
        raise ValueError("Public credentials missing: set PUBLIC_API_SECRET_KEY and PUBLIC_ACCOUNT_NUMBER.")
    return PublicApiClient(ApiKeyAuthConfig(api_secret_key=key), config=PublicApiClientConfiguration(default_account_number=acct))


def _val(x) -> str:
    return str(getattr(x, "value", x)).upper()


def _num(x) -> float:
    return float(x) if x is not None else 0.0


class PublicBroker(LiveBroker):
    name = "public"

    def __init__(self, client=None, *, api_secret_key: str | None = None, account_number: str | None = None,
                 risk: RiskLimits | None = None, safety: LiveSafety | None = None, **kw):
        super().__init__(risk=risk, safety=safety, **kw)
        if self.safety.mode == "paper":
            raise ValueError("Public has no paper-trading environment. Use mode='dry-run' (pre-flight only) or 'live'.")
        self.client = client or make_client(api_secret_key, account_number)

    def _request(self, order: Order, limit_price: float, order_id: str):
        from public_api_sdk import (InstrumentType, OpenCloseIndicator, OrderExpirationRequest, OrderInstrument,
                                    OrderRequest, OrderSide, OrderType, TimeInForce)
        return OrderRequest(
            order_id=order_id,
            instrument=OrderInstrument(symbol=osi_symbol(order.instrument), type=InstrumentType.OPTION),
            order_side=OrderSide.BUY if order.side == "BUY" else OrderSide.SELL,
            order_type=OrderType.LIMIT, expiration=OrderExpirationRequest(time_in_force=TimeInForce.DAY),
            quantity=order.qty, limit_price=Decimal(str(limit_price)),
            open_close_indicator=OpenCloseIndicator.CLOSE if self.is_closing(order) else OpenCloseIndicator.OPEN)

    def _preflight(self, order: Order, limit_price: float) -> None:
        from public_api_sdk import PreflightRequest
        req = self._request(order, limit_price, str(uuid.uuid4()))
        self.client.perform_preflight_calculation(PreflightRequest(**{
            k: getattr(req, k) for k in ("instrument", "order_side", "order_type", "expiration", "quantity",
                                         "limit_price", "open_close_indicator")}))

    def _place(self, order: Order, limit_price: float, client_id: str) -> FillReport:
        oid = str(uuid.UUID(client_id))                   # Public wants a UUID order id: also our idempotency key
        new = self.client.place_order(self._request(order, limit_price, oid))

        def fetch() -> FillReport:
            o = self.client.get_order(new.order_id)
            return FillReport(_num(o.filled_quantity), _num(o.average_price), _val(o.status), order_id=str(new.order_id))

        return self._track(fetch, lambda: self.client.cancel_order(new.order_id), TERMINAL)

    def positions(self) -> dict[Instrument, tuple[float, int]]:
        out = {}
        for p in self.client.get_portfolio().positions:
            if _val(p.instrument.type) != "OPTION":
                continue
            inst = parse_osi(p.instrument.symbol)
            if inst is None:
                continue
            qty = int(_num(p.quantity))
            cb = getattr(p, "cost_basis", None)
            # Unknown entry price must never be 0 (that would fake a huge unrealised profit): fall back to the last price.
            avg = abs(_num(getattr(cb, "unit_cost", None))) or abs(_num(getattr(p, "last_price", None)))
            out[inst] = (avg, qty)                        # a negative quantity is a short position
        return out


class PublicFeed(PollingFeed):
    """Quotes from Public's market-data endpoints (one batched `get_quotes` call per second)."""

    def __init__(self, underlier: str, expiry: date, strikes, *, client, **kw):
        super().__init__(underlier, expiry, strikes, **kw)
        self.client = client

    @classmethod
    def create(cls, ticker: str, client=None, *, api_secret_key: str | None = None, account_number: str | None = None,
               today: date | None = None, **kw) -> "PublicFeed":
        from public_api_sdk import InstrumentType, OptionChainRequest, OptionExpirationsRequest, OrderInstrument
        client = client or make_client(api_secret_key, account_number)
        ticker = ticker.upper()
        inst = OrderInstrument(symbol=ticker, type=InstrumentType.EQUITY)
        today = today or date.today()
        expiries = sorted(d for d in (_as_date(e) for e in client.get_option_expirations(OptionExpirationsRequest(instrument=inst)).expirations) if d >= today)
        if not expiries:
            raise ValueError(f"Public lists no upcoming option expirations for {ticker}.")
        expiry = expiries[0]
        chain = client.get_option_chain(OptionChainRequest(instrument=inst, expiration_date=expiry.isoformat()))
        strikes = sorted({int(i.strike) for i in (parse_osi(_chain_symbol(o), ticker) for o in list(chain.calls) + list(chain.puts))
                          if i and float(i.strike).is_integer()})
        if len(strikes) < 4:
            raise ValueError(f"Too few whole-dollar strikes for {ticker} {expiry}: {strikes}")
        return cls(ticker, expiry, strikes, client=client, **kw)

    def fetch_underlying(self) -> Quote:
        from public_api_sdk import InstrumentType, OrderInstrument
        q = self.client.get_quotes([OrderInstrument(symbol=self.underlier, type=InstrumentType.EQUITY)])[0]
        return Quote(_num(q.bid) or None, _num(q.ask) or None, _num(q.last) or None, ts=self.clock())

    def fetch_options(self, instruments: list[Instrument]) -> dict[Instrument, Quote]:
        from public_api_sdk import InstrumentType, OrderInstrument
        by_symbol = {osi_symbol(i): i for i in instruments}
        res = self.client.get_quotes([OrderInstrument(symbol=s, type=InstrumentType.OPTION) for s in by_symbol])
        now = self.clock()
        return {by_symbol[q.instrument.symbol]: Quote(_num(q.bid) or None, _num(q.ask) or None, _num(q.last) or None, ts=now)
                for q in res if q.instrument.symbol in by_symbol}


def _as_date(x) -> date:
    return x if isinstance(x, date) and not isinstance(x, datetime) else date.fromisoformat(str(x)[:10])


def _chain_symbol(o) -> str:
    inst = getattr(o, "instrument", None)
    return getattr(inst, "symbol", None) or getattr(o, "symbol", "") or str(o)
