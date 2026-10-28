# src/stratbt/adapters/kite.py
"""
Zerodha Kite Connect — NSE (and BSE) index and stock options, via the official `kiteconnect` SDK.

How Kite differs from the other venues, and how this adapter handles it:

* **No paper environment.** Zerodha has no sandbox, so there are two modes: `dry-run` (real quotes and decisions;
  each order is checked with Kite's own order-margins endpoint and then NOT sent) and `live` (real money,
  behind the usual confirmation token and loss limit). `paper` is refused.
* **Quantity is in units, not lots.** An order of 1 contract is `lot_size` units. The lot size and tick size come
  from the instruments master, and the broker's P&L multiplier is the lot size, so PnL is in rupees.
* **Symbols come from the instruments master.** Kite trading symbols (weekly and monthly formats differ) are never
  constructed, only looked up in the `NFO`/`BFO` instruments dump.
* **Daily login.** The access token expires every day. `stratbt kite-login` walks through getting a new one.
* **Costs.** Zerodha charges a flat brokerage per executed order (default ₹20 here). Statutory charges (STT,
  exchange fees, GST, stamp duty) are NOT modelled, so reported PnL is somewhat better than the contract note.
* **Intraday products.** The default product is NRML (positions are yours until you close them). With MIS, Zerodha
  auto-squares positions around 15:20, before this adapter's default 15:25 flatten time; if you use MIS, set an
  earlier session close.

Credentials, from the environment: KITE_API_KEY and KITE_ACCESS_TOKEN (KITE_API_SECRET for the login helper).
Install: pip install 'strat-backtester[kite]'.

This adapter follows Zerodha's published API documentation and the SDK's signatures; it has been exercised only
against a fake client, never against a live Kite account. Two conventions are inferred from the docs' examples
rather than stated, so confirm them with a one-lot test: that a short position has a negative `quantity`, and the
index quote key for NIFTY BANK.
"""
from __future__ import annotations

import logging
import math
import os
from datetime import date, datetime, time
from urllib.parse import parse_qs, urlparse

from stratbt.adapters.base import FillReport, LiveBroker, LiveSafety, PollingFeed, Quote
from stratbt.instruments import Instrument
from stratbt.risk import RiskLimits
from stratbt.strategy.base import Order

logger = logging.getLogger(__name__)

TERMINAL = frozenset({"COMPLETE", "REJECTED", "CANCELLED"})
TICK = 0.05
NSE_OPEN, KITE_CLOSE = time(9, 15), time(15, 25)        # flatten window ends 5 minutes before the 15:30 close

# Underlying index -> quote key. NIFTY BANK / FIN SERVICE / MID SELECT follow Zerodha's naming but were not in the
# docs page consulted; the feed verifies the key against a live quote at start-up and says so if it is wrong.
INDEX_KEYS = {
    "NIFTY": "NSE:NIFTY 50", "BANKNIFTY": "NSE:NIFTY BANK", "FINNIFTY": "NSE:NIFTY FIN SERVICE",
    "MIDCPNIFTY": "NSE:NIFTY MID SELECT", "SENSEX": "BSE:SENSEX", "BANKEX": "BSE:BANKEX",
}


def make_client(api_key: str | None = None, access_token: str | None = None, *, verify: bool = True):
    """A KiteConnect client from arguments or KITE_API_KEY / KITE_ACCESS_TOKEN; checks the token is still valid."""
    from kiteconnect import KiteConnect
    key, token = api_key or os.environ.get("KITE_API_KEY"), access_token or os.environ.get("KITE_ACCESS_TOKEN")
    if not key or not token:
        raise ValueError("Kite credentials missing: set KITE_API_KEY and KITE_ACCESS_TOKEN. "
                         "The access token expires daily; get a fresh one with `stratbt kite-login`.")
    kite = KiteConnect(api_key=key, access_token=token)
    if verify:
        try:
            kite.profile()
        except Exception as e:
            raise ValueError(f"Kite rejected the access token ({type(e).__name__}: {e}). "
                             "It expires every day; run `stratbt kite-login` for a new one.") from e
    return kite


def login_url(api_key: str | None = None, kite=None) -> str:
    from kiteconnect import KiteConnect
    kite = kite or KiteConnect(api_key=api_key or os.environ["KITE_API_KEY"])
    return kite.login_url()


def request_token_from(text: str) -> str:
    """Accept either the bare request token or the full redirect URL Kite sends you to after login."""
    text = text.strip()
    if "request_token=" in text:
        tokens = parse_qs(urlparse(text).query).get("request_token")
        if tokens:
            return tokens[0]
        return text.split("request_token=")[1].split("&")[0]
    return text


def exchange_request_token(request_token: str, api_secret: str | None = None, api_key: str | None = None, kite=None) -> str:
    """Trade the one-time request token for today's access token."""
    from kiteconnect import KiteConnect
    kite = kite or KiteConnect(api_key=api_key or os.environ["KITE_API_KEY"])
    secret = api_secret or os.environ.get("KITE_API_SECRET")
    if not secret:
        raise ValueError("Set KITE_API_SECRET (from your Kite Connect app page) to complete the login.")
    return kite.generate_session(request_token_from(request_token), api_secret=secret)["access_token"]


class KiteInstruments:
    """The day's option contracts for one exchange, from Kite's instruments master (regenerated daily by Zerodha)."""

    def __init__(self, kite, exchange: str = "NFO", rows: list[dict] | None = None):
        self.exchange = exchange
        self.rows = rows if rows is not None else kite.instruments(exchange)
        self._by_inst: dict[Instrument, dict] = {}
        self._by_symbol: dict[str, Instrument] = {}
        seg = f"{exchange}-OPT"
        for r in self.rows:
            if r.get("segment") != seg or r.get("instrument_type") not in ("CE", "PE") or not r.get("expiry"):
                continue
            strike = float(r["strike"])
            inst = Instrument(str(r["name"]), _as_date(r["expiry"]), int(strike) if strike.is_integer() else strike, r["instrument_type"])
            self._by_inst[inst] = r
            self._by_symbol[r["tradingsymbol"]] = inst

    def row(self, inst: Instrument) -> dict:
        try:
            return self._by_inst[inst]
        except KeyError:
            raise ValueError(f"{inst.underlier} {inst.expiry} {inst.strike}{inst.option_type} is not in Kite's {self.exchange} "
                             "instruments list") from None

    def instrument(self, tradingsymbol: str) -> Instrument | None:
        return self._by_symbol.get(tradingsymbol)

    def chain(self, name: str, today: date | None = None) -> tuple[date, list[int], int, float]:
        """(front expiry, whole-number strikes, lot size, tick size) for the underlying `name`."""
        today = today or date.today()
        mine = [(i, r) for i, r in self._by_inst.items() if i.underlier == name and i.expiry >= today]
        if not mine:
            raise ValueError(f"Kite lists no upcoming {self.exchange} options for '{name}'. "
                             "Use the exchange name (NIFTY, BANKNIFTY, RELIANCE, …).")
        expiry = min(i.expiry for i, _ in mine)
        front = [(i, r) for i, r in mine if i.expiry == expiry]
        strikes = sorted({int(i.strike) for i, _ in front if float(i.strike).is_integer()})
        if len(strikes) < 4:
            raise ValueError(f"Too few strikes for {name} {expiry}: {strikes}")
        return expiry, strikes, int(front[0][1]["lot_size"]), float(front[0][1].get("tick_size") or TICK)


def _as_date(x) -> date:
    return x.date() if isinstance(x, datetime) else x if isinstance(x, date) else date.fromisoformat(str(x)[:10])


class KiteBroker(LiveBroker):
    name = "kite"

    def __init__(self, kite, table: KiteInstruments, *, lot_size: int, product: str = "NRML", brokerage_per_order: float = 20.0,
                 risk: RiskLimits | None = None, safety: LiveSafety | None = None, **kw):
        safety = safety or LiveSafety(tick=TICK)
        if safety.mode == "paper":
            raise ValueError("Zerodha has no paper-trading environment. Use mode='dry-run' (margin check only) or 'live'.")
        super().__init__(multiplier=float(lot_size), risk=risk, safety=safety, **kw)
        self.kite, self.table, self.lot_size = kite, table, int(lot_size)
        self.product, self.brokerage = product, brokerage_per_order

    # -- order parameters -----------------------------------------------------
    def _params(self, order: Order, limit_price: float) -> dict:
        row = self.table.row(order.instrument)
        return dict(variety="regular", exchange=self.table.exchange, tradingsymbol=row["tradingsymbol"],
                    transaction_type=order.side, quantity=int(order.qty) * int(row["lot_size"]), product=self.product,
                    order_type="LIMIT", price=limit_price, validity="DAY")

    def _preflight(self, order: Order, limit_price: float) -> None:
        """Dry-run: ask Kite what this order would need (validates the symbol, lot multiple and margin) without placing it."""
        p = self._params(order, limit_price)
        self.kite.order_margins([{k: p[k] for k in ("exchange", "tradingsymbol", "transaction_type", "variety", "product",
                                                    "order_type", "quantity", "price")}])

    def _place(self, order: Order, limit_price: float, client_id: str) -> FillReport:
        p = self._params(order, limit_price)
        lot = p["quantity"] // int(order.qty)
        order_id = str(self.kite.place_order(**p, tag=client_id[:20]))        # Kite tags: up to 20 chars, our idempotency key

        def fetch() -> FillReport:
            h = self.kite.order_history(order_id)[-1]
            units = float(h.get("filled_quantity") or 0)
            lots = math.floor(units / lot)
            if units and units != lots * lot:
                logger.error("order %s filled %s units, not a whole number of %s-unit lots", order_id, units, lot)
            return FillReport(lots, float(h.get("average_price") or 0), str(h.get("status", "")).upper(),
                              fee=self.brokerage if lots else 0.0, order_id=order_id)

        return self._track(fetch, lambda: self.kite.cancel_order("regular", order_id), TERMINAL)

    def positions(self) -> dict[Instrument, tuple[float, int]]:
        out = {}
        for p in self.kite.positions().get("net", []):
            if p.get("exchange") != self.table.exchange or p.get("product") != self.product:
                continue
            inst = self.table.instrument(p.get("tradingsymbol", ""))
            units = float(p.get("quantity") or 0)
            if inst is None or units == 0:
                continue
            lot = int(self.table.row(inst)["lot_size"])
            avg = float(p.get("average_price") or 0) or float(p.get("last_price") or 0)    # never 0: it would fake a profit
            out[inst] = (avg, int(units / lot))                                            # negative quantity = short (per the docs' example)
        return out


class KiteFeed(PollingFeed):
    """
    One `kite.quote` call per second returns the underlying and every watched option together (the
    endpoint takes up to 500 instruments and is throttled, so it is never called more than once a tick).
    """

    def __init__(self, underlier: str, expiry: date, strikes, *, kite, table: KiteInstruments, index_key: str,
                 lot_size: int, tick_size: float = TICK, **kw):
        kw.setdefault("tz", "Asia/Kolkata")
        kw.setdefault("session_open", NSE_OPEN)
        kw.setdefault("session_close", KITE_CLOSE)
        super().__init__(underlier, expiry, strikes, **kw)
        self.kite, self.table, self.index_key = kite, table, index_key
        self.lot_size, self.tick_size = lot_size, tick_size

    @classmethod
    def create(cls, ticker: str, kite=None, *, exchange: str = "NFO", index_key: str | None = None,
               today: date | None = None, table: KiteInstruments | None = None, **kw) -> "KiteFeed":
        kite = kite or make_client()
        name = ticker.upper()
        table = table or KiteInstruments(kite, exchange)
        expiry, strikes, lot, tick = table.chain(name, today)
        key = index_key or INDEX_KEYS.get(name) or f"NSE:{name}"
        probe = kite.quote(key)
        if key not in probe:
            raise ValueError(f"Kite returned no quote for '{key}'. Pass the correct underlying key, e.g. "
                             "'NSE:NIFTY 50' for an index or 'NSE:RELIANCE' for a stock.")
        return cls(name, expiry, strikes, kite=kite, table=table, index_key=key, lot_size=lot, tick_size=tick, **kw)

    @staticmethod
    def _best(side: list) -> float | None:
        px = float(side[0].get("price") or 0) if side else 0.0
        return px or None

    def _key(self, inst: Instrument) -> str:
        return f"{self.table.exchange}:{self.table.row(inst)['tradingsymbol']}"

    def _parse(self, data: dict, key: str) -> Quote | None:
        d = data.get(key)
        if d is None:                                         # Kite omits instruments it has no data for
            return None
        depth = d.get("depth") or {}
        last = float(d.get("last_price") or 0) or None
        return Quote(self._best(depth.get("buy", [])), self._best(depth.get("sell", [])), last, ts=self.clock())

    def _fetch(self, wanted: list[Instrument]) -> None:
        keys = {self._key(i): i for i in wanted}
        data = self.kite.quote(self.index_key, *keys)
        under = self._parse(data, self.index_key)
        if under is None:
            raise RuntimeError(f"no quote for the underlying '{self.index_key}'")
        self._under = under
        for key, inst in keys.items():
            q = self._parse(data, key)
            if q is not None:
                self._quotes[inst] = q

    def fetch_underlying(self) -> Quote:
        q = self._parse(self.kite.quote(self.index_key), self.index_key)
        if q is None:
            raise RuntimeError(f"no quote for the underlying '{self.index_key}'")
        return q

    def fetch_options(self, instruments: list[Instrument]) -> dict[Instrument, Quote]:
        keys = {self._key(i): i for i in instruments}
        data = self.kite.quote(*keys)
        return {keys[k]: q for k in keys if (q := self._parse(data, k)) is not None}
