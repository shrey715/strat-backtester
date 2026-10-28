# src/stratbt/adapters/ibkr.py
"""
Interactive Brokers, via `ib_async` (the maintained successor to ib_insync).

IBKR covers most markets, including NSE. It needs Trader Workstation or IB Gateway running and logged in
with API access enabled. Paper trading uses a separate paper account (its id starts with "DU") on its own
port: TWS paper 7497, TWS live 7496, Gateway paper 4002, Gateway live 4001.

Mode and account must agree, and this adapter checks: dry-run/paper refuse a non-paper account, and live
refuses a paper one, so a wrong port can't silently turn a rehearsal into real trading or vice versa.

Install: pip install 'strat-backtester[ibkr]'.  Not exercised against a real TWS/Gateway session here.
"""
from __future__ import annotations

from datetime import date, datetime

from stratbt.adapters.base import FillReport, LiveBroker, LiveSafety, PollingFeed, Quote
from stratbt.instruments import Instrument
from stratbt.risk import RiskLimits
from stratbt.strategy.base import Order

DONE = frozenset({"Filled", "Cancelled", "ApiCancelled", "Inactive"})
PAPER_PORTS = {"dry-run": 7497, "paper": 7497, "live": 7496}


def connect(host: str = "127.0.0.1", port: int | None = None, client_id: int = 7, mode: str = "dry-run", timeout: float = 8.0):
    from ib_async import IB
    ib = IB()
    ib.connect(host, port or PAPER_PORTS[mode], clientId=client_id, timeout=timeout)
    return ib


def check_account_matches_mode(ib, mode: str) -> None:
    accounts = list(ib.managedAccounts())
    if not accounts:
        raise RuntimeError("IBKR reports no managed accounts.")
    paper = [a.startswith("DU") for a in accounts]
    if mode in ("dry-run", "paper") and not all(paper):
        raise PermissionError(f"mode='{mode}' but IBKR accounts {accounts} include a non-paper account. Connect to the paper port.")
    if mode == "live" and any(paper):
        raise PermissionError(f"mode='live' but IBKR accounts {accounts} include a paper account. Connect to the live port.")


def _px(v) -> float | None:
    """IBKR reports -1 / nan when there is no data."""
    return float(v) if v is not None and v == v and v > 0 else None


class IBKRBroker(LiveBroker):
    name = "ibkr"

    def __init__(self, ib=None, *, host: str = "127.0.0.1", port: int | None = None, client_id: int = 7,
                 exchange: str = "SMART", currency: str = "USD", multiplier: str = "100", account: str | None = None,
                 risk: RiskLimits | None = None, safety: LiveSafety | None = None, **kw):
        safety = safety or LiveSafety()
        if ib is None:
            ib = connect(host, port, client_id, safety.mode)
        # ib_async only processes its network events inside ib.sleep(), so order tracking must wait with it
        extra = {"sleep": ib.sleep} if callable(getattr(ib, "sleep", None)) else {}
        super().__init__(multiplier=float(multiplier), risk=risk, safety=safety, **extra, **kw)
        check_account_matches_mode(ib, safety.mode)
        self.ib, self.exchange, self.currency, self.mult, self.account = ib, exchange, currency, str(multiplier), account
        self._contracts: dict[Instrument, object] = {}

    def contract(self, inst: Instrument):
        if inst not in self._contracts:
            from ib_async import Option
            c = Option(symbol=inst.underlier, lastTradeDateOrContractMonth=f"{inst.expiry:%Y%m%d}", strike=float(inst.strike),
                       right="C" if inst.option_type == "CE" else "P", exchange=self.exchange, multiplier=self.mult,
                       currency=self.currency)
            qualified = self.ib.qualifyContracts(c)
            if not qualified or not getattr(qualified[0], "conId", 0):
                raise ValueError(f"IBKR could not qualify {inst.filename}")
            self._contracts[inst] = qualified[0]
        return self._contracts[inst]

    def _place(self, order: Order, limit_price: float, client_id: str) -> FillReport:
        from ib_async import LimitOrder
        o = LimitOrder("BUY" if order.side == "BUY" else "SELL", order.qty, limit_price, tif="DAY", orderRef=client_id)
        if self.account:
            o.account = self.account
        trade = self.ib.placeOrder(self.contract(order.instrument), o)

        def fetch() -> FillReport:
            st = trade.orderStatus
            return FillReport(float(st.filled or 0), float(st.avgFillPrice or 0), str(st.status), order_id=str(getattr(trade.order, "orderId", "")))

        return self._track(fetch, lambda: self.ib.cancelOrder(trade.order), DONE)

    def positions(self) -> dict[Instrument, tuple[float, int]]:
        out = {}
        for p in self.ib.positions():
            c = p.contract
            if getattr(c, "secType", "") != "OPT":
                continue
            expiry = datetime.strptime(str(c.lastTradeDateOrContractMonth)[:8], "%Y%m%d").date()
            strike = int(c.strike) if float(c.strike).is_integer() else float(c.strike)
            inst = Instrument(c.symbol, expiry, strike, "CE" if c.right in ("C", "CALL") else "PE")
            mult = float(c.multiplier or self.mult)
            out[inst] = (float(p.avgCost) / mult, int(p.position))      # IBKR reports cost per contract, not per share
        return out


class IBKRFeed(PollingFeed):
    """Streaming market data from TWS/Gateway, sampled once a second."""

    def __init__(self, underlier: str, expiry: date, strikes, *, ib, underlying_contract, option_factory, **kw):
        super().__init__(underlier, expiry, strikes, **kw)
        self.ib, self._underlying, self._option = ib, underlying_contract, option_factory
        self._tickers: dict = {}
        self._under_ticker = None

    @classmethod
    def create(cls, ticker: str, ib=None, *, host: str = "127.0.0.1", port: int | None = None, client_id: int = 8,
               mode: str = "dry-run", exchange: str = "SMART", currency: str = "USD", multiplier: str = "100",
               underlying_kind: str = "Stock", delayed: bool = False, today: date | None = None, **kw) -> "IBKRFeed":
        import ib_async
        ib = ib or connect(host, port, client_id, mode)
        if delayed:
            ib.reqMarketDataType(3)
        ticker = ticker.upper()
        under = getattr(ib_async, underlying_kind)(ticker, exchange if underlying_kind == "Stock" else "", currency)
        under = ib.qualifyContracts(under)[0]
        chains = ib.reqSecDefOptParams(ticker, "", "STK" if underlying_kind == "Stock" else "IND", under.conId)
        chain = max((c for c in chains if c.exchange == exchange), key=lambda c: len(c.strikes), default=None) or max(chains, key=lambda c: len(c.strikes))
        today = today or date.today()
        expiries = sorted(d for d in (datetime.strptime(e, "%Y%m%d").date() for e in chain.expirations) if d >= today)
        if not expiries:
            raise ValueError(f"IBKR lists no upcoming expirations for {ticker}.")
        strikes = sorted({int(k) for k in chain.strikes if float(k).is_integer()})
        make = lambda inst: ib_async.Option(inst.underlier, f"{inst.expiry:%Y%m%d}", float(inst.strike),
                                            "C" if inst.option_type == "CE" else "P", exchange, multiplier, currency)
        return cls(ticker, expiries[0], strikes, ib=ib, underlying_contract=under, option_factory=make, **kw)

    def _ticker_for(self, contract):
        return self.ib.reqMktData(contract, "", False, False)

    def fetch_underlying(self) -> Quote:
        if self._under_ticker is None:
            self._under_ticker = self._ticker_for(self._underlying)
        self.ib.sleep(0)
        t = self._under_ticker
        return Quote(_px(t.bid), _px(t.ask), _px(t.last) or _px(t.close), ts=self.clock())

    def fetch_options(self, instruments: list[Instrument]) -> dict[Instrument, Quote]:
        for inst in instruments:
            if inst not in self._tickers:
                c = self.ib.qualifyContracts(self._option(inst))
                if c and getattr(c[0], "conId", 0):
                    self._tickers[inst] = self._ticker_for(c[0])
        self.ib.sleep(0)
        now = self.clock()
        return {i: Quote(_px(t.bid), _px(t.ask), _px(t.last), ts=now) for i, t in self._tickers.items() if i in instruments}

    def close(self) -> None:
        for t in list(self._tickers.values()) + ([self._under_ticker] if self._under_ticker else []):
            try:
                self.ib.cancelMktData(t.contract)
            except Exception:
                pass
