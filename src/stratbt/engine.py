# src/stratbt/engine.py

import logging
from typing import Callable

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import date

from stratbt.broker import PaperBroker, Rejection
from stratbt.execution import ExecutionModel
from stratbt.instruments import Instrument
from stratbt.position import Trade
from stratbt.providers.base import DataProvider
from stratbt.risk import RiskLimits
from stratbt.strategy.base import BaseStrategy, Order

logger = logging.getLogger(__name__)

@dataclass
class DayResult:
    """Everything produced by one underlier for one day of trading."""
    date: date
    underlier: str
    trades: list[Trade]          # ordered list of transactions
    mtm: pd.Series                # per-second net MTM equity (after fees)
    final_pnl: float              # final net PnL for the day (realized - fees)
    holdings_log: pd.DataFrame    # what was held, second by second
    fees: float = 0.0             # total fees paid (already deducted from final_pnl)
    rejections: list[Rejection] = field(default_factory=list)  # orders the broker refused
    multiplier: float = 1.0       # currency value of one price point per contract

class BacktestResult:
    """
    Aggregated results across one or more DayResults (any mix of days and
    underliers). Each property below re-derives its view from the raw
    day_results on every access rather than caching, since a BacktestResult
    is typically built once and read a handful of times, not in a hot loop.
    """

    def __init__(self, day_results: list[DayResult], unit: str = "pts"):
        """
        Parameters
        ----------
        day_results : list[DayResult]
            One entry per (day, underlier) combination that was run. May be
            empty (an empty backtest is valid; every property below returns
            an appropriately-shaped empty result rather than raising).
        unit : str, default "pts"
            Label for PnL numbers ("pts" for index points, "USD", ...).
        """
        self._results = day_results
        self.unit = unit

    @property
    def day_results(self) -> list[DayResult]:
        """A copy of the raw per-day, per-underlier results list."""
        return list(self._results)

    @property
    def trades_df(self) -> pd.DataFrame:
        """
        Returns
        -------
        pd.DataFrame
            One row per trade across all days/underliers, sorted by
            timestamp. Columns: timestamp, date, underlier, instrument,
            expiry, strike, option_type, side, price, qty, fee. Empty (but
            with these columns) if there are no trades.
        """
        rows = []
        for dr in self._results:
            for t in dr.trades:
                rows.append({
                    "timestamp": t.timestamp,
                    "date": dr.date,
                    "underlier": dr.underlier,
                    "instrument": t.instrument.filename,
                    "expiry": t.instrument.expiry,
                    "strike": t.instrument.strike,
                    "option_type": t.instrument.option_type,
                    "side": t.side,
                    "price": t.price,
                    "qty": t.qty,
                    "fee": t.fee,
                })
        if not rows:
            return pd.DataFrame(columns=[
                "timestamp", "date", "underlier", "instrument", "expiry",
                "strike", "option_type", "side", "price", "qty", "fee"
            ])
        df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
        return df

    @property
    def notional_df(self) -> pd.DataFrame:
        """
        trades_df plus a "notional" column: price * qty * contract multiplier
        for each trade, i.e. the currency value changing hands.
        """
        mult = {(dr.date, dr.underlier): dr.multiplier for dr in self._results}
        df = self.trades_df.copy()
        df["notional"] = df["price"] * df["qty"] * [
            mult[(d, u)] for d, u in zip(df["date"], df["underlier"])
        ]
        return df

    @property
    def daily_pnl(self) -> pd.DataFrame:
        """
        Returns
        -------
        pd.DataFrame
            Daily final PnL, pivoted: index = date, columns = underlier.
            Empty (zero rows, zero columns) if there are no results — the
            pivoted shape has no underlier names to report, so it must not
            fall back to the long-format ["date","underlier","pnl"] schema
            used to build it (metrics functions iterate `.columns` expecting
            underlier tickers).
        """
        rows = []
        for dr in self._results:
            rows.append({
                "date": dr.date,
                "underlier": dr.underlier,
                "pnl": dr.final_pnl
            })
        if not rows:
            return pd.DataFrame(index=pd.Index([], name="date"))
        df = pd.DataFrame(rows).pivot(index="date", columns="underlier", values="pnl")
        return df

    @property
    def mtm_df(self) -> pd.DataFrame:
        """
        Returns
        -------
        pd.DataFrame
            Per-second MTM equity, pivoted: DatetimeIndex, columns =
            underlier. Multiple days for the same underlier are
            concatenated in day_results order. Empty if there are no results.
        """
        by_underlier: dict[str, list[pd.Series]] = {}
        for dr in self._results:
            by_underlier.setdefault(dr.underlier, []).append(dr.mtm)
        combined = {}
        for underlier, series_list in by_underlier.items():
            combined[underlier] = pd.concat(series_list)
        if not combined:
            return pd.DataFrame(columns=[])
        df = pd.DataFrame(combined)
        return df

    @property
    def holdings_log_df(self) -> pd.DataFrame:
        """
        Returns
        -------
        pd.DataFrame
            Per-second holdings log concatenated across all days and
            underliers, with "date" and "underlier" columns added. Only
            includes seconds where a position was actually open. Empty if
            no position was ever held.
        """
        frames = []
        for dr in self._results:
            if dr.holdings_log.empty:
                continue
            df = dr.holdings_log.copy()
            df["date"] = dr.date
            df["underlier"] = dr.underlier
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames).sort_index()

StrategyFactory = Callable[[str, date, np.ndarray], BaseStrategy]
ProgressCallback = Callable[[str, date, int, int], None]

class BacktestEngine:
    """
    Drives the backtest loop, day by day and underlier by underlier: for
    each, load the day's underlying series, resolve the front expiry and
    strikes, construct a fresh strategy and PaperBroker, then walk every
    second of the session asking the strategy for orders and recording MTM.

    Option series are loaded lazily and cached per day inside run_day —
    never opened from inside the per-second loop.
    """

    def __init__(
        self,
        provider: DataProvider,
        strategy_factory: StrategyFactory,
        execution: ExecutionModel | None = None,
        risk: RiskLimits | None = None,
        flatten_at_close: bool = True,
    ):
        """
        Parameters
        ----------
        provider : DataProvider
            Source of days, expiries, strikes and prices.
        strategy_factory : StrategyFactory
            Callable(underlier, expiry, strikes) -> BaseStrategy, invoked
            once per (day, underlier) to construct a fresh strategy instance
            — strategies are stateful, so state must never carry over
            between days. See `stratbt.strategy.registry` to build one by name.
        execution : ExecutionModel, optional
            Fill delay, slippage and fees. Defaults to ExecutionModel()
            (next-second fills, no costs). Use ExecutionModel.instant() to
            reproduce same-second optimistic fills.
        risk : RiskLimits, optional
            Position/loss guard-rails enforced by the broker.
        flatten_at_close : bool, default True
            Close anything still open at the last second of the session, so
            every day's PnL is fully realized (no overnight positions).
        """
        self.provider = provider
        self.strategy_factory = strategy_factory
        self.execution = execution or ExecutionModel()
        self.risk = risk or RiskLimits()
        self.flatten_at_close = flatten_at_close

    def run(self, underlier: str, progress: ProgressCallback | None = None) -> BacktestResult:
        """
        Run one underlier across every trading day the provider knows about.

        Parameters
        ----------
        underlier : str
            e.g. "NIFTY" or "SPY".
        progress : callable, optional
            Called as progress(underlier, day, done, total) after each day.

        Returns
        -------
        BacktestResult
        """
        return self.run_all([underlier], progress=progress)

    def run_all(
        self,
        underliers:  list[str] | None = None,
        start_date:  date | None = None,
        end_date:    date | None = None,
        progress:    ProgressCallback | None = None,
    ) -> BacktestResult:
        """
        Run one or more underliers, optionally filtered to a date range.

        Parameters
        ----------
        underliers : list[str], optional
            Defaults to every underlier the provider offers.
        start_date, end_date : date, optional
            Inclusive date filter.
        progress : callable, optional
            Called as progress(underlier, day, done, total) after each day.

        Returns
        -------
        BacktestResult
            Combined results across every (underlier, day) pair run.
        """
        if underliers is None:
            underliers = self.provider.underliers

        days = self.provider.trading_days
        if start_date:
            days = [d for d in days if d >= start_date]
        if end_date:
            days = [d for d in days if d <= end_date]

        total = len(underliers) * len(days)
        all_results = []
        for underlier in underliers:
            for day in days:
                dr = self.run_day(day, underlier)
                logger.info("%s %s: net PnL %.2f, %d trades", underlier, day, dr.final_pnl, len(dr.trades))
                all_results.append(dr)
                if progress:
                    progress(underlier, day, len(all_results), total)
        return BacktestResult(all_results, unit=self.provider.unit)

    def run_day(self, day: date, underlier: str) -> DayResult:
        """
        Run one underlier for one trading day, second by second.

        Parameters
        ----------
        day : date
        underlier : str

        Returns
        -------
        DayResult

        Raises
        ------
        ValueError
            If the day/underlier has no data (propagated from the provider).
        RuntimeError
            If an order that would close a position has no fill price —
            this would leave the position open with no way to close it,
            which indicates a data bug rather than a normal market condition.
        """
        provider = self.provider
        front_exp = provider.front_expiry(day, underlier)
        strikes = provider.strikes(day, underlier, front_exp)

        underlying = provider.underlying_prices(day, underlier)
        grid = underlying.index
        underlying_values = underlying.to_numpy()
        last_idx = len(grid) - 1

        strategy = self.strategy_factory(underlier, front_exp, strikes)
        broker = PaperBroker(
            multiplier=provider.multiplier(underlier),
            execution=self.execution,
            risk=self.risk,
            allow_short=getattr(strategy, "allow_short", False),
        )
        multiplier = broker.position.multiplier

        option_cache: dict[Instrument, np.ndarray | None] = {}
        filename_cache: dict[Instrument, str] = {}
        current_idx = 0

        def option_price(inst: Instrument) -> float | None:
            """
            Lazily load and cache one instrument's price series as a numpy
            array, then return its value at the second currently being processed.
            """
            if inst not in option_cache:
                series = provider.option_prices(day, inst)
                option_cache[inst] = series.to_numpy() if series is not None else None
            arr = option_cache[inst]
            if arr is None:
                return None
            val = arr[current_idx]
            return None if np.isnan(val) else float(val)

        def cached_filename(inst: Instrument) -> str:
            """
            Instrument.filename formats a date via strftime() on every call;
            memoizing it matters because the holdings log recomputes it for
            both legs on every second the position is held.
            """
            if inst not in filename_cache:
                filename_cache[inst] = inst.filename
            return filename_cache[inst]

        mtm_values = np.zeros(len(grid))
        holdings_records = []
        pending: list[tuple[int, Order]] = []   # (fill index, order) in flight

        for current_idx, ts in enumerate(grid):
            # 1. fill any in-flight orders that are due now
            if pending:
                due = [o for i, o in pending if i <= current_idx]
                pending = [(i, o) for i, o in pending if i > current_idx]
                for order in due:
                    broker.submit(order, ts, option_price(order.instrument))

            # 2. ask for new orders (not while earlier orders are still in flight)
            if not pending:
                holdings = broker.holdings
                if broker.halted:
                    orders = broker.flatten_orders()
                else:
                    fut_price = float(underlying_values[current_idx])
                    orders = strategy.on_second(ts, fut_price, option_price, holdings)
                if self.flatten_at_close and current_idx == last_idx:
                    orders = list(orders) + [
                        o for o in broker.flatten_orders()
                        if not any(x.instrument == o.instrument for x in orders)
                    ]
                for order in orders:
                    fill_idx = min(current_idx + self.execution.delay_s, last_idx)
                    if fill_idx == current_idx:
                        broker.submit(order, ts, option_price(order.instrument))
                    else:
                        pending.append((fill_idx, order))

            # Single snapshot of post-trade holdings, reused by both MTM and
            # the log below — position.holdings copies its internal dict on
            # every access, and this loop runs ~20,000 times per day.
            holdings = broker.holdings
            equity = self._compute_mtm(broker.position.net_realized, holdings, option_price, multiplier)
            mtm_values[current_idx] = equity
            broker.check_risk(equity)

            if holdings:
                insts = list(holdings.keys())
                ce_inst = next((i for i in insts if i.option_type == "CE"), None)
                pe_inst = next((i for i in insts if i.option_type == "PE"), None)
                holdings_records.append({
                    "timestamp":    ts,
                    "strike":       ce_inst.strike if ce_inst else (pe_inst.strike if pe_inst else None),
                    "ce_instrument": cached_filename(ce_inst) if ce_inst else None,
                    "pe_instrument": cached_filename(pe_inst) if pe_inst else None,
                    "ce_entry":     holdings[ce_inst][0] if ce_inst else None,
                    "pe_entry":     holdings[pe_inst][0] if pe_inst else None,
                    "ce_mtm_price": option_price(ce_inst) if ce_inst else None,
                    "pe_mtm_price": option_price(pe_inst) if pe_inst else None,
                })

        mtm_series = pd.Series(mtm_values, index=grid, name=f"{underlier}_mtm")
        holdings_log = pd.DataFrame(holdings_records).set_index("timestamp") if holdings_records else pd.DataFrame()
        position = broker.position
        return DayResult(
            date=day,
            underlier=underlier,
            trades=position.trades,
            mtm=mtm_series,
            final_pnl=position.net_realized,
            holdings_log=holdings_log,
            fees=position.fees,
            rejections=list(broker.rejections),
            multiplier=multiplier,
        )

    def _compute_mtm(
        self,
        realized: float,
        holdings: dict[Instrument, tuple[float, int]],
        option_price: Callable,
        multiplier: float = 1.0,
    ) -> float:
        """
        Parameters
        ----------
        realized : float
            Net realized PnL (after fees) as of this second.
        holdings : dict[Instrument, tuple[float, int]]
            An already-fetched holdings snapshot (not a Position) so callers
            already holding one in the per-second loop don't fetch it twice.
        option_price : Callable[[Instrument], float | None]
        multiplier : float, default 1.0
            Currency value of one price point per contract.

        Returns
        -------
        float
            realized + unrealized PnL on `holdings` at current prices.
            Instruments with no available price contribute zero rather than
            raising — a momentary price gap on a held instrument must not
            blow up the whole backtest.
        """
        if not holdings:
            return realized
        unrealized = 0.0
        for inst, (entry_price, qty) in holdings.items():
            price = option_price(inst)
            if price is not None:
                unrealized += (price - entry_price) * qty * multiplier
        return realized + unrealized
