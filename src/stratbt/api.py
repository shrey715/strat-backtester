# src/stratbt/api.py
from datetime import date

from stratbt.engine import BacktestEngine, BacktestResult, ProgressCallback
from stratbt.execution import ExecutionModel
from stratbt.providers.base import DataProvider
from stratbt.providers.synthetic import SyntheticProvider
from stratbt.risk import RiskLimits
from stratbt.strategy import registry


def run_backtest(
    strategy: str = "atm_straddle",
    provider: DataProvider | None = None,
    params: dict | None = None,
    execution: ExecutionModel | None = None,
    risk: RiskLimits | None = None,
    underliers: list[str] | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    progress: ProgressCallback | None = None,
) -> BacktestResult:
    """
    One-call backtest by strategy name.

    Parameters
    ----------
    strategy : str
        A key from `stratbt.strategy.registry.available()`.
    provider : DataProvider, optional
        Defaults to a seeded SyntheticProvider, so this works with no data.
    params : dict, optional
        Strategy parameter overrides, e.g. {"width": 3}.
    execution, risk : optional
        See ExecutionModel / RiskLimits.
    underliers, start_date, end_date, progress :
        See BacktestEngine.run_all.

    Examples
    --------
    >>> result = run_backtest("iron_condor", params={"width": 3})   # doctest: +SKIP
    """
    provider = provider or SyntheticProvider()
    spec = registry.get(strategy)
    engine = BacktestEngine(provider, spec.factory(provider, **(params or {})),
                            execution=execution, risk=risk)
    result = engine.run_all(underliers, start_date, end_date, progress=progress)
    result.meta.update(strategy=spec.label, strategy_key=spec.key, allow_short=spec.cls.allow_short, unlimited_risk=spec.cls.unlimited_risk, params=spec.resolve(**(params or {})))
    return result
