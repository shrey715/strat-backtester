"""stratbt — a pluggable options-strategy backtesting engine."""
from stratbt.api import run_backtest
from stratbt.broker import Broker, Fill, PaperBroker
from stratbt.engine import BacktestEngine, BacktestResult, DayResult
from stratbt.execution import ExecutionModel
from stratbt.providers import DataProvider, NSETickProvider, SyntheticProvider
from stratbt.risk import RiskLimits
from stratbt.strategy import BaseStrategy, Order
from stratbt.strategy.registry import ParamSpec, StrategySpec, register

__version__ = "0.2.0"

__all__ = [
    "run_backtest", "BacktestEngine", "BacktestResult", "DayResult",
    "DataProvider", "NSETickProvider", "SyntheticProvider",
    "ExecutionModel", "RiskLimits", "Broker", "PaperBroker", "Fill",
    "BaseStrategy", "Order", "StrategySpec", "ParamSpec", "register",
]
