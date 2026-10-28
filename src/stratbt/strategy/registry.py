# src/stratbt/strategy/registry.py
"""
Name -> strategy lookup with typed, UI-friendly parameter specs.

The CLI and dashboard are generated from this registry, so registering a
strategy is all it takes to make it runnable everywhere:

    from stratbt.strategy.registry import StrategySpec, ParamSpec, register

    register(StrategySpec(
        key="my_strategy", label="My Strategy", cls=MyStrategy,
        description="...", params=[ParamSpec("lookback_s", 60, 1, 600, 1)],
    ))

Third-party packages can ship strategies without touching this repo by
exposing a StrategySpec (or a list of them) under the `stratbt.strategies`
entry-point group.
"""
import inspect
import logging
from dataclasses import dataclass, field
from importlib.metadata import entry_points

from stratbt.strategy.atm_straddle import ATMStraddleStrategy
from stratbt.strategy.base import BaseStrategy
from stratbt.strategy.fixed_window import FixedWindowStraddleStrategy
from stratbt.strategy.structures import IronCondor, LongStrangle, ShortStraddle, ShortStrangle

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParamSpec:
    """A tunable constructor argument."""
    name: str
    default: float
    min: float
    max: float
    step: float = 1
    label: str = ""

    @property
    def is_int(self) -> bool:
        return all(float(x).is_integer() for x in (self.default, self.min, self.max, self.step))

    def coerce(self, value):
        v = float(value)
        if not self.min <= v <= self.max:
            raise ValueError(f"{self.name}={v} outside [{self.min}, {self.max}]")
        return int(v) if self.is_int else v


@dataclass(frozen=True)
class StrategySpec:
    key: str
    label: str
    cls: type[BaseStrategy]
    description: str
    params: list[ParamSpec] = field(default_factory=list)

    def resolve(self, **overrides) -> dict:
        """Validated constructor kwargs: defaults overlaid with `overrides`."""
        known = {p.name: p for p in self.params}
        unknown = set(overrides) - set(known)
        if unknown:
            raise ValueError(f"{self.key}: unknown parameter(s) {sorted(unknown)}; valid: {sorted(known)}")
        out = {p.name: p.coerce(p.default) for p in self.params}
        out.update({k: known[k].coerce(v) for k, v in overrides.items()})
        return out

    def factory(self, provider=None, **overrides):
        """
        Build the `(underlier, expiry, strikes) -> BaseStrategy` callable the
        engine wants. If a provider is given, its session open/close times
        are passed to strategies that accept them.
        """
        kwargs = self.resolve(**overrides)
        accepted = inspect.signature(self.cls.__init__).parameters
        if provider is not None:
            if "session_open" in accepted:
                kwargs["session_open"] = provider.session_open
            if "session_close" in accepted:
                kwargs["session_close"] = provider.session_close

        def make(underlier, expiry, strikes):
            return self.cls(underlier, expiry, strikes, **kwargs)
        return make


_REGISTRY: dict[str, StrategySpec] = {}
_plugins_loaded = False


def register(spec: StrategySpec, replace: bool = False) -> StrategySpec:
    if spec.key in _REGISTRY and not replace:
        raise ValueError(f"Strategy '{spec.key}' is already registered")
    _REGISTRY[spec.key] = spec
    return spec


def _load_plugins() -> None:
    global _plugins_loaded
    if _plugins_loaded:
        return
    _plugins_loaded = True
    for ep in entry_points(group="stratbt.strategies"):
        try:
            obj = ep.load()
            for spec in (obj if isinstance(obj, (list, tuple)) else [obj]):
                register(spec, replace=True)
        except Exception:
            logger.exception("Failed to load strategy plugin %s", ep.name)


def available() -> dict[str, StrategySpec]:
    """All registered strategies (built-ins plus entry-point plugins), by key."""
    _load_plugins()
    return dict(_REGISTRY)


def get(key: str) -> StrategySpec:
    specs = available()
    if key not in specs:
        raise KeyError(f"Unknown strategy '{key}'. Available: {', '.join(sorted(specs))}")
    return specs[key]


_ENTRY = ParamSpec("entry_delay_s", 300, 0, 3600, 60, "Entry delay (seconds after open)")
_EXIT = ParamSpec("exit_buffer_s", 300, 0, 3600, 60, "Exit buffer (seconds before close)")
_WIDTH = ParamSpec("width", 2, 1, 20, 1, "Width (strikes from ATM)")
_WING = ParamSpec("wing", 2, 1, 20, 1, "Wing (strikes beyond the short legs)")
_TP = ParamSpec("take_profit", 0.0, 0.0, 100000.0, 0.5, "Take-profit (price points; 0 = off)")
_SL = ParamSpec("stop_loss", 0.0, 0.0, 100000.0, 0.5, "Stop-loss (price points; 0 = off)")

for _spec in (
    StrategySpec("atm_straddle", "ATM Straddle (Rolling)", ATMStraddleStrategy,
                 "Hold the ATM straddle at every second; roll whenever the nearest strike changes. "
                 "Maximum roll frequency."),
    StrategySpec("fixed_window", "Fixed Window Straddle", FixedWindowStraddleStrategy,
                 "Buy the ATM straddle N seconds after the open, hold without rolling, exit M seconds "
                 "before the close.", [_ENTRY, _EXIT]),
    StrategySpec("short_straddle", "Short Straddle", ShortStraddle,
                 "Sell the ATM straddle once, hold to the exit time. Collects theta; unlimited risk.",
                 [_ENTRY, _EXIT, _TP, _SL]),
    StrategySpec("long_strangle", "Long Strangle", LongStrangle,
                 "Buy an OTM call and put `width` strikes from ATM, hold to the exit time.",
                 [_ENTRY, _EXIT, _WIDTH, _TP, _SL]),
    StrategySpec("short_strangle", "Short Strangle", ShortStrangle,
                 "Sell an OTM call and put `width` strikes from ATM, hold to the exit time. Unlimited risk.",
                 [_ENTRY, _EXIT, _WIDTH, _TP, _SL]),
    StrategySpec("iron_condor", "Iron Condor", IronCondor,
                 "Short strangle at `width` strikes with long protective wings `wing` strikes further out. "
                 "Defined risk.", [_ENTRY, _EXIT, _WIDTH, _WING, _TP, _SL]),
):
    register(_spec)
