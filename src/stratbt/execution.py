# src/stratbt/execution.py
from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionModel:
    """
    How an order turns into a fill. Defaults remove look-ahead but charge
    nothing; use `realistic()` for a cost-bearing retail-style setup or
    `instant()` for the legacy same-second, zero-cost fill.

    Attributes
    ----------
    delay_s : int
        Seconds between the strategy's decision and the fill. A strategy
        sees the price of second t only once second t is over, so the
        earliest honest fill is the price of second t+1 (delay_s=1). While
        orders are in flight the engine does not consult the strategy.
        Orders issued at the final second fill immediately (no later bar).
    half_spread_bps : float
        Half the bid/ask spread as basis points of price. BUYs pay it, SELLs give it up.
    min_half_spread : float
        Absolute floor for the half-spread, in price units (e.g. 0.01 tick).
    commission_per_contract : float
        Currency charged per contract traded.
    commission_bps : float
        Basis points of notional (price * qty * multiplier) charged per fill.
    """
    delay_s: int = 1
    half_spread_bps: float = 0.0
    min_half_spread: float = 0.0
    commission_per_contract: float = 0.0
    commission_bps: float = 0.0

    def __post_init__(self):
        if self.delay_s < 0:
            raise ValueError("delay_s must be >= 0")
        for f in ("half_spread_bps", "min_half_spread", "commission_per_contract", "commission_bps"):
            if getattr(self, f) < 0:
                raise ValueError(f"{f} must be >= 0")

    @classmethod
    def instant(cls) -> "ExecutionModel":
        """Fill at the same second's last price with zero costs (optimistic; legacy behaviour)."""
        return cls(delay_s=0)

    @classmethod
    def realistic(cls) -> "ExecutionModel":
        """Next-second fills, ~25 bps half-spread (1c floor), $0.65/contract — illustrative retail costs."""
        return cls(delay_s=1, half_spread_bps=25.0, min_half_spread=0.01,
                   commission_per_contract=0.65)

    def fill_price(self, side: str, mid: float) -> float:
        """Price actually paid/received for `side` against reference price `mid`."""
        slip = max(mid * self.half_spread_bps / 10_000.0, self.min_half_spread)
        return mid + slip if side == "BUY" else max(mid - slip, 0.0)

    def fee(self, price: float, qty: int, multiplier: float) -> float:
        """Total fee for one fill, in currency."""
        return (self.commission_per_contract * qty
                + self.commission_bps / 10_000.0 * price * qty * multiplier)
