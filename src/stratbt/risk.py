# src/stratbt/risk.py
from dataclasses import dataclass


@dataclass(frozen=True)
class RiskLimits:
    """
    Hard guard-rails enforced between a strategy and the market — in
    backtests exactly as they should be in live trading. All None = no limits.

    Attributes
    ----------
    max_daily_loss : float, optional
        Currency amount (positive). Once net equity for the day falls to
        -max_daily_loss the broker flattens everything and rejects new
        opening orders for the rest of the day.
    max_open_contracts : int, optional
        Cap on total absolute contracts held across all instruments.
        Orders that would exceed it are rejected; orders that reduce
        exposure are always allowed.
    max_order_qty : int, optional
        Cap on the size of a single order.
    """
    max_daily_loss: float | None = None
    max_open_contracts: int | None = None
    max_order_qty: int | None = None

    def __post_init__(self):
        for f in ("max_daily_loss", "max_open_contracts", "max_order_qty"):
            v = getattr(self, f)
            if v is not None and v <= 0:
                raise ValueError(f"{f} must be positive")
