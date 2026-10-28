# src/stratbt/instruments.py
from datetime import date
from dataclasses import dataclass

@dataclass(frozen=True)
class Instrument:
    """
    One tradeable option contract, identified by underlier/expiry/strike/type.

    Frozen and hashable so it can be used as a dict key (Position._holdings,
    engine.py's option_cache) — two Instruments with the same field values
    are treated as the same instrument.
    """
    underlier: str      # "NIFTY" or "BANKNIFTY"
    expiry: date        # actual date object — not a string like "221103"
    strike: int         # e.g. 18150
    option_type: str    # "CE" or "PE"

    def __post_init__(self):
        # Instruments are dict keys in the per-second hot loop; hash once, not on every lookup.
        object.__setattr__(self, "_hash", hash((self.underlier, self.expiry, self.strike, self.option_type)))

    def __hash__(self) -> int:
        return self._hash  # type: ignore[attr-defined]

    @property
    def filename(self)->str:
        # returns e.g. "NIFTY22110318150CE.csv"
        return f"{self.underlier}{self.expiry.strftime('%y%m%d')}{self.strike}{self.option_type}.csv"
