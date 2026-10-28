# src/stratbt/instruments.py
from datetime import date, datetime
from dataclasses import dataclass
import re

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

    @property
    def filename(self)->str:
        # returns e.g. "NIFTY22110318150CE.csv"
        return f"{self.underlier}{self.expiry.strftime('%y%m%d')}{self.strike}{self.option_type}.csv"
    
INSTRUMENT_RE = re.compile(r"^([A-Z]+)(\d{6})(\d+)(CE|PE)$")

def parse_instrument(filename: str) -> Instrument:
    """
    Parse an NSE option filename into its component fields.

    Parameters
    ----------
    filename : str
        e.g. "NIFTY22110318150CE.csv" or "NIFTY22110318150CE"
        (".csv" suffix is optional). Format: UNDERLIER + YYMMDD + STRIKE + CE|PE.

    Returns
    -------
    Instrument
        The parsed underlier, expiry date, strike, and option type.

    Raises
    ------
    ValueError
        If filename doesn't match the expected pattern.
    """
    if filename.endswith(".csv"):
        filename = filename[:-4]  # strip ".csv"
    m = INSTRUMENT_RE.match(filename)
    if not m:
        raise ValueError(f"Invalid instrument filename: {filename}")
    underlier, expiry_str, strike_str, option_type = m.groups()
    expiry = datetime.strptime(expiry_str, "%y%m%d").date()
    return Instrument(
        underlier=underlier,
        expiry=expiry,
        strike=int(strike_str),
        option_type=option_type
    )