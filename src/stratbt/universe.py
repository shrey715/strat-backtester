# src/stratbt/universe.py
import numpy as np
from pathlib import Path
from datetime import date, datetime
from collections import defaultdict

from stratbt.instruments import parse_instrument

class Universe:
    """
    Scans the data directory once at construction and indexes every trading
    day, underlier, expiry, and strike found. All lookups after that are O(1)
    dict/array access — no filesystem I/O happens outside __init__.
    """

    def __init__(self, data_dir: Path, underliers: list[str] | None = None):
        """
        Parameters
        ----------
        data_dir : Path
            Root directory containing one "NSE_YYYYMMDD" folder per trading
            day, each with an "Options" subfolder of instrument CSVs.
        underliers : list[str], optional
            Underliers to index (default: ["NIFTY", "BANKNIFTY"]). Instrument
            files for any other underlier are skipped during the scan.
        """
        if underliers is None:
            underliers = ["NIFTY", "BANKNIFTY"]
        self._data_dir = data_dir
        self._underliers = set(underliers)

        self._data: dict[date, dict[str, dict[date, np.ndarray]]] = {}
        self._trading_days: list[date] = []

        self._scan()

    def _scan(self):
        """
        Walk data_dir once, building date -> underlier -> expiry -> sorted
        strikes. Non-instrument files and unrequested underliers are
        silently skipped; malformed folder names are not (a day folder that
        doesn't match "NSE_YYYYMMDD" raises during the strptime parse).
        """
        acc: dict[date, dict[str, dict[date, set[int]]]] = {} # temp accumulator date -> underlier -> expiry -> set(strike)
        for day_dir in sorted(self._data_dir.iterdir()):
            if not day_dir.is_dir() or not day_dir.name.startswith("NSE"):
                continue
            trading_date = datetime.strptime(day_dir.name.replace("NSE_", ""), "%Y%m%d").date()
            options_dir = day_dir / "Options"
            if not options_dir.exists():
                continue
            acc[trading_date] = defaultdict(lambda: defaultdict(set))
            for f in options_dir.iterdir():
                try:
                    inst = parse_instrument(f.name)
                except ValueError:
                    continue
                if inst.underlier not in self._underliers:
                    continue
                acc[trading_date][inst.underlier][inst.expiry].add(inst.strike)
        # convert sets to sorted numpy arrays
        for trading_date, underlier_dict in acc.items():
            self._trading_days.append(trading_date)
            self._data[trading_date] = {}
            for underlier, expiry_dict in underlier_dict.items():
                self._data[trading_date][underlier] = {
                    exp: np.array(sorted(strikes)) for exp, strikes in expiry_dict.items()
                }

    @property
    def underliers(self) -> set[str]:
        """The underliers this Universe indexes."""
        return set(self._underliers)

    @property
    def trading_days(self) -> list[date]:
        """All trading days found during the scan, in ascending order."""
        return self._trading_days

    def front_expiry(self, day: date, underlier: str) -> date:
        """
        The nearest expiry on or after a given trading day.

        Parameters
        ----------
        day : date
            Trading day to look up.
        underlier : str
            e.g. "NIFTY" or "BANKNIFTY".

        Returns
        -------
        date
            The earliest expiry with expiry >= day.

        Raises
        ------
        ValueError
            If day/underlier wasn't seen during the scan, or every known
            expiry for that underlier falls before day.
        """
        try:
            expiry_map = self._data[day][underlier]
        except KeyError:
            raise ValueError(f"No data for day {day} and underlier {underlier}")

        upcoming = [exp for exp in expiry_map.keys() if exp >= day]
        if not upcoming:
            raise ValueError(f"No upcoming expiries for day {day} and underlier {underlier}")
        return min(upcoming)

    def strikes(self, day: date, underlier: str, expiry: date) -> np.ndarray:
        """
        Sorted strike prices available for one day/underlier/expiry.

        Parameters
        ----------
        day : date
        underlier : str
        expiry : date

        Returns
        -------
        np.ndarray
            Sorted strikes (int), suitable for np.searchsorted.

        Raises
        ------
        ValueError
            If the day/underlier/expiry combination wasn't seen during the scan.
        """
        try:
            return self._data[day][underlier][expiry]
        except KeyError:
            raise ValueError(f"No data for day {day}, underlier {underlier} and expiry {expiry}")

    def day_dir(self, day: date) -> Path:
        """
        Parameters
        ----------
        day : date

        Returns
        -------
        Path
            The "NSE_YYYYMMDD" directory for that trading day.
        """
        return self._data_dir / f"NSE_{day.strftime('%Y%m%d')}"
