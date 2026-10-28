# tests/test_instruments.py
from datetime import date

from stratbt.instruments import Instrument


def test_filename_format():
    inst = Instrument("NIFTY", date(2022, 11, 3), 14550, "PE")
    assert inst.filename == "NIFTY22110314550PE.csv"


def test_equal_instruments_hash_equal_and_work_as_dict_keys():
    a = Instrument("SPY", date(2024, 1, 5), 450, "CE")
    b = Instrument("SPY", date(2024, 1, 5), 450, "CE")
    assert a == b and hash(a) == hash(b)
    assert {a: 1}[b] == 1


def test_different_fields_are_different_instruments():
    base = Instrument("SPY", date(2024, 1, 5), 450, "CE")
    for other in (Instrument("QQQ", date(2024, 1, 5), 450, "CE"), Instrument("SPY", date(2024, 1, 12), 450, "CE"),
                  Instrument("SPY", date(2024, 1, 5), 451, "CE"), Instrument("SPY", date(2024, 1, 5), 450, "PE")):
        assert base != other
        assert len({base, other}) == 2


def test_tickers_with_punctuation_are_allowed():
    assert Instrument("^NSEI", date(2026, 1, 6), 24000, "CE").filename.startswith("^NSEI")
    assert Instrument("RELIANCE.NS", date(2026, 1, 6), 1400, "PE").underlier == "RELIANCE.NS"


def test_frozen():
    import dataclasses
    import pytest
    with pytest.raises(dataclasses.FrozenInstanceError):
        Instrument("SPY", date(2024, 1, 5), 450, "CE").strike = 1  # type: ignore[misc]
