# tests/test_instruments.py
import pytest
from datetime import date
from stratbt.instruments import Instrument, parse_instrument

def test_parse_nifty():
    inst = parse_instrument("NIFTY22110314550PE.csv")
    assert inst.underlier == "NIFTY"
    assert inst.expiry == date(2022, 11, 3)
    assert inst.strike == 14550
    assert inst.option_type == "PE"

def test_parse_banknifty():
    inst = parse_instrument("BANKNIFTY22112443200CE.csv")
    assert inst.underlier == "BANKNIFTY"
    assert inst.expiry == date(2022, 11, 24)
    assert inst.strike == 43200
    assert inst.option_type == "CE"

def test_parse_finnifty():
    inst = parse_instrument("FINNIFTY22110719500CE.csv")
    assert inst.underlier == "FINNIFTY"
    assert inst.expiry == date(2022, 11, 7)
    assert inst.strike == 19500
    assert inst.option_type == "CE"

def test_strip_csv_extension():
    with_ext    = parse_instrument("NIFTY22110314550PE.csv")
    without_ext = parse_instrument("NIFTY22110314550PE")
    assert with_ext == without_ext

def test_round_trip_nifty():
    original = "NIFTY22110314550PE.csv"
    assert parse_instrument(original).filename == original

def test_round_trip_banknifty():
    original = "BANKNIFTY22112443200CE.csv"
    assert parse_instrument(original).filename == original

def test_round_trip_finnifty():
    original = "FINNIFTY22110719500CE.csv"
    assert parse_instrument(original).filename == original

def test_invalid_raises():
    with pytest.raises(ValueError):
        parse_instrument("NOTVALID.csv")

def test_empty_string_raises():
    with pytest.raises(ValueError):
        parse_instrument("")

def test_futures_file_raises():
    with pytest.raises(ValueError):
        parse_instrument("BANKNIFTY-I.csv")

def test_instrument_is_hashable():
    inst = parse_instrument("NIFTY22110314550PE.csv")
    d = {inst: 42}
    assert d[inst] == 42

def test_equality():
    a = parse_instrument("NIFTY22110314550PE.csv")
    b = parse_instrument("NIFTY22110314550PE.csv")
    assert a == b
