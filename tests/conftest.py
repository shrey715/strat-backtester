# tests/conftest.py
from pathlib import Path

import pytest

NSE_DATA_DIR = Path(__file__).parent.parent / "data" / "allData"

# Tests that read the real NSE tick dataset (not shipped with the repo).
_NSE_MODULES = {"test_loader.py", "test_universe.py", "test_nse_provider.py"}
_NSE_FIXTURES = {
    "universe", "provider", "engine_atm", "engine_nothing", "engine_buy_once",
    "day_result_banknifty", "full_result_banknifty",
}


def _has_nse_data() -> bool:
    return NSE_DATA_DIR.is_dir() and any(NSE_DATA_DIR.glob("NSE_*"))


def pytest_collection_modifyitems(config, items):
    if _has_nse_data():
        return
    skip = pytest.mark.skip(reason=f"NSE tick data not found in {NSE_DATA_DIR}")
    for item in items:
        if item.path.name in _NSE_MODULES or _NSE_FIXTURES & set(item.fixturenames):
            item.add_marker(skip)
