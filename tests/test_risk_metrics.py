# tests/test_risk_metrics.py
#
# Rigorous coverage for the risk / distribution / drawdown metrics added on top
# of the original scalar suite. Every expected value is hand-derived from first
# principles (see comments) rather than read back from the function under test.

import math

import numpy as np
import pandas as pd
import pytest
from datetime import date

from stratbt.engine import BacktestResult, DayResult
from stratbt.metrics import (
    value_at_risk,
    conditional_value_at_risk,
    pnl_skewness,
    pnl_kurtosis,
    ulcer_index,
    recovery_factor,
    omega_ratio,
    sharpe_confidence_interval,
    risk_table,
    drawdown_table,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
)


# ---------------------------------------------------------------------------
# Lightweight builder — the risk metrics only read result.daily_pnl, which is
# derived purely from (date, underlier, final_pnl). No MTM grid needed.
# ---------------------------------------------------------------------------

def make_result(pnl_map: dict, start_date: date = date(2022, 11, 1)) -> BacktestResult:
    """pnl_map: { underlier: [pnl_day1, pnl_day2, ...] }, business-dated."""
    n_days = max(len(v) for v in pnl_map.values())
    dates  = pd.bdate_range(start_date, periods=n_days).date
    day_results = []
    for underlier, pnls in pnl_map.items():
        for i, pnl in enumerate(pnls):
            ts  = pd.Timestamp(f"{dates[i]} 09:15:00")
            mtm = pd.Series([0.0, float(pnl)], index=pd.date_range(ts, periods=2, freq="1s"))
            day_results.append(DayResult(
                date=dates[i], underlier=underlier, trades=[],
                mtm=mtm, final_pnl=float(pnl), holdings_log=pd.DataFrame(),
            ))
    return BacktestResult(day_results)


# ===========================================================================
# Value at Risk
# ===========================================================================

class TestValueAtRisk:

    def test_known_five_day_case(self):
        # PnL sorted: [-100,-50,0,50,100]. quantile(0.05) via linear interp:
        # pos = 0.05*(5-1) = 0.2 -> -100 + 0.2*(-50-(-100)) = -90. VaR = 90.
        r = make_result({"NIFTY": [-100.0, -50.0, 0.0, 50.0, 100.0]})
        assert value_at_risk(r)["NIFTY"] == pytest.approx(90.0)

    def test_reported_as_positive_loss(self):
        # A loss-making series must yield a positive VaR magnitude.
        r = make_result({"NIFTY": [-50.0, -40.0, -30.0, -20.0, -10.0]})
        assert value_at_risk(r)["NIFTY"] > 0

    def test_all_positive_series_gives_negative_var(self):
        # Every outcome is a gain, so even the 5% tail is positive -> VaR < 0.
        r = make_result({"NIFTY": [10.0, 20.0, 30.0, 40.0, 50.0]})
        assert value_at_risk(r)["NIFTY"] < 0

    def test_confidence_level_monotonic(self):
        # A higher confidence (deeper tail) never reports a smaller loss.
        r = make_result({"NIFTY": list(np.linspace(-100, 100, 21))})
        assert value_at_risk(r, 0.99)["NIFTY"] >= value_at_risk(r, 0.95)["NIFTY"]

    def test_matches_numpy_quantile(self):
        pnl = [-10.0, -3.0, 2.0, 4.0, 7.0, 9.0, -6.0, 1.0]
        r = make_result({"NIFTY": pnl})
        expected = -np.quantile(np.array(pnl), 0.05)
        assert value_at_risk(r)["NIFTY"] == pytest.approx(expected)

    def test_nan_days_ignored(self):
        # BANKNIFTY has fewer days; the pivot pads NaN. VaR must ignore them.
        r = make_result({"NIFTY": [1.0, 2.0, 3.0, 4.0, 5.0], "BANKNIFTY": [1.0, 2.0, 3.0]})
        bn = np.array([1.0, 2.0, 3.0])
        assert value_at_risk(r)["BANKNIFTY"] == pytest.approx(-np.quantile(bn, 0.05))

    def test_returns_series_per_underlier(self):
        r = make_result({"NIFTY": [1.0, -2.0, 3.0], "BANKNIFTY": [4.0, 5.0, -6.0]})
        out = value_at_risk(r)
        assert set(out.index) == {"NIFTY", "BANKNIFTY"}


# ===========================================================================
# Conditional VaR (expected shortfall)
# ===========================================================================

class TestConditionalVaR:

    def test_known_five_day_case(self):
        # threshold = quantile(0.05) = -90. tail = [-100]. CVaR = 100.
        r = make_result({"NIFTY": [-100.0, -50.0, 0.0, 50.0, 100.0]})
        assert conditional_value_at_risk(r)["NIFTY"] == pytest.approx(100.0)

    def test_cvar_at_least_as_extreme_as_var(self):
        # Expected shortfall averages the worst tail, so it must be >= VaR.
        r = make_result({"NIFTY": [-80.0, -60.0, -10.0, 5.0, 20.0, 40.0, -100.0]})
        assert conditional_value_at_risk(r)["NIFTY"] >= value_at_risk(r)["NIFTY"]

    def test_tail_mean_definition(self):
        pnl = [-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0]
        r = make_result({"NIFTY": pnl})
        arr = np.array(pnl)
        thr = np.quantile(arr, 0.05)
        expected = -arr[arr <= thr].mean()
        assert conditional_value_at_risk(r)["NIFTY"] == pytest.approx(expected)

    def test_nan_days_ignored(self):
        r = make_result({"NIFTY": [1.0, 2.0, 3.0, 4.0], "BANKNIFTY": [-5.0, -10.0]})
        bn = np.array([-5.0, -10.0])
        thr = np.quantile(bn, 0.05)
        assert conditional_value_at_risk(r)["BANKNIFTY"] == pytest.approx(-bn[bn <= thr].mean())


# ===========================================================================
# Skewness
# ===========================================================================

class TestSkewness:

    def test_symmetric_is_zero(self):
        r = make_result({"NIFTY": [-2.0, -1.0, 0.0, 1.0, 2.0]})
        assert pnl_skewness(r)["NIFTY"] == pytest.approx(0.0, abs=1e-9)

    def test_right_tail_positive(self):
        # A long right tail (one big win) skews positive.
        r = make_result({"NIFTY": [-1.0, -1.0, -1.0, -1.0, 100.0]})
        assert pnl_skewness(r)["NIFTY"] > 0

    def test_left_tail_negative(self):
        r = make_result({"NIFTY": [1.0, 1.0, 1.0, 1.0, -100.0]})
        assert pnl_skewness(r)["NIFTY"] < 0

    def test_matches_pandas_reference(self):
        pnl = [10.0, -5.0, 20.0, -8.0, 15.0, -3.0]
        r = make_result({"NIFTY": pnl})
        assert pnl_skewness(r)["NIFTY"] == pytest.approx(pd.Series(pnl).skew())

    def test_nan_days_ignored(self):
        r = make_result({"NIFTY": [1.0, 2.0, 3.0, 4.0, 5.0], "BANKNIFTY": [-2.0, 0.0, 2.0]})
        assert pnl_skewness(r)["BANKNIFTY"] == pytest.approx(pd.Series([-2.0, 0.0, 2.0]).skew())


# ===========================================================================
# Kurtosis
# ===========================================================================

class TestKurtosis:

    def test_matches_pandas_reference(self):
        pnl = [10.0, -5.0, 20.0, -8.0, 15.0, -3.0]
        r = make_result({"NIFTY": pnl})
        assert pnl_kurtosis(r)["NIFTY"] == pytest.approx(pd.Series(pnl).kurt())

    def test_fat_tails_positive_excess(self):
        # Mostly clustered with two extreme outliers -> leptokurtic (excess > 0).
        r = make_result({"NIFTY": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -50.0, 50.0]})
        assert pnl_kurtosis(r)["NIFTY"] > 0

    def test_fewer_than_four_points_is_nan(self):
        r = make_result({"NIFTY": [1.0, 2.0, 3.0]})
        assert math.isnan(pnl_kurtosis(r)["NIFTY"])

    def test_nan_days_ignored(self):
        r = make_result({"NIFTY": [1.0, -2.0, 3.0, -4.0, 5.0, -6.0],
                         "BANKNIFTY": [2.0, 4.0, 6.0, 8.0]})
        assert pnl_kurtosis(r)["BANKNIFTY"] == pytest.approx(pd.Series([2.0, 4.0, 6.0, 8.0]).kurt())


# ===========================================================================
# Ulcer Index
# ===========================================================================

class TestUlcerIndex:

    def test_monotonic_up_is_zero(self):
        # Never below the running peak -> no drawdown -> UI = 0.
        r = make_result({"NIFTY": [10.0, 10.0, 10.0, 10.0]})
        assert ulcer_index(r)["NIFTY"] == pytest.approx(0.0)

    def test_known_case(self):
        # PnL [-10,-10,-10] -> cum [-10,-20,-30], peak [-10,-10,-10],
        # dd [0,10,20], UI = sqrt((0+100+400)/3) = sqrt(500/3).
        r = make_result({"NIFTY": [-10.0, -10.0, -10.0]})
        assert ulcer_index(r)["NIFTY"] == pytest.approx(math.sqrt(500 / 3))

    def test_always_non_negative(self):
        r = make_result({"NIFTY": [5.0, -20.0, 8.0, -3.0, 12.0, -30.0]})
        assert ulcer_index(r)["NIFTY"] >= 0

    def test_single_drawdown_by_hand(self):
        # PnL [+20,-30,+5] -> cum [20,-10,-5], peak [20,20,20],
        # dd [0,30,25], UI = sqrt((0+900+625)/3) = sqrt(1525/3).
        r = make_result({"NIFTY": [20.0, -30.0, 5.0]})
        assert ulcer_index(r)["NIFTY"] == pytest.approx(math.sqrt(1525 / 3))


# ===========================================================================
# Recovery Factor
# ===========================================================================

class TestRecoveryFactor:

    def test_known_case(self):
        # total = -30, mdd = 20 -> RF = -1.5.
        r = make_result({"NIFTY": [-10.0, -10.0, -10.0]})
        assert recovery_factor(r)["NIFTY"] == pytest.approx(-1.5)

    def test_zero_drawdown_is_nan(self):
        # Monotonic up -> mdd = 0 -> RF undefined (NaN, not inf/crash).
        r = make_result({"NIFTY": [10.0, 10.0, 10.0]})
        assert math.isnan(recovery_factor(r)["NIFTY"])

    def test_profitable_with_drawdown_positive(self):
        # Net positive total with a real drawdown -> positive RF.
        r = make_result({"NIFTY": [100.0, -40.0, 80.0]})  # total 140, mdd 40
        assert recovery_factor(r)["NIFTY"] == pytest.approx(140.0 / 40.0)

    def test_consistent_with_max_drawdown(self):
        pnl = [50.0, -80.0, 30.0, -10.0, 60.0]
        r = make_result({"NIFTY": pnl})
        expected = sum(pnl) / max_drawdown(r)["NIFTY"]
        assert recovery_factor(r)["NIFTY"] == pytest.approx(expected)


# ===========================================================================
# Omega Ratio
# ===========================================================================

class TestOmegaRatio:

    def test_known_case(self):
        # gains = 10+10 = 20, losses = 5+5 = 10 -> Omega = 2.0.
        r = make_result({"NIFTY": [10.0, -5.0, 10.0, -5.0]})
        assert omega_ratio(r)["NIFTY"] == pytest.approx(2.0)

    def test_equals_profit_factor_at_zero_threshold(self):
        # Per the docstring: Omega(0) == profit_factor for any daily series.
        for pnl in ([12.0, -4.0, 7.0, -9.0, 3.0], [-1.0, -2.0, 5.0, 8.0, -3.0, 6.0]):
            r = make_result({"NIFTY": pnl})
            assert omega_ratio(r)["NIFTY"] == pytest.approx(profit_factor(r)["NIFTY"])

    def test_no_losses_is_inf(self):
        r = make_result({"NIFTY": [10.0, 20.0, 30.0]})
        assert omega_ratio(r)["NIFTY"] == float("inf")

    def test_all_losses_is_zero(self):
        r = make_result({"NIFTY": [-10.0, -20.0, -30.0]})
        assert omega_ratio(r)["NIFTY"] == pytest.approx(0.0)

    def test_threshold_shifts_ratio_down(self):
        # Raising the threshold reclassifies marginal wins as shortfalls,
        # so Omega must not increase.
        r = make_result({"NIFTY": [10.0, -5.0, 10.0, -5.0]})
        assert omega_ratio(r, threshold=3.0)["NIFTY"] <= omega_ratio(r, threshold=0.0)["NIFTY"]

    def test_threshold_explicit_value(self):
        # threshold=2: excess gains = 8+8 = 16, shortfalls = 7+7 = 14 -> 16/14.
        r = make_result({"NIFTY": [10.0, -5.0, 10.0, -5.0]})
        assert omega_ratio(r, threshold=2.0)["NIFTY"] == pytest.approx(16.0 / 14.0)


# ===========================================================================
# Sharpe bootstrap confidence interval
# ===========================================================================

class TestSharpeConfidenceInterval:

    def test_point_matches_canonical_sharpe(self):
        # Regression guard: the CI's point estimate must equal sharpe_ratio()
        # (both sample std, ddof=1) — they diverged before the ddof fix.
        r = make_result({"NIFTY": [10.0, -5.0, 20.0, -8.0, 15.0, -3.0]})
        ci = sharpe_confidence_interval(r, n_boot=200)
        assert ci.loc["NIFTY", "sharpe"] == pytest.approx(sharpe_ratio(r)["NIFTY"])

    def test_ci_low_below_high(self):
        r = make_result({"NIFTY": [10.0, -5.0, 20.0, -8.0, 15.0, -3.0]})
        ci = sharpe_confidence_interval(r, n_boot=500)
        assert ci.loc["NIFTY", "ci_low"] <= ci.loc["NIFTY", "ci_high"]

    def test_reproducible_with_seed(self):
        r = make_result({"NIFTY": [10.0, -5.0, 20.0, -8.0, 15.0, -3.0]})
        a = sharpe_confidence_interval(r, n_boot=300, seed=7)
        b = sharpe_confidence_interval(r, n_boot=300, seed=7)
        pd.testing.assert_frame_equal(a, b)

    def test_different_seed_changes_interval(self):
        r = make_result({"NIFTY": [10.0, -5.0, 20.0, -8.0, 15.0, -3.0]})
        a = sharpe_confidence_interval(r, n_boot=300, seed=1)
        b = sharpe_confidence_interval(r, n_boot=300, seed=2)
        assert a.loc["NIFTY", "ci_low"] != b.loc["NIFTY", "ci_low"]

    def test_constant_pnl_no_crash(self):
        # Zero variance -> std 0 everywhere. Every column must be NaN, and no
        # warning may leak from the all-NaN bootstrap array.
        r = make_result({"NIFTY": [5.0, 5.0, 5.0, 5.0]})
        with np.errstate(all="raise"):
            ci = sharpe_confidence_interval(r, n_boot=100)
        assert math.isnan(ci.loc["NIFTY", "sharpe"])
        assert math.isnan(ci.loc["NIFTY", "ci_low"])
        assert math.isnan(ci.loc["NIFTY", "ci_high"])

    def test_all_columns_present(self):
        r = make_result({"NIFTY": [1.0, -2.0, 3.0, -4.0], "BANKNIFTY": [2.0, 3.0, -1.0, 5.0]})
        ci = sharpe_confidence_interval(r, n_boot=100)
        assert list(ci.columns) == ["sharpe", "ci_low", "ci_high"]
        assert set(ci.index) == {"NIFTY", "BANKNIFTY"}


# ===========================================================================
# Risk table aggregation
# ===========================================================================

class TestRiskTable:

    def _r(self):
        return make_result({
            "NIFTY":     [10.0, -5.0, 20.0, -8.0, 15.0, -30.0, 4.0],
            "BANKNIFTY": [-2.0, 3.0, -1.0, 5.0, -4.0, 6.0, -3.0],
        })

    def test_one_row_per_underlier(self):
        tbl = risk_table(self._r())
        assert set(tbl.index) == {"NIFTY", "BANKNIFTY"}

    def test_columns_present(self):
        tbl = risk_table(self._r())
        assert list(tbl.columns) == [
            "var_95", "cvar_95", "skew", "kurtosis",
            "ulcer_index", "recovery_factor", "omega",
        ]

    def test_values_match_component_functions(self):
        r   = self._r()
        tbl = risk_table(r)
        for ul in ("NIFTY", "BANKNIFTY"):
            assert tbl.loc[ul, "var_95"]      == pytest.approx(round(value_at_risk(r)[ul], 2))
            assert tbl.loc[ul, "cvar_95"]     == pytest.approx(round(conditional_value_at_risk(r)[ul], 2))
            assert tbl.loc[ul, "skew"]        == pytest.approx(round(pnl_skewness(r)[ul], 3))
            assert tbl.loc[ul, "kurtosis"]    == pytest.approx(round(pnl_kurtosis(r)[ul], 3))
            assert tbl.loc[ul, "ulcer_index"] == pytest.approx(round(ulcer_index(r)[ul], 2))
            assert tbl.loc[ul, "omega"]       == pytest.approx(round(omega_ratio(r)[ul], 3))

    def test_empty_result_returns_empty_not_a_crash(self):
        # Regression test: BacktestResult([]) has zero underlier columns in
        # daily_pnl, so the `rows` list stays empty and pd.DataFrame(rows)
        # has no columns at all — .set_index("underlier") used to raise
        # KeyError instead of returning a properly-shaped empty table.
        r   = BacktestResult([])
        tbl = risk_table(r)
        assert tbl.empty
        assert tbl.index.name == "underlier"
        assert list(tbl.columns) == [
            "var_95", "cvar_95", "skew", "kurtosis",
            "ulcer_index", "recovery_factor", "omega",
        ]


# ===========================================================================
# Drawdown episode table
# ===========================================================================

class TestDrawdownTable:

    def test_two_episodes_traced_correctly(self):
        # PnL [+10,-30,+5,+40,-10] on business days from Nov 1 (Tue):
        #   cum  [10,-20,-15,25,15]
        #   dd   [ 0, 30, 25, 0,10]
        # Episode A: days 2-3, depth 30, trough Nov 2, recovers Nov 4.
        # Episode B: day 5 (last row), depth 10, trough Nov 7, still open -> None.
        r  = make_result({"NIFTY": [10.0, -30.0, 5.0, 40.0, -10.0]})
        dt = drawdown_table(r).reset_index(drop=True)
        assert len(dt) == 2
        # sorted by depth desc
        assert dt.loc[0, "depth"] == pytest.approx(30.0)
        assert dt.loc[0, "trough_date"] == date(2022, 11, 2)
        assert dt.loc[0, "recovery_date"] == date(2022, 11, 4)
        assert dt.loc[1, "depth"] == pytest.approx(10.0)
        assert dt.loc[1, "trough_date"] == date(2022, 11, 7)
        assert dt.loc[1, "recovery_date"] is None

    def test_no_drawdown_returns_empty(self):
        r = make_result({"NIFTY": [10.0, 20.0, 30.0]})
        assert drawdown_table(r).empty

    def test_ongoing_drawdown_recovery_none(self):
        # Ends underwater -> the final episode never recovers.
        r  = make_result({"NIFTY": [50.0, -10.0, -10.0]})
        dt = drawdown_table(r)
        assert len(dt) == 1
        assert dt.iloc[0]["recovery_date"] is None
        assert dt.iloc[0]["depth"] == pytest.approx(20.0)

    def test_recovered_drawdown_has_recovery_date(self):
        # Dips then makes a new high -> recovery_date is the new-high day.
        r  = make_result({"NIFTY": [50.0, -30.0, 100.0]})
        dt = drawdown_table(r)
        assert len(dt) == 1
        assert dt.iloc[0]["recovery_date"] == date(2022, 11, 3)
        assert dt.iloc[0]["depth"] == pytest.approx(30.0)

    def test_sorted_by_depth_descending(self):
        # Three separate episodes of increasing depth, then recovery each time.
        r  = make_result({"NIFTY": [100.0, -10.0, 20.0, -40.0, 50.0, -25.0, 30.0]})
        dt = drawdown_table(r)
        depths = dt["depth"].tolist()
        assert depths == sorted(depths, reverse=True)

    def test_top_n_limit_per_underlier(self):
        r  = make_result({"NIFTY": [100.0, -10.0, 20.0, -40.0, 50.0, -25.0, 30.0]})
        dt = drawdown_table(r, top_n=2)
        assert len(dt) == 2
        # keeps the two deepest
        assert dt["depth"].tolist() == [40.0, 25.0]

    def test_multiple_underliers_independent(self):
        r = make_result({
            "NIFTY":     [50.0, -30.0, 100.0],   # one recovered drawdown
            "BANKNIFTY": [10.0, 20.0, 30.0],     # none
        })
        dt = drawdown_table(r)
        assert set(dt["underlier"]) == {"NIFTY"}

    def test_depth_equals_max_drawdown(self):
        # The single largest episode depth must equal the global max drawdown.
        pnl = [50.0, -80.0, 30.0, -10.0, 60.0]
        r   = make_result({"NIFTY": pnl})
        dt  = drawdown_table(r)
        assert dt["depth"].max() == pytest.approx(max_drawdown(r)["NIFTY"])
