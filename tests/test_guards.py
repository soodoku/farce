"""
Regression tests for the guards added after the correctness audit.

Each test corresponds to a defect that shipped: a silently-dropped control
column, a standard error computed from two treated units, a placebo that
cherry-picked its own draws, an album list that drifted from the paper's, and
a joint test that treated correlated day coefficients as independent.
"""

import numpy as np
import pandas as pd
import pytest

from src.gates import check_album_list, check_daily_series, window_overlap
from src.s03_design import _modal_weekday, _sign_flip_joint_test
from src.s07_falsification import cherry_pick_benchmark, friday_placebo_test
from src.utils import MIN_ALBUMS_FOR_SE, album_stats, build_design_matrix


def _frame(n=400, seed=0):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({"date": pd.date_range("2019-01-01", periods=n, freq="D")})
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["holiday"] = 0
    df["holiday_adj"] = 0
    df["fatalities"] = 100 + rng.normal(0, 10, n)
    df["resid_global"] = rng.normal(0, 13, n)
    return df


class TestAlbumStats:
    def test_uses_sample_sd_not_population_sd(self):
        v = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
        assert album_stats(v)["se"] == pytest.approx(np.std(v, ddof=1) / np.sqrt(10))

    def test_critical_values_come_from_t_not_normal(self):
        # the day +1 event-study coefficient: mean 8.79, se 4.52, so t = 1.94.
        # The shipped code reported t = 2.05 (ddof=0) and called it significant
        # against a normal critical value of 1.96.
        spread = np.linspace(-1, 1, 10)
        v = 8.79 + spread * (4.52 * np.sqrt(10) / spread.std(ddof=1))
        st = album_stats(v)
        assert st["se"] == pytest.approx(4.52, abs=0.01)
        assert st["t_stat"] == pytest.approx(1.94, abs=0.02)
        assert st["p_value"] > 0.05
        assert st["ci_lower"] < 0 < st["ci_upper"]

    def test_no_standard_error_below_five_units(self):
        st = album_stats([14.7, 3.5])
        assert st["n"] == 2
        assert np.isnan(st["se"])
        assert np.isnan(st["t_stat"])
        assert st["effect"] == pytest.approx(9.1)

    def test_threshold_is_enforced_exactly(self):
        assert np.isnan(album_stats(list(range(MIN_ALBUMS_FOR_SE - 1)))["se"])
        assert np.isfinite(album_stats(list(range(MIN_ALBUMS_FOR_SE)))["se"])


class TestDesignMatrix:
    def test_absent_control_raises_instead_of_being_dropped(self):
        with pytest.raises(KeyError, match="absent from the frame"):
            build_design_matrix(_frame(), controls=["pct_rain"])

    def test_control_with_missing_values_raises(self):
        df = _frame()
        df["pct_rain"] = 0.1
        df.loc[3, "pct_rain"] = np.nan
        with pytest.raises(ValueError, match="missing values"):
            build_design_matrix(df, controls=["pct_rain"])

    def test_present_control_is_used(self):
        df = _frame()
        df["pct_rain"] = 0.1
        assert "pct_rain" in build_design_matrix(df, controls=["pct_rain"]).columns


class TestFridayPlacebo:
    def test_random_fridays_are_centred_on_zero(self):
        out = friday_placebo_test(_frame(n=900), n_sims=300, seed=1)
        assert abs(out["null_mean"]).max() < 2.5

    def test_cherry_picking_is_not_a_friday_result(self):
        # the defect: top-10-of-100 was reported as a random-Friday placebo.
        # It is an order statistic, so any weekday gives the same answer.
        out = cherry_pick_benchmark(_frame(n=900), n_sims=300, seed=1)
        fri = out.loc[out["pool"] == "Fridays", "selected_mean"].iloc[0]
        tue = out.loc[out["pool"] == "Tuesdays", "selected_mean"].iloc[0]
        assert fri > 10 and tue > 10
        assert abs(fri - tue) < 0.5 * fri


class TestEventTime:
    def test_day_minus_six_is_a_saturday(self):
        # the manuscript called day -6 "the previous Friday"
        assert _modal_weekday(-6) == "Saturday"
        assert _modal_weekday(-7) == "Friday"
        assert _modal_weekday(0) == "Friday"
        assert _modal_weekday(7) == "Friday"

    def test_sign_flip_test_is_not_fooled_by_one_spike(self):
        rng = np.random.RandomState(0)
        panel = rng.normal(0, 13, (10, 10))
        panel[:, 4] += 16
        p_all, _ = _sign_flip_joint_test(panel, n_perms=2000)
        p_drop, _ = _sign_flip_joint_test(
            panel[:, [j for j in range(10) if j != 4]], n_perms=2000
        )
        assert p_drop > p_all


class TestGates:
    def test_album_list_matches_the_paper(self):
        assert check_album_list() == 10

    def test_daily_series_rejects_a_gap(self):
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2019-01-01", "2019-01-02", "2019-01-04"]),
                "fatalities": [100, 100, 100],
            }
        )
        with pytest.raises(AssertionError, match="missing"):
            check_daily_series(df)

    def test_daily_series_rejects_duplicates(self):
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2019-01-01", "2019-01-01"]),
                "fatalities": [100, 100],
            }
        )
        with pytest.raises(AssertionError, match="duplicate"):
            check_daily_series(df)

    def test_overlapping_release_windows_are_reported(self):
        overlap = window_overlap(window=10)
        pairs = set(zip(overlap["album"], overlap["contaminated_by"]))
        assert ("Certified Lover Boy", "Donda") in pairs
        assert len(overlap) == 6
