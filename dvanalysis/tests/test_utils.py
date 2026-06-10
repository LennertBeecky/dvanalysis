"""Tests for dvanalysis.utils — shared math and mask utilities."""
import numpy as np
import pytest

from dvanalysis.utils.math import (
    hampel_filter_1d,
    mad_scale,
    nanstat,
    safe_nanargmax,
    safe_nanargmin,
    fill_missing_interp,
    to_percent,
)
from dvanalysis.utils.masks import (
    contiguous_true_runs,
    support_good_timepoints,
    resolve_min_valid_abs,
)


# ---------------------------------------------------------------------------
# math.py
# ---------------------------------------------------------------------------

class TestHampelFilter:
    def test_no_outliers(self):
        y = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        y_out, is_out = hampel_filter_1d(y, k=2, nsigmas=3.0)
        np.testing.assert_array_equal(is_out, False)
        np.testing.assert_array_almost_equal(y_out, y)

    def test_detects_spike(self):
        # Need enough spread so MAD > 0; use noisy data with a clear spike
        rng = np.random.default_rng(0)
        y = rng.normal(0, 1, 50)
        y[25] = 100.0  # clear outlier in noisy data
        y_out, is_out = hampel_filter_1d(y, k=5, nsigmas=3.0)
        assert is_out[25]
        assert abs(y_out[25]) < 10.0  # replaced with local median

    def test_nan_handling(self):
        rng = np.random.default_rng(1)
        y = rng.normal(0, 1, 20)
        y[3] = np.nan
        y[10] = 50.0  # outlier
        y_out, is_out = hampel_filter_1d(y, k=5, nsigmas=3.0)
        assert np.isnan(y_out[3])  # NaN preserved
        assert is_out[10]  # outlier detected

    def test_replace_nan_mode(self):
        rng = np.random.default_rng(2)
        y = rng.normal(0, 1, 30)
        y[15] = 100.0
        y_out, _ = hampel_filter_1d(y, k=5, nsigmas=3.0, replace="nan")
        assert np.isnan(y_out[15])

    def test_empty(self):
        y = np.array([], dtype=float)
        y_out, is_out = hampel_filter_1d(y, k=3, nsigmas=3.0)
        assert y_out.size == 0


class TestMadScale:
    def test_normal_data(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 10000)
        s = mad_scale(x)
        assert abs(s - 1.0) < 0.1  # should be close to 1 for standard normal

    def test_constant(self):
        x = np.ones(10)
        s = mad_scale(x)
        assert s == 0.0  # MAD of constant is 0

    def test_with_nans(self):
        x = np.array([1.0, 2.0, np.nan, 3.0, 4.0])
        s = mad_scale(x)
        assert np.isfinite(s)


class TestNanstat:
    def test_mean(self):
        x = np.array([1.0, 2.0, np.nan, 4.0])
        assert abs(nanstat(x, "mean") - 7.0 / 3.0) < 1e-10

    def test_median(self):
        x = np.array([1.0, 2.0, np.nan, 4.0])
        assert abs(nanstat(x, "median") - 2.0) < 1e-10

    def test_max(self):
        assert nanstat(np.array([1.0, 3.0, 2.0]), "max") == 3.0

    def test_min(self):
        assert nanstat(np.array([1.0, 3.0, 2.0]), "min") == 1.0

    def test_empty(self):
        assert np.isnan(nanstat(np.array([]), "mean"))

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            nanstat(np.array([1.0]), "mode")


class TestSafeArgmax:
    def test_normal(self):
        assert safe_nanargmax(np.array([1.0, 3.0, 2.0])) == 1

    def test_all_nan(self):
        assert safe_nanargmax(np.array([np.nan, np.nan])) is None

    def test_argmin(self):
        assert safe_nanargmin(np.array([3.0, 1.0, 2.0])) == 1

    def test_argmin_all_nan(self):
        assert safe_nanargmin(np.array([np.nan])) is None


class TestFillMissing:
    def test_interpolates_gaps(self):
        X = np.array([[1.0, 10.0], [np.nan, np.nan], [3.0, 30.0]])
        M = np.array([[True, True], [False, False], [True, True]])
        out = fill_missing_interp(X, M)
        np.testing.assert_almost_equal(out[1, 0], 2.0)
        np.testing.assert_almost_equal(out[1, 1], 20.0)

    def test_all_valid_unchanged(self):
        X = np.ones((5, 3))
        M = np.ones((5, 3), dtype=bool)
        out = fill_missing_interp(X, M)
        np.testing.assert_array_equal(out, X)

    def test_too_few_valid_unchanged(self):
        X = np.array([[1.0], [np.nan], [np.nan]])
        M = np.array([[True], [False], [False]])
        out = fill_missing_interp(X, M)
        assert np.isnan(out[1, 0])  # can't interpolate with only 1 valid


class TestToPercent:
    def test_delta_over_baseline(self):
        y = np.array([100.0, 105.0, 95.0])
        p = to_percent(y, 100.0, mode="delta_over_baseline")
        np.testing.assert_almost_equal(p, [0.0, 5.0, -5.0])

    def test_ratio(self):
        y = np.array([100.0, 105.0])
        p = to_percent(y, 100.0, mode="ratio")
        np.testing.assert_almost_equal(p, [100.0, 105.0])

    def test_zero_baseline_nan(self):
        p = to_percent(np.array([1.0]), 0.0, mode="delta_over_baseline")
        assert np.all(np.isnan(p))


# ---------------------------------------------------------------------------
# masks.py
# ---------------------------------------------------------------------------

class TestContiguousTrueRuns:
    def test_single_run(self):
        m = np.array([False, True, True, True, False])
        runs = contiguous_true_runs(m)
        assert len(runs) == 1
        np.testing.assert_array_equal(runs[0], [1, 2, 3])

    def test_multiple_runs(self):
        m = np.array([True, True, False, True, False, True])
        runs = contiguous_true_runs(m)
        assert len(runs) == 3

    def test_all_false(self):
        runs = contiguous_true_runs(np.zeros(5, dtype=bool))
        assert len(runs) == 0

    def test_all_true(self):
        runs = contiguous_true_runs(np.ones(5, dtype=bool))
        assert len(runs) == 1
        assert runs[0].size == 5


class TestSupportGoodTimepoints:
    def test_threshold(self):
        M = np.array([
            [True, True, True],
            [True, False, False],
            [True, True, True],
        ])
        good = support_good_timepoints(M, min_valid_abs=2)
        np.testing.assert_array_equal(good, [True, False, True])

    def test_all_pass(self):
        M = np.ones((5, 3), dtype=bool)
        good = support_good_timepoints(M, min_valid_abs=1)
        assert np.all(good)


class TestResolveMinValidAbs:
    def test_fraction_mode(self):
        M = np.ones((10, 20), dtype=bool)
        assert resolve_min_valid_abs(M, min_abs=0, min_frac=0.5) == 10

    def test_absolute_mode(self):
        M = np.ones((10, 20), dtype=bool)
        assert resolve_min_valid_abs(M, min_abs=5, min_frac=None) == 5

    def test_fraction_rounds_up(self):
        M = np.ones((10, 3), dtype=bool)
        assert resolve_min_valid_abs(M, min_abs=0, min_frac=0.5) == 2  # ceil(1.5)
