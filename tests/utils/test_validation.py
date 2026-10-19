import numpy as np
import pytest

from pysad.utils.validation import (
    _finite_float,
    _int_at_least,
    check_feature_bounds,
    check_univariate,
)


def test_int_at_least_accepts_valid_integers():
    assert _int_at_least(5, "param", 1) == 5
    assert _int_at_least(np.int64(10), "param", 1) == 10
    assert _int_at_least(1, "param", 1) == 1


@pytest.mark.parametrize("bad_val", [True, False, 1.5, "5", None])
def test_int_at_least_rejects_non_integers(bad_val):
    with pytest.raises(TypeError, match="must be an int"):
        _int_at_least(bad_val, "param", 1)


def test_int_at_least_rejects_values_below_minimum():
    with pytest.raises(ValueError, match="must be at least 2"):
        _int_at_least(1, "param", 2)


def test_finite_float_accepts_valid_numbers():
    assert _finite_float(3.14, "param") == 3.14
    assert _finite_float(5, "param") == 5.0
    assert _finite_float(np.float64(2.5), "param") == 2.5


@pytest.mark.parametrize("bad_val", [True, False, "3.14", None])
def test_finite_float_rejects_non_numbers(bad_val):
    with pytest.raises(TypeError, match="must be a real number"):
        _finite_float(bad_val, "param")


@pytest.mark.parametrize("bad_float", [np.nan, np.inf, -np.inf])
def test_finite_float_rejects_non_finite_values(bad_float):
    with pytest.raises(ValueError, match="must be finite"):
        _finite_float(bad_float, "param")


def test_check_feature_bounds_accepts_valid_bounds():
    mins, maxes = check_feature_bounds([0.0, 1.0], [5.0, 10.0])
    np.testing.assert_array_equal(mins, [0.0, 1.0])
    np.testing.assert_array_equal(maxes, [5.0, 10.0])
    assert mins.dtype == np.float64
    assert maxes.dtype == np.float64


def test_check_feature_bounds_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        check_feature_bounds([], [])


def test_check_feature_bounds_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same length"):
        check_feature_bounds([0.0], [1.0, 2.0])


def test_check_feature_bounds_rejects_multidimensional():
    with pytest.raises(ValueError, match="1-D"):
        check_feature_bounds([[0.0]], [[1.0]])


def test_check_feature_bounds_rejects_non_finite():
    with pytest.raises(ValueError, match="must be finite"):
        check_feature_bounds([np.nan], [1.0])
    with pytest.raises(ValueError, match="must be finite"):
        check_feature_bounds([0.0], [np.inf])


def test_check_feature_bounds_rejects_min_greater_than_max():
    with pytest.raises(ValueError, match="must not exceed"):
        check_feature_bounds([5.0], [2.0])


def test_check_univariate_accepts_scalar_and_single_element_array():
    assert check_univariate(3.14, "TestModel") == 3.14
    assert check_univariate([2.0], "TestModel") == 2.0
    assert check_univariate(np.array([5.0]), "TestModel") == 5.0
    assert check_univariate(np.array([[7.0]]), "TestModel") == 7.0


@pytest.mark.parametrize("bad_shape", [np.array([]), np.array([1.0, 2.0]), np.zeros((2, 2))])
def test_check_univariate_rejects_wrong_shapes(bad_shape):
    with pytest.raises(ValueError, match="TestModel is univariate"):
        check_univariate(bad_shape, "TestModel")


def test_check_univariate_reject_nan():
    assert np.isnan(check_univariate(np.nan, "TestModel"))
    with pytest.raises(ValueError, match="TestModel does not accept NaN values"):
        check_univariate(np.nan, "TestModel", reject_nan=True)
