"""Property-based tests for the guarantees every preprocessing transformer makes.

Each test draws a stream of feature vectors with Hypothesis and checks one guarantee for every
concrete class in ``pysad.transform.preprocessing.__all__``:

- ``transform_partial`` does not change the state, so two calls with the same input give the same
  output, and the outputs that follow do not change;
- the batch methods (``fit``, ``transform``, ``fit_transform``) agree with the per-instance
  methods;
- finite inputs give finite outputs once the transformer is initialized.

Each transformer also keeps the invariants documented in its class docstring: a scaler returns the
input unchanged before the first fit, a constant feature scales to zeros, an instance scaler is
stateless, a unit-norm scaler maps nonzero vectors to unit norm, and a shingle ends with the input
point and never holds a later point.
"""

import inspect

import numpy as np
import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

import pysad.transform.preprocessing

MAX_STREAM_LENGTH = 20
WARMUP_POINTS = 30

# Small pools make constant streams and repeated vectors likely.
FINITE = st.one_of(
    st.sampled_from([0.0, 1.0, -1.0, 0.5]),
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
)

PROPERTIES = settings(
    max_examples=8,
    deadline=None,
    database=None,
    suppress_health_check=[HealthCheck.too_slow],
    # The same examples on every run, so a failure in CI reproduces locally.
    derandomize=True,
)

# Constructor arguments by class.
PARAMS = {
    "ShingleTransformer": {"shingle_size": 3},
    "SeasonalTrendDecomposer": {"season_length": 5},
    "OneShotSTL": {"period": 5, "num_init_periods": 3},
    "ModifiedSTLResidualTransformer": {"period": 5, "window_size": 15, "robust": False},
    "FunctionTransformer": {"func": None, "fit_func": None},
    "DrainLogParser": {"depth": 3, "st": 0.4, "max_child": 5},
}

# Univariate transformers accept one feature per point.
UNIVARIATE = {"SeasonalTrendDecomposer", "OneShotSTL", "ModifiedSTLResidualTransformer"}

LOG_LINES = st.sampled_from(
    [
        "User alice logged in from 192.168.1.10",
        "Error connecting to host1:8080",
        "Processing request 42 took 120ms",
        "Cache miss for key key7",
        "Database query executed in 5s",
    ]
)


def _preprocessors():
    """The names of the concrete classes that the subpackage exports."""
    module = pysad.transform.preprocessing
    return [name for name in module.__all__ if not inspect.isabstract(getattr(module, name))]


# DrainLogParser parses log lines, not feature vectors, so it is tested separately.
VECTOR_PREPROCESSORS = [name for name in _preprocessors() if name != "DrainLogParser"]


def _make(name):
    return getattr(pysad.transform.preprocessing, name)(**PARAMS.get(name, {}))


def _draw_num_features(data):
    return data.draw(st.integers(1, 4), label="num_features")


def _point(num_features):
    return hnp.arrays(np.float64, num_features, elements=FINITE)


def _points(num_features, min_length=1):
    return st.integers(min_length, MAX_STREAM_LENGTH).flatmap(
        lambda length: hnp.arrays(np.float64, (length, num_features), elements=FINITE)
    )


def _expected_output_shape(name, num_features):
    if name == "ShingleTransformer":
        return (3 * num_features,)
    if name in ("OneShotSTL", "ModifiedSTLResidualTransformer"):
        return (1,)
    return (num_features,)


@pytest.mark.parametrize("name", VECTOR_PREPROCESSORS)
@PROPERTIES
@given(data=st.data())
def test_transform_partial_does_not_change_state(name, data):
    if name in UNIVARIATE:
        num_features = 1
    else:
        num_features = _draw_num_features(data)
    scores = data.draw(_points(num_features, min_length=2), label="scores")
    probe = data.draw(_point(num_features), label="probe")
    fitted, rest = scores[: len(scores) // 2], scores[len(scores) // 2 :]
    transformer = _make(name).fit(fitted)
    fresh = _make(name).fit(fitted)

    for row in [fitted[-1], probe]:
        first = transformer.transform_partial(row)
        np.testing.assert_array_equal(transformer.transform_partial(row), first)
        np.testing.assert_array_equal(fresh.transform_partial(row), first)

    np.testing.assert_array_equal(
        transformer.fit_transform(rest), fresh.fit_transform(rest)
    )


@pytest.mark.parametrize("name", VECTOR_PREPROCESSORS)
@PROPERTIES
@given(data=st.data())
def test_batch_methods_agree_with_partial_methods(name, data):
    if name in UNIVARIATE:
        num_features = 1
    else:
        num_features = _draw_num_features(data)
    scores = data.draw(_points(num_features), label="scores")
    new_scores = data.draw(_points(num_features), label="new_scores")

    transformer = _make(name)
    streamed = [transformer.fit_transform_partial(row) for row in scores]
    batch = _make(name).fit_transform(scores)
    if name == "SeasonalTrendDecomposer":
        # SeasonalTrendDecomposer fits with a vectorized cumsum, so the batch result matches the
        # per-instance path up to floating-point rounding.
        np.testing.assert_allclose(batch, streamed, atol=1e-12, rtol=1e-12)
    else:
        np.testing.assert_array_equal(batch, streamed)

    transformer = _make(name)
    for row in scores:
        transformer.fit_partial(row)
    streamed = [transformer.transform_partial(row) for row in new_scores]
    batch = _make(name).fit(scores).transform(new_scores)
    if name == "SeasonalTrendDecomposer":
        np.testing.assert_allclose(batch, streamed, atol=1e-12, rtol=1e-12)
    else:
        np.testing.assert_array_equal(batch, streamed)


@pytest.mark.parametrize("name", VECTOR_PREPROCESSORS)
@PROPERTIES
@given(data=st.data())
def test_output_shape_matches_input(name, data):
    if name in UNIVARIATE:
        num_features = 1
    else:
        num_features = _draw_num_features(data)
    point = data.draw(_point(num_features), label="point")
    output = _make(name).fit_transform_partial(point)
    assert output.shape == _expected_output_shape(name, num_features)


@pytest.mark.parametrize("name", VECTOR_PREPROCESSORS)
@PROPERTIES
@given(points=st.lists(_point(1), min_size=WARMUP_POINTS, max_size=WARMUP_POINTS))
def test_outputs_are_finite_for_finite_inputs(name, points):
    transformer = _make(name)
    for point in points:
        output = transformer.fit_transform_partial(point)
        assert np.all(np.isfinite(output))


@pytest.mark.parametrize("value", [0.0, -3.0, 100.0])
@pytest.mark.parametrize("num_features", [1, 3, 10])
def test_standard_scaler_constant_feature_scales_to_zeros(value, num_features):
    import math

    from pysad.transform.preprocessing import StandardScaler

    X = np.full(num_features, value)
    scaler = StandardScaler()
    scaler.fit_partial(X)

    with np.errstate(all="raise"):
        scaled = scaler.transform_partial(X)

    np.testing.assert_array_equal(scaled, np.zeros(num_features))
    mean = scaler.means_[0].get() if scaler.means_ is not None else float("nan")
    variance = scaler.variances_[0].get() if scaler.variances_ is not None else float("nan")
    expected = (
        0.0
        if variance == 0.0
        else (float(value) - mean) / math.sqrt(variance)
    )
    np.testing.assert_allclose(scaled[0], expected)


@PROPERTIES
@given(X=_point(4))
@example(X=np.array([1.0, 2.0, 3.0]))
def test_standard_scaler_before_fit_returns_input_unchanged(X):
    from pysad.transform.preprocessing import StandardScaler

    np.testing.assert_array_equal(StandardScaler().transform_partial(X), X)


@pytest.mark.parametrize("value", [0.0, -3.0, 100.0])
@pytest.mark.parametrize("num_features", [1, 3, 10])
def test_min_max_scaler_constant_feature_scales_to_zeros(value, num_features):
    from pysad.transform.preprocessing import MinMaxScaler

    X = np.full(num_features, value)
    scaler = MinMaxScaler()
    scaler.fit_partial(X)

    with np.errstate(all="raise"):
        scaled = scaler.transform_partial(X)

    np.testing.assert_array_equal(scaled, np.zeros(num_features))


@PROPERTIES
@given(X=_point(4))
@example(X=np.array([1.0, 2.0, 3.0]))
def test_min_max_scaler_before_fit_returns_input_unchanged(X):
    from pysad.transform.preprocessing import MinMaxScaler

    np.testing.assert_array_equal(MinMaxScaler().transform_partial(X), X)


@PROPERTIES
@given(
    low=st.floats(-10.0, 10.0, allow_nan=False, allow_infinity=False),
    high=st.floats(-10.0, 10.0, allow_nan=False, allow_infinity=False),
)
@example(low=0.0, high=2.0)
def test_min_max_scaler_values_in_observed_range_map_to_unit_interval(low, high):
    from pysad.transform.preprocessing import MinMaxScaler

    lo, hi = min(low, high), max(low, high)
    if hi == lo:
        hi = lo + 1.0
    scaler = MinMaxScaler()
    scaler.fit_partial(np.array([lo]))
    scaler.fit_partial(np.array([hi]))
    for value in [lo, (lo + hi) / 2.0, hi]:
        output = scaler.transform_partial(np.array([value]))
        assert 0.0 <= output[0] <= 1.0


@PROPERTIES
@given(X=_point(4))
@example(X=np.array([1.0, 1.0, 1.0]))
@example(X=np.array([5.0]))
def test_instance_standard_scaler_constant_vector_scales_to_zeros(X):
    from pysad.transform.preprocessing import InstanceStandardScaler

    if X.size and (X == X[0]).all() and np.isfinite(X).all():
        np.testing.assert_array_equal(
            InstanceStandardScaler().transform_partial(X), np.zeros_like(X, dtype=np.float64)
        )


@PROPERTIES
@given(X=_point(4))
@example(X=np.array([1.0, 2.0, 3.0]))
def test_instance_standard_scaler_is_stateless(X):
    from pysad.transform.preprocessing import InstanceStandardScaler

    scaler = InstanceStandardScaler()
    scaler.fit_partial(np.array([9.0, 9.0, 9.0]))
    first = scaler.transform_partial(X)
    scaler.fit_partial(np.array([1.0, 1.0, 1.0]))
    second = scaler.transform_partial(X)
    np.testing.assert_array_equal(first, second)


@PROPERTIES
@given(X=_point(4))
@example(X=np.array([1.0, 2.0, 3.0]))
def test_instance_standard_scaler_output_has_zero_mean_and_unit_std(X):
    from pysad.transform.preprocessing import InstanceStandardScaler

    if (X == X[0]).all() and np.isfinite(X).all():
        return
    output = InstanceStandardScaler().transform_partial(X)
    assert abs(output.mean()) < 1e-12
    assert abs(output.std() - 1.0) < 1e-12


@PROPERTIES
@given(X=_point(4))
@example(X=np.array([0.0, 0.0, 0.0]))
@example(X=np.array([3.0, 4.0]))
def test_instance_unit_norm_scaler_zero_vector_unchanged(X):
    from pysad.transform.preprocessing import InstanceUnitNormScaler

    if np.linalg.norm(X) == 0:
        np.testing.assert_array_equal(InstanceUnitNormScaler().transform_partial(X), X)


@PROPERTIES
@given(X=_point(4))
@example(X=np.array([3.0, 4.0]))
def test_instance_unit_norm_scaler_nonzero_vector_has_unit_norm(X):
    from pysad.transform.preprocessing import InstanceUnitNormScaler

    if np.linalg.norm(X) == 0:
        return
    output = InstanceUnitNormScaler().transform_partial(X)
    assert abs(np.linalg.norm(output) - 1.0) < 1e-12


@PROPERTIES
@given(X=_point(4))
@example(X=np.array([1.0, 2.0, 3.0]))
def test_identity_scaler_returns_input_unchanged(X):
    from pysad.transform.preprocessing import IdentityScaler

    np.testing.assert_array_equal(IdentityScaler().transform_partial(X), X)


@PROPERTIES
@given(X=_point(4))
@example(X=np.array([1.0, 2.0, 3.0]))
def test_function_transformer_none_func_passes_input_through(X):
    from pysad.transform.preprocessing import FunctionTransformer

    np.testing.assert_array_equal(FunctionTransformer().transform_partial(X), X)


@PROPERTIES
@given(X=_point(4), factor=st.floats(0.1, 10.0, allow_nan=False, allow_infinity=False))
@example(X=np.array([1.0, 2.0, 3.0]), factor=2.0)
def test_function_transformer_applies_func_and_returns_1d_float64(X, factor):
    from pysad.transform.preprocessing import FunctionTransformer

    output = FunctionTransformer(func=lambda x: factor * x).transform_partial(X)
    assert type(output) is np.ndarray
    assert output.dtype == np.float64
    assert output.ndim == 1
    np.testing.assert_array_equal(output, factor * X)


@PROPERTIES
@given(data=st.data())
def test_shingle_transformer_structure(data):
    from pysad.transform.preprocessing import ShingleTransformer

    shingle_size = data.draw(st.sampled_from([1, 2, 3]), label="shingle_size")
    num_features = data.draw(st.integers(1, 3), label="num_features")
    point = data.draw(_point(num_features), label="point")
    earlier = data.draw(st.lists(_point(num_features), min_size=0, max_size=4), label="earlier")

    transformer = ShingleTransformer(shingle_size=shingle_size)
    for row in earlier:
        transformer.fit_partial(row)

    output = transformer.transform_partial(point)
    assert output.shape == (shingle_size * num_features,)
    np.testing.assert_array_equal(output[-num_features:], point)

    # The shingle holds earlier points oldest first, then the point.
    expected_earlier = list(earlier)
    missing = shingle_size - 1 - len(expected_earlier)
    fill = expected_earlier[0] if expected_earlier else point
    expected = np.concatenate([*[fill] * missing, *expected_earlier, point])
    np.testing.assert_array_equal(output, expected)

    # Transforming again does not change the history.
    np.testing.assert_array_equal(transformer.transform_partial(point), output)
    assert len(transformer.recent) == min(len(earlier), shingle_size - 1)


@PROPERTIES
@given(data=st.data())
def test_shingle_transformer_fit_adds_point(data):
    from pysad.transform.preprocessing import ShingleTransformer

    num_features = data.draw(st.integers(1, 3), label="num_features")
    point = data.draw(_point(num_features), label="point")
    transformer = ShingleTransformer(shingle_size=3)
    before = len(transformer.recent)
    transformer.fit_transform_partial(point)
    assert len(transformer.recent) == before + 1


@PROPERTIES
@given(points=st.lists(_point(2), min_size=WARMUP_POINTS, max_size=WARMUP_POINTS))
def test_seasonal_trend_decomposer_output_is_finite(points):
    from pysad.transform.preprocessing import SeasonalTrendDecomposer

    transformer = SeasonalTrendDecomposer(season_length=5)
    for point in points:
        output = transformer.fit_transform_partial(point)
        assert output.shape == (2,)
        assert np.all(np.isfinite(output))


@PROPERTIES
@given(points=st.lists(_point(1), min_size=WARMUP_POINTS, max_size=WARMUP_POINTS))
def test_one_shot_stl_returns_zeros_during_initialization(points):
    from pysad.transform.preprocessing import OneShotSTL

    transformer = OneShotSTL(period=5, num_init_periods=3)
    for index, point in enumerate(points[: 5 * 3]):
        output = transformer.fit_transform_partial(point)
        assert output.shape == (1,)
        np.testing.assert_array_equal(output, np.zeros(1))


@PROPERTIES
@given(points=st.lists(_point(1), min_size=WARMUP_POINTS, max_size=WARMUP_POINTS))
def test_one_shot_stl_output_is_finite_after_initialization(points):
    from pysad.transform.preprocessing import OneShotSTL

    transformer = OneShotSTL(period=5, num_init_periods=3)
    for point in points:
        output = transformer.fit_transform_partial(point)
        assert output.shape == (1,)
        assert np.all(np.isfinite(output))


@PROPERTIES
@given(points=st.lists(_point(1), min_size=WARMUP_POINTS, max_size=WARMUP_POINTS))
def test_modified_stl_residual_returns_zeros_during_initialization(points):
    from pysad.transform.preprocessing import ModifiedSTLResidualTransformer

    # The window holds at most 2 * period = 10 points during initialization.
    transformer = ModifiedSTLResidualTransformer(period=5, window_size=15, robust=False)
    for point in points[: 2 * 5]:
        output = transformer.fit_transform_partial(point)
        assert output.shape == (1,)
        np.testing.assert_array_equal(output, np.zeros(1))


@PROPERTIES
@given(points=st.lists(_point(1), min_size=WARMUP_POINTS, max_size=WARMUP_POINTS))
def test_modified_stl_residual_output_is_finite_after_initialization(points):
    from pysad.transform.preprocessing import ModifiedSTLResidualTransformer

    transformer = ModifiedSTLResidualTransformer(period=5, window_size=15, robust=False)
    for point in points:
        output = transformer.fit_transform_partial(point)
        assert output.shape == (1,)
        assert np.all(np.isfinite(output))


# DrainLogParser takes log lines, not feature vectors, so it is tested separately.


@PROPERTIES
@given(lines=st.lists(LOG_LINES, min_size=1, max_size=12), probe=LOG_LINES)
def test_drain_log_parser_event_ids_are_stable_and_in_range(lines, probe):
    from pysad.transform.preprocessing import DrainLogParser

    parser = DrainLogParser(depth=3, st=0.4, max_child=5)
    for line in lines:
        output = parser.fit_transform_partial(line)
        assert output.shape == (1,)
        assert np.isfinite(output[0])
        assert output[0] >= -1.0

    first = parser.transform_partial(probe)
    np.testing.assert_array_equal(parser.transform_partial(probe), first)
    assert first.shape == (1,)
    assert np.isfinite(first[0])
    assert first[0] >= -1.0


@PROPERTIES
@given(lines=st.lists(LOG_LINES, min_size=1, max_size=12))
def test_drain_log_parser_batch_agrees_with_partial(lines):
    from pysad.transform.preprocessing import DrainLogParser

    parser = DrainLogParser(depth=3, st=0.4, max_child=5)
    streamed = [parser.fit_transform_partial(line) for line in lines]
    np.testing.assert_array_equal(
        parser.fit_transform(np.array(lines, dtype=object)), streamed
    )
