"""Property-based tests for the guarantees every stream generator makes, whatever the input.

Each test draws a stream of feature vectors with Hypothesis and checks one guarantee for every
concrete class in ``pysad.transform.stream_generators.__all__``:

- ``transform_partial`` does not change the state, so two calls with the same input give the same
  output, and the outputs that follow do not change;
- the batch methods (``fit``, ``transform``, ``fit_transform``) agree with the per-instance
  methods;
- finite inputs give finite outputs;
- normal points are returned unchanged;
- anomaly points are shifted from the input.
"""

import copy
import inspect

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

import pysad.transform.stream_generators

SEED = 61
MAX_STREAM_LENGTH = 30

# Small pools make constant features and repeated vectors likely.
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
    "SlowPoisonStream": {"random_state": SEED, "poison_rate": 0.2, "warmup": 10, "magnitude": 3.0},
    "StealthBurstStream": {"random_state": SEED, "burst_every": 10, "burst_length": 3, "magnitude": 3.0},
}


def _stream_generators():
    """The names of the concrete classes that the subpackage exports."""
    module = pysad.transform.stream_generators
    return [name for name in module.__all__ if not inspect.isabstract(getattr(module, name))]


def _make(name):
    return getattr(pysad.transform.stream_generators, name)(**PARAMS.get(name, {}))


def _draw_num_features(data):
    return data.draw(st.integers(1, 4), label="num_features")


def _point(num_features):
    return hnp.arrays(np.float64, num_features, elements=FINITE)


def _points(num_features, min_length=1):
    return st.integers(min_length, MAX_STREAM_LENGTH).flatmap(
        lambda length: hnp.arrays(np.float64, (length, num_features), elements=FINITE)
    )


# Stream generators accept feature vectors of any dimensionality.
# They are initialized on the first point, so we need at least 2 points for the warmup tests.
MIN_WARMUP_POINTS = 15


@pytest.mark.parametrize("name", _stream_generators())
@PROPERTIES
@given(data=st.data())
def test_transform_partial_does_not_change_state(name, data):
    """Transform_partial is deterministic: same input gives same output."""
    num_features = _draw_num_features(data)
    scores = data.draw(_points(num_features, min_length=2), label="scores")
    probe = data.draw(_point(num_features), label="probe")

    generator = _make(name)
    fitted = generator.fit(scores[: len(scores) // 2])
    fresh = copy.deepcopy(fitted)

    for row in [scores[0], probe]:
        first = fitted.transform_partial(row)
        np.testing.assert_array_equal(fitted.transform_partial(row), first)
        np.testing.assert_array_equal(fresh.transform_partial(row), first)

    rest = scores[len(scores) // 2 :]
    np.testing.assert_array_equal(
        fitted.fit_transform(rest), copy.deepcopy(fresh).fit_transform(rest)
    )


@pytest.mark.parametrize("name", _stream_generators())
@PROPERTIES
@given(data=st.data())
def test_batch_methods_agree_with_partial_methods(name, data):
    """Batch transform agrees with per-instance transform_partial."""
    num_features = _draw_num_features(data)
    scores = data.draw(_points(num_features), label="scores")
    new_scores = data.draw(_points(num_features), label="new_scores")

    generator = _make(name)
    streamed = np.array([generator.fit_transform_partial(row) for row in scores])
    batch = _make(name).fit_transform(scores)
    np.testing.assert_allclose(batch, streamed, rtol=1e-12, atol=1e-12)

    generator = _make(name)
    for row in scores:
        generator.fit_partial(row)
    streamed = np.array([generator.transform_partial(row) for row in new_scores])
    batch = _make(name).fit(scores).transform(new_scores)
    np.testing.assert_allclose(batch, streamed, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("name", _stream_generators())
@PROPERTIES
@given(data=st.data())
def test_outputs_are_finite_for_finite_inputs(name, data):
    """All outputs are finite when inputs are finite."""
    num_features = _draw_num_features(data)
    generator = _make(name)

    # Initialize the generator with warmup points.
    for _ in range(MIN_WARMUP_POINTS):
        point = data.draw(_point(num_features), label="init_point")
        output = generator.fit_transform_partial(point)
        assert np.all(np.isfinite(output)), output


@pytest.mark.parametrize("name", _stream_generators())
@PROPERTIES
@given(data=st.data())
def test_output_shape_matches_input(name, data):
    """Transform_partial returns output with same shape as input."""
    num_features = _draw_num_features(data)
    point = data.draw(_point(num_features), label="point")

    generator = _make(name)
    output = generator.transform_partial(point)

    np.testing.assert_array_equal(output.shape, point.shape)
    assert np.all(np.isfinite(output)), output


@pytest.mark.parametrize("name", _stream_generators())
@PROPERTIES
@given(data=st.data())
def test_normal_points_unchanged(name, data):
    """Normal points (when not an anomaly) are returned unchanged."""
    num_features = _draw_num_features(data)
    points = data.draw(_points(num_features, min_length=MIN_WARMUP_POINTS), label="points")

    generator = _make(name)
    for point in points:
        output = generator.fit_transform_partial(point)
        if not generator.is_anomaly:
            np.testing.assert_array_equal(output, point)


@pytest.mark.parametrize("name", _stream_generators())
@PROPERTIES
@given(data=st.data())
def test_is_anomaly_is_boolean(name, data):
    """is_anomaly is always a boolean."""
    num_features = _draw_num_features(data)
    points = data.draw(_points(num_features, min_length=MIN_WARMUP_POINTS), label="points")

    generator = _make(name)
    for point in points:
        generator.fit_transform_partial(point)
        assert isinstance(generator.is_anomaly, bool)


# Specific tests for SlowPoisonStream behavior

@PROPERTIES
@given(data=st.data())
def test_slow_poison_rate_zero_never_poisons(data):
    """With poison_rate=0, no points are poisoned."""
    num_features = _draw_num_features(data)
    generator = pysad.transform.stream_generators.SlowPoisonStream(
        random_state=SEED, poison_rate=0.0, warmup=10, magnitude=3.0
    )

    for _ in range(20):
        point = data.draw(_point(num_features), label="point")
        generator.fit_transform_partial(point)
        assert generator.is_anomaly is False


@PROPERTIES
@given(data=st.data())
def test_slow_poison_rate_one_always_poisons(data):
    """With poison_rate=1 and warmup=0, all points are poisoned."""
    num_features = _draw_num_features(data)
    generator = pysad.transform.stream_generators.SlowPoisonStream(
        random_state=SEED, poison_rate=1.0, warmup=0, magnitude=3.0
    )

    for _ in range(20):
        point = data.draw(_point(num_features), label="point")
        generator.fit_transform_partial(point)
        assert generator.is_anomaly is True


@PROPERTIES
@given(data=st.data())
def test_slow_poison_anomaly_is_shifted(data):
    """Poisoned points are shifted from the original input."""
    num_features = _draw_num_features(data)
    generator = pysad.transform.stream_generators.SlowPoisonStream(
        random_state=123, poison_rate=1.0, warmup=0, magnitude=5.0
    )

    for _ in range(20):
        point = data.draw(_point(num_features), label="point")
        output = generator.fit_transform_partial(point)
        # When fully poisoned, the point should be shifted.
        assert generator.is_anomaly is True
        assert not np.allclose(output, point, rtol=1e-10, atol=1e-10)


# Specific tests for StealthBurstStream behavior

@PROPERTIES
@given(data=st.data())
def test_stealth_burst_burst_pattern(data):
    """Bursts occur at the expected positions."""
    num_features = _draw_num_features(data)
    bursts_every = 10
    burst_length = 3
    generator = pysad.transform.stream_generators.StealthBurstStream(
        random_state=SEED, burst_every=bursts_every, burst_length=burst_length, magnitude=3.0
    )

    expected_anomalies = []
    for i in range(30):
        point = data.draw(_point(num_features), label=f"point_{i}")
        generator.fit_transform_partial(point)
        # Burst points are at indices: burst_every, burst_every+1, ..., burst_every+burst_length-1,
        # then again at 2*burst_every, etc.
        is_burst = (i >= bursts_every) and ((i - bursts_every) % bursts_every < burst_length)
        expected_anomalies.append(is_burst)

    # The pattern should match.
    actual_anomalies = []
    generator = pysad.transform.stream_generators.StealthBurstStream(
        random_state=SEED, burst_every=bursts_every, burst_length=burst_length, magnitude=3.0
    )
    for i in range(30):
        point = np.zeros(num_features)
        generator.fit_transform_partial(point)
        actual_anomalies.append(generator.is_anomaly)

    np.testing.assert_array_equal(actual_anomalies, expected_anomalies)