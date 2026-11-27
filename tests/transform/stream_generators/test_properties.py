"""Generated streams exercise anomaly labels, seeded replay, and batch equivalence."""

import inspect

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

import pysad.transform.stream_generators

SEED = 61
PROPERTIES = settings(max_examples=20, deadline=None, database=None, derandomize=True)
FINITE = st.one_of(
    st.sampled_from([0.0, 1.0, -1.0, 0.5]),
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
)
PARAMS = {
    "SlowPoisonStream": {"poison_rate": 0.2, "warmup": 10, "magnitude": 3.0},
    "StealthBurstStream": {"burst_every": 10, "burst_length": 3, "magnitude": 3.0},
}
GENERATORS = [
    name
    for name in pysad.transform.stream_generators.__all__
    if not inspect.isabstract(getattr(pysad.transform.stream_generators, name))
]


def _make(name):
    return getattr(pysad.transform.stream_generators, name)(random_state=SEED, **PARAMS[name])


@st.composite
def _points(draw):
    shape = (draw(st.integers(15, 35)), draw(st.integers(1, 4)))
    return draw(hnp.arrays(np.float64, shape, elements=FINITE))


@pytest.mark.parametrize("name", GENERATORS)
@PROPERTIES
@given(points=_points())
def test_seeded_replay_preserves_outputs_labels_and_inputs(name, points):
    original = points.copy()
    generator = _make(name)
    replay = _make(name)
    for point in points:
        output = generator.fit_transform_partial(point)
        np.testing.assert_array_equal(replay.fit_transform_partial(point), output)
        assert generator.is_anomaly == replay.is_anomaly
        assert isinstance(generator.is_anomaly, bool)
        assert output.shape == point.shape
        assert output.dtype == np.float64
        assert np.isfinite(output).all()
        if not generator.is_anomaly:
            np.testing.assert_array_equal(output, point)
    np.testing.assert_array_equal(points, original)


@pytest.mark.parametrize("name", GENERATORS)
@PROPERTIES
@given(points=_points())
def test_batch_methods_agree_with_partial_methods(name, points):
    generator = _make(name)
    streamed = [generator.fit_transform_partial(point) for point in points]
    np.testing.assert_array_equal(_make(name).fit_transform(points), streamed)

    generator = _make(name)
    for point in points:
        generator.fit_partial(point)
    streamed = [generator.transform_partial(point) for point in points]
    np.testing.assert_array_equal(_make(name).fit(points).transform(points), streamed)


@pytest.mark.parametrize("name", GENERATORS)
@PROPERTIES
@given(points=_points())
def test_zero_magnitude_preserves_points(name, points):
    params = {**PARAMS[name], "magnitude": 0.0}
    generator = getattr(pysad.transform.stream_generators, name)(random_state=SEED, **params)
    np.testing.assert_array_equal(generator.fit_transform(points), points)


@pytest.mark.parametrize("rate", [0.0, 1.0])
@PROPERTIES
@given(points=_points())
def test_poison_rate_endpoints_set_labels(rate, points):
    from pysad.transform.stream_generators import SlowPoisonStream

    generator = SlowPoisonStream(random_state=SEED, poison_rate=rate, warmup=0)
    for point in points:
        generator.fit_transform_partial(point)
        assert generator.is_anomaly is bool(rate)


@PROPERTIES
@given(
    point=hnp.arrays(np.float64, st.integers(1, 4), elements=st.integers(-100, 100)),
    magnitude=st.floats(0.1, 10.0, allow_nan=False, allow_infinity=False),
)
def test_constant_stream_poison_shift_has_requested_magnitude(point, magnitude):
    from pysad.transform.stream_generators import SlowPoisonStream

    generator = SlowPoisonStream(random_state=SEED, poison_rate=1.0, warmup=0, magnitude=magnitude)
    for _ in range(5):
        output = generator.fit_transform_partial(point)
        assert np.linalg.norm(output - point) == pytest.approx(magnitude)


@PROPERTIES
@given(points=_points(), burst_every=st.integers(2, 10), burst_length=st.integers(1, 12))
def test_burst_labels_follow_schedule(points, burst_every, burst_length):
    from pysad.transform.stream_generators import StealthBurstStream

    generator = StealthBurstStream(
        random_state=SEED, burst_every=burst_every, burst_length=burst_length
    )
    for index, point in enumerate(points):
        generator.fit_transform_partial(point)
        expected = index >= burst_every and (index - burst_every) % burst_every < burst_length
        assert generator.is_anomaly is expected
