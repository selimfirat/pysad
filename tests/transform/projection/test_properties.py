"""Generated vectors exercise projection shape, linearity, and batch equivalence."""

import copy
import inspect

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

import pysad.transform.projection

SEED = 61
MAX_STREAM_LENGTH = 30

# Small pools make constant features and repeated vectors likely.
FINITE = st.one_of(
    st.sampled_from([0.0, 1.0, -1.0, 0.5]),
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
)

PROPERTIES = settings(
    max_examples=20,
    deadline=None,
    database=None,
    # The same examples on every run, so a failure in CI reproduces locally.
    derandomize=True,
)

# Number of components for each projector.
NUM_COMPONENTS = 2

# Constructor arguments by class.
PARAMS = {
    "GaussianRandomProjector": {"num_components": NUM_COMPONENTS, "random_state": SEED},
    "SparseRandomProjector": {"num_components": NUM_COMPONENTS, "random_state": SEED},
    "StreamhashProjector": {"num_components": NUM_COMPONENTS},
}


def _projectors():
    """The names of the concrete classes that the subpackage exports."""
    module = pysad.transform.projection
    return [name for name in module.__all__ if not inspect.isabstract(getattr(module, name))]


def _make(name):
    return getattr(pysad.transform.projection, name)(**PARAMS.get(name, {}))


def _draw_num_features(data):
    return data.draw(st.integers(2, 6), label="num_features")


def _point(num_features):
    return hnp.arrays(np.float64, num_features, elements=FINITE)


def _points(num_features, min_length=1):
    return st.integers(min_length, MAX_STREAM_LENGTH).flatmap(
        lambda length: hnp.arrays(np.float64, (length, num_features), elements=FINITE)
    )


@pytest.mark.parametrize("name", _projectors())
@PROPERTIES
@given(data=st.data())
def test_transform_partial_does_not_change_state(name, data):
    num_features = _draw_num_features(data)
    scores = data.draw(_points(num_features, min_length=2), label="scores")
    probe = data.draw(_point(num_features), label="probe")

    projector = _make(name)
    fitted = projector.fit(scores[: len(scores) // 2])
    fresh = copy.deepcopy(fitted)

    for row in [scores[0], probe]:
        first = fitted.transform_partial(row)
        np.testing.assert_array_equal(fitted.transform_partial(row), first)
        np.testing.assert_array_equal(fresh.transform_partial(row), first)

    rest = scores[len(scores) // 2 :]
    np.testing.assert_array_equal(fitted.fit_transform(rest), fresh.fit_transform(rest))


@pytest.mark.parametrize("name", _projectors())
@PROPERTIES
@given(data=st.data())
def test_batch_methods_agree_with_partial_methods(name, data):
    num_features = _draw_num_features(data)
    scores = data.draw(_points(num_features), label="scores")
    new_scores = data.draw(_points(num_features), label="new_scores")

    projector = _make(name)
    streamed = np.array([projector.fit_transform_partial(row) for row in scores])
    batch = _make(name).fit_transform(scores)
    np.testing.assert_allclose(batch, streamed, rtol=1e-12, atol=1e-12)

    projector = _make(name)
    for row in scores:
        projector.fit_partial(row)
    streamed = np.array([projector.transform_partial(row) for row in new_scores])
    batch = _make(name).fit(scores).transform(new_scores)
    np.testing.assert_allclose(batch, streamed, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("name", _projectors())
@PROPERTIES
@given(data=st.data())
def test_outputs_are_finite_for_finite_inputs(name, data):
    num_features = _draw_num_features(data)
    projector = _make(name)

    # Initialize the projector with at least 2 points.
    for _ in range(2):
        point = data.draw(_point(num_features), label="init_point")
        projector.fit_partial(point)

    for _ in range(5):
        point = data.draw(_point(num_features), label="point")
        output = projector.transform_partial(point)
        assert np.all(np.isfinite(output)), output


@pytest.mark.parametrize("name", _projectors())
@PROPERTIES
@given(data=st.data())
def test_transform_partial_output_shape(name, data):
    num_features = _draw_num_features(data)
    point = data.draw(_point(num_features), label="point")

    projector = _make(name)
    output = projector.transform_partial(point)

    expected_shape = (NUM_COMPONENTS,)
    assert output.shape == expected_shape
    assert output.dtype == np.float64


@pytest.mark.parametrize("name", _projectors())
@PROPERTIES
@given(data=st.data())
def test_projection_is_linear(name, data):
    num_features = _draw_num_features(data)
    left = data.draw(_point(num_features), label="left")
    right = data.draw(_point(num_features), label="right")
    projector = _make(name).fit_partial(left)
    np.testing.assert_allclose(
        projector.transform_partial(left + right),
        projector.transform_partial(left) + projector.transform_partial(right),
        rtol=1e-12,
        atol=1e-8,
    )
