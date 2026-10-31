"""Property-based tests for the projectors that take a ``random_state``, whatever the data.

Each test draws a stream with Hypothesis and checks, for every class in
``pysad.transform.projection.__all__`` whose constructor takes ``random_state``, that two
projectors with the same seed project the stream to the same output.
"""

import inspect

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

import pysad.transform.projection

SEED = 61
NUM_COMPONENTS = 2
MAX_FEATURES = 6
MAX_STREAM_LENGTH = 30

FINITE_VALUES = st.one_of(
    st.sampled_from([0.0, 1.0, -1.0, 0.5]),
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
)

PROPERTIES = settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
    # The same examples on every run, so a failure in CI reproduces locally.
    derandomize=True,
)


def _seeded_projectors():
    """The names of the concrete exported classes whose constructor takes `random_state`."""
    module = pysad.transform.projection
    names = []
    for name in module.__all__:
        cls = getattr(module, name)
        if not inspect.isabstract(cls) and "random_state" in inspect.signature(cls).parameters:
            names.append(name)

    return names


def _make(name):
    return getattr(pysad.transform.projection, name)(
        num_components=NUM_COMPONENTS, random_state=SEED
    )


def _streams():
    # At least NUM_COMPONENTS features, so the projection does not add dimensions.
    shapes = st.tuples(st.integers(1, MAX_STREAM_LENGTH), st.integers(NUM_COMPONENTS, MAX_FEATURES))
    return shapes.flatmap(lambda shape: hnp.arrays(np.float64, shape, elements=FINITE_VALUES))


@pytest.mark.parametrize("name", _seeded_projectors())
@PROPERTIES
@given(X=_streams())
def test_random_state_makes_projection_reproducible(name, X):
    np.testing.assert_array_equal(_make(name).fit_transform(X), _make(name).fit_transform(X))

    first, second = _make(name), _make(name)
    for x in X:
        np.testing.assert_array_equal(
            first.fit_transform_partial(x), second.fit_transform_partial(x)
        )
