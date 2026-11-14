"""Property-based tests for the guarantees every ensembler makes, whatever the scores.

Each test draws a matrix of scores from several detectors with Hypothesis and checks one guarantee
for every concrete class in ``pysad.transform.ensemble.__all__``:

- ``transform_partial`` does not change the state, so two calls with the same scores give the
  same output, and the outputs that follow do not change;
- the batch methods (``fit``, ``transform``, ``fit_transform``) agree with the per-instance
  methods.
"""

import copy
import inspect

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

import pysad.transform.ensemble

SEED = 61
MAX_STREAM_LENGTH = 30
# The bucket ensemblers split the detectors into 5 buckets by default, which needs a multiple of 5.
NUM_DETECTORS = [5, 10]

# Small pools make constant columns and repeated rows likely.
FINITE_SCORES = st.one_of(
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

# Constructor arguments by class. The bucket ensemblers draw the buckets at random on every call,
# so they get a seed.
PARAMS = {
    "AverageOfMaximumScoreEnsembler": {"random_state": SEED},
    "MaximumOfAverageScoreEnsembler": {"random_state": SEED},
}


def _ensemblers():
    """The names of the concrete classes that the subpackage exports."""
    module = pysad.transform.ensemble
    return [name for name in module.__all__ if not inspect.isabstract(getattr(module, name))]


def _make(name):
    return getattr(pysad.transform.ensemble, name)(**PARAMS.get(name, {}))


def _scores(num_detectors, min_length=1):
    return st.integers(min_length, MAX_STREAM_LENGTH).flatmap(
        lambda length: hnp.arrays(np.float64, (length, num_detectors), elements=FINITE_SCORES)
    )


@pytest.mark.parametrize("name", _ensemblers())
@PROPERTIES
@given(data=st.data())
def test_transform_partial_does_not_change_state(name, data):
    num_detectors = data.draw(st.sampled_from(NUM_DETECTORS), label="num_detectors")
    scores = data.draw(_scores(num_detectors, min_length=2), label="scores")
    probe = data.draw(hnp.arrays(np.float64, num_detectors, elements=FINITE_SCORES), label="probe")
    fitted, rest = scores[: len(scores) // 2], scores[len(scores) // 2 :]
    ensembler = _make(name).fit(fitted)
    untouched = copy.deepcopy(ensembler)

    for row in [fitted[-1], probe]:
        first = ensembler.transform_partial(row)
        assert ensembler.transform_partial(row) == first
        assert copy.deepcopy(untouched).transform_partial(row) == first

    np.testing.assert_array_equal(
        ensembler.fit_transform(rest), copy.deepcopy(untouched).fit_transform(rest)
    )


@pytest.mark.parametrize("name", _ensemblers())
@PROPERTIES
@given(data=st.data())
def test_batch_methods_agree_with_partial_methods(name, data):
    num_detectors = data.draw(st.sampled_from(NUM_DETECTORS), label="num_detectors")
    scores = data.draw(_scores(num_detectors), label="scores")
    new_scores = data.draw(_scores(num_detectors), label="new_scores")

    ensembler = _make(name)
    streamed = [ensembler.fit_transform_partial(row) for row in scores]
    np.testing.assert_array_equal(_make(name).fit_transform(scores), streamed)

    ensembler = _make(name)
    for row in scores:
        ensembler.fit_partial(row)
    streamed = [ensembler.transform_partial(row) for row in new_scores]
    np.testing.assert_array_equal(_make(name).fit(scores).transform(new_scores), streamed)
