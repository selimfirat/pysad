"""Property-based tests for the guarantees every postprocessor makes, whatever the scores.

Each test draws a stream of scores with Hypothesis and checks one guarantee for every concrete
class in ``pysad.transform.postprocessing.__all__``:

- ``transform_partial`` does not change the state, so two calls with the same score give the
  same output, and the outputs that follow do not change;
- the batch methods (``fit``, ``transform``, ``fit_transform``) agree with the per-score methods;
- in the same state, a higher score never gets a lower output. Higher output means more anomalous
  for every postprocessor (AGENTS.md), and each postprocessor maps a score either to a statistic
  of the previous scores, which does not depend on the score, or to its Z-score, which grows with
  it (class docstrings).
"""

import copy
import inspect

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

import pysad.transform.postprocessing

MAX_STREAM_LENGTH = 30

# Small pools make constant streams and repeated scores likely.
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

# Constructor arguments by class. A small window makes the running postprocessors evict scores.
PARAMS = {
    "RunningAveragePostprocessor": {"window_size": 5},
    "RunningMaxPostprocessor": {"window_size": 5},
    "RunningMedianPostprocessor": {"window_size": 5},
    "RunningZScorePostprocessor": {"window_size": 5},
}


def _postprocessors():
    """The names of the concrete classes that the subpackage exports."""
    module = pysad.transform.postprocessing
    return [name for name in module.__all__ if not inspect.isabstract(getattr(module, name))]


def _make(name):
    return getattr(pysad.transform.postprocessing, name)(**PARAMS.get(name, {}))


def _scores(min_length=1):
    return st.integers(min_length, MAX_STREAM_LENGTH).flatmap(
        lambda length: hnp.arrays(np.float64, length, elements=FINITE_SCORES)
    )


@pytest.mark.parametrize("name", _postprocessors())
@PROPERTIES
@given(scores=_scores(min_length=2), probe=FINITE_SCORES)
def test_transform_partial_does_not_change_state(name, scores, probe):
    fitted, rest = scores[: len(scores) // 2], scores[len(scores) // 2 :]
    postprocessor = _make(name).fit(fitted)
    untouched = copy.deepcopy(postprocessor)

    for score in [fitted[-1], probe]:
        first = postprocessor.transform_partial(score)
        assert postprocessor.transform_partial(score) == first
        assert untouched.transform_partial(score) == first

    np.testing.assert_array_equal(
        postprocessor.fit_transform(rest), copy.deepcopy(untouched).fit_transform(rest)
    )


@pytest.mark.parametrize("name", _postprocessors())
@PROPERTIES
@given(scores=_scores(), new_scores=_scores())
def test_batch_methods_agree_with_partial_methods(name, scores, new_scores):
    postprocessor = _make(name)
    streamed = [postprocessor.fit_transform_partial(score) for score in scores]
    np.testing.assert_array_equal(_make(name).fit_transform(scores), streamed)

    postprocessor = _make(name)
    for score in scores:
        postprocessor.fit_partial(score)
    streamed = [postprocessor.transform_partial(score) for score in new_scores]
    np.testing.assert_array_equal(_make(name).fit(scores).transform(new_scores), streamed)


@pytest.mark.parametrize("name", _postprocessors())
@PROPERTIES
@given(scores=_scores(), probes=st.lists(FINITE_SCORES, min_size=1, max_size=10))
def test_higher_score_never_gets_lower_output(name, scores, probes):
    postprocessor = _make(name)
    for score in scores:
        postprocessor.fit_partial(score)

    # Each probe gets its own copy, so all probes see the same state. The fitted scores are
    # probes too, since a postprocessor transforms a score right after it fits it.
    probes = np.sort(np.concatenate([scores, probes]))
    outputs = [copy.deepcopy(postprocessor).transform_partial(probe) for probe in probes]

    assert all(np.diff(outputs) >= 0), list(zip(probes, outputs, strict=True))
