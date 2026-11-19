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

# Nonnegative scores for NonparametricDynamicThresholdPostprocessor.
NONNEGATIVE_SCORES = st.one_of(
    st.sampled_from([0.0, 1.0, 0.5]),
    st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
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
    "AveragePostprocessor": {"window_size": 5},
    "MaxPostprocessor": {"window_size": 5},
    "MedianPostprocessor": {"window_size": 5},
    "ZScorePostprocessor": {"window_size": 5},
    "NonparametricDynamicThresholdPostprocessor": {"window_size": 10},
    "QuantileFilter": {"q": 0.5},
    "ThresholdFilter": {"threshold": 0.5},
}


def _postprocessors():
    """The names of the concrete classes that the subpackage exports."""
    module = pysad.transform.postprocessing
    return [name for name in module.__all__ if not inspect.isabstract(getattr(module, name))]


def _make(name):
    return getattr(pysad.transform.postprocessing, name)(**PARAMS.get(name, {}))


def _scores(min_length=1, nonnegative=False):
    elements = NONNEGATIVE_SCORES if nonnegative else FINITE_SCORES
    return st.integers(min_length, MAX_STREAM_LENGTH).flatmap(
        lambda length: hnp.arrays(np.float64, length, elements=elements)
    )


def _needs_nonnegative(name: str) -> bool:
    return name == "NonparametricDynamicThresholdPostprocessor"


@st.composite
def _name_and_scores(draw, min_length=1):
    name = draw(st.sampled_from(_postprocessors()))
    nonnegative = _needs_nonnegative(name)
    scores = draw(_scores(min_length=min_length, nonnegative=nonnegative))
    return name, scores


@st.composite
def _name_and_scores_probe(draw):
    name = draw(st.sampled_from(_postprocessors()))
    nonnegative = _needs_nonnegative(name)
    scores = draw(_scores(min_length=2, nonnegative=nonnegative))
    probe = draw(NONNEGATIVE_SCORES if nonnegative else FINITE_SCORES)
    return name, scores, probe


@st.composite
def _name_scores_new_scores(draw):
    name = draw(st.sampled_from(_postprocessors()))
    nonnegative = _needs_nonnegative(name)
    scores = draw(_scores(nonnegative=nonnegative))
    new_scores = draw(_scores(nonnegative=nonnegative))
    return name, scores, new_scores


@st.composite
def _name_scores_probes(draw):
    name = draw(st.sampled_from(_postprocessors()))
    nonnegative = _needs_nonnegative(name)
    scores = draw(_scores(nonnegative=nonnegative))
    probe_elements = NONNEGATIVE_SCORES if nonnegative else FINITE_SCORES
    probes = draw(st.lists(probe_elements, min_size=1, max_size=10))
    return name, scores, probes


@PROPERTIES
@given(data=_name_and_scores_probe())
def test_transform_partial_does_not_change_state(data):
    name, scores, probe = data
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


@PROPERTIES
@given(data=_name_scores_new_scores())
def test_batch_methods_agree_with_partial_methods(data):
    name, scores, new_scores = data
    postprocessor = _make(name)
    streamed = [postprocessor.fit_transform_partial(score) for score in scores]
    np.testing.assert_array_equal(_make(name).fit_transform(scores), streamed)

    postprocessor = _make(name)
    for score in scores:
        postprocessor.fit_partial(score)
    streamed = [postprocessor.transform_partial(score) for score in new_scores]
    np.testing.assert_array_equal(_make(name).fit(scores).transform(new_scores), streamed)


@PROPERTIES
@given(data=_name_scores_probes())
def test_higher_score_never_gets_lower_output(data):
    name, scores, probes = data
    postprocessor = _make(name)
    for score in scores:
        postprocessor.fit_partial(score)

    # Each probe gets its own copy, so all probes see the same state. The fitted scores are
    # probes too, since a postprocessor transforms a score right after it fits it.
    probes = np.sort(np.concatenate([scores, probes]))
    outputs = [copy.deepcopy(postprocessor).transform_partial(probe) for probe in probes]

    assert all(np.diff(outputs) >= 0), list(zip(probes, outputs, strict=True))
