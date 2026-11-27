"""Generated scores exercise postprocessor state, ordering, and batch equivalence."""

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

# Nonnegative scores for NonparametricDynamicThresholdPostprocessor.
NONNEGATIVE_SCORES = st.one_of(
    st.sampled_from([0.0, 1.0, 0.5]),
    st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
)

PROPERTIES = settings(
    max_examples=8,
    deadline=None,
    database=None,
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


@pytest.mark.parametrize("name", _postprocessors())
@PROPERTIES
@given(data=st.data())
def test_transform_partial_does_not_change_state(name, data):
    scores = data.draw(_scores(min_length=2, nonnegative=_needs_nonnegative(name)))
    probe = data.draw(NONNEGATIVE_SCORES if _needs_nonnegative(name) else FINITE_SCORES)
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
@given(data=st.data())
def test_batch_methods_agree_with_partial_methods(name, data):
    scores = data.draw(_scores(nonnegative=_needs_nonnegative(name)))
    new_scores = data.draw(_scores(nonnegative=_needs_nonnegative(name)))
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
@given(data=st.data())
def test_higher_score_never_gets_lower_output(name, data):
    scores = data.draw(_scores(nonnegative=_needs_nonnegative(name)))
    elements = NONNEGATIVE_SCORES if _needs_nonnegative(name) else FINITE_SCORES
    probes = data.draw(st.lists(elements, min_size=1, max_size=10))
    postprocessor = _make(name)
    for score in scores:
        postprocessor.fit_partial(score)

    # Copies give every probe the same fitted state.
    probes = np.sort(np.concatenate([scores, probes]))
    outputs = [copy.deepcopy(postprocessor).transform_partial(probe) for probe in probes]

    assert all(np.diff(outputs) >= 0), list(zip(probes, outputs, strict=True))


@pytest.mark.parametrize("name", _postprocessors())
@PROPERTIES
@given(data=st.data())
def test_finite_scores_return_finite_float_outputs(name, data):
    scores = data.draw(_scores(min_length=20, nonnegative=_needs_nonnegative(name)))
    postprocessor = _make(name)
    for score in scores:
        output = postprocessor.fit_transform_partial(score)
        assert isinstance(output, float)
        assert np.isfinite(output)
    output = _make(name).fit_transform(scores)
    assert output.shape == scores.shape
    assert output.dtype == np.float64
    assert np.isfinite(output).all()
