"""Property-based tests for the guarantees every calibrator makes, whatever the scores.

Each test draws a stream of scores with Hypothesis and checks one guarantee for every concrete
class in ``pysad.transform.probability_calibration.__all__``:

- ``transform_partial`` does not change the state, so two calls with the same score give the
  same output, and the outputs that follow do not change;
- the batch methods (``fit``, ``transform``, ``fit_transform``) agree with the per-score methods;
- in the same state, a higher score never gets a lower output. Higher output means more anomalous
  for every calibrator (AGENTS.md). ``ConformalProbabilityCalibrator`` returns the fraction of the
  window below the score, and ``GaussianTailProbabilityCalibrator`` returns
  ``1 - Q((short-term average - mean) / std)``, where the short-term average includes the score
  (class docstrings).
"""

import copy
import inspect

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

import pysad.transform.probability_calibration

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

# Constructor arguments by class. Small windows make the calibrators evict scores.
PARAMS = {
    "ConformalProbabilityCalibrator": {"window_size": 10},
    "GaussianTailProbabilityCalibrator": {"window_size": 10, "short_window_size": 3},
}

# Open bugs that a property finds, by test and class. Each is a strict xfail, so the test starts
# failing once the bug is fixed, as a reminder to remove it here.
KNOWN_FAILURES = {
    "test_transform_partial_does_not_change_state": {
        "GaussianTailProbabilityCalibrator": (
            "transform_partial treats a score as already fitted only on its first call after "
            "fit_partial, so a second call with the same score counts it twice in the short-term "
            "average."
        ),
    },
    "test_batch_methods_agree_with_partial_methods": {
        "GaussianTailProbabilityCalibrator": (
            "After fit_partial, transform_partial counts a score equal to the last fitted one "
            "only once in the short-term average, but after fit it adds the score again."
        ),
    },
    "test_higher_score_never_gets_lower_output": {
        "GaussianTailProbabilityCalibrator": (
            "Right after fit_partial, the fitted score is counted once in the short-term "
            "average, but any other score is added to it, which drops the oldest score, so a "
            "slightly higher score can get a lower output."
        ),
    },
}


def _calibrators(test_name):
    """The concrete classes that the subpackage exports, with the known failures of `test_name`."""
    module = pysad.transform.probability_calibration
    known_failures = KNOWN_FAILURES.get(test_name, {})
    params = []
    for name in module.__all__:
        if inspect.isabstract(getattr(module, name)):
            continue
        marks = []
        if name in known_failures:
            marks.append(pytest.mark.xfail(reason=known_failures[name], strict=True))
        params.append(pytest.param(name, marks=marks))

    return params


def _make(name):
    return getattr(pysad.transform.probability_calibration, name)(**PARAMS.get(name, {}))


def _fit_partial(calibrator, scores):
    for score in scores:
        calibrator.fit_partial(score)

    return calibrator


def _scores(min_length=1):
    return st.integers(min_length, MAX_STREAM_LENGTH).flatmap(
        lambda length: hnp.arrays(np.float64, length, elements=FINITE_SCORES)
    )


@pytest.mark.parametrize("name", _calibrators("test_transform_partial_does_not_change_state"))
@PROPERTIES
@given(scores=_scores(min_length=2), probe=FINITE_SCORES)
def test_transform_partial_does_not_change_state(name, scores, probe):
    fitted, rest = scores[: len(scores) // 2], scores[len(scores) // 2 :]
    # The last fitted score is transformed first, as fit_transform_partial does.
    calibrator = _fit_partial(_make(name), fitted)
    untouched = copy.deepcopy(calibrator)

    for score in [fitted[-1], probe]:
        first = calibrator.transform_partial(score)
        assert calibrator.transform_partial(score) == first
        assert copy.deepcopy(untouched).transform_partial(score) == first

    np.testing.assert_array_equal(
        calibrator.fit_transform(rest), copy.deepcopy(untouched).fit_transform(rest)
    )


@pytest.mark.parametrize("name", _calibrators("test_batch_methods_agree_with_partial_methods"))
@PROPERTIES
@given(scores=_scores(), new_scores=_scores())
def test_batch_methods_agree_with_partial_methods(name, scores, new_scores):
    calibrator = _make(name)
    streamed = [calibrator.fit_transform_partial(score) for score in scores]
    np.testing.assert_array_equal(_make(name).fit_transform(scores), streamed)

    # The first new score repeats the last fitted one, as when it is transformed right after it is
    # fitted.
    new_scores = np.concatenate([scores[-1:], new_scores])
    calibrator = _fit_partial(_make(name), scores)
    streamed = [calibrator.transform_partial(score) for score in new_scores]
    np.testing.assert_array_equal(_make(name).fit(scores).transform(new_scores), streamed)


@pytest.mark.parametrize("name", _calibrators("test_higher_score_never_gets_lower_output"))
@PROPERTIES
@given(scores=_scores(), probes=st.lists(FINITE_SCORES, min_size=1, max_size=10))
def test_higher_score_never_gets_lower_output(name, scores, probes):
    calibrator = _fit_partial(_make(name), scores)

    # Each probe gets its own copy, so all probes see the same state. The fitted scores are
    # probes too, since a calibrator transforms a score right after it fits it.
    probes = np.sort(np.concatenate([scores, probes]))
    outputs = [copy.deepcopy(calibrator).transform_partial(probe) for probe in probes]

    assert all(np.diff(outputs) >= 0), list(zip(probes, outputs, strict=True))
