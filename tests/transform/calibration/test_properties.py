"""Test calibrator bounds, state, batch methods, and score formulas with generated streams."""

import copy
import inspect

import numpy as np
import pytest
from hypothesis import HealthCheck, example, given, settings
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
    max_examples=8,
    deadline=None,
    database=None,
    suppress_health_check=[HealthCheck.too_slow],
    # The same examples on every run, so a failure in CI reproduces locally.
    derandomize=True,
)

# Constructor arguments by class. Small windows make the calibrators evict scores.
PARAMS = {
    "AnomalyLikelihoodProbabilityCalibrator": {
        "learning_period": 2,
        "num_estimation_samples": 3,
        "window_size": 10,
        "short_window_size": 3,
        "reestimation_period": 2,
    },
    "ConformalProbabilityCalibrator": {"window_size": 10},
    "ConformalTSFMCalibrator": {"alpha": 0.1, "gamma": 0.005, "window_size": 10},
    "GaussianTailProbabilityCalibrator": {"window_size": 10, "short_window_size": 3},
    "GroupConditionalConformalCalibrator": {"window_size": 10, "max_regimes": 4},
    "SPOTProbabilityCalibrator": {"num_init": 5, "max_peaks": 10},
}


def _calibrators():
    """The concrete classes that the subpackage exports."""
    module = pysad.transform.probability_calibration
    params = []
    for name in module.__all__:
        if inspect.isabstract(getattr(module, name)):
            continue
        params.append(pytest.param(name))

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


@pytest.mark.parametrize("name", _calibrators())
@PROPERTIES
@given(scores=_scores(min_length=2), probe=FINITE_SCORES)
@example(scores=np.linspace(0.0, 1.0, 30), probe=2.0)
def test_transform_partial_does_not_change_state(name, scores, probe):
    fitted, rest = scores[: len(scores) // 2], scores[len(scores) // 2 :]
    # Probe the last fitted score and a new score against the same state.
    calibrator = _fit_partial(_make(name), fitted)
    untouched = copy.deepcopy(calibrator)

    for score in [fitted[-1], probe]:
        first = calibrator.transform_partial(score)
        assert calibrator.transform_partial(score) == first
        assert copy.deepcopy(untouched).transform_partial(score) == first

    np.testing.assert_array_equal(
        calibrator.fit_transform(rest), copy.deepcopy(untouched).fit_transform(rest)
    )


@pytest.mark.parametrize("name", _calibrators())
@PROPERTIES
@given(scores=_scores(), new_scores=_scores())
@example(scores=np.linspace(0.0, 1.0, 30), new_scores=np.array([-1.0, 0.5, 2.0]))
def test_batch_methods_agree_with_partial_methods(name, scores, new_scores):
    calibrator = _make(name)
    streamed = [calibrator.fit_transform_partial(score) for score in scores]
    np.testing.assert_allclose(_make(name).fit_transform(scores), streamed, rtol=1e-12, atol=1e-15)

    # The first new score repeats the last fitted one, as when it is transformed right after it is
    # fitted.
    new_scores = np.concatenate([scores[-1:], new_scores])
    calibrator = _fit_partial(_make(name), scores)
    streamed = [calibrator.transform_partial(score) for score in new_scores]
    np.testing.assert_allclose(
        _make(name).fit(scores).transform(new_scores), streamed, rtol=1e-12, atol=1e-15
    )


# AnomalyLikelihood uses both tails, so raw-score monotonicity does not apply.
@pytest.mark.parametrize(
    "name",
    [
        name
        for name in pysad.transform.probability_calibration.__all__
        if name != "AnomalyLikelihoodProbabilityCalibrator"
    ],
)
@PROPERTIES
@given(scores=_scores(), probes=st.lists(FINITE_SCORES, min_size=1, max_size=10))
def test_higher_score_never_gets_lower_output(name, scores, probes):
    calibrator = _fit_partial(_make(name), scores)

    # Each probe gets its own copy, so all probes see the same state. The fitted scores are
    # probes too, since a calibrator transforms a score right after it fits it.
    probes = np.sort(np.concatenate([scores, probes]))
    outputs = [copy.deepcopy(calibrator).transform_partial(probe) for probe in probes]

    assert all(np.diff(outputs) >= 0), list(zip(probes, outputs, strict=True))


@pytest.mark.parametrize("name", _calibrators())
@PROPERTIES
@given(scores=_scores(), probes=_scores(min_length=0))
@example(scores=np.linspace(0.0, 1.0, 30), probes=np.array([-1.0, 0.5, 2.0]))
def test_outputs_are_probabilities(name, scores, probes):
    calibrator = _make(name)
    assert calibrator.output_kind == "probability"
    for score in scores:
        output = calibrator.fit_transform_partial(float(score))
        assert type(output) is float
        assert 0.0 <= output <= 1.0

    for score in probes:
        output = calibrator.transform_partial(float(score))
        assert type(output) is float
        assert 0.0 <= output <= 1.0

    for output, inputs in [
        (calibrator.transform(probes), probes),
        (_make(name).fit_transform(scores), scores),
    ]:
        assert output.shape == inputs.shape
        assert output.dtype == np.float64
        assert output.ndim == 1
        assert np.isfinite(output).all()
        assert ((output >= 0.0) & (output <= 1.0)).all()


@pytest.mark.parametrize("name", _calibrators())
@PROPERTIES
@given(scores=_scores())
def test_empty_batches_preserve_future_outputs(name, scores):
    calibrator = _make(name).fit(scores)
    untouched = copy.deepcopy(calibrator)
    empty = np.empty(0, dtype=np.float64)
    assert calibrator.fit(empty) is calibrator
    for output in [calibrator.transform(empty), calibrator.fit_transform(empty)]:
        assert output.shape == (0,)
        assert output.dtype == np.float64
    np.testing.assert_array_equal(calibrator.fit_transform(scores), untouched.fit_transform(scores))


@pytest.mark.parametrize(
    "name",
    [
        "ConformalProbabilityCalibrator",
        "ConformalTSFMCalibrator",
        "GroupConditionalConformalCalibrator",
    ],
)
@PROPERTIES
@given(scores=_scores(), probe=FINITE_SCORES, window_size=st.integers(1, 10))
def test_conformal_rank_counts_only_lower_scores_in_recent_window(name, scores, probe, window_size):
    params = {**PARAMS[name], "window_size": window_size}
    calibrator = getattr(pysad.transform.probability_calibration, name)(**params)
    for index, score in enumerate(scores):
        assert calibrator.fit_partial(float(score)) is calibrator
        recent = scores[max(0, index + 1 - window_size) : index + 1]
        for value in [score, probe]:
            expected = np.count_nonzero(recent < value) / len(recent)
            assert calibrator.transform_partial(float(value)) == expected


@PROPERTIES
@given(
    score=st.integers(-100, 100),
    count=st.integers(1, 30),
    window_size=st.one_of(st.none(), st.integers(1, 10)),
)
def test_gaussian_constant_stream_has_exact_zero_variance_limits(score, count, window_size):
    from pysad.transform.probability_calibration import GaussianTailProbabilityCalibrator

    calibrator = GaussianTailProbabilityCalibrator(window_size=window_size, short_window_size=1)
    outputs = calibrator.fit_transform(np.full(count, score, dtype=np.float64))
    np.testing.assert_array_equal(outputs, np.full(count, 0.5))
    assert calibrator.transform_partial(score - 1.0) == 0.0
    assert calibrator.transform_partial(float(score)) == 0.5
    assert calibrator.transform_partial(score + 1.0) == 1.0


@PROPERTIES
@given(score=st.integers(1, 100), distance=st.integers(1, 100), count=st.integers(5, 30))
def test_anomaly_likelihood_uses_both_tails(score, distance, count):
    from pysad.transform.probability_calibration import AnomalyLikelihoodProbabilityCalibrator

    calibrator = AnomalyLikelihoodProbabilityCalibrator(
        learning_period=2,
        num_estimation_samples=3,
        window_size=10,
        short_window_size=1,
        reestimation_period=2,
    )
    outputs = calibrator.fit_transform(np.full(count, score, dtype=np.float64))
    np.testing.assert_array_equal(outputs[:5], np.full(5, 0.5))
    center = calibrator.transform_partial(float(score))
    low = calibrator.transform_partial(score - float(distance))
    high = calibrator.transform_partial(score + float(distance))
    assert low == high
    assert low > center


@pytest.mark.parametrize("depth", [0, 3])
@PROPERTIES
@given(scale=st.floats(0.5, 10.0), offset=st.floats(-10.0, 10.0), q=st.floats(0.001, 0.05))
def test_spot_threshold_matches_tail_risk(depth, scale, offset, q):
    from pysad.transform.probability_calibration import SPOTProbabilityCalibrator

    calibrator = SPOTProbabilityCalibrator(num_init=20, level=0.8, depth=depth, q=q)
    calibrator.fit(offset + scale * np.arange(20 + depth, dtype=np.float64) ** 2)
    assert np.isfinite(calibrator.threshold_)
    assert calibrator.transform_partial(calibrator.threshold_) == pytest.approx(1.0 - q)


@pytest.mark.parametrize("method", ["simple", "momentum"])
@PROPERTIES
@given(scores=_scores(), alpha=st.floats(0.01, 0.99), gamma=st.floats(0.001, 2.0))
def test_adaptive_conformal_keeps_alpha_in_bounds(method, scores, alpha, gamma):
    from pysad.transform.probability_calibration import ConformalTSFMCalibrator

    calibrator = ConformalTSFMCalibrator(alpha=alpha, gamma=gamma, window_size=3, method=method)
    for score in scores:
        calibrator.fit_partial(float(score))
        assert 0.0 <= calibrator.alpha_t <= 1.0


@PROPERTIES
@given(scores=_scores(), other_scores=_scores(), probe=FINITE_SCORES)
def test_group_conformal_keeps_regime_scores_separate(scores, other_scores, probe):
    from pysad.transform.probability_calibration import GroupConditionalConformalCalibrator

    calibrator = GroupConditionalConformalCalibrator(window_size=3, max_regimes=2)
    for score in scores:
        calibrator.fit_partial(float(score), regime="a")
    expected = calibrator.transform_partial(probe, regime="a")
    for score in other_scores:
        calibrator.fit_partial(float(score), regime="b")
        assert calibrator.transform_partial(probe, regime="a") == expected
    recent = other_scores[-3:]
    assert calibrator.transform_partial(probe, regime="b") == np.count_nonzero(
        recent < probe
    ) / len(recent)
