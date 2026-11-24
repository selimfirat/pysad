"""Behavior tests for the regime-stratified conformal calibrator.

These tests cover behavior the shared calibrator contracts do not: per-regime stratification,
regime eviction, and the default-regime equivalence with ConformalProbabilityCalibrator.
"""

import numpy as np
import pytest

from pysad.transform.probability_calibration import (
    ConformalProbabilityCalibrator,
    GroupConditionalConformalCalibrator,
)


def test_default_regime_matches_conformal_probability_calibrator():
    scores = np.array([1.0, 2.0, 2.0, 4.0, 3.0, 10.0])
    group = GroupConditionalConformalCalibrator(window_size=10)
    single = ConformalProbabilityCalibrator(window_size=10)
    np.testing.assert_array_equal(group.fit_transform(scores), single.fit_transform(scores))


def test_first_point_of_a_fresh_regime_is_not_anomalous():
    calibrator = GroupConditionalConformalCalibrator(window_size=10)
    assert calibrator.fit_transform_partial(0.7, regime="a") == 0.0


def test_per_regime_scores_do_not_leak_across_regimes():
    calibrator = GroupConditionalConformalCalibrator(window_size=10)
    for score in np.full(20, 1.0):
        calibrator.fit_partial(float(score), regime="a")
    for score in np.full(20, 9.0):
        calibrator.fit_partial(float(score), regime="b")

    low_regime = calibrator.transform_partial(5.0, regime="a")
    high_regime = calibrator.transform_partial(5.0, regime="b")
    assert low_regime > high_regime


def test_transform_partial_is_pure_per_regime():
    calibrator = GroupConditionalConformalCalibrator(window_size=10)
    for score in [1.0, 2.0, 3.0]:
        calibrator.fit_partial(score, regime="a")

    first = calibrator.transform_partial(2.0, regime="a")
    assert calibrator.transform_partial(2.0, regime="a") == first


def test_monotonic_within_a_regime():
    calibrator = GroupConditionalConformalCalibrator(window_size=10)
    for score in np.linspace(0.0, 10.0, 11):
        calibrator.fit_partial(float(score), regime="a")

    probes = [-1.0, 0.0, 5.0, 5.0, 10.0, 11.0]
    outputs = [calibrator.transform_partial(p, regime="a") for p in probes]
    assert outputs == sorted(outputs)


def test_window_eviction_within_a_regime():
    calibrator = GroupConditionalConformalCalibrator(window_size=3)
    for score in [1.0, 2.0, 3.0]:
        calibrator.fit_partial(score, regime="a")
    before = calibrator.transform_partial(2.5, regime="a")
    calibrator.fit_partial(4.0, regime="a")
    after = calibrator.transform_partial(2.5, regime="a")
    assert before == pytest.approx(2.0 / 3.0)
    assert after == pytest.approx(1.0 / 3.0)


def test_max_regimes_evicts_the_least_recently_used_regime():
    calibrator = GroupConditionalConformalCalibrator(window_size=10, max_regimes=2)
    for score in np.full(10, 1.0):
        calibrator.fit_partial(float(score), regime="a")
    for score in np.full(10, 9.0):
        calibrator.fit_partial(float(score), regime="b")

    before_a = calibrator.transform_partial(5.0, regime="a")
    before_b = calibrator.transform_partial(5.0, regime="b")

    for score in np.full(10, 5.0):
        calibrator.fit_partial(float(score), regime="c")

    after_a = calibrator.transform_partial(5.0, regime="a")
    after_b = calibrator.transform_partial(5.0, regime="b")
    after_c = calibrator.transform_partial(6.0, regime="c")

    assert before_a == pytest.approx(1.0)
    assert before_b == pytest.approx(0.0)
    # Regime "a" was evicted, so its calibrated output resets to zero.
    assert after_a == pytest.approx(0.0)
    # Regime "b" is still tracked and unchanged.
    assert after_b == pytest.approx(0.0)
    # Regime "c" is active.
    assert after_c == pytest.approx(1.0)


def test_nan_scores_are_not_learned_and_transform_to_zero():
    calibrator = GroupConditionalConformalCalibrator(window_size=5)
    calibrator.fit_partial(float("nan"), regime="a")
    assert calibrator.transform_partial(1.0, regime="a") == 0.0
    assert calibrator.transform_partial(float("nan"), regime="a") == 0.0


def test_batch_transform_matches_partial_for_default_regime():
    rng = np.random.default_rng(3)
    scores = rng.uniform(size=50)
    batch = GroupConditionalConformalCalibrator(window_size=20)
    streamed = GroupConditionalConformalCalibrator(window_size=20)
    np.testing.assert_allclose(
        batch.fit_transform(scores),
        [streamed.fit_transform_partial(float(s)) for s in scores],
    )
    probes = np.array([0.1, 0.5, 0.9])
    np.testing.assert_allclose(
        batch.transform(probes),
        [streamed.transform_partial(float(p)) for p in probes],
    )


def test_invalid_parameters_raise():
    with pytest.raises(ValueError, match="window_size must be at least 1."):
        GroupConditionalConformalCalibrator(window_size=0)
    with pytest.raises(ValueError, match="max_regimes must be at least 1."):
        GroupConditionalConformalCalibrator(max_regimes=0)
