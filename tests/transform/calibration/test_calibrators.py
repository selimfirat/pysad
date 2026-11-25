def test_spot_grimshaw_matches_scipy_fit():
    import numpy as np
    from scipy.stats import genpareto

    from pysad.transform.probability_calibration import SPOTProbabilityCalibrator

    rng = np.random.RandomState(0)
    for shape in [-0.3, 0.0, 0.3]:
        peaks = genpareto.rvs(shape, scale=2.0, size=300, random_state=rng)
        # The 0.7 quantile of the scores is 0, so the peaks are the GPD sample.
        scores = np.r_[np.zeros(701), peaks]
        calibrator = SPOTProbabilityCalibrator(num_init=len(scores), level=0.7).fit(scores)

        expected_shape, _, expected_scale = genpareto.fit(peaks, floc=0)
        assert calibrator.t_ == 0.0
        np.testing.assert_allclose(calibrator.gamma_, expected_shape, atol=1e-3)
        np.testing.assert_allclose(calibrator.sigma_, expected_scale, rtol=1e-3)


def test_spot_threshold_matches_exponential_quantile():
    import numpy as np
    import pytest

    from pysad.transform.probability_calibration import SPOTProbabilityCalibrator

    q = 1e-4
    scores = np.random.RandomState(0).exponential(size=20000)
    calibrator = SPOTProbabilityCalibrator(q=q, num_init=20000).fit(scores)

    # The 1 - q quantile of the standard exponential distribution is -log(q).
    assert calibrator.threshold_ == pytest.approx(-np.log(q), rel=0.1)
    # A score at the threshold has the calibrated value 1 - q.
    assert calibrator.transform_partial(calibrator.threshold_) == pytest.approx(1 - q)
    assert calibrator.transform_partial(calibrator.t_) == 0.0
    assert calibrator.transform_partial(calibrator.threshold_ + 1) > 1 - q


def test_spot_constant_scores():
    import numpy as np

    from pysad.transform.probability_calibration import SPOTProbabilityCalibrator

    calibrator = SPOTProbabilityCalibrator(num_init=50)
    calibrated = calibrator.fit_transform(np.full(200, 0.3))
    assert np.all(calibrated == 0.0)
    assert calibrator.transform_partial(1.0) == 1.0
    assert calibrator.transform(np.array([1.0]))[0] == 1.0


def test_spot_keeps_only_the_last_max_peaks():
    import numpy as np

    from pysad.transform.probability_calibration import SPOTProbabilityCalibrator

    calibrator = SPOTProbabilityCalibrator(q=1e-4, num_init=10, level=0.5, max_peaks=5)
    scores = np.r_[np.arange(10, dtype=np.float64), np.arange(5.0, 7.5, 0.25)]
    calibrated = np.array([calibrator.fit_transform_partial(score) for score in scores])

    assert list(calibrator.peaks_) == [1.75, 2.0, 2.25, 2.5, 2.75]
    assert np.isfinite(calibrated).all()
    assert np.all((0.0 <= calibrated) & (calibrated <= 1.0))


def test_spot_rejects_risk_above_initial_level():
    import pytest

    from pysad.transform.probability_calibration import SPOTProbabilityCalibrator

    for q in [0.0, 0.03, 0.5]:
        with pytest.raises(ValueError, match="q must be between 0 and 1 - level"):
            SPOTProbabilityCalibrator(q=q, level=0.98)


def test_anomaly_likelihood_matches_nupic():
    import numpy as np

    from pysad.transform.probability_calibration import AnomalyLikelihoodProbabilityCalibrator

    rng = np.random.RandomState(0)
    scores = rng.beta(0.5, 5, 300)
    scores[150:160] = 1.0
    scores[250] = 0.0
    calibrator = AnomalyLikelihoodProbabilityCalibrator(
        learning_period=40, num_estimation_samples=30, window_size=100, reestimation_period=20
    )
    likelihoods = calibrator.fit_transform(scores)

    # The values of AnomalyLikelihood.anomalyProbability in NuPIC 1.0.5 on the same scores.
    expected = {
        69: 0.5,
        70: 0.5988556426282219,
        71: 0.6945526496941197,
        150: 0.9997448042202933,
        151: 0.9999999996520234,
        152: 0.999,
        200: 0.5238192980493048,
        250: 0.7297238102116057,
        299: 0.6638713747866467,
    }
    # NuPIC approximates the square root of 2 with 1.4142.
    np.testing.assert_allclose(likelihoods[list(expected)], list(expected.values()), atol=1e-5)
    assert np.all(likelihoods[:70] == 0.5)


def test_conformal_calibrator_sorted_window_matches_reference_across_evictions_and_ties():
    from collections import deque

    import numpy as np

    from pysad.transform.probability_calibration import ConformalProbabilityCalibrator

    rng = np.random.default_rng(123)
    # Include ties, negative values, and multiple window turnovers.
    scores = rng.choice([-2.0, 0.0, 0.0, 1.5, 1.5, 3.0, 7.2], size=200)
    for window_size in [1, 5, 25]:
        cal_partial = ConformalProbabilityCalibrator(window_size=window_size)
        win = deque(maxlen=window_size)
        expected = []
        actual_partial = []
        for s in scores:
            win.append(float(s))
            expected.append(sum(1 for v in win if v < s) / len(win))
            actual_partial.append(cal_partial.fit_transform_partial(float(s)))
        np.testing.assert_array_equal(actual_partial, expected)

        cal_batch = ConformalProbabilityCalibrator(window_size=window_size)
        np.testing.assert_array_equal(cal_batch.fit_transform(scores), expected)

        probes = np.array([-5.0, -2.0, 0.0, 1.0, 1.5, 10.0])
        np.testing.assert_array_equal(
            cal_batch.transform(probes),
            [cal_partial.transform_partial(float(p)) for p in probes],
        )


def test_all_calibrators_batch_transform_matches_partial():
    import numpy as np

    from pysad.transform.probability_calibration import (
        AnomalyLikelihoodProbabilityCalibrator,
        ConformalProbabilityCalibrator,
        GaussianTailProbabilityCalibrator,
        SPOTProbabilityCalibrator,
    )

    rng = np.random.default_rng(99)
    train = rng.exponential(size=120)
    probes = rng.exponential(size=40)
    for cls, kwargs in [
        (ConformalProbabilityCalibrator, {"window_size": 30}),
        (GaussianTailProbabilityCalibrator, {"window_size": 40, "short_window_size": 5}),
        (GaussianTailProbabilityCalibrator, {"window_size": None, "short_window_size": 1}),
        (
            AnomalyLikelihoodProbabilityCalibrator,
            {"learning_period": 20, "num_estimation_samples": 20, "reestimation_period": 10},
        ),
        (SPOTProbabilityCalibrator, {"num_init": 40, "level": 0.9, "depth": 5}),
    ]:
        cal_u = cls(**kwargs)
        np.testing.assert_allclose(
            cal_u.transform(probes),
            [cal_u.transform_partial(float(p)) for p in probes],
        )
        cal_f = cls(**kwargs).fit(train)
        np.testing.assert_allclose(
            cal_f.transform(probes),
            [cal_f.transform_partial(float(p)) for p in probes],
        )


def test_calibrators_match_batch_for_numpy_float64_scalars():
    import numpy as np

    from pysad.transform.probability_calibration import (
        AnomalyLikelihoodProbabilityCalibrator,
        GaussianTailProbabilityCalibrator,
    )

    scores = np.array([3.0, 1.6, 1.6, 3.0, 1.6, 1.6])
    probes = np.array([3.0, 1.6, 1.6])
    for cls, kwargs in [
        (GaussianTailProbabilityCalibrator, {"window_size": 10, "short_window_size": 3}),
        (
            AnomalyLikelihoodProbabilityCalibrator,
            {
                "learning_period": 0,
                "num_estimation_samples": 2,
                "window_size": 10,
                "reestimation_period": 1,
                "short_window_size": 3,
            },
        ),
    ]:
        cal = cls(**kwargs)
        np.testing.assert_array_equal(
            cls(**kwargs).fit_transform(scores),
            [cal.fit_transform_partial(s) for s in scores],
        )
        cal_batch = cls(**kwargs).fit(scores)
        np.testing.assert_array_equal(
            cal_batch.transform(probes),
            [cal.transform_partial(p) for p in probes],
        )
