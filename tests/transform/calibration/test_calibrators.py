def test_calibrators():
    import numpy as np

    from pysad.transform.probability_calibration import (
        ConformalProbabilityCalibrator,
        GaussianTailProbabilityCalibrator,
        SPOTProbabilityCalibrator,
    )
    from pysad.utils import fix_seed

    fix_seed(61)

    scores = np.random.rand(100)

    calibrators = {
        GaussianTailProbabilityCalibrator: {},
        ConformalProbabilityCalibrator: {},
        SPOTProbabilityCalibrator: {"num_init": 40, "level": 0.9, "depth": 5},
    }

    for calibrator_cls, args in calibrators.items():
        calibrator = calibrator_cls(**args)
        calibrated_scores = calibrator.fit_transform(scores)

        assert calibrated_scores.shape == scores.shape
        assert not np.isnan(calibrated_scores).any()

        calibrator = calibrator_cls(**args).fit(scores)
        assert type(calibrator) is calibrator_cls
        calibrated_scores = calibrator.fit_transform(scores)

        assert calibrated_scores.shape == scores.shape
        assert not np.isnan(calibrated_scores).any()


def test_conformal_calibrator_values():
    import numpy as np
    import pytest

    from pysad.transform.probability_calibration import ConformalProbabilityCalibrator

    # One minus the conformal p-value: the fraction of the window below the target.
    for target, expected in [(0.0, 0.0), (5.5, 0.5), (10.0, 0.9)]:
        calibrator = ConformalProbabilityCalibrator(window_size=300)
        calibrator.fit(np.arange(1, 10, dtype=np.float64))
        assert calibrator.fit_transform_partial(target) == pytest.approx(expected)

    # The first point of a fresh stream is never anomalous.
    assert ConformalProbabilityCalibrator().fit_transform_partial(0.7) == 0.0

    # Nor is any point of a constant stream.
    calibrated_scores = ConformalProbabilityCalibrator().fit_transform(np.full(50, 0.3))
    assert np.all(calibrated_scores == 0.0)


def test_gaussian_tail_calibrator_global_statistics():
    import numpy as np
    from scipy.stats import norm

    from pysad.transform.probability_calibration import GaussianTailProbabilityCalibrator

    scores = np.random.RandomState(0).normal(0, 10, 100)
    # A window smaller than the stream makes the windowed and global variance
    # differ, so this test fails if the variance meter is windowed (#107).
    calibrator = GaussianTailProbabilityCalibrator(
        running_statistics=False, window_size=10, short_window_size=1
    )

    for score in scores:
        calibrator.fit_partial(score)

    mean = scores.mean()
    std = scores.std(ddof=0)

    last_score = scores[-1]
    actual = calibrator.transform_partial(last_score)
    expected = norm.cdf(last_score, loc=mean, scale=std)

    np.testing.assert_allclose(actual, expected)


def test_gaussian_tail_calibrator_running_statistics():
    import numpy as np
    from scipy.stats import norm

    from pysad.transform.probability_calibration import GaussianTailProbabilityCalibrator

    window_size = 10
    scores = np.random.RandomState(0).normal(0, 10, 100)
    calibrator = GaussianTailProbabilityCalibrator(
        running_statistics=True, window_size=window_size, short_window_size=1
    )

    for score in scores:
        calibrator.fit_partial(score)

    windowed_scores = scores[-window_size:]
    mean = windowed_scores.mean()
    std = windowed_scores.std(ddof=0)

    last_score = scores[-1]
    actual = calibrator.transform_partial(last_score)
    expected = norm.cdf(last_score, loc=mean, scale=std)

    np.testing.assert_allclose(actual, expected)


def test_gaussian_tail_calibrator_values():
    import numpy as np
    import pytest

    from pysad.transform.probability_calibration import GaussianTailProbabilityCalibrator

    # The fitted scores have mean 0 and standard deviation 1.
    calibrator = GaussianTailProbabilityCalibrator(running_statistics=False, short_window_size=1)
    calibrator.fit(np.array([-1.0, 1.0]))

    # As for model scores, higher values mean more anomalous.
    assert calibrator.transform_partial(0.0) == pytest.approx(0.5)
    assert calibrator.transform_partial(3.0) == pytest.approx(0.9986501, rel=1e-4)
    assert calibrator.transform_partial(-3.0) == pytest.approx(0.0013499, rel=1e-4)


def test_gaussian_tail_calibrator_short_window():
    import numpy as np
    from scipy.stats import norm

    from pysad.transform.probability_calibration import GaussianTailProbabilityCalibrator

    scores = np.random.RandomState(0).normal(0, 10, 100)
    calibrator = GaussianTailProbabilityCalibrator(running_statistics=False, short_window_size=10)

    for score in scores:
        calibrator.fit_partial(score)

    mean = scores.mean()
    std = scores.std(ddof=0)
    short_mean = scores[-10:].mean()

    last_score = scores[-1]
    actual = calibrator.transform_partial(last_score)
    expected = norm.cdf(short_mean, loc=mean, scale=std)

    np.testing.assert_allclose(actual, expected)


def test_gaussian_tail_calibrator_noise_spike_resistance():
    import numpy as np

    from pysad.transform.probability_calibration import GaussianTailProbabilityCalibrator
    from pysad.utils import fix_seed

    fix_seed(0)
    noise = np.random.randn(2000)
    single_spike = np.r_[noise, 4.0, np.zeros(9)]
    sustained = np.r_[noise, np.full(10, 2.0)]

    single_pysad = GaussianTailProbabilityCalibrator().fit_transform(single_spike)[-10:]
    sustained_pysad = GaussianTailProbabilityCalibrator().fit_transform(sustained)[-10:]

    # A single spike's likelihood is smoothed down by the short-term window (W' = 10)
    assert single_pysad.max() < 0.70
    # A sustained series of elevated scores raises an alert
    assert sustained_pysad.max() > 0.95


def test_gaussian_tail_calibrator_zero_variance():
    import numpy as np

    from pysad.transform.probability_calibration import GaussianTailProbabilityCalibrator

    for unit in [1e-3, 1.0, 1e3]:
        calibrator = GaussianTailProbabilityCalibrator().fit(np.zeros(100))
        assert calibrator.transform_partial(unit) == 1.0
        assert calibrator.transform_partial(-unit) == 0.0
        assert calibrator.transform_partial(0.0) == 0.5


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


def test_dspot_keeps_false_alarm_rate_under_drift():
    import numpy as np

    from pysad.transform.probability_calibration import SPOTProbabilityCalibrator

    q = 1e-3
    rng = np.random.RandomState(0)
    scores = rng.normal(size=20000) + np.arange(20000) * 1e-3

    spot = SPOTProbabilityCalibrator(q=q).fit_transform(scores)[2000:]
    dspot = SPOTProbabilityCalibrator(q=q, depth=10).fit_transform(scores)[2000:]

    assert (spot > 1 - q).mean() > 0.1
    assert (dspot > 1 - q).mean() < 0.01


def test_spot_does_not_learn_anomalies():
    import numpy as np

    from pysad.transform.probability_calibration import SPOTProbabilityCalibrator

    calibrator = SPOTProbabilityCalibrator(num_init=1000)
    calibrator.fit(np.random.RandomState(0).exponential(size=1000))
    threshold = calibrator.threshold_

    assert calibrator.fit_transform_partial(threshold + 100) > 1 - 1e-6
    assert calibrator.threshold_ == threshold


def test_spot_constant_scores():
    import numpy as np

    from pysad.transform.probability_calibration import SPOTProbabilityCalibrator

    calibrated = SPOTProbabilityCalibrator(num_init=50).fit_transform(np.full(200, 0.3))
    assert np.all(calibrated == 0.0)


def test_spot_rejects_risk_above_initial_level():
    import pytest

    from pysad.transform.probability_calibration import SPOTProbabilityCalibrator

    for q in [0.0, 0.03, 0.5]:
        with pytest.raises(ValueError, match="q must be between 0 and 1 - level"):
            SPOTProbabilityCalibrator(q=q, level=0.98)
