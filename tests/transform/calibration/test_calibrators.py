def test_calibrators():
    import numpy as np

    from pysad.transform.probability_calibration import (
        ConformalProbabilityCalibrator,
        GaussianTailProbabilityCalibrator,
    )
    from pysad.utils import fix_seed

    fix_seed(61)

    scores = np.random.rand(100)

    calibrators = {GaussianTailProbabilityCalibrator: {}, ConformalProbabilityCalibrator: {}}

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


def test_conformal_calibrator_p_values():
    import numpy as np
    import pytest

    from pysad.transform.probability_calibration import ConformalProbabilityCalibrator

    for target, expected in [(0.0, 1.0), (5.5, 0.5), (10.0, 0.1)]:
        calibrator = ConformalProbabilityCalibrator(windowed=True, window_size=300)
        calibrator.fit(np.arange(1, 10, dtype=np.float64))
        assert calibrator.fit_transform_partial(target) == pytest.approx(expected)

    # The first point of a fresh stream is never anomalous.
    assert ConformalProbabilityCalibrator().fit_transform_partial(0.7) == 1.0

    # A constant stream gets p = 1, not p = 0.
    calibrated_scores = ConformalProbabilityCalibrator().fit_transform(np.full(50, 0.3))
    assert np.all(calibrated_scores == 1.0)


def test_gaussian_tail_calibrator_global_statistics():
    import numpy as np
    from scipy.stats import norm

    from pysad.transform.probability_calibration import GaussianTailProbabilityCalibrator

    scores = np.random.RandomState(0).normal(0, 10, 100)
    # A window smaller than the stream makes the windowed and global variance
    # differ, so this test fails if the variance meter is windowed (#107).
    calibrator = GaussianTailProbabilityCalibrator(running_statistics=False, window_size=10)

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
    calibrator = GaussianTailProbabilityCalibrator(running_statistics=True, window_size=window_size)

    for score in scores:
        calibrator.fit_partial(score)

    windowed_scores = scores[-window_size:]
    mean = windowed_scores.mean()
    std = windowed_scores.std(ddof=0)

    last_score = scores[-1]
    actual = calibrator.transform_partial(last_score)
    expected = norm.cdf(last_score, loc=mean, scale=std)

    np.testing.assert_allclose(actual, expected)
