
def test_calibrators():
    from pysad.transform.probability_calibration import GaussianTailProbabilityCalibrator
    import numpy as np
    from pysad.transform.probability_calibration import ConformalProbabilityCalibrator
    from pysad.utils import fix_seed
    fix_seed(61)

    scores = np.random.rand(100)

    calibrators = {
        GaussianTailProbabilityCalibrator: {},
        ConformalProbabilityCalibrator: {}
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
