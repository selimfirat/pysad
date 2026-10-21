import pickle

import numpy as np
import pytest

import pysad.transform.probability_calibration as probability_calibration
from pysad.utils import fix_seed

SEED = 61
NUM_FIT = 60
NUM_NEXT = 20

CALIBRATOR_PARAMS = {
    "ConformalProbabilityCalibrator": {"window_size": 20},
    "GaussianTailProbabilityCalibrator": {"window_size": 20, "short_window_size": 5},
}


def _scores():
    rng = np.random.RandomState(SEED)
    return rng.rand(NUM_FIT + NUM_NEXT)


def _fitted_calibrator(calibrator_name, scores):
    fix_seed(SEED)
    calibrator_cls = getattr(probability_calibration, calibrator_name)
    calibrator = calibrator_cls(**CALIBRATOR_PARAMS.get(calibrator_name, {}))
    for score in scores[:NUM_FIT]:
        calibrator.fit_partial(score)

    return calibrator


def _next_outputs(calibrator, scores):
    return np.array(
        [calibrator.fit_transform_partial(score) for score in scores[NUM_FIT:]], dtype=np.float64
    )


@pytest.mark.parametrize("calibrator_name", probability_calibration.__all__)
def test_pickle_round_trip(calibrator_name):
    scores = _scores()
    calibrator = _fitted_calibrator(calibrator_name, scores)

    restored = pickle.loads(pickle.dumps(calibrator))

    assert type(restored) is type(calibrator)
    np.testing.assert_allclose(_next_outputs(restored, scores), _next_outputs(calibrator, scores))
