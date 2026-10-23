import pickle

import numpy as np
import pytest

import pysad.transform.postprocessing as postprocessing
from pysad.utils import fix_seed

SEED = 61
NUM_FIT = 60
NUM_NEXT = 20

POSTPROCESSOR_PARAMS = {
    "RunningAveragePostprocessor": {"window_size": 16},
    "RunningMaxPostprocessor": {"window_size": 16},
    "RunningMedianPostprocessor": {"window_size": 16},
    "RunningZScorePostprocessor": {"window_size": 16},
}


def _scores():
    rng = np.random.RandomState(SEED)
    return rng.rand(NUM_FIT + NUM_NEXT)


def _fitted_postprocessor(postprocessor_name, scores):
    fix_seed(SEED)
    postprocessor_cls = getattr(postprocessing, postprocessor_name)
    postprocessor = postprocessor_cls(**POSTPROCESSOR_PARAMS.get(postprocessor_name, {}))
    for score in scores[:NUM_FIT]:
        postprocessor.fit_partial(score)

    return postprocessor


def _next_outputs(postprocessor, scores):
    return np.array(
        [postprocessor.fit_transform_partial(score) for score in scores[NUM_FIT:]],
        dtype=np.float64,
    )


@pytest.mark.parametrize("postprocessor_name", postprocessing.__all__)
def test_pickle_round_trip(postprocessor_name):
    scores = _scores()
    postprocessor = _fitted_postprocessor(postprocessor_name, scores)

    restored = pickle.loads(pickle.dumps(postprocessor))

    assert type(restored) is type(postprocessor)
    np.testing.assert_allclose(
        _next_outputs(restored, scores), _next_outputs(postprocessor, scores)
    )
