import pickle

import numpy as np
import pytest

import pysad.transform.ensemble as ensemble
from pysad.utils import fix_seed

SEED = 61
NUM_FIT = 60
NUM_NEXT = 20
NUM_DETECTORS = 6

ENSEMBLER_PARAMS = {
    "AverageOfMaximumScoreEnsembler": {"n_buckets": 3},
    "MaximumOfAverageScoreEnsembler": {"n_buckets": 3},
}


def _scores():
    rng = np.random.RandomState(SEED)
    return rng.rand(NUM_FIT + NUM_NEXT, NUM_DETECTORS)


def _fitted_ensembler(ensembler_name, scores):
    fix_seed(SEED)
    ensembler_cls = getattr(ensemble, ensembler_name)
    ensembler = ensembler_cls(**ENSEMBLER_PARAMS.get(ensembler_name, {}))
    for row in scores[:NUM_FIT]:
        ensembler.fit_partial(row)

    return ensembler


def _next_outputs(ensembler, scores):
    # AverageOfMaximumScoreEnsembler and MaximumOfAverageScoreEnsembler shuffle detectors into
    # buckets using the global NumPy random state, so both sides must start from the same seed.
    fix_seed(SEED + 1)
    return np.array([ensembler.fit_transform_partial(row) for row in scores[NUM_FIT:]])


@pytest.mark.parametrize("ensembler_name", ensemble.__all__)
def test_pickle_round_trip(ensembler_name):
    scores = _scores()
    ensembler = _fitted_ensembler(ensembler_name, scores)

    restored = pickle.loads(pickle.dumps(ensembler))

    assert type(restored) is type(ensembler)
    np.testing.assert_allclose(_next_outputs(restored, scores), _next_outputs(ensembler, scores))
