import pickle

import numpy as np
import pytest

import pysad.transform.projection as projection
from pysad.utils import fix_seed

SEED = 61
NUM_FIT = 60
NUM_NEXT = 20

# Abstract base class, not meant to be instantiated directly.
PICKLE_SKIP_PROJECTORS = {"BaseSKLearnProjector"}

PROJECTOR_PARAMS = {
    "GaussianRandomProjector": {"num_components": 2, "random_state": SEED},
    "SparseRandomProjector": {"num_components": 2, "random_state": SEED},
    "StreamhashProjector": {"num_components": 5},
}


def _data():
    rng = np.random.RandomState(SEED)
    return rng.rand(NUM_FIT + NUM_NEXT, 3)


def _fitted_projector(projector_name, X):
    fix_seed(SEED)
    projector_cls = getattr(projection, projector_name)
    projector = projector_cls(**PROJECTOR_PARAMS.get(projector_name, {}))
    for xi in X[:NUM_FIT]:
        projector.fit_partial(xi)

    return projector


def _next_outputs(projector, X):
    fix_seed(SEED + 1)
    return np.array([projector.fit_transform_partial(xi) for xi in X[NUM_FIT:]])


@pytest.mark.parametrize(
    "projector_name", [name for name in projection.__all__ if name not in PICKLE_SKIP_PROJECTORS]
)
def test_pickle_round_trip(projector_name):
    X = _data()
    projector = _fitted_projector(projector_name, X)

    restored = pickle.loads(pickle.dumps(projector))

    assert type(restored) is type(projector)
    np.testing.assert_allclose(_next_outputs(restored, X), _next_outputs(projector, X))
