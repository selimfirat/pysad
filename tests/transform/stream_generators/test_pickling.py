import pickle

import numpy as np
import pytest

import pysad.transform.stream_generators as stream_generators
from pysad.utils import fix_seed

SEED = 61
NUM_FIT = 60
NUM_NEXT = 20

GENERATOR_PARAMS = {
    "SlowPoisonStream": {
        "poison_rate": 0.3,
        "warmup": 10,
        "magnitude": 4.0,
        "random_state": SEED,
    },
    "StealthBurstStream": {
        "burst_every": 20,
        "burst_length": 5,
        "magnitude": 3.0,
        "random_state": SEED,
    },
}


def _data() -> np.ndarray:
    rng = np.random.RandomState(SEED)
    return rng.rand(NUM_FIT + NUM_NEXT, 3)


def _fitted_generator(generator_name: str, X: np.ndarray):
    fix_seed(SEED)
    generator_cls = getattr(stream_generators, generator_name)
    generator = generator_cls(**GENERATOR_PARAMS.get(generator_name, {}))
    for xi in X[:NUM_FIT]:
        generator.fit_partial(xi)
    return generator


def _next_outputs(generator, X: np.ndarray) -> np.ndarray:
    return np.array([np.asarray(generator.fit_transform_partial(xi)) for xi in X[NUM_FIT:]])


@pytest.mark.parametrize("generator_name", stream_generators.__all__)
def test_pickle_round_trip(generator_name: str) -> None:
    X = _data()
    generator = _fitted_generator(generator_name, X)

    restored = pickle.loads(pickle.dumps(generator))

    assert type(restored) is type(generator)
    np.testing.assert_allclose(_next_outputs(restored, X), _next_outputs(generator, X))
