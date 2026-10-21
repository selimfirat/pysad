import pickle

import numpy as np
import pytest

import pysad.transform.preprocessing as preprocessing
from pysad.utils import fix_seed

SEED = 61
NUM_FIT = 60
NUM_NEXT = 20

# These decompose a single scalar series, unlike the other transformers, which take a feature
# vector per instance.
UNIVARIATE_TRANSFORMERS = {"ModifiedSTLResidualTransformer", "SeasonalTrendDecomposer"}

TRANSFORMER_PARAMS = {
    "ModifiedSTLResidualTransformer": {"period": 4, "window_size": 16},
    "SeasonalTrendDecomposer": {"season_length": 4},
    "ShingleTransformer": {"shingle_size": 3},
}


def _data(transformer_name):
    rng = np.random.RandomState(SEED)
    num_instances = NUM_FIT + NUM_NEXT
    if transformer_name in UNIVARIATE_TRANSFORMERS:
        t = np.arange(num_instances)
        return (0.5 + 0.4 * np.sin(2 * np.pi * t / 4) + 0.05 * rng.rand(num_instances)).reshape(
            -1, 1
        )

    return rng.rand(num_instances, 3)


def _fitted_transformer(transformer_name, X):
    fix_seed(SEED)
    transformer_cls = getattr(preprocessing, transformer_name)
    transformer = transformer_cls(**TRANSFORMER_PARAMS.get(transformer_name, {}))
    for xi in X[:NUM_FIT]:
        transformer.fit_partial(xi)

    return transformer


def _next_outputs(transformer, X):
    return np.array([np.asarray(transformer.fit_transform_partial(xi)) for xi in X[NUM_FIT:]])


@pytest.mark.parametrize("transformer_name", preprocessing.__all__)
def test_pickle_round_trip(transformer_name):
    X = _data(transformer_name)
    transformer = _fitted_transformer(transformer_name, X)

    restored = pickle.loads(pickle.dumps(transformer))

    assert type(restored) is type(transformer)
    np.testing.assert_allclose(_next_outputs(restored, X), _next_outputs(transformer, X))
