import copy
import pickle

import numpy as np
import pytest

import pysad.models
from pysad.utils import fix_seed

SEED = 61
NUM_FIT = 60
NUM_NEXT = 20

UNIVARIATE_MODELS = {"KNNCAD", "MedianAbsoluteDeviation", "RelativeEntropy", "SeasonalESD", "SeasonalHybridESD", "StandardAbsoluteDeviation"}

MODEL_PARAMS = {
    "HalfSpaceTrees": {"feature_mins": [0.0, 0.0, 0.0], "feature_maxes": [1.0, 1.0, 1.0], "window_size": 20, "num_trees": 5, "max_depth": 8},
    "IForestASD": {"window_size": 32},
    "Inqmad": {"input_shape": 3, "dim_x": 32, "gamma": 100},
    "KNNCAD": {"probationary_period": 50},
    # Non-overlapping windows of 5 close 4 times among the NUM_NEXT compared values, and 80 buckets
    # split the noise so that some of them score 1.0 (see test_relative_entropy_compared_scores_use_learned_state).
    "RelativeEntropy": {"min_val": 0.0, "max_val": 1.0, "window_size": 5, "num_bins": 80, "step": 5},
    "RobustRandomCutForest": {"tree_size": 32},
    "RSHash": {"feature_mins": [0.0, 0.0, 0.0], "feature_maxes": [1.0, 1.0, 1.0]},
    "SeasonalESD": {"period": 4, "window_size": 16, "max_anomalies": 3},
    "SeasonalHybridESD": {"period": 4, "window_size": 16, "max_anomalies": 3},
    "xStream": {"n_chains": 10, "depth": 10, "window_size": 20},
}

PICKLE_XFAIL_MODELS = {
    "Inqmad": "Inqmad keeps jax-jitted functions as attributes, which cannot be pickled.",
}


def _data(model_name):
    rng = np.random.RandomState(SEED)
    num_instances = NUM_FIT + NUM_NEXT
    if model_name in UNIVARIATE_MODELS:
        t = np.arange(num_instances)
        X = (0.5 + 0.4 * np.sin(2 * np.pi * t / 4) + 0.05 * rng.rand(num_instances)).reshape(-1, 1)
    else:
        X = rng.rand(num_instances, 3)
    y = (rng.rand(num_instances) > 0.9).astype(int)

    return X, y


def _make_model(model_name, X):
    model_cls = getattr(pysad.models, model_name)
    if model_name == "LocalOutlierProbability":
        return model_cls(X[:20])

    return model_cls(**MODEL_PARAMS.get(model_name, {}))


def _fitted_model(model_name, X, y):
    fix_seed(SEED)
    model = _make_model(model_name, X)
    for xi, yi in zip(X[:NUM_FIT], y[:NUM_FIT], strict=True):
        model.fit_partial(xi, yi)

    return model


def _next_scores(model, X, y):
    # Models such as RobustRandomCutForest and RandomModel draw from the global numpy generator.
    fix_seed(SEED + 1)
    scores = [model.fit_score_partial(xi, yi) for xi, yi in zip(X[NUM_FIT:], y[NUM_FIT:], strict=True)]

    return np.array([np.asarray(score, dtype=np.float64).ravel() for score in scores])


def _assert_same_scores(original, restored, X, y):
    expected = _next_scores(original, X, y)
    actual = _next_scores(restored, X, y)

    np.testing.assert_allclose(actual, expected)


def _model_params(xfail_models):
    params = []
    for model_name in pysad.models.__all__:
        marks = []
        if model_name in xfail_models:
            marks.append(pytest.mark.xfail(reason=xfail_models[model_name], strict=True))
        params.append(pytest.param(model_name, marks=marks))

    return params


@pytest.mark.parametrize("model_name", _model_params(PICKLE_XFAIL_MODELS))
def test_pickle_round_trip(model_name):
    X, y = _data(model_name)
    model = _fitted_model(model_name, X, y)

    restored = pickle.loads(pickle.dumps(model))

    assert type(restored) is type(model)
    _assert_same_scores(model, restored, X, y)


@pytest.mark.parametrize("model_name", pysad.models.__all__)
def test_deepcopy(model_name):
    X, y = _data(model_name)
    model = _fitted_model(model_name, X, y)

    copied = copy.deepcopy(model)

    _assert_same_scores(model, copied, X, y)


def test_relative_entropy_compared_scores_use_learned_state():
    # Only a value that closes a window can score nonzero. If no compared value did, a round trip
    # that lost RelativeEntropy's hypotheses would still give the same (all 0.0) scores.
    X, y = _data("RelativeEntropy")
    model = _fitted_model("RelativeEntropy", X, y)

    assert _next_scores(model, X, y).max() == 1.0


def test_rrcf_pickle_round_trip():
    from pysad.models import RobustRandomCutForest

    X, y = _data("RobustRandomCutForest")
    model = _fitted_model("RobustRandomCutForest", X, y)

    restored = pickle.loads(pickle.dumps(model))

    assert restored.index == model.index == NUM_FIT
    assert len(restored.forest) == model.num_trees
    for original_tree, restored_tree in zip(model.forest, restored.forest, strict=True):
        assert original_tree.rng is np.random
        assert restored_tree.rng is np.random
        assert sorted(restored_tree.leaves) == sorted(original_tree.leaves)

    # tree_size < NUM_FIT, so the next instances also exercise forgetting old points after restoring.
    _assert_same_scores(model, restored, X, y)

    restored_again = pickle.loads(pickle.dumps(restored))
    assert isinstance(restored_again, RobustRandomCutForest)
