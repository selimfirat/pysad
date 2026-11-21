import inspect
import pickle

import numpy as np
import pytest

import pysad.models
from pysad.models import RandomModel
from pysad.utils import fix_seed
from tests.models.conftest import (
    EDGE_MODELS,
    TIMESTAMPED_MODELS,
    UNIVARIATE_MODELS,
    active_model_names,
)

NUM_FEATURES = 3

MODEL_PARAMS = {
    "ApproxStorm": {
        "window_size": 20,
        "max_radius": 0.5,
        "num_neighbors": 3,
        "sample_fraction": 0.25,
    },
    "D3": {"window_size": 20, "sample_fraction": 0.5, "max_radius": 0.3},
    "EXPoSE": {"num_random_features": 50},
    "FeedbackHST": {
        "feature_mins": [0.0] * NUM_FEATURES,
        "feature_maxes": [1.0] * NUM_FEATURES,
        "window_size": 10,
        "num_trees": 4,
        "max_depth": 4,
        "feedback_budget": 10,
    },
    "HalfSpaceTrees": {
        "feature_mins": [0.0] * NUM_FEATURES,
        "feature_maxes": [1.0] * NUM_FEATURES,
        "window_size": 10,
        "num_trees": 4,
        "max_depth": 4,
    },
    "HTM": {
        "min_val": 0.0,
        "max_val": 1.0,
        "probationary_period": 15,
        "num_columns": 64,
        "num_cells_per_column": 4,
    },
    "IDKS": {
        "input_shape": NUM_FEATURES,
        "window_size": 20,
        "num_samples": 4,
        "num_partitions": 10,
    },
    "IForestASD": {"window_size": 20, "n_estimators": 5},
    "Inqmad": {"input_shape": NUM_FEATURES, "dim_x": 16, "initial_size": 15},
    # Short grace periods, so that the autoencoders are built and trained within the data.
    "KitNet": {"grace_feature_mapping": 10, "grace_anomaly_detector": 15},
    "LODA": {"num_random_cuts": 10},
    "MIDAS": {"num_buckets": 4, "tick_size": 5},
    "MiLOF": {"num_neighbors": 5, "window_size": 20, "num_summaries": 5},
    "MemStream": {"window_size": 8, "num_epochs": 20},
    "OeSNNUAD": {"window_size": 10},
    "PCBIsolationForest": {"num_trees": 5, "window_size": 20, "stat_size": 6},
    "OnlineIsolationForest": {"num_trees": 5, "max_leaf_samples": 4, "window_size": 20},
    "RobustRandomCutForest": {"num_trees": 4, "tree_size": 16},
    "RSForest": {"window_size": 20, "num_trees": 5, "max_depth": 6},
    "RSHash": {
        "feature_mins": [0.0] * NUM_FEATURES,
        "feature_maxes": [1.0] * NUM_FEATURES,
        "num_components": 10,
    },
    "SAND": {"subsequence_length": 4, "batch_size": 24},
    "xStream": {"num_components": 10, "n_chains": 5, "depth": 5, "window_size": 10},
    "FoundationFeatureDetector": {
        "model_name": "patchtst",
        "model_size": "small",
        "context_length": 16,
        "patch_length": 4,
        "patch_stride": 2,
        "warmup_windows": 2,
        "ema_alpha": 0.1,
    },
}

NAMES = [
    name
    for name in pysad.models.__all__
    if "random_state" in inspect.signature(getattr(pysad.models, name).__init__).parameters
]
ACTIVE_NAMES = [name for name in NAMES if name in set(active_model_names())]
MODELS = {
    name: (lambda name=name, **kw: getattr(pysad.models, name)(**MODEL_PARAMS.get(name, {}), **kw))
    for name in NAMES
}


def _data(model):
    X = np.random.RandomState(0).rand(80, NUM_FEATURES)
    name = type(model).__name__
    if name in TIMESTAMPED_MODELS:
        return np.column_stack([3600.0 * np.arange(len(X)), X[:, 0]])
    if name in EDGE_MODELS:
        return np.floor(10.0 * X[:, :2])
    return X[:, :1] if name in UNIVARIATE_MODELS else X


def _scores(model):
    return np.array([np.ravel(model.fit_score_partial(x))[0] for x in _data(model)])


@pytest.mark.parametrize("name", ACTIVE_NAMES)
def test_random_state_determinism_and_global_isolation(name):
    fix_seed(11)
    expected_global = np.random.rand(3)

    fix_seed(11)
    scores_5a = _scores(MODELS[name](random_state=5))
    np.testing.assert_array_equal(np.random.rand(3), expected_global)

    scores_5b = _scores(MODELS[name](random_state=5))
    np.testing.assert_array_equal(scores_5a, scores_5b)

    # The global state after fix_seed(5) produces the same stream as RandomState(5).
    fix_seed(5)
    default = _scores(MODELS[name]())
    np.testing.assert_array_equal(default, scores_5a)
    np.testing.assert_array_equal(
        default, _scores(MODELS[name](random_state=np.random.RandomState(5)))
    )

    if name == "FRAHST":
        # Binary alarms can agree even when the learned bases differ.
        models = [MODELS[name](random_state=seed) for seed in (5, 6)]
        for model in models:
            model.fit_partial(_data(model)[0])
        assert not np.array_equal(models[0].basis_, models[1].basis_)
        return
    assert not np.array_equal(scores_5a, _scores(MODELS[name](random_state=6)))


@pytest.mark.parametrize("name", ACTIVE_NAMES)
def test_models_seeded_side_by_side_do_not_interfere(name):
    alone = _scores(MODELS[name](random_state=5))

    model = MODELS[name](random_state=5)
    other = MODELS[name](random_state=6)
    side_by_side = []
    for x in _data(model):
        side_by_side.append(np.ravel(model.fit_score_partial(x))[0])
        other.fit_score_partial(x)

    np.testing.assert_array_equal(side_by_side, alone)


@pytest.mark.parametrize("name", ACTIVE_NAMES)
def test_seeded_model_pickles_with_its_generator(name):
    model = MODELS[name](random_state=5)
    data = _data(model)
    for x in data[:40]:
        model.fit_score_partial(x)

    restored = pickle.loads(pickle.dumps(model))
    for x in data[40:]:
        np.testing.assert_array_equal(model.fit_score_partial(x), restored.fit_score_partial(x))


def test_random_model_default_keeps_following_global_state_after_pickling():
    model = pickle.loads(pickle.dumps(RandomModel()))

    fix_seed(3)
    score = model.score_partial(None)
    fix_seed(3)

    assert score == np.random.uniform()


def test_kitnet_autoencoders_share_one_generator():
    model = MODELS["KitNet"](random_state=5, max_size_ae=2)
    for x in np.random.RandomState(0).rand(25, 8):
        model.fit_partial(x)

    # Each autoencoder used to seed its own RandomState(1234), so same-sized ones started out identical.
    weights = [ae.W for ae in model.model.ensemble_layer]
    same_shape = [
        (a, b) for i, a in enumerate(weights) for b in weights[i + 1 :] if a.shape == b.shape
    ]
    assert same_shape
    for a, b in same_shape:
        assert not np.array_equal(a, b)
