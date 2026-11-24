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
    num_features_for,
    short_stream_params,
)

NUM_FEATURES = 3

NAMES = [
    name
    for name in pysad.models.__all__
    if "random_state" in inspect.signature(getattr(pysad.models, name).__init__).parameters
]
ACTIVE_NAMES = [name for name in NAMES if name in set(active_model_names())]
MODELS = {
    name: (
        lambda name=name, **kw: getattr(pysad.models, name)(
            **{**short_stream_params(name, num_features_for(name, NUM_FEATURES)), **kw}
        )
    )
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


def test_robust_random_cut_forest_score_partial_does_not_advance_rng():
    """score_partial must not advance the model's RNG (issue #977)."""
    from pysad.models import RobustRandomCutForest

    X = np.random.default_rng(42).normal(size=(700, 4))

    # Test 1: Two score_partial calls on the same point give identical scores.
    a = RobustRandomCutForest(random_state=0)
    for x in X[:400]:
        a.fit_partial(x)
    s1 = a.score_partial(X[400])
    s2 = a.score_partial(X[400])
    assert s1 == s2

    # Test 2: Extra score_partial calls on one model do not affect later scores.
    a = RobustRandomCutForest(random_state=0)
    b = RobustRandomCutForest(random_state=0)
    for x in X[:400]:
        a.fit_partial(x)
        b.fit_partial(x)
    for x in X[400:420]:
        a.score_partial(x)  # model a gets extra score calls
    for x in X[400:600]:
        a.fit_partial(x)
        b.fit_partial(x)
    sa = np.array([a.score_partial(x) for x in X[600:620]])
    sb = np.array([b.score_partial(x) for x in X[600:620]])
    np.testing.assert_array_equal(sa, sb)
