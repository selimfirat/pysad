import pickle

import numpy as np
import pytest

from pysad.models import (
    LODA,
    HalfSpaceTrees,
    Inqmad,
    KitNet,
    RandomModel,
    RobustRandomCutForest,
    RSHash,
    xStream,
)
from pysad.utils import fix_seed

NUM_FEATURES = 3

MODELS = {
    "HalfSpaceTrees": lambda **kw: HalfSpaceTrees(
        [0.0] * NUM_FEATURES,
        [1.0] * NUM_FEATURES,
        window_size=20,
        num_trees=5,
        max_depth=8,
        **kw,
    ),
    "Inqmad": lambda **kw: Inqmad(input_shape=NUM_FEATURES, dim_x=32, gamma=100, **kw),
    # Short grace periods, so that the autoencoders are built and trained within the data.
    "KitNet": lambda **kw: KitNet(grace_feature_mapping=20, grace_anomaly_detector=30, **kw),
    "LODA": lambda **kw: LODA(num_random_cuts=20, **kw),
    "RandomModel": lambda **kw: RandomModel(**kw),
    "RobustRandomCutForest": lambda **kw: RobustRandomCutForest(tree_size=32, **kw),
    "RSHash": lambda **kw: RSHash([0.0] * NUM_FEATURES, [1.0] * NUM_FEATURES, **kw),
    "xStream": lambda **kw: xStream(n_chains=10, depth=10, window_size=20, **kw),
}

NAMES = list(MODELS)
PICKLABLE_NAMES = [
    pytest.param(
        name,
        marks=pytest.mark.xfail(
            strict=True,
            reason="dA.W_prime is a view of dA.W that pickling turns into a copy, so training after loading diverges.",
        ),
    )
    if name == "KitNet"
    else name
    for name in MODELS
]


def _data():
    return np.random.RandomState(0).rand(80, NUM_FEATURES)


def _scores(model):
    return np.array([np.ravel(model.fit_score_partial(x))[0] for x in _data()])


@pytest.mark.parametrize("name", NAMES)
def test_same_seed_gives_same_scores(name):
    np.testing.assert_array_equal(
        _scores(MODELS[name](random_state=5)), _scores(MODELS[name](random_state=5))
    )


@pytest.mark.parametrize("name", NAMES)
def test_different_seeds_give_different_scores(name):
    assert not np.array_equal(
        _scores(MODELS[name](random_state=5)), _scores(MODELS[name](random_state=6))
    )


@pytest.mark.parametrize("name", NAMES)
def test_seed_leaves_global_state_alone(name):
    fix_seed(11)
    expected = np.random.rand(3)

    fix_seed(11)
    _scores(MODELS[name](random_state=5))

    np.testing.assert_array_equal(np.random.rand(3), expected)


@pytest.mark.parametrize("name", NAMES)
def test_models_seeded_side_by_side_do_not_interfere(name):
    alone = _scores(MODELS[name](random_state=5))

    model = MODELS[name](random_state=5)
    other = MODELS[name](random_state=6)
    side_by_side = []
    for x in _data():
        side_by_side.append(np.ravel(model.fit_score_partial(x))[0])
        other.fit_score_partial(x)

    np.testing.assert_array_equal(side_by_side, alone)


@pytest.mark.parametrize("name", NAMES)
def test_default_draws_from_global_state(name):
    # The global state after fix_seed(5) produces the same stream as RandomState(5).
    fix_seed(5)
    default = _scores(MODELS[name]())

    np.testing.assert_array_equal(default, _scores(MODELS[name](random_state=5)))
    np.testing.assert_array_equal(
        default, _scores(MODELS[name](random_state=np.random.RandomState(5)))
    )


@pytest.mark.parametrize("name", PICKLABLE_NAMES)
def test_seeded_model_pickles_with_its_generator(name):
    data = _data()
    model = MODELS[name](random_state=5)
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
    weights = [ae.W for ae in model.model.ensembleLayer]
    same_shape = [
        (a, b) for i, a in enumerate(weights) for b in weights[i + 1 :] if a.shape == b.shape
    ]
    assert same_shape
    for a, b in same_shape:
        assert not np.array_equal(a, b)
