import copy
import inspect
import pickle

import numpy as np
import pytest

import pysad.models
from pysad.transform.postprocessing import ZScorePostprocessor
from pysad.transform.preprocessing import ShingleTransformer
from pysad.transform.probability_calibration import ConformalProbabilityCalibrator
from pysad.utils import fix_seed
from tests.models.conftest import (
    EDGE_MODELS,
    TIMESTAMPED_MODELS,
    UNIVARIATE_MODELS,
    active_model_names,
    short_stream_params,
)

SEED = 61
NUM_FIT = 60
NUM_NEXT = 20
NUM_NEW = 20

MODEL_PARAMS = {
    # Short grace periods, so that the autoencoders are built and trained within the data.
    "KitNet": {"grace_feature_mapping": 20, "grace_anomaly_detector": 50},
    "KNNCAD": {"probationary_period": 50},
    "MatrixProfile": {"window_length": 4, "history_size": 30},
    "DAMP": {"window_length": 4, "history_size": 20},
    "SpectralResidual": {"window_size": 8, "num_average_points": 3},
    # The window slides before and after the round trip.
    "STARE": {"window_size": 30, "slide_size": 5, "num_neighbors": 3, "cell_size": 0.6},
    # The window slides before and after the round trip.
    "ThreshLEAP": {"window_size": 30, "max_radius": 0.6, "num_neighbors": 3},
    # Micro-clusters form, expire and dissolve within the first instances.
    "MCOD": {"window_size": 30, "max_radius": 0.6, "num_neighbors": 2},
    "MemStream": {"window_size": 8, "num_epochs": 20},
    # Ticks start and end before and after the round trip.
    "MIDAS": {"tick_size": 7},
    # Cells empty out and fill again within the first instances.
    "NETS": {"window_size": 30, "max_radius": 0.6},
    # Output neurons merge and replace the oldest one before and after the round trip.
    "OeSNNUAD": {"window_size": 20, "num_output_neurons": 4},
    "OnlineIsolationForest": {"num_trees": 10, "max_leaf_samples": 8, "window_size": 32},
    # A window that wraps before and after the round trip.
    "OnlineOneClassSVM": {"window_size": 32},
    "PCBIsolationForest": {
        "num_trees": 10,
        "window_size": 32,
        "stat_size": 10,
    },
    # Non-overlapping windows of 5 close 4 times among the NUM_NEXT compared values, and 80 buckets
    # split the noise so that some of them score 1.0 (see test_relative_entropy_compared_scores_use_learned_state).
    "RelativeEntropy": {
        "min_val": 0.0,
        "max_val": 1.0,
        "window_size": 5,
        "num_bins": 80,
        "step": 5,
    },
}

PICKLE_XFAIL_MODELS = {}


def _data(model_name):
    rng = np.random.RandomState(SEED)
    num_instances = NUM_FIT + NUM_NEXT
    if model_name in UNIVARIATE_MODELS:
        t = np.arange(num_instances)
        X = (0.5 + 0.4 * np.sin(2 * np.pi * t / 4) + 0.05 * rng.rand(num_instances)).reshape(-1, 1)
    elif model_name in TIMESTAMPED_MODELS:
        t = np.arange(num_instances)
        values = 0.5 + 0.4 * np.sin(2 * np.pi * t / 4) + 0.05 * rng.rand(num_instances)
        X = np.column_stack([3600.0 * t, values])
    elif model_name in EDGE_MODELS:
        X = rng.randint(0, 5, size=(num_instances, 2)).astype(np.float64)
    else:
        X = rng.rand(num_instances, 3)
    y = (rng.rand(num_instances) > 0.9).astype(int)

    return X, y


_RANDOM_STATE_SUPPORT: dict[type, bool] = {}


def _model_accepts_random_state(model_cls: type) -> bool:
    accepted = _RANDOM_STATE_SUPPORT.get(model_cls)
    if accepted is None:
        accepted = "random_state" in inspect.signature(model_cls.__init__).parameters
        _RANDOM_STATE_SUPPORT[model_cls] = accepted
    return accepted


def _make_model(model_name, X, random_state=None):
    model_cls = getattr(pysad.models, model_name)
    if model_name == "LocalOutlierProbability":
        return model_cls(X[:20])
    if model_name == "StreamPipeline":
        return model_cls(
            model=pysad.models.LODA(),
            preprocessors=[ShingleTransformer(shingle_size=2)],
            postprocessors=[
                ZScorePostprocessor(window_size=20),
                ConformalProbabilityCalibrator(window_size=20),
            ],
        )

    params = dict(MODEL_PARAMS.get(model_name) or short_stream_params(model_name, X.shape[1]))
    if random_state is not None and _model_accepts_random_state(model_cls):
        params["random_state"] = random_state
    return model_cls(**params)


def _fitted_model(model_name, X, y):
    fix_seed(SEED)
    model = _make_model(model_name, X)
    for xi, yi in zip(X[:NUM_FIT], y[:NUM_FIT], strict=True):
        model.fit_partial(xi, yi)

    return model


def _next_scores(model, X, y):
    # Models such as RobustRandomCutForest and RandomModel draw from the global numpy generator.
    fix_seed(SEED + 1)
    scores = [
        model.fit_score_partial(xi, yi) for xi, yi in zip(X[NUM_FIT:], y[NUM_FIT:], strict=True)
    ]

    return np.array([np.asarray(score, dtype=np.float64).ravel() for score in scores])


def _assert_same_scores(original, restored, X, y):
    expected = _next_scores(original, X, y)
    actual = _next_scores(restored, X, y)

    np.testing.assert_allclose(actual, expected)


def _new_data(model_name):
    """Fresh unseen stream used to test continued learning after a round trip."""
    rng = np.random.RandomState(SEED + 777)
    num_instances = NUM_NEW
    if model_name in UNIVARIATE_MODELS:
        t = np.arange(num_instances)
        X = (0.5 + 0.4 * np.sin(2 * np.pi * t / 4) + 0.05 * rng.rand(num_instances)).reshape(-1, 1)
    elif model_name in TIMESTAMPED_MODELS:
        t = np.arange(num_instances)
        values = 0.5 + 0.4 * np.sin(2 * np.pi * t / 4) + 0.05 * rng.rand(num_instances)
        X = np.column_stack([3600.0 * t, values])
    elif model_name in EDGE_MODELS:
        X = rng.randint(0, 5, size=(num_instances, 2)).astype(np.float64)
    else:
        X = rng.rand(num_instances, 3)
    y = (rng.rand(num_instances) > 0.9).astype(int)

    return X, y


def _fit_scores(model, X, y, seed):
    # fit_score_partial learns from each instance and returns its score; reseeding makes the draw
    # sequence identical for the original and the restored model.
    fix_seed(seed)
    scores = [model.fit_score_partial(xi, yi) for xi, yi in zip(X, y, strict=True)]

    return np.array([np.asarray(score, dtype=np.float64).ravel() for score in scores])


def _model_params(xfail_models):
    params = []
    for model_name in active_model_names():
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


@pytest.mark.parametrize("model_name", active_model_names())
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


# Both tests below give each model a fixed random_state, so the model owns its RNG and pickling
# copies an identical RNG snapshot alongside the learned state. The restored model reproduces the
# original on every platform, and no model needs to be xfailed here.
UNFITTED_XFAIL_MODELS = {}

SCORE_ONLY_XFAIL_MODELS = {}


def _edge_model_params(xfail_models):
    params = []
    for model_name in active_model_names():
        marks = []
        if model_name in xfail_models:
            marks.append(pytest.mark.xfail(reason=xfail_models[model_name], strict=True))
        params.append(pytest.param(model_name, marks=marks))

    return params


@pytest.mark.parametrize("model_name", _edge_model_params(UNFITTED_XFAIL_MODELS))
def test_pickle_unfitted_model(model_name):
    # Pickling a model before any fit_partial call must preserve its initial state.
    X, y = _data(model_name)
    original = _make_model(model_name, X, random_state=SEED)
    restored = pickle.loads(pickle.dumps(original))

    assert type(restored) is type(original)
    np.testing.assert_allclose(
        _fit_scores(restored, X, y, SEED + 2),
        _fit_scores(original, X, y, SEED + 2),
    )


@pytest.mark.parametrize("model_name", _edge_model_params(SCORE_ONLY_XFAIL_MODELS))
def test_pickle_after_score_only(model_name):
    # score_partial does not learn; pickling after scoring only must still round trip.
    X, y = _data(model_name)
    original = _make_model(model_name, X, random_state=SEED)
    for xi in X[:NUM_FIT]:
        original.score_partial(xi)

    restored = pickle.loads(pickle.dumps(original))

    assert type(restored) is type(original)
    np.testing.assert_allclose(
        _fit_scores(restored, X[NUM_FIT:], y[NUM_FIT:], SEED + 2),
        _fit_scores(original, X[NUM_FIT:], y[NUM_FIT:], SEED + 2),
    )


@pytest.mark.parametrize("model_name", _edge_model_params({}))
def test_pickle_mid_stream_then_learn_new_data(model_name):
    # After a mid-stream round trip, the restored model must learn new instances identically.
    X, y = _data(model_name)
    X_new, y_new = _new_data(model_name)
    original = _fitted_model(model_name, X, y)
    restored = pickle.loads(pickle.dumps(original))

    assert type(restored) is type(original)
    np.testing.assert_allclose(
        _fit_scores(restored, X_new, y_new, SEED + 2),
        _fit_scores(original, X_new, y_new, SEED + 2),
    )


@pytest.mark.parametrize("model_name", _edge_model_params({}))
def test_pickle_at_end_of_stream_then_learn_new_data(model_name):
    # A round trip at the end of the stream must preserve state for continued learning.
    X, y = _data(model_name)
    X_new, y_new = _new_data(model_name)
    original = _make_model(model_name, X)
    fix_seed(SEED)
    for xi, yi in zip(X, y, strict=True):
        original.fit_partial(xi, yi)

    restored = pickle.loads(pickle.dumps(original))

    assert type(restored) is type(original)
    np.testing.assert_allclose(
        _fit_scores(restored, X_new, y_new, SEED + 2),
        _fit_scores(original, X_new, y_new, SEED + 2),
    )
