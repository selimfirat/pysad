import copy
import pickle

import numpy as np
import pytest

import pysad.models
from pysad.transform.postprocessing import RunningZScorePostprocessor
from pysad.transform.preprocessing import ShingleTransformer
from pysad.transform.probability_calibration import ConformalProbabilityCalibrator
from pysad.utils import fix_seed

SEED = 61
NUM_FIT = 60
NUM_NEXT = 20

UNIVARIATE_MODELS = {
    "ChangeFinder",
    "DAMP",
    "HoltWinters",
    "KNNCAD",
    "MatrixProfile",
    "MedianAbsoluteDeviation",
    "PEWMA",
    "RelativeEntropy",
    "SAND",
    "SeasonalESD",
    "SeasonalHybridESD",
    "SpectralResidual",
    "StandardAbsoluteDeviation",
}

# These models take each instance as [timestamp, value].
TIMESTAMPED_MODELS = {"HTM"}

# These models take each instance as an edge [source, destination].
EDGE_MODELS = {"MIDAS"}

MODEL_PARAMS = {
    # The tree splits, merges and discards entries before and after the round trip.
    "AnyOut": {"max_height": 2, "decay": 0.1, "snapshot_interval": 5},
    # Safe inliers form, are removed from the sample and expire within the first instances.
    "ApproxStorm": {
        "window_size": 30,
        "max_radius": 0.6,
        "num_neighbors": 3,
        "sample_fraction": 0.2,
    },
    # Core points form and expire within the first instances.
    "CPOD": {"window_size": 30, "max_radius": 0.6},
    # Sample values expire and their chains move on before and after the round trip.
    "D3": {"window_size": 30, "sample_fraction": 0.5, "max_radius": 0.6},
    "HalfSpaceTrees": {
        "feature_mins": [0.0, 0.0, 0.0],
        "feature_maxes": [1.0, 1.0, 1.0],
        "window_size": 20,
        "num_trees": 5,
        "max_depth": 8,
    },
    # The anomaly likelihood leaves its probationary period before the round trip.
    "HTM": {"min_val": 0.0, "max_val": 1.0, "probationary_period": 40, "num_columns": 256},
    "IForestASD": {"window_size": 32},
    # Violations count toward failures before and after the round trip.
    "HoltWinters": {"period": 4, "threshold": 1},
    # The window is summarized before and after the round trip.
    "DILOF": {"num_neighbors": 5, "window_size": 24},
    # A window that slides before and after the round trip.
    "IncrementalLOF": {"num_neighbors": 5, "window_size": 20},
    # The initial sample completes before the round trip, so the restored model has rho and tau.
    "Inqmad": {"input_shape": 3, "dim_x": 32, "initial_size": 30},
    # Clusters form and dissolve as the window slides before and after the round trip.
    "KELOS": {"window_size": 30, "radius": 0.6, "num_neighbors": 3},
    # Its autoencoders still train on the instances scored after the round trip.
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
    "OnlineIsolationForest": {"num_trees": 10, "max_leaf_samples": 8, "window_size": 32},
    # A window that wraps before and after the round trip.
    "OnlineOneClassSVM": {"window_size": 32},
    # Non-overlapping windows of 5 close 4 times among the NUM_NEXT compared values, and 80 buckets
    # split the noise so that some of them score 1.0 (see test_relative_entropy_compared_scores_use_learned_state).
    "RelativeEntropy": {
        "min_val": 0.0,
        "max_val": 1.0,
        "window_size": 5,
        "num_bins": 80,
        "step": 5,
    },
    "RobustRandomCutForest": {"tree_size": 32},
    "RSForest": {"window_size": 20, "num_trees": 5, "max_depth": 8},
    "RSHash": {"feature_mins": [0.0, 0.0, 0.0], "feature_maxes": [1.0, 1.0, 1.0]},
    # Batches complete before and after the round trip.
    "SAND": {"subsequence_length": 4, "batch_size": 25},
    "SeasonalESD": {"period": 4, "window_size": 16, "max_anomalies": 0.19},
    "SeasonalHybridESD": {"period": 4, "window_size": 16, "max_anomalies": 0.19},
    "xStream": {"n_chains": 10, "depth": 10, "window_size": 20},
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


def _make_model(model_name, X):
    model_cls = getattr(pysad.models, model_name)
    if model_name == "LocalOutlierProbability":
        return model_cls(X[:20])
    if model_name == "StreamPipeline":
        return model_cls(
            model=pysad.models.LODA(),
            preprocessors=[ShingleTransformer(shingle_size=2)],
            postprocessors=[
                RunningZScorePostprocessor(window_size=20),
                ConformalProbabilityCalibrator(window_size=20),
            ],
        )

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
    scores = [
        model.fit_score_partial(xi, yi) for xi, yi in zip(X[NUM_FIT:], y[NUM_FIT:], strict=True)
    ]

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
        assert sorted(restored_tree.leaves) == sorted(original_tree.leaves)

    # tree_size < NUM_FIT, so the next instances also exercise forgetting old points after restoring.
    _assert_same_scores(model, restored, X, y)

    restored_again = pickle.loads(pickle.dumps(restored))
    assert isinstance(restored_again, RobustRandomCutForest)
