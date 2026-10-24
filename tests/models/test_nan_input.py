"""Every model rejects a NaN instance from every public entry point.

Before #402 the detectors handled NaN in different ways: four of them scored NaN
for the rest of the stream after one NaN instance, three raised three different
errors, and the rest returned a number that depended on how each one compared
NaN. ``BaseModel`` now raises a ``ValueError`` that names the model before any
of its methods run, so a rejected instance leaves the model unchanged and the
stream continues with finite scores. These tests pin that guarantee for every
model in ``pysad.models.__all__``.
"""

import numpy as np
import pytest

import pysad.models
from pysad.utils import fix_seed

NUM_INSTANCES = 12

UNIVARIATE_MODELS = {
    "KNNCAD",
    "MedianAbsoluteDeviation",
    "RelativeEntropy",
    "SeasonalESD",
    "SeasonalHybridESD",
    "StandardAbsoluteDeviation",
}


def _num_features(name):
    return 1 if name in UNIVARIATE_MODELS else 3


def _params(name, num_features):
    """The constructor arguments of `name`, sized to `num_features` and short streams."""
    ones, zeros = [1.0] * num_features, [0.0] * num_features
    params = {
        "HalfSpaceTrees": {
            "feature_mins": np.array(zeros),
            "feature_maxes": np.array(ones),
            "window_size": 10,
            "num_trees": 5,
            "max_depth": 5,
        },
        "IForestASD": {"window_size": 20, "n_estimators": 10},
        "Inqmad": {"input_shape": num_features, "dim_x": 16},
        "KitNet": {"grace_feature_mapping": 10, "grace_anomaly_detector": 10},
        "KNNCAD": {"probationary_period": 48},
        "LocalOutlierProbability": {
            "initial_X": np.random.RandomState(0).rand(12, num_features),
            "num_neighbors": 10,
        },
        "RelativeEntropy": {"min_val": 0.0, "max_val": 1.0, "window_size": 5, "num_bins": 5},
        "RobustRandomCutForest": {"num_trees": 3, "shingle_size": 2, "tree_size": 16},
        "RSHash": {
            "feature_mins": np.array(zeros),
            "feature_maxes": np.array(ones),
            "num_components": 10,
        },
        "SeasonalESD": {"period": 2, "window_size": 8, "max_anomalies": 0.25},
        "SeasonalHybridESD": {"period": 2, "window_size": 8, "max_anomalies": 0.25},
        "xStream": {"num_components": 10, "n_chains": 10, "depth": 5, "window_size": 10},
    }

    return params.get(name, {})


def _make_model(name):
    fix_seed(61)

    return getattr(pysad.models, name)(**_params(name, _num_features(name)))


def _data(name, num_features):
    fix_seed(61)
    X = np.random.rand(NUM_INSTANCES, num_features)
    y = np.zeros(NUM_INSTANCES, dtype=np.int32) if name == "PerfectModel" else None

    return X, y


def _nan_instance(num_features):
    """An in-range instance whose first feature is NaN, so only NaN can be the reason for a raise."""
    instance = np.full(num_features, 0.5)
    instance[0] = np.nan

    return instance


@pytest.mark.parametrize("name", list(pysad.models.__all__), ids=list(pysad.models.__all__))
def test_partial_methods_reject_nan(name):
    num_features = _num_features(name)

    for method in ("fit_partial", "score_partial", "fit_score_partial"):
        model = _make_model(name)
        with pytest.raises(ValueError, match=f"{name} does not accept NaN values"):
            getattr(model, method)(_nan_instance(num_features))


@pytest.mark.parametrize("name", list(pysad.models.__all__), ids=list(pysad.models.__all__))
def test_batch_methods_reject_nan(name):
    num_features = _num_features(name)
    X, y = _data(name, num_features)
    X_nan_last = np.vstack([X, _nan_instance(num_features)])
    y_nan_last = None if y is None else np.append(y, np.int32(1))

    for method, args in (
        ("fit", (X_nan_last, y_nan_last)),
        ("score", (X_nan_last,)),
        ("fit_score", (X_nan_last, y_nan_last)),
    ):
        model = _make_model(name)
        # warm up on the clean part, so a model that needs state to score raises because of the NaN
        model.fit(X, y)
        with pytest.raises(ValueError, match=f"{name} does not accept NaN values"):
            getattr(model, method)(*args)


@pytest.mark.parametrize("name", list(pysad.models.__all__), ids=list(pysad.models.__all__))
def test_stream_continues_after_a_rejected_nan(name):
    """The issue's poisoning case: one NaN no longer silences the scores that follow."""
    num_features = _num_features(name)
    X, y = _data(name, num_features)
    model = _make_model(name)

    with pytest.raises(ValueError, match=f"{name} does not accept NaN values"):
        model.fit_score_partial(_nan_instance(num_features))

    scores = model.fit_score(X, y)

    assert scores.shape == (NUM_INSTANCES,)
    assert np.all(np.isfinite(scores))
