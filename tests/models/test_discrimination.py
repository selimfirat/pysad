"""A ranking check for every model in ``pysad.models.__all__``.

Each model scores a seeded stream in which 5% of the instances after a warmup are anomalies. A model
passes when its scores after the warmup rank the anomalies better than chance: its AUROC beats the
AUROC of ``RandomModel`` on the same stream, and its AUPR beats the anomaly rate, which is the AUPR
of a score with no information. A model with inverted scores fails the check.
"""

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

import pysad.models
from pysad.utils import fix_seed

SEED = 0
NUM_INSTANCES = 600
NUM_FEATURES = 3
# The anomalies start after the warmup, and only the scores after it count.
WARMUP = 200
ANOMALY_RATE = 0.05

UNIVARIATE_MODELS = {
    "KNNCAD",
    "MedianAbsoluteDeviation",
    "RelativeEntropy",
    "SeasonalESD",
    "SeasonalHybridESD",
    "StandardAbsoluteDeviation",
}

# NullModel and PerfectModel do not score the data, and RandomModel is the chance reference.
REFERENCE_MODELS = {"NullModel", "PerfectModel", "RandomModel"}

# Open bugs that the check finds, by model. Each is a strict xfail, so the test starts failing once
# the bug is fixed, as a reminder to remove it here.
KNOWN_FAILURES = {}


def _anomalies(rng):
    """The indices of the anomalies, all after the warmup, and the labels of the stream."""
    num_anomalies = int(ANOMALY_RATE * (NUM_INSTANCES - WARMUP))
    anomalies = rng.choice(np.arange(WARMUP, NUM_INSTANCES), size=num_anomalies, replace=False)
    y = np.zeros(NUM_INSTANCES, dtype=np.int64)
    y[anomalies] = 1
    return anomalies, y


def _multivariate_stream():
    """Standard normal instances, with the anomalies moved 6 units in a random direction."""
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(NUM_INSTANCES, NUM_FEATURES))
    anomalies, y = _anomalies(rng)
    directions = rng.normal(size=(len(anomalies), NUM_FEATURES))
    X[anomalies] += 6.0 * directions / np.linalg.norm(directions, axis=1, keepdims=True)
    return X, y


def _univariate_stream():
    """A seasonal series of period 24, with the anomalies as spikes of 3 in either direction."""
    rng = np.random.default_rng(SEED)
    X = np.sin(2 * np.pi * np.arange(NUM_INSTANCES) / 24) + rng.normal(0.0, 0.1, NUM_INSTANCES)
    anomalies, y = _anomalies(rng)
    X[anomalies] += rng.choice([-3.0, 3.0], size=len(anomalies))
    return X.reshape(-1, 1), y


def _model_params(model_name, X):
    """The constructor arguments of `model_name` for the stream `X`."""
    feature_mins, feature_maxes = X.min(axis=0), X.max(axis=0)
    params = {
        "HalfSpaceTrees": {"feature_mins": feature_mins, "feature_maxes": feature_maxes},
        "IForestASD": {"window_size": WARMUP},
        "Inqmad": {"input_shape": NUM_FEATURES, "dim_x": 32},
        # Grace periods that end with the warmup.
        "KitNet": {"grace_feature_mapping": 50, "grace_anomaly_detector": 150},
        "KNNCAD": {"probationary_period": 100},
        "LocalOutlierProbability": {"initial_X": X[:100]},
        "RelativeEntropy": {"min_val": feature_mins[0], "max_val": feature_maxes[0]},
        "RobustRandomCutForest": {"num_trees": 10, "tree_size": 64},
        "RSHash": {"feature_mins": feature_mins, "feature_maxes": feature_maxes},
        # The spikes go in both directions.
        "SeasonalESD": {"period": 24, "window_size": 96, "direction": "both"},
        "SeasonalHybridESD": {"period": 24, "window_size": 96, "direction": "both"},
    }

    return params.get(model_name, {})


def _scores(model_name, X):
    fix_seed(SEED)
    model = getattr(pysad.models, model_name)(**_model_params(model_name, X))
    # Models such as RobustRandomCutForest and RandomModel draw from the global numpy generator.
    fix_seed(SEED + 1)
    return model.fit_score(X)[WARMUP:]


def _models():
    params = []
    for name in pysad.models.__all__:
        if name in REFERENCE_MODELS:
            continue
        marks = []
        if name in KNOWN_FAILURES:
            marks.append(pytest.mark.xfail(reason=KNOWN_FAILURES[name], strict=True))
        params.append(pytest.param(name, marks=marks))

    return params


@pytest.mark.parametrize("model_name", _models())
def test_scores_rank_anomalies_above_chance(model_name):
    X, y = _univariate_stream() if model_name in UNIVARIATE_MODELS else _multivariate_stream()
    scores = _scores(model_name, X)
    random_scores = _scores("RandomModel", X)
    y = y[WARMUP:]

    assert roc_auc_score(y, scores) > roc_auc_score(y, random_scores)
    assert average_precision_score(y, scores) > y.mean()
