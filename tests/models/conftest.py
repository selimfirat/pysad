import os
import sys
from typing import Any

import numpy as np
import pytest

import pysad.models
from pysad.transform.postprocessing import RunningZScorePostprocessor
from pysad.transform.preprocessing import ShingleTransformer

# Univariate models validate their input through check_univariate in their defining module.
UNIVARIATE_MODELS = frozenset(
    name
    for name in pysad.models.__all__
    if hasattr(sys.modules[getattr(pysad.models, name).__module__], "check_univariate")
)

# These models take each instance as [timestamp, value].
TIMESTAMPED_MODELS = frozenset({"HTM"})

# These models take each instance as an edge [source, destination].
EDGE_MODELS = frozenset({"MIDAS"})


def num_features_for(name: str, default: int = 3) -> int:
    """Returns the number of input features expected by model `name`."""
    if name in TIMESTAMPED_MODELS or name in EDGE_MODELS:
        return 2
    return 1 if name in UNIVARIATE_MODELS else default


def short_stream_params(name: str, num_features: int = 3) -> dict[str, Any]:
    """Returns constructor arguments for `name` sized to `num_features` and short test streams."""
    ones, zeros = [1.0] * num_features, [0.0] * num_features
    params: dict[str, dict[str, Any]] = {
        "HalfSpaceTrees": {
            "feature_mins": np.array(zeros),
            "feature_maxes": np.array(ones),
            "window_size": 10,
            "num_trees": 5,
            "max_depth": 5,
        },
        "HTM": {"min_val": 0.0, "max_val": 1.0, "num_columns": 256},
        "HoltWinters": {"period": 2},
        "IForestASD": {"window_size": 20, "n_estimators": 10},
        "Inqmad": {"input_shape": num_features, "dim_x": 16},
        "MemStream": {"window_size": 4, "num_epochs": 20},
        "KitNet": {"grace_feature_mapping": 10, "grace_anomaly_detector": 10},
        "KNNCAD": {"probationary_period": 48},
        "MatrixProfile": {"window_length": 3},
        "DAMP": {"window_length": 3, "probationary_period": 4},
        "SpectralResidual": {"window_size": 8, "num_average_points": 3},
        "LocalOutlierProbability": {
            "initial_X": np.random.RandomState(0).rand(12, num_features),
            "num_neighbors": 10,
        },
        "OnlineIsolationForest": {"num_trees": 5, "max_leaf_samples": 4, "window_size": 10},
        "RelativeEntropy": {"min_val": 0.0, "max_val": 1.0, "window_size": 5, "num_bins": 5},
        "RobustRandomCutForest": {"num_trees": 3, "shingle_size": 2, "tree_size": 16},
        "RSForest": {"window_size": 5, "num_trees": 5, "max_depth": 5},
        "RSHash": {
            "feature_mins": np.array(zeros),
            "feature_maxes": np.array(ones),
            "num_components": 10,
        },
        "SAND": {"subsequence_length": 2, "centroid_length": 4, "num_clusters": 2, "batch_size": 6},
        "SeasonalESD": {"period": 2, "window_size": 8, "max_anomalies": 0.25},
        "SeasonalHybridESD": {"period": 2, "window_size": 8, "max_anomalies": 0.25},
        "StreamPipeline": {
            "model": pysad.models.LODA(),
            "preprocessors": [ShingleTransformer(shingle_size=2)],
            "postprocessors": [RunningZScorePostprocessor(window_size=10)],
        },
        "xStream": {"num_components": 10, "n_chains": 10, "depth": 5, "window_size": 10},
    }
    return params.get(name, {})


@pytest.fixture
def test_path():  # adapted from https://github.com/scikit-multiflow/scikit-multiflow/blob/master/tests/anomaly_detection/test_half_space_trees.py
    return os.path.dirname(os.path.abspath(__file__))
