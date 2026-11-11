import os
import sys

import pytest

import pysad.models

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


@pytest.fixture
def test_path():  # adapted from https://github.com/scikit-multiflow/scikit-multiflow/blob/master/tests/anomaly_detection/test_half_space_trees.py
    return os.path.dirname(os.path.abspath(__file__))
