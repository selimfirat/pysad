import os
import pytest


@pytest.fixture
def test_path(): # adapted from https://github.com/scikit-multiflow/scikit-multiflow/blob/master/tests/anomaly_detection/test_half_space_trees.py
    return os.path.dirname(os.path.abspath(__file__))
