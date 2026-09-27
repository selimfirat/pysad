import os

import numpy as np
import pytest

from pysad.utils import fix_seed

# Scores produced by the original per-chain, per-depth loop implementation of xStream. The
# vectorized implementation must reproduce them exactly for the same seeds.
REFERENCE_SCORES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "xstream_reference_scores.npz")


def _uniform_one_feature(model_cls):
    fix_seed(61)
    X = np.random.rand(150, 1)

    return model_cls().fit_score(X)


def _five_features_small_window(model_cls):
    fix_seed(61)
    X = np.random.randn(300, 5)

    # 300 / 20 = 15 swaps of the reference and current windows.
    return model_cls(window_size=20).fit_score(X)


def _three_features_separate_fit_and_score(model_cls):
    fix_seed(61)
    X = np.random.randn(200, 3)
    X[::25] *= 10.0

    model = model_cls(num_components=20, n_chains=30, depth=10, window_size=50)
    scores_before_fit = model.score(X[:10])
    scores_after_fit = model.fit(X).score(X)

    return np.concatenate([scores_before_fit, scores_after_fit])


SCENARIOS = {
    "uniform_one_feature": _uniform_one_feature,
    "five_features_small_window": _five_features_small_window,
    "three_features_separate_fit_and_score": _three_features_separate_fit_and_score,
}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_xstream_scores_match_reference(name):
    from pysad.models import xStream

    with np.load(REFERENCE_SCORES_PATH) as reference:
        expected = reference[name]

    scores = SCENARIOS[name](xStream)

    assert np.array_equal(scores, expected)
