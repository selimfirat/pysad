"""Edge-case tests for the StreamingRHF detector.

StreamingRHF builds its forest on the first ``window_size`` instances. Until then every score is
``0.0``. Constant and zero-variance streams must score ``0.0`` without floating-point warnings or
errors across forest rebuild boundaries (:cite:`nesic2022streamrhf`).
"""

import numpy as np
import pytest

WINDOW = 6
NUM_TREES = 5
HEIGHT = 3


def _make_model(random_state=0):
    from pysad.models import StreamingRHF

    return StreamingRHF(
        window_size=WINDOW,
        num_trees=NUM_TREES,
        height=HEIGHT,
        random_state=random_state,
    )


def _random_stream(num_instances, num_features=2, seed=0):
    rng = np.random.RandomState(seed)
    return rng.rand(num_instances, num_features)


def _constant_stream(num_instances, num_features=2):
    return np.full((num_instances, num_features), 0.5)


def _zero_variance_stream(num_instances, num_features=3):
    row = (np.arange(1, num_features + 1) * 0.25).astype(np.float64)
    return np.tile(row, (num_instances, 1))


def test_score_partial_is_zero_before_the_forest_is_built():
    model = _make_model(random_state=1)
    xi = np.full(2, 0.5)
    assert model.score_partial(xi) == 0.0
    for _ in range(WINDOW - 1):
        assert model.fit_score_partial(xi) == 0.0
    assert model.score_partial(xi) == 0.0


@pytest.mark.parametrize("num_features", [1, 2, 5])
def test_scores_across_rebuilds_are_finite_without_fp_errors(num_features):
    model = _make_model(random_state=2)
    X = _random_stream(3 * WINDOW + 4, num_features=num_features, seed=3)
    with np.errstate(all="raise"):
        scores = model.fit_score(X)

    assert scores.shape == (len(X),)
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0.0)
    assert float(scores[WINDOW:].max()) > 0.0


@pytest.mark.parametrize(
    "stream_factory", [_constant_stream, _zero_variance_stream], ids=["constant", "zero_variance"]
)
def test_constant_like_stream_always_scores_zero(stream_factory):
    model = _make_model(random_state=4)
    X = stream_factory(3 * WINDOW, num_features=3)
    with np.errstate(all="raise"):
        scores = model.fit_score(X)

    assert np.all(scores == 0.0)
