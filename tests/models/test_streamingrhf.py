"""Edge-case tests for the StreamingRHF detector.

StreamingRHF builds its forest on the first ``window_size`` instances. Until then every score is
``0.0``. After the build, instances that route to small leaves score higher, and every
``window_size`` insertions the forest is rebuilt from the most recent window so old points are
forgotten. These tests pin those guarantees for constant, zero-variance, single-instance and long
streams that cross the rebuild boundary. They follow :cite:`nesic2022streamrhf` (Nesic et al.,
AICCSA 2022).
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
def test_scores_after_build_are_finite_and_non_negative(num_features):
    model = _make_model(random_state=2)
    X = _random_stream(WINDOW + 10, num_features=num_features, seed=3)
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


def test_score_partial_does_not_change_model_state():
    model = _make_model(random_state=5)
    X = _random_stream(20, seed=6)
    for xi in X:
        model.fit_partial(xi)

    n_active = model.n_active_
    point_count = model.point_count_
    xi = np.full(2, 0.5)
    first = model.score_partial(xi)
    second = model.score_partial(xi)

    assert type(first) is float
    assert first == second
    assert np.isfinite(first)
    assert model.n_active_ == n_active
    assert model.point_count_ == point_count


def test_forest_rebuilds_after_two_windows_and_keeps_scoring():
    model = _make_model(random_state=7)
    X = _random_stream(2 * WINDOW + 5, seed=8)
    with np.errstate(all="raise"):
        scores = model.fit_score(X)

    assert scores.shape == (2 * WINDOW + 5,)
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0.0)
    # The rebuild happened at 2 * window_size insertions and the count has grown since then.
    assert WINDOW < model.n_active_ < 2 * WINDOW
    assert model.n_active_ == WINDOW + 5
    assert model.point_count_ == 2 * WINDOW + 5


def test_same_random_state_gives_same_scores():
    X = _random_stream(20, seed=9)
    expected = None
    for _ in range(2):
        model = _make_model(random_state=10)
        with np.errstate(all="raise"):
            scores = model.fit_score(X.copy())
        if expected is None:
            expected = scores
        else:
            np.testing.assert_array_equal(scores, expected)


@pytest.mark.parametrize(
    "method", ["predict_partial", "fit_predict_partial", "predict", "fit_predict"]
)
def test_predict_methods_raise_type_error(method):
    model = _make_model(random_state=11)
    for _ in range(WINDOW):
        model.fit_partial(np.full(2, 0.5))

    with pytest.raises(TypeError):
        getattr(model, method)(np.zeros(2))


def test_batch_and_partial_scores_agree_after_build():
    model_a = _make_model(random_state=12)
    model_b = _make_model(random_state=12)
    X = _random_stream(2 * WINDOW + 4, seed=13)
    with np.errstate(all="raise"):
        batch_scores = model_a.fit_score(X)
        partial_scores = np.array([model_b.fit_score_partial(xi) for xi in X], dtype=np.float64)

    np.testing.assert_array_equal(batch_scores, partial_scores)
