import numpy as np
import pytest


@pytest.mark.parametrize("window_size", [30, 1000])
def test_scores_equal_sklearn_lof_on_the_window(window_size):
    from sklearn.neighbors import LocalOutlierFactor

    from pysad.models import IncrementalLOF

    rng = np.random.RandomState(0)
    X = rng.normal(size=(300, 3))
    queries = 2.0 * rng.normal(size=(40, 3))

    # With a window of 30, the window slides 270 times, so the deletion update runs too.
    model = IncrementalLOF(num_neighbors=7, window_size=window_size).fit(X)
    reference = LocalOutlierFactor(n_neighbors=7, novelty=True).fit(X[-window_size:])

    np.testing.assert_allclose(model.score(queries), -reference.score_samples(queries))


def test_scores_zero_until_the_window_holds_num_neighbors_plus_one():
    from pysad.models import IncrementalLOF

    X = np.random.RandomState(0).normal(size=(10, 2))
    scores = IncrementalLOF(num_neighbors=5, window_size=20).fit_score(X)

    np.testing.assert_array_equal(scores[:6], 0.0)
    assert (scores[6:] > 0.0).all()


def test_duplicates_get_finite_scores():
    from pysad.models import IncrementalLOF

    X = np.ones((50, 2))
    scores = IncrementalLOF(num_neighbors=5, window_size=20).fit_score(X)

    np.testing.assert_allclose(scores[6:], 1.0)


def test_outlier_scores_higher():
    from pysad.models import IncrementalLOF

    X = np.random.RandomState(0).normal(size=(200, 2))
    model = IncrementalLOF(window_size=100).fit(X)

    assert model.score_partial(np.array([6.0, 6.0])) > model.score_partial(np.zeros(2))


def test_rejects_a_window_too_small_for_the_neighbors():
    from pysad.models import IncrementalLOF

    with pytest.raises(ValueError, match="window_size"):
        IncrementalLOF(num_neighbors=10, window_size=11)
