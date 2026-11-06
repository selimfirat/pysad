import numpy as np
import pytest


def test_scores_equal_incremental_lof_until_the_window_is_full():
    from pysad.models import DILOF, IncrementalLOF

    X = np.random.RandomState(0).normal(size=(99, 3))

    scores = DILOF(num_neighbors=7, window_size=100, threshold=None).fit_score(X)
    reference = IncrementalLOF(num_neighbors=7, window_size=100).fit_score(X)

    np.testing.assert_allclose(scores, reference)


def test_summarization_keeps_a_quarter_of_the_oldest_half_and_the_newest_half():
    from pysad.models import DILOF

    X = np.random.RandomState(0).normal(size=(40, 2))
    model = DILOF(num_neighbors=3, window_size=40, threshold=None).fit(X)

    assert model.num_instances_ == 30
    np.testing.assert_array_equal(model.X_[10:30], X[20:])
    # The summary is a subset of the oldest half, in arrival order.
    summary = [np.flatnonzero((X[:20] == x).all(axis=1))[0] for x in model.X_[:10]]
    assert summary == sorted(set(summary))


def test_window_stays_bounded():
    from pysad.models import DILOF

    X = np.random.RandomState(0).normal(size=(500, 2))
    model = DILOF(num_neighbors=5, window_size=40)

    for x in X:
        model.fit_partial(x)
        assert model.num_instances_ < 40


def test_skipping_keeps_a_sequence_of_outliers_out_of_the_window():
    from pysad.models import DILOF

    rng = np.random.RandomState(0)
    X = np.vstack([rng.normal(size=(100, 2)), 8.0 + 0.01 * rng.normal(size=(30, 2))])

    skipping = DILOF(num_neighbors=5, window_size=200).fit_score(X)
    no_skipping = DILOF(num_neighbors=5, window_size=200, threshold=None).fit_score(X)

    # Without skipping, the outliers form a dense cluster and their LOF falls to about 1.
    assert (skipping[-10:] > 5.0).all()
    assert (no_skipping[-10:] < 2.0).all()


def test_outlier_scores_higher():
    from pysad.models import DILOF

    X = np.random.RandomState(0).normal(size=(500, 2))
    model = DILOF(window_size=100).fit(X)

    assert model.score_partial(np.array([6.0, 6.0])) > model.score_partial(np.zeros(2))


def test_rejects_a_window_too_small_for_the_summary():
    from pysad.models import DILOF

    with pytest.raises(ValueError, match="window_size"):
        DILOF(num_neighbors=10, window_size=21)


def test_an_instance_that_skipping_skips_scores_at_least_the_last_outlier():
    from pysad.models import DILOF, IncrementalLOF

    rng = np.random.RandomState(0)
    X = np.vstack([rng.normal(size=(100, 2)), [[6.0, 6.0]]])
    model = DILOF(num_neighbors=5, window_size=200).fit(X)
    reference = IncrementalLOF(num_neighbors=5, window_size=200).fit(X)
    duplicate = np.array([6.0, 6.0])

    # Against the window, which holds the outlier, the LOF of its duplicate is lower than its own.
    assert model.score_partial(duplicate) > reference.score_partial(duplicate)
    assert model.score_partial(np.zeros(2)) == reference.score_partial(np.zeros(2))
