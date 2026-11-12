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


def test_skipping_follows_the_last_skipped_instance_and_resumes_learning():
    from pysad.models import DILOF, IncrementalLOF

    X = np.arange(8.0)[:, None]
    model = DILOF(num_neighbors=2, window_size=40).fit(X)
    reference = IncrementalLOF(num_neighbors=2, window_size=40).fit(X)
    outlier = np.array([20.0])
    model.fit_partial(outlier)
    reference.fit_partial(outlier)

    # Each step is below the mean nearest-neighbor distance (21/9), but the
    # sequence travels farther than that from the inserted outlier.
    for value in (20.6, 21.2, 21.8, 22.4, 23.0):
        instance = np.array([value])
        assert model.score_partial(instance) > 1.5
        model.fit_partial(instance)

    probes = np.array([[0.5], [3.5], [10.0], [30.0]])
    np.testing.assert_allclose(model.score(probes), reference.score(probes))

    # An instance outside the skipping radius ends the sequence and is learned.
    normal = np.array([3.0])
    model.fit_partial(normal)
    reference.fit_partial(normal)
    model.fit_partial(np.array([20.6]))
    reference.fit_partial(np.array([20.6]))
    np.testing.assert_allclose(model.score(probes), reference.score(probes))


@pytest.mark.parametrize("window_size", [24, 25])
def test_scores_match_static_lof_after_repeated_summaries(window_size):
    from sklearn.neighbors import LocalOutlierFactor

    from pysad.models import DILOF

    rng = np.random.RandomState(12)
    X = rng.normal(size=(200, 3))
    probes = rng.normal(size=(10, 3))
    model = DILOF(num_neighbors=3, window_size=window_size, threshold=None)

    for instance in X:
        model.fit_partial(instance)
        if model.num_instances_ == window_size - window_size // 2 + window_size // 4:
            # Compare the public scores to an independent LOF implementation fitted
            # on the retained observations, including after each later summarization.
            reference = LocalOutlierFactor(n_neighbors=3, novelty=True).fit(
                model.X_[: model.num_instances_]
            )
            np.testing.assert_allclose(model.score(probes), -reference.score_samples(probes))


def test_duplicate_instances_stay_finite_through_repeated_summaries():
    from pysad.models import DILOF

    model = DILOF(num_neighbors=2, window_size=12, threshold=None)
    X = np.vstack([np.zeros((30, 2)), np.ones((30, 2)), np.zeros((30, 2))])

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        scores = model.fit_score(X)
        probes = model.score(np.array([[0.0, 0.0], [0.5, 0.5], [8.0, 8.0]]))

    assert np.isfinite(scores).all()
    assert np.isfinite(probes).all()
    assert probes[2] > probes[0]


@pytest.mark.parametrize("scale", [1e-12, 1e-6, 1e6])
def test_scores_and_skipping_do_not_depend_on_the_unit_of_the_values(scale):
    from pysad.models import DILOF

    rng = np.random.RandomState(0)
    X = rng.normal(size=(600, 2))
    X[[300, 301, 302, 450]] += 8.0

    # The window of 60 is summarized several times, and the outliers start the skipping scheme.
    model = DILOF(num_neighbors=5, window_size=60)
    scaled_model = DILOF(num_neighbors=5, window_size=60)
    skipped = []
    for instance in X:
        model.fit_partial(instance)
        scaled_model.fit_partial(scale * instance)
        skipped.append((model._last_outlier is None, scaled_model._last_outlier is None))

    assert any(not unscaled for unscaled, _ in skipped)
    assert all(unscaled == scaled for unscaled, scaled in skipped)
    np.testing.assert_allclose(
        scaled_model.X_[: scaled_model.num_instances_], scale * model.X_[: model.num_instances_]
    )
    probes = rng.normal(size=(20, 2))
    np.testing.assert_allclose(scaled_model.score(scale * probes), model.score(probes), rtol=1e-9)
