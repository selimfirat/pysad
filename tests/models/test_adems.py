import numpy as np
import pytest


def test_scores_equal_the_residual_on_the_exact_singular_vectors():
    from pysad.models import ADeMS

    rng = np.random.RandomState(0)
    X = rng.normal(size=(200, 2)) @ rng.normal(size=(2, 6))  # rank 2
    queries = rng.normal(size=(20, 6))

    # A sketch with more columns than the rank of the data never shrinks, so it is exact.
    model = ADeMS(num_components=2, sketch_size=6).fit(X)
    U = np.linalg.svd(X.T)[0][:, :2]
    expected = np.linalg.norm(queries - queries @ U @ U.T, axis=1)

    np.testing.assert_allclose(model.score(queries), expected, atol=1e-10)


def test_threshold_keeps_anomalies_out_of_the_sketch():
    from pysad.models import ADeMS

    initial_X = np.tile([1.0, 0.0, 0.0], (10, 1))
    anomalies = np.tile([0.0, 1.0, 0.0], (30, 1))

    filtered = ADeMS(threshold=0.5, initial_X=initial_X).fit(anomalies)
    unfiltered = ADeMS(initial_X=initial_X).fit(anomalies)

    assert filtered.score_partial(anomalies[0]) == pytest.approx(1.0)
    assert unfiltered.score_partial(anomalies[0]) < 0.5


def test_scores_scale_with_the_instance_and_stay_finite():
    from pysad.models import ADeMS

    X = np.random.RandomState(0).normal(size=(50, 4))
    model = ADeMS().fit(X)

    np.testing.assert_allclose(model.score(1e-150 * X) / 1e-150, model.score(1e150 * X) / 1e150)
    assert model.score_partial(np.zeros(4)) == 0.0


def test_rejects_a_threshold_without_initial_X():
    from pysad.models import ADeMS

    with pytest.raises(ValueError, match="initial_X"):
        ADeMS(threshold=0.5)


@pytest.mark.parametrize(
    ("num_components", "sketch_size", "num_features"),
    [(5, 4, 100), (40, None, 100), (11, 50, 10)],
)
def test_rejects_more_components_than_sketch_columns(num_components, sketch_size, num_features):
    from pysad.models import ADeMS

    # The sketch size is the given sketch_size, ceil(sqrt(num_features)) by default, and at most num_features.
    model = ADeMS(num_components=num_components, sketch_size=sketch_size)
    with pytest.raises(ValueError, match="sketch size"):
        model.fit_partial(np.ones(num_features))


def test_unfitted_scoring_does_not_choose_the_training_dimension():
    from pysad.models import ADeMS

    model = ADeMS()
    assert model.score_partial(np.array([3.0, 4.0])) == 5.0
    model.fit_partial(np.array([1.0, 0.0, 0.0]))
    assert model.score_partial(np.array([0.0, 3.0, 4.0])) == 5.0


@pytest.mark.parametrize("initial_X", [np.empty((0, 3)), np.zeros((4, 3))])
def test_empty_or_zero_training_set(initial_X):
    from pysad.models import ADeMS

    model = ADeMS(initial_X=initial_X)
    assert model.score_partial(np.array([0.0, 3.0, 4.0])) == 5.0
    model.fit_partial(np.array([1.0, 0.0, 0.0]))
    assert model.score_partial(np.array([1.0, 0.0, 0.0])) == 0.0


def test_detupdate_scores_match_algorithm_3_with_shrinkage():
    from pysad.models import ADeMS

    rng = np.random.RandomState(7)
    stream = rng.normal(size=(40, 5))
    query = rng.normal(size=5)
    covariance = np.zeros((5, 5))
    model = ADeMS(num_components=2, sketch_size=3)
    for x in stream:
        # Independent covariance/eigenvalue formulation of Algorithm 3.
        values, vectors = np.linalg.eigh(covariance + np.outer(x, x))
        values, vectors = values[::-1], vectors[:, ::-1]
        k = min(2, int((values > values[0] * 1e-12).sum()))
        basis = vectors[:, :k]
        expected = np.linalg.norm(query - basis @ (basis.T @ query))
        covariance = (vectors[:, :3] * np.maximum(values[:3] - values[2], 0.0)) @ vectors[:, :3].T
        model.fit_partial(x)
        assert model.score_partial(query) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_detupdate_is_stable_for_extreme_magnitudes(scale):
    from pysad.models import ADeMS

    rng = np.random.RandomState(8)
    X = rng.normal(size=(30, 5))
    expected = ADeMS(num_components=2, sketch_size=3).fit_score(X)
    actual = ADeMS(num_components=2, sketch_size=3).fit_score(scale * X) / scale
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, atol=1e-12)
