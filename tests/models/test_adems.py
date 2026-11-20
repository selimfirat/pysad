import numpy as np
import pytest


def test_scores_equal_the_residual_on_the_exact_singular_vectors() -> None:
    from pysad.models import ADeMS

    rng = np.random.RandomState(0)
    X = rng.normal(size=(200, 2)) @ rng.normal(size=(2, 6))  # rank 2
    queries = rng.normal(size=(20, 6))

    # A sketch with more columns than the rank of the data never shrinks, so it is exact.
    model = ADeMS(num_components=2, sketch_size=6).fit(X)

    # Normalize queries as the model does.
    def normalize_rows(X: np.ndarray) -> np.ndarray:
        scale = np.abs(X).max(axis=1, keepdims=True)
        scale[scale == 0] = 1.0
        Y = X / scale
        norms = np.linalg.norm(Y, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return Y / norms

    unit_queries = normalize_rows(queries)
    U = np.linalg.svd(X.T)[0][:, :2]
    expected = np.linalg.norm(unit_queries - unit_queries @ U @ U.T, axis=1)

    np.testing.assert_allclose(model.score(queries), expected, atol=1e-10)


def test_threshold_keeps_anomalies_out_of_the_sketch() -> None:
    from pysad.models import ADeMS

    initial_X = np.tile([1.0, 0.0, 0.0], (10, 1))
    anomalies = np.tile([0.0, 1.0, 0.0], (30, 1))

    filtered = ADeMS(threshold=0.5, initial_X=initial_X).fit(anomalies)
    unfiltered = ADeMS(initial_X=initial_X).fit(anomalies)

    assert filtered.score_partial(anomalies[0]) == pytest.approx(1.0)
    assert unfiltered.score_partial(anomalies[0]) < 0.5


def test_scores_do_not_depend_on_the_scale_of_an_instance() -> None:
    from pysad.models import ADeMS

    X = np.random.RandomState(0).normal(size=(50, 4))
    model = ADeMS().fit(X)

    np.testing.assert_allclose(model.score(1e-150 * X), model.score(1e150 * X))
    assert model.score_partial(np.zeros(4)) == 0.0


def test_pickling_mid_stream_keeps_later_scores() -> None:
    import pickle

    from pysad.models import ADeMS

    X = np.random.RandomState(0).normal(size=(60, 8))
    for split in range(1, X.shape[0]):
        model = ADeMS().fit(X[:split])
        restored = pickle.loads(pickle.dumps(model))

        np.testing.assert_array_equal(
            restored.fit_score(X[split:]), model.fit_score(X[split:]), err_msg=f"split={split}"
        )


def test_rejects_a_threshold_without_initial_X() -> None:
    from pysad.models import ADeMS

    with pytest.raises(ValueError, match="initial_X"):
        ADeMS(threshold=0.5)


@pytest.mark.parametrize(
    ("num_components", "sketch_size", "num_features"),
    [(5, 4, 100), (40, None, 100), (11, 50, 10)],
)
def test_rejects_more_components_than_sketch_columns(
    num_components: int, sketch_size: int | None, num_features: int
) -> None:
    from pysad.models import ADeMS

    # The sketch size is the given sketch_size, ceil(sqrt(num_features)) by default, and at most num_features.
    model = ADeMS(num_components=num_components, sketch_size=sketch_size)
    with pytest.raises(ValueError, match="sketch size"):
        model.fit_partial(np.ones(num_features))


def test_unfitted_scoring_does_not_choose_the_training_dimension() -> None:
    from pysad.models import ADeMS

    model = ADeMS()
    # Non-zero instance scores 1 (its normalized norm) before any fit.
    assert model.score_partial(np.array([3.0, 4.0])) == 1.0
    model.fit_partial(np.array([1.0, 0.0, 0.0]))
    # After fitting, a vector orthogonal to the sketch scores 1.
    assert model.score_partial(np.array([0.0, 3.0, 4.0])) == 1.0


@pytest.mark.parametrize("initial_X", [np.empty((0, 3)), np.zeros((4, 3))])
def test_empty_or_zero_training_set(initial_X: np.ndarray) -> None:
    from pysad.models import ADeMS

    model = ADeMS(initial_X=initial_X)
    # Non-zero instance scores 1 before any fit.
    assert model.score_partial(np.array([0.0, 3.0, 4.0])) == 1.0
    model.fit_partial(np.array([1.0, 0.0, 0.0]))
    assert model.score_partial(np.array([1.0, 0.0, 0.0])) == 0.0


def test_detupdate_scores_match_algorithm_3_with_shrinkage() -> None:
    from pysad.models import ADeMS

    rng = np.random.RandomState(7)
    stream = rng.normal(size=(40, 5))
    query = rng.normal(size=5)
    covariance = np.zeros((5, 5))
    model = ADeMS(num_components=2, sketch_size=3)
    for x in stream:
        # Normalize x as the model does.
        scale = np.abs(x).max()
        if scale == 0.0:
            x_norm = x
        else:
            x_norm = x / scale
            nrm = np.linalg.norm(x_norm)
            if nrm > 0:
                x_norm = x_norm / nrm
            else:
                x_norm = x
        # Independent covariance/eigenvalue formulation of Algorithm 3 on normalized input.
        values, vectors = np.linalg.eigh(covariance + np.outer(x_norm, x_norm))
        values, vectors = values[::-1], vectors[:, ::-1]
        k = min(2, int((values > values[0] * 1e-12).sum()))
        basis = vectors[:, :k]
        # Normalize query for scoring.
        scale_q = np.abs(query).max()
        if scale_q == 0.0:
            query_norm = query
        else:
            query_norm = query / scale_q
            nrm_q = np.linalg.norm(query_norm)
            if nrm_q > 0:
                query_norm = query_norm / nrm_q
            else:
                query_norm = query
        expected = np.linalg.norm(query_norm - basis @ (basis.T @ query_norm))
        covariance = (vectors[:, :3] * np.maximum(values[:3] - values[2], 0.0)) @ vectors[:, :3].T
        model.fit_partial(x)
        assert model.score_partial(query) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_detupdate_is_stable_for_extreme_magnitudes(scale: float) -> None:
    from pysad.models import ADeMS

    rng = np.random.RandomState(8)
    X = rng.normal(size=(30, 5))
    expected = ADeMS(num_components=2, sketch_size=3).fit_score(X)
    actual = ADeMS(num_components=2, sketch_size=3).fit_score(scale * X)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, atol=1e-12)
