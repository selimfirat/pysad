import numpy as np
import pytest


def test_scores_equal_the_residual_on_the_exact_singular_vectors():
    from pysad.models import ADeMS

    rng = np.random.RandomState(0)
    X = rng.normal(size=(200, 2)) @ rng.normal(size=(2, 6))  # rank 2
    queries = rng.normal(size=(20, 6))

    # A sketch with more columns than the rank of the data never shrinks, so it is exact.
    model = ADeMS(num_components=2, sketch_size=6).fit(X)
    U = np.linalg.svd((X / np.linalg.norm(X, axis=1, keepdims=True)).T)[0][:, :2]
    unit = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    expected = np.linalg.norm(unit - unit @ U @ U.T, axis=1)

    np.testing.assert_allclose(model.score(queries), expected, atol=1e-10)


def test_threshold_keeps_anomalies_out_of_the_sketch():
    from pysad.models import ADeMS

    initial_X = np.tile([1.0, 0.0, 0.0], (10, 1))
    anomalies = np.tile([0.0, 1.0, 0.0], (30, 1))

    filtered = ADeMS(threshold=0.5, initial_X=initial_X).fit(anomalies)
    unfiltered = ADeMS(initial_X=initial_X).fit(anomalies)

    assert filtered.score_partial(anomalies[0]) == pytest.approx(1.0)
    assert unfiltered.score_partial(anomalies[0]) < 0.5


def test_scores_do_not_depend_on_the_scale_of_an_instance():
    from pysad.models import ADeMS

    X = np.random.RandomState(0).normal(size=(50, 4))
    model = ADeMS().fit(X)

    np.testing.assert_allclose(model.score(1e-150 * X), model.score(1e150 * X))
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
