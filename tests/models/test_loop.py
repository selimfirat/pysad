import pytest


@pytest.mark.parametrize("extent", [1, 2, 3])
@pytest.mark.parametrize("num_neighbors", [1, 10])
def test_loop_matches_pynomaly(num_neighbors, extent):
    import numpy as np
    from PyNomaly import loop

    from pysad.models import LocalOutlierProbability

    rng = np.random.default_rng(0)
    initial_X = rng.normal(size=(60, 3))
    # Ties among the neighbor distances.
    initial_X[:10] = np.round(initial_X[:10])
    X = 2 * rng.normal(size=(40, 3))

    reference = loop.LocalOutlierProbability(initial_X, extent=extent, n_neighbors=num_neighbors)
    reference.fit()
    expected = [float(reference.stream(x)) for x in X]

    scores = LocalOutlierProbability(initial_X, num_neighbors=num_neighbors, extent=extent).score(X)

    assert np.array_equal(scores, expected)
