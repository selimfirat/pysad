import pytest


@pytest.mark.parametrize(
    "params",
    [
        {"window_size": 50, "max_radius": 0.5},
        {"window_size": 1, "max_radius": 0.5},
        {"window_size": 30, "max_radius": 0.0},
        {"window_size": 100, "calibration_size": 150},
    ],
)
@pytest.mark.parametrize("num_features", [1, 2, 6])
def test_nets_scores_equal_exact_storm(params, num_features):
    """NETS gives the scores of ExactStorm while cells fill, empty out and the radius changes."""
    import numpy as np

    from pysad.models import NETS, ExactStorm

    rng = np.random.default_rng(0)
    # Dense groups whose centers drift, sparse noise, and repeated instances.
    t = np.arange(1000)
    centers = np.sin(t[:, None] / 150 + np.arange(num_features)) * (1 + t % 3)[:, None]
    X = centers + 0.2 * rng.normal(size=(1000, num_features))
    X[::10] = 4 * rng.normal(size=(100, num_features))
    X[5::17] = X[4::17][: len(X[5::17])]

    expected = ExactStorm(**params).fit_score(X)

    np.testing.assert_array_equal(NETS(**params).fit_score(X), expected)
    np.testing.assert_array_equal(NETS(**params).fit(X).score(X[:50]), ExactStorm(**params).fit(X).score(X[:50]))


def test_nets_empty_window():
    import numpy as np

    from pysad.models import NETS

    assert NETS().score_partial(np.array([0.0, 0.0])) == 1.0
    assert NETS(max_radius=1.0).fit_score_partial(np.array([0.0, 0.0])) == 1.0


def test_nets_neighbor_counting():
    """The score is the fraction of the other window instances farther than max_radius."""
    import numpy as np

    from pysad.models import NETS

    model = NETS(window_size=10, max_radius=1.0)
    model.fit(np.array([[0.0], [0.1], [0.2], [0.3], [5.0], [-0.9]]))

    assert model.score_partial(np.array([0.0])) == pytest.approx(1 / 6)
    assert model.score_partial(np.array([5.5])) == pytest.approx(5 / 6)
    assert model.score_partial(np.array([20.0])) == 1.0
