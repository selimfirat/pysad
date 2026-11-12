import pytest


@pytest.mark.parametrize(
    "params",
    [
        {"window_size": 50, "max_radius": 0.5},
        {"window_size": 1, "max_radius": 0.5},
        {"window_size": 100, "calibration_size": 150},
    ],
)
@pytest.mark.parametrize("num_neighbors", [0, 3, 10])
def test_mcod_scores_equal_exact_storm(params, num_neighbors):
    """MCOD gives the scores of ExactStorm while micro-clusters form, drift out of the window and dissolve."""
    import numpy as np

    from pysad.models import MCOD, ExactStorm

    rng = np.random.default_rng(0)
    # Dense groups whose centers drift, sparse noise, and repeated instances.
    t = np.arange(1000)
    centers = np.stack([np.sin(t / 150), np.cos(t / 150)], axis=1) * (1 + t % 3)[:, None]
    X = centers + 0.2 * rng.normal(size=(1000, 2))
    X[::10] = 4 * rng.normal(size=(100, 2))
    X[5::17] = X[4::17][: len(X[5::17])]

    expected = ExactStorm(**params).fit_score(X)
    model = MCOD(num_neighbors=num_neighbors, **params)

    np.testing.assert_array_equal(model.fit_score(X), expected)
    np.testing.assert_array_equal(model.score(X[:50]), ExactStorm(**params).fit(X).score(X[:50]))


def test_mcod_empty_window():
    import numpy as np

    from pysad.models import MCOD

    assert MCOD().score_partial(np.array([0.0, 0.0])) == 1.0
    assert MCOD(max_radius=1.0).fit_score_partial(np.array([0.0, 0.0])) == 1.0


def test_mcod_neighbor_counting():
    """The score is the fraction of the other window instances farther than max_radius."""
    import numpy as np

    from pysad.models import MCOD

    model = MCOD(window_size=10, max_radius=1.0, num_neighbors=1)
    model.fit(np.array([[0.0], [0.1], [0.2], [0.3], [5.0]]))

    assert model.score_partial(np.array([0.0])) == pytest.approx(1 / 5)
    assert model.score_partial(np.array([5.5])) == pytest.approx(4 / 5)
    assert model.score_partial(np.array([20.0])) == 1.0


@pytest.mark.parametrize(
    "model_name",
    ["CPOD", "D3", "KELOS", "MCOD", "NETS", "OnlineOneClassSVM", "xStream"],
)
@pytest.mark.parametrize("window_size", [0, -1])
def test_non_positive_window_size_raises(model_name, window_size):
    from pysad import models

    cls = getattr(models, model_name)
    with pytest.raises(ValueError, match="window_size"):
        cls(window_size=window_size)
