def _direct_score(window, x, num_neighbors, cell_size, density_window=None):
    """The local outlier score of Definition 2.3, computed from scratch on the grid of the window.

    If given, `density_window` is the window of the last update of all the nearest kernel centers, which were skipped since.
    """
    import numpy as np

    side = cell_size / np.sqrt(window.shape[1])

    def grid(window):
        cells, weights = np.unique(np.floor(window / side), axis=0, return_counts=True)
        return (cells + 0.5) * side, weights

    def nearest(centers, point):
        return np.argsort(np.linalg.norm(centers - point, axis=1), kind="stable")[:num_neighbors]

    def density(point, cell_center, window):
        centers, weights = grid(window)
        idx = nearest(centers, cell_center)
        w = weights[idx] / weights[idx].sum()
        dists = np.abs(centers[idx] - point)
        h = w @ dists
        h[h <= 0.0] = 0.001  # the bandwidth of the reference implementation
        return w @ np.prod(np.exp(-0.5 * (dists / h) ** 2) / (np.sqrt(2 * np.pi) * h), axis=1)

    centers, _ = grid(window)
    cell_center = (np.floor(x / side) + 0.5) * side
    density_window = window if density_window is None else density_window
    neighbor_densities = np.array(
        [density(c, c, density_window) for c in centers[nearest(centers, cell_center)]]
    )
    return (neighbor_densities.mean() - density(x, cell_center, window)) / neighbor_densities.std()


def test_stare_with_skip_threshold_0_matches_a_direct_computation_on_the_window():
    """Region [5, 5.9) repeats each slide, so its kernel centers are skipped; cell [5.9, 6) and region [0, 1) change.

    In one dimension, with every cell of a region nonempty and an odd number of neighbors, the nearest kernel centers have no ties.
    """
    import numpy as np

    from pysad.models import STARE

    rng = np.random.RandomState(0)
    slide_size, num_slides, num_neighbors, cell_size = 40, 4, 3, 0.1
    stationary = np.repeat(np.arange(5.05, 5.9, 0.1), [1, 2, 3, 2, 1, 3, 2, 1, 3])  # 18 points
    slides = []
    for _ in range(12):
        num_tail = rng.randint(4)
        slides.append(
            np.concatenate(
                [
                    stationary,
                    np.full(num_tail, 5.95),
                    rng.uniform(0.0, 1.0, slide_size - len(stationary) - num_tail),
                ]
            )
        )
    X = np.concatenate(slides)[:, None]

    model = STARE(
        window_size=num_slides * slide_size,
        slide_size=slide_size,
        num_neighbors=num_neighbors,
        cell_size=cell_size,
        skip_threshold=0.0,
    ).fit(X)

    window = X[-num_slides * slide_size :]
    queries = np.concatenate([np.arange(0.05, 1.0, 0.1), np.arange(5.05, 6.0, 0.1), [0.33, 2.0]])
    expected = [_direct_score(window, q, num_neighbors, cell_size) for q in queries[:, None]]
    np.testing.assert_allclose(model.score(queries[:, None]), expected, rtol=1e-9)


def test_stare_uses_the_reference_bandwidth_in_a_constant_feature():
    """The kernel centers and the instances have one value in the second feature, so the density at a kernel center has bandwidth 0 there."""
    import numpy as np

    from pysad.models import STARE

    rng = np.random.RandomState(3)
    window_size, num_neighbors, cell_size = 200, 3, 0.1
    X = np.column_stack([rng.uniform(0.0, 1.0, 2 * window_size), np.full(2 * window_size, 0.5)])

    model = STARE(
        window_size=window_size,
        slide_size=40,
        num_neighbors=num_neighbors,
        cell_size=cell_size,
        skip_threshold=0.0,
    ).fit(X)

    queries = np.column_stack([np.arange(0.05, 1.0, 0.1), np.full(10, 0.5)])
    expected = [_direct_score(X[-window_size:], q, num_neighbors, cell_size) for q in queries]
    np.testing.assert_allclose(model.score(queries), expected, rtol=1e-9)


def test_stare_scores_a_skipped_cell_with_its_current_nearest_kernel_centers():
    """The window is one slide. Cells 0 to 2 hold 20 instances each, cell 4 expires, and cell 9 appears while cell 7 stays.

    The error of every kernel center near cell 0 stays below skip_threshold, so their local densities are from the first window, but the expired cell 4 is no longer one of the nearest kernel centers of cell 0; cell 7 is.
    """
    import numpy as np

    from pysad.models import STARE

    cluster = np.repeat([0.05, 0.15, 0.25], 20)
    first = np.concatenate([cluster, [0.45, 0.75]])[:, None]
    second = np.concatenate([cluster, [0.75, 0.95]])[:, None]
    model = STARE(
        window_size=len(first),
        slide_size=len(first),
        num_neighbors=4,
        cell_size=0.1,
        skip_threshold=0.1,
    ).fit(np.vstack([first, second]))

    x = np.array([0.03])
    expected = _direct_score(second, x, num_neighbors=4, cell_size=0.1, density_window=first)
    np.testing.assert_allclose(model.score_partial(x), expected, rtol=1e-9)


def test_stare_scores_a_point_away_from_the_clusters_highest():
    import numpy as np

    from pysad.models import STARE

    rng = np.random.RandomState(1)
    X = np.concatenate([rng.normal(0.3, 0.03, (300, 2)), rng.normal(0.7, 0.03, (300, 2))])
    model = STARE(window_size=600, slide_size=30, num_neighbors=10, cell_size=0.05).fit(
        rng.permutation(X)
    )

    scores = model.score(np.vstack([X[::50], [[0.5, 0.5]]]))

    assert scores.argmax() == len(scores) - 1


def test_stare_scores_against_the_window_of_the_last_complete_slide():
    import numpy as np

    from pysad.models import STARE

    X = np.random.RandomState(2).uniform(size=(30, 2))
    model = STARE(window_size=20, slide_size=10, num_neighbors=3, cell_size=0.3)

    assert model.fit(X[:9]).score_partial(X[0]) == 0.0
    model.fit(X[9:20])
    before = model.score(X)
    model.fit(X[20:29])
    np.testing.assert_array_equal(model.score(X), before)
    assert not np.array_equal(model.fit_partial(X[29]).score(X), before)


def test_stare_rejects_a_window_that_is_not_a_multiple_of_the_slide():
    import pytest

    from pysad.models import STARE

    with pytest.raises(ValueError, match="multiple of slide_size"):
        STARE(window_size=100, slide_size=30)
