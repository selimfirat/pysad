import numpy as np
import pytest


def test_rs_forest_estimates_the_density_of_the_previous_window():
    from scipy.stats import norm

    from pysad.models import RSForest

    window_size = 4096
    X = np.random.default_rng(1).normal(size=(window_size, 1))
    model = RSForest(
        window_size=window_size, num_trees=100, max_depth=8, node_size_limit=0, random_state=0
    ).fit(X)

    # The score is the negative density relative to the volume of the estimated range: the
    # shortest interval that holds 90% of the first window, widened by three standard deviations.
    values = np.sort(X[:, 0])
    m = window_size * 9 // 10
    start = (values[m:] - values[: window_size - m]).argmin()
    volume = values[start + m] - values[start] + 6.0 * X.std()

    points = np.array([-1.0, 0.0, 1.0])
    densities = [-model.score_partial(np.array([x])) / volume for x in points]
    np.testing.assert_allclose(densities, norm.pdf(points), rtol=0.05)


def test_rs_forest_scores_zero_until_the_first_window_is_full():
    from pysad.models import RSForest

    X = np.random.default_rng(0).normal(size=(40, 2))
    scores = RSForest(window_size=20, num_trees=5, max_depth=6, random_state=0).fit_score(X)

    np.testing.assert_array_equal(scores[:20], 0.0)
    assert (scores[20:] < 0.0).all()


@pytest.mark.parametrize(("label", "dense"), [(None, True), (1, False)])
def test_rs_forest_learns_the_latest_window_without_labeled_anomalies(label, dense):
    from pysad.models import RSForest

    window_size = 50
    rng = np.random.default_rng(0)
    model = RSForest(window_size=window_size, num_trees=10, max_depth=6, random_state=0)
    model.fit(rng.normal(size=(window_size, 2)))
    # A window of points at a new place becomes the reference, unless each one is labeled an anomaly.
    shifted = 2.0 + 0.1 * rng.normal(size=(window_size, 2))
    for x in shifted:
        model.fit_partial(x, label)

    assert (model.score_partial(np.array([2.0, 2.0])) < 0.0) == dense


def test_rs_forest_rejects_a_zero_window_size():
    from pysad.models import RSForest

    with pytest.raises(ValueError, match="window_size"):
        RSForest(window_size=0)


def test_rs_forest_replaces_the_reference_only_at_window_boundaries():
    from pysad.models import RSForest

    rng = np.random.default_rng(42)
    window_size = 64
    model = RSForest(
        window_size=window_size, num_trees=10, max_depth=8, node_size_limit=2, random_state=0
    ).fit(rng.normal(loc=-1.0, scale=0.1, size=(window_size, 2)))
    probes = np.array([[-1.0, -1.0], [1.0, 1.0]])
    before = model.score(probes)
    assert before[0] < before[1]

    for center in (1.0, -1.0):
        window = rng.normal(loc=center, scale=0.1, size=(window_size, 2))
        model.fit(window[:-1])
        np.testing.assert_array_equal(model.score(probes), before)
        model.fit_partial(window[-1])
        after = model.score(probes)
        assert (after[0] < after[1]) == (center == -1.0)
        before = after


def test_rs_forest_uses_the_termination_node_and_window_normalization():
    from pysad.models import RSForest

    window_size = 16
    model = RSForest(
        window_size=window_size,
        num_trees=5,
        max_depth=6,
        node_size_limit=window_size,
        random_state=0,
    ).fit(np.random.default_rng(0).normal(size=(window_size, 2)))
    probes = np.array([[0.0, 0.0], [100.0, -100.0]])
    # All paths terminate at the root: its relative volume is 1.
    np.testing.assert_array_equal(model.score(probes), -1.0)
    for i in range(window_size):
        model.fit_partial(probes[0], y=i % 2)
    # Labeled anomalies do not enter the profile, but N remains the window size.
    np.testing.assert_array_equal(model.score(probes), -0.5)


def test_rs_forest_initialization_ignores_labeled_anomalies():
    from pysad.models import RSForest

    X = np.random.default_rng(0).normal(size=(32, 2))
    model = RSForest(window_size=16, num_trees=5, max_depth=6, random_state=0)
    assert model.fit_score_partial(np.array([1e6, -1e6]), y=1) == 0.0
    scores = model.fit_score(X)
    expected = RSForest(window_size=16, num_trees=5, max_depth=6, random_state=0).fit_score(X)
    np.testing.assert_array_equal(scores, expected)


@pytest.mark.parametrize("scale", [0.0, 1e-120, 1e120])
def test_rs_forest_scores_remain_finite_at_different_feature_scales(scale):
    from pysad.models import RSForest

    X = scale * np.random.default_rng(0).normal(size=(128, 3))
    model = RSForest(window_size=16, num_trees=5, max_depth=8, random_state=0)
    scores = model.fit_score(X)
    assert np.isfinite(scores).all()
    if scale:
        expected = RSForest(window_size=16, num_trees=5, max_depth=8, random_state=0).fit_score(
            X / scale
        )
        np.testing.assert_allclose(scores, expected)


def test_rs_forest_serialized_memory_stays_bounded_across_windows():
    import pickle

    from pysad.models import RSForest

    rng = np.random.default_rng(0)
    model = RSForest(window_size=16, num_trees=5, max_depth=6, random_state=0)
    model.fit(rng.normal(size=(16, 3)))
    initial_size = len(pickle.dumps(model))
    for _ in range(20):
        model.fit(rng.normal(size=(160, 3)))
        assert len(pickle.dumps(model)) == initial_size
