import pytest


@pytest.mark.parametrize("case_index", range(8))
def test_matches_authors_reference_scores(case_index):
    import json
    from pathlib import Path

    import numpy as np

    from pysad.models import OnlineIsolationForest

    # Golden scores from the unmodified authors' implementation. The fixture
    # records its source revision. Single-point batches use n_jobs=1 and
    # np.random.seed(seed); only its undefined first prequential score is omitted.
    fixture = Path(__file__).parent / "data" / "online_isolation_forest_reference.json"
    case = json.loads(fixture.read_text())["cases"][case_index]
    n, d = case["shape"]
    if case["name"] == "drift":
        X = np.random.RandomState(61).normal(size=(n, d))
        X[n // 2 :] += 4.0
    elif case["name"] == "constant":
        X = np.ones((n, d))
    else:
        X = np.tile([0.0, 1.0, 1.0, 0.0], n // 4).reshape(-1, 1)
    model = OnlineIsolationForest(**case["params"], random_state=case["seed"])
    before, after = [], []
    for x in X:
        before.append(model.score_partial(x))
        model.fit_partial(x)
        after.append(model.score_partial(x))

    for actual, expected in [(before, case["before"]), (after, case["after"])]:
        valid = np.array([value is not None for value in expected])
        np.testing.assert_allclose(
            np.array(actual)[valid], np.array(expected, dtype=float)[valid], rtol=1e-12, atol=1e-14
        )
    np.testing.assert_allclose(model.score(X), case["final"], rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_trees": 0},
        {"max_leaf_samples": 0},
        {"window_size": 0},
        {"growth_criterion": "invalid"},
        {"subsample": 0.0},
        {"subsample": 1.5},
        {"branching_factor": 1},
        {"split": "oblique"},
    ],
)
def test_invalid_parameters(kwargs):
    from pysad.models import OnlineIsolationForest

    with pytest.raises(ValueError):
        OnlineIsolationForest(**kwargs)


def test_window_forgetting():
    import numpy as np

    from pysad.models import OnlineIsolationForest

    window_size = 100
    model = OnlineIsolationForest(
        num_trees=16, max_leaf_samples=8, window_size=window_size, random_state=0
    )

    rng = np.random.RandomState(0)
    cluster_a = rng.normal(loc=0.0, scale=0.2, size=(window_size, 2))
    cluster_b = rng.normal(loc=10.0, scale=0.2, size=(window_size, 2))

    # Fit cluster A
    for x in cluster_a:
        model.fit_partial(x)

    # Point in cluster A is normal, point in cluster B is anomalous
    score_a_initial = model.score_partial(np.array([0.0, 0.0]))
    score_b_initial = model.score_partial(np.array([10.0, 10.0]))
    assert score_b_initial > score_a_initial

    # Now stream cluster B for full window size, forgetting cluster A
    for x in cluster_b:
        model.fit_partial(x)

    # Now cluster B is normal and cluster A is anomalous
    score_a_after = model.score_partial(np.array([0.0, 0.0]))
    score_b_after = model.score_partial(np.array([10.0, 10.0]))
    assert score_a_after > score_b_after


def test_pickling_preserves_scoring_with_fixed_growth():
    import pickle

    import numpy as np

    from pysad.models import OnlineIsolationForest

    rng = np.random.RandomState(0)
    X = rng.normal(size=(80, 3))

    model = OnlineIsolationForest(
        num_trees=16,
        max_leaf_samples=8,
        window_size=50,
        random_state=42,
        growth_criterion="fixed",
        branching_factor=4,
        subsample=0.5,
    )
    for x in X[:40]:
        model.fit_partial(x)

    pickled = pickle.loads(pickle.dumps(model))

    scores_orig = [model.fit_score_partial(x) for x in X[40:]]
    scores_pickled = [pickled.fit_score_partial(x) for x in X[40:]]

    np.testing.assert_allclose(scores_orig, scores_pickled)


def test_scores_use_the_reference_active_count_normalization():
    import numpy as np

    from pysad.models import OnlineIsolationForest

    model = OnlineIsolationForest(num_trees=1, max_leaf_samples=2, window_size=8, random_state=0)
    model.fit(np.array([[0.0], [2.0]]))

    np.testing.assert_array_equal(model.score(np.array([[0.0], [2.0]])), [1.0, 1.0])
    model.fit_partial(np.array([1.0]))
    # The third point permits a split at depth zero. Both leaves have depth
    # one, normalized by log(3 / 2) / log(4), rather than by the full window.
    expected = 2.0 ** (-1.0 / (np.log(1.5) / np.log(4.0) + np.finfo(float).eps))
    np.testing.assert_allclose(model.score(np.array([[0.0], [2.0]])), expected)


@pytest.mark.parametrize("subsample", [0.5])
def test_reusing_an_input_buffer_keeps_the_same_scores(subsample):
    import numpy as np

    from pysad.models import OnlineIsolationForest

    X = np.random.RandomState(0).normal(size=(100, 2))
    params = dict(
        num_trees=4, max_leaf_samples=4, window_size=20, subsample=subsample, random_state=0
    )
    expected = OnlineIsolationForest(**params).fit_score(X)
    model = OnlineIsolationForest(**params)
    buffer = np.empty(2)
    actual = []
    for x in X:
        buffer[:] = x
        actual.append(model.fit_score_partial(buffer))

    np.testing.assert_array_equal(actual, expected)


def test_unfitted_model_scores_zero_then_learns():
    import numpy as np

    from pysad.models import OnlineIsolationForest

    model = OnlineIsolationForest()
    x = np.array([1.0, 2.0])
    assert model.score_partial(x) == 0.0
    assert model.fit_score_partial(x) == 0.0
    assert model.score_partial(x) == 1.0


def test_initial_score_avoids_zero_normalization():
    import numpy as np

    from pysad.models import OnlineIsolationForest

    model = OnlineIsolationForest(num_trees=1, max_leaf_samples=1, window_size=4, random_state=0)
    model.fit_partial(np.array([1.0]))
    assert model.score_partial(np.array([1.0])) == 1.0


@pytest.mark.parametrize("subsample", [1.0, 0.5])
def test_memory_stays_bounded_over_a_long_stream(subsample):
    import tracemalloc

    import numpy as np

    from pysad.models import OnlineIsolationForest

    model = OnlineIsolationForest(
        num_trees=2, max_leaf_samples=4, window_size=20, subsample=subsample, random_state=0
    )
    rng = np.random.RandomState(0)
    tracemalloc.start()
    try:
        for _ in range(400):
            model.fit_score_partial(rng.rand(3))
        initial = tracemalloc.get_traced_memory()[0]
        for _ in range(2000):
            model.fit_score_partial(rng.rand(3))
        final = tracemalloc.get_traced_memory()[0]
    finally:
        tracemalloc.stop()
    assert final - initial < 64 * 1024


def test_subsampling_expires_the_originally_selected_points():
    import numpy as np

    from pysad.models import OnlineIsolationForest

    class SamplingRandomState(np.random.RandomState):
        def __init__(self):
            super().__init__(0)
            self.draws = 0

        def random_sample(self, size=None):
            # Select the first two learning updates, then reject every draw,
            # including the reference's independent forgetting draws.
            self.draws += 1
            return 0.0 if self.draws <= 2 else 1.0

    model = OnlineIsolationForest(
        num_trees=1,
        max_leaf_samples=1,
        window_size=4,
        subsample=0.5,
        random_state=SamplingRandomState(),
    )
    model.fit(np.array([[0.0], [1.0], [0.0], [0.0]]))
    assert model.score_partial(np.array([0.0])) < 1.0
    model.fit(np.array([[0.0], [0.0]]))
    # Both selected points expired. Rejecting the forgetting draws must not
    # leave those old points in the tree.
    assert model.score_partial(np.array([0.0])) == 1.0


def test_packed_capacity_growth_across_trees_and_eviction():
    """Growing packed node capacity in one tree updates _p_leaf_pl for later trees in the same step."""
    import numpy as np

    from pysad.models import OnlineIsolationForest

    rng = np.random.default_rng(0)
    X = rng.uniform(size=(260, 4))
    model = OnlineIsolationForest(
        num_trees=16, max_leaf_samples=2, window_size=64, subsample=1.0, random_state=0
    )
    scores = model.fit_score(X)
    assert scores.shape == (260,)
    assert np.all(np.isfinite(scores))
    assert np.all((scores >= 0.0) & (scores <= 1.0))


def test_online_isolation_forest_explain_does_not_change_state():
    import numpy as np

    from pysad.models import OnlineIsolationForest
    from pysad.utils import fix_seed

    fix_seed(0)
    X = np.random.rand(200, 5)
    model = OnlineIsolationForest(num_trees=10, window_size=100, random_state=0)
    model.fit(X[:100])

    before = model.score_partial(X[101])
    model.explain_partial(X[101])
    after = model.score_partial(X[101])

    assert before == after


def test_online_isolation_forest_explain_is_reproducible():
    import numpy as np

    from pysad.models import OnlineIsolationForest
    from pysad.utils import fix_seed

    fix_seed(0)
    X = np.random.rand(200, 5)

    model_a = OnlineIsolationForest(num_trees=10, window_size=100, random_state=0)
    model_a.fit(X[:100])
    weights_a = model_a.explain_partial(X[101])

    model_b = OnlineIsolationForest(num_trees=10, window_size=100, random_state=0)
    model_b.fit(X[:100])
    weights_b = model_b.explain_partial(X[101])

    np.testing.assert_array_equal(weights_a, weights_b)


def test_online_isolation_forest_explain_shap_depends_on_the_instance():
    import numpy as np

    from pysad.models import OnlineIsolationForest
    from pysad.utils import fix_seed

    fix_seed(0)
    X = np.random.rand(200, 5)
    model = OnlineIsolationForest(num_trees=10, window_size=100, random_state=0)
    model.fit(X[:100])

    normal = model.explain_partial(X[101])
    outside = model.explain_partial(np.array([100.0, -50.0, 25.0, -75.0, 200.0]))

    assert not np.allclose(normal, outside)


def test_online_isolation_forest_explain_returns_zeros_before_any_fit():
    import numpy as np

    from pysad.models import OnlineIsolationForest

    model = OnlineIsolationForest(num_trees=10, window_size=100, random_state=0)

    weights = model.explain_partial(np.array([0.3, 0.6, 0.1, 0.2, 0.5]))

    assert weights.shape == (5,)
    assert np.all(weights == 0.0)


def test_online_isolation_forest_explain_rejects_nan():
    import numpy as np

    from pysad.models import OnlineIsolationForest

    X = np.random.RandomState(0).rand(200, 5)
    model = OnlineIsolationForest(num_trees=10, window_size=100, random_state=0).fit(X[:100])

    anomalous = X[101].copy()
    anomalous[1] = np.nan
    with pytest.raises(ValueError, match="OnlineIsolationForest does not accept NaN values"):
        model.explain_partial(anomalous)


def test_online_isolation_forest_explain_non_binary_trees_returns_zeros():
    """TreeSHAP currently only supports binary trees; non-binary should return zeros."""
    import numpy as np

    from pysad.models import OnlineIsolationForest
    from pysad.utils import fix_seed

    fix_seed(0)
    X = np.random.rand(200, 5)
    model = OnlineIsolationForest(num_trees=10, window_size=100, branching_factor=4, random_state=0)
    model.fit(X[:100])

    weights = model.explain_partial(X[101])

    assert weights.shape == (5,)
    assert np.all(weights == 0.0)


def test_online_isolation_forest_explain_satisfies_local_accuracy():
    import numpy as np

    from pysad.models import OnlineIsolationForest

    rng = np.random.RandomState(7)
    X = rng.randn(180, 4)
    model = OnlineIsolationForest(
        num_trees=15, max_leaf_samples=8, window_size=120, random_state=7
    ).fit(X[:150])

    def _expected_depth(node, tree):
        if node.children is None:
            mls = tree.max_leaf_samples
            ns = node.data_size
            pl = 0.0 if ns <= mls else np.log(ns / mls) * tree._inv_log
            return node.depth + pl
        left_child, right_child = node.children
        total = left_child.data_size + right_child.data_size
        left_frac = left_child.data_size / total if total > 0 else 0.5
        return left_frac * _expected_depth(left_child, tree) + (1.0 - left_frac) * _expected_depth(
            right_child, tree
        )

    for idx in range(150, 160):
        x = X[idx]
        phi = model.explain_partial(x)
        assert phi.shape == (4,)
        assert phi.dtype == np.float64
        diffs = [
            0.0 if tree.max_depth == 0 else tree.predict_one(x) - _expected_depth(tree.root, tree)
            for tree in model.trees_
        ]
        np.testing.assert_allclose(phi.sum(), float(np.mean(diffs)), atol=1e-10)


def test_online_isolation_tree_searchsorted_child_raises_on_missing_split_values():
    """_searchsorted_child raises RuntimeError when split_values is None."""
    import numpy as np

    from pysad.models.online_isolation_forest import _OnlineIsolationNode, _OnlineIsolationTree

    tree = _OnlineIsolationTree(
        max_leaf_samples=4, growth_criterion="fixed", subsample=1.0, branching_factor=4
    )
    node = _OnlineIsolationNode(np.array([[0.0, 0.0]]), depth=0, threshold=4)
    node.split_values = None
    with pytest.raises(RuntimeError, match="Non-binary node is missing split_values"):
        tree._searchsorted_child(node, np.array([0.5, 0.5]))


def test_online_isolation_tree_learn_path_raises_on_none_root():
    """_learn_path raises RuntimeError when root is not initialized."""
    import numpy as np

    from pysad.models.online_isolation_forest import _OnlineIsolationTree

    tree = _OnlineIsolationTree(
        max_leaf_samples=4, growth_criterion="fixed", subsample=1.0, branching_factor=2
    )
    with pytest.raises(RuntimeError, match="Tree root is not initialized"):
        tree._learn_path(np.array([0.5, 0.5]), np.random.RandomState(0))


def test_online_isolation_tree_unlearn_raises_on_none_root():
    """_unlearn raises RuntimeError when root is not initialized."""
    import numpy as np

    from pysad.models.online_isolation_forest import _OnlineIsolationTree

    tree = _OnlineIsolationTree(
        max_leaf_samples=4, growth_criterion="fixed", subsample=1.0, branching_factor=2
    )
    with pytest.raises(RuntimeError, match="Tree root is not initialized"):
        tree.unlearn_one(np.array([0.5, 0.5]), np.random.RandomState(0), selected=True)


def test_online_isolation_tree_refresh_bounds_on_leaf():
    """_refresh_bounds returns early for leaf nodes."""
    import numpy as np

    from pysad.models.online_isolation_forest import _OnlineIsolationNode, _OnlineIsolationTree

    tree = _OnlineIsolationTree(
        max_leaf_samples=4, growth_criterion="fixed", subsample=1.0, branching_factor=2
    )
    leaf = _OnlineIsolationNode(np.array([[0.0, 0.0]]), depth=0, threshold=4)
    tree._refresh_bounds(leaf)  # Should return without error


def test_online_isolation_tree_predict_one_returns_zero_for_none_root():
    """predict_one returns 0.0 when root is None."""
    import numpy as np

    from pysad.models.online_isolation_forest import _OnlineIsolationTree

    tree = _OnlineIsolationTree(
        max_leaf_samples=4, growth_criterion="fixed", subsample=1.0, branching_factor=2
    )
    assert tree.predict_one(np.array([0.5, 0.5])) == 0.0


def test_online_isolation_tree_maybe_split_leaf_on_root():
    """_maybe_split_leaf updates root when parent is None (splitting the root)."""
    import numpy as np

    from pysad.models.online_isolation_forest import _OnlineIsolationTree

    tree = _OnlineIsolationTree(
        max_leaf_samples=1, growth_criterion="fixed", subsample=1.0, branching_factor=2
    )
    # Create a root node with enough data to trigger split
    # Need depth_limit > 0 to allow splitting, so we need enough samples
    # With max_leaf_samples=1 and fixed growth, threshold=1 at depth 0
    # data_size=2 >= threshold=1, and depth_limit = log(2/1)/log(4) = 0.5 > depth=0
    root = tree._build(np.array([[0.0, 0.0], [1.0, 1.0]]), depth=0, rng=np.random.RandomState(0))
    tree.root = root
    tree.data_size = 2
    tree.depth_limit = 0.5
    # Now call _maybe_split_leaf on the root (parent=None)
    # The root has data_size=2 >= threshold=1, so it should split
    selected, changed, leaf = tree._maybe_split_leaf(
        root, parent=None, child_idx=0, rng=np.random.RandomState(1)
    )
    assert changed is True
    assert leaf is None
    assert tree.root is not root  # Root should be replaced


def test_online_isolation_forest_getstate_setstate_global_rng():
    """__getstate__ and __setstate__ handle global RNG correctly."""
    import pickle

    import numpy as np

    from pysad.models import OnlineIsolationForest

    model = OnlineIsolationForest(
        num_trees=4, max_leaf_samples=4, window_size=10, random_state=None
    )
    model.fit_partial(np.array([0.5, 0.5]))
    # Use global RNG
    model.rng_ = np.random.mtrand._rand
    pickled = pickle.dumps(model)
    restored = pickle.loads(pickled)
    assert restored.rng_ is np.random.mtrand._rand


def test_online_isolation_forest_ensure_packed_cap_expands():
    """_ensure_packed_cap expands the packed arrays when needed."""
    from pysad.models import OnlineIsolationForest

    model = OnlineIsolationForest(num_trees=4, max_leaf_samples=4, window_size=10, random_state=0)
    # Initial capacity is 16
    assert model._p_cap == 16
    model._ensure_packed_cap(20)
    assert model._p_cap == 32
    # Arrays should be expanded
    assert model._p_dim.shape == (4, 32)
    assert model._p_split.shape == (4, 32)
    assert model._p_left.shape == (4, 32)
    assert model._p_right.shape == (4, 32)
    assert model._p_leaf_pl.shape == (4, 32)
    # Old data should be preserved
    assert model._p_dim[0, 0] == 0


def test_online_isolation_forest_pack_tree_with_none_root():
    """_pack_tree handles None root by setting defaults."""
    from pysad.models import OnlineIsolationForest

    model = OnlineIsolationForest(num_trees=4, max_leaf_samples=4, window_size=10, random_state=0)
    # Tree 0 has no root yet
    model._pack_tree(0)
    assert model.trees_[0].max_depth == 0
    assert model.trees_[0].node_count == 0
    assert model._p_dim[0, 0] == 0
    assert model._p_split[0, 0] == 0.0
    assert model._p_left[0, 0] == 0
    assert model._p_right[0, 0] == 0
    assert model._p_leaf_pl[0, 0] == 0.0


def test_online_isolation_forest_update_packed_leaf_updates_leaf_pl():
    """_update_packed_leaf updates leaf path length when leaf is provided."""
    import numpy as np

    from pysad.models import OnlineIsolationForest

    model = OnlineIsolationForest(num_trees=4, max_leaf_samples=4, window_size=10, random_state=0)
    # Fit one point to create a tree
    model.fit_partial(np.array([0.5, 0.5]))
    # Get the leaf node
    tree = model.trees_[0]
    assert tree.root is not None
    leaf = tree.root
    # Update leaf data_size
    leaf.data_size = 5
    # Update packed leaf
    model._update_packed_leaf(0, changed=False, leaf=leaf)
    # Check that leaf_pl was updated
    expected_pl = 0.0 if leaf.data_size <= 4 else np.log(leaf.data_size / 4) * model._inv_log
    assert model._p_leaf_pl[0, leaf.slot] == leaf.depth + expected_pl


def test_online_isolation_forest_tree_left_fractions_returns_default_for_none_root():
    """_tree_left_fractions returns default 0.5 fractions for None root."""
    from pysad.models import OnlineIsolationForest

    model = OnlineIsolationForest(num_trees=4, max_leaf_samples=4, window_size=10, random_state=0)
    fractions = model._tree_left_fractions(0, 1)
    assert fractions.shape == (1,)
    assert fractions[0] == 0.5


def test_online_isolation_forest_packed_tree_for_shap_returns_defaults_for_zero_depth():
    """_packed_tree_for_shap returns defaults for tree with max_depth == 0."""
    import numpy as np

    from pysad.models import OnlineIsolationForest

    model = OnlineIsolationForest(num_trees=4, max_leaf_samples=4, window_size=10, random_state=0)
    # Tree 0 has no root (max_depth == 0)
    features, thresholds, left, right, values, fractions = model._packed_tree_for_shap(0)
    np.testing.assert_array_equal(features, np.array([-1], dtype=np.intp))
    np.testing.assert_array_equal(thresholds, np.array([0.0], dtype=np.float64))
    np.testing.assert_array_equal(left, np.array([0], dtype=np.intp))
    np.testing.assert_array_equal(right, np.array([0], dtype=np.intp))
    np.testing.assert_array_equal(values, np.array([0.0], dtype=np.float64))
    np.testing.assert_array_equal(fractions, np.array([0.5], dtype=np.float64))


def test_online_isolation_forest_non_binary_tree_learn_and_score():
    """Test learn_one, _learn_path, _build, _unlearn, _refresh_bounds, predict_one, score_partial with branching_factor > 2."""
    import numpy as np

    from pysad.models.online_isolation_forest import _OnlineIsolationTree

    # Test non-binary _build with sufficient depth_limit
    tree = _OnlineIsolationTree(
        max_leaf_samples=1, growth_criterion="fixed", subsample=1.0, branching_factor=4
    )
    # Set depth_limit high enough to allow splitting at depth 0
    tree.depth_limit = 2.0
    root = tree._build(
        np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
        depth=0,
        rng=np.random.RandomState(0),
    )
    assert root.split_values is not None
    assert len(root.split_values) == 3  # branching_factor - 1
    assert root.children is not None
    assert len(root.children) == 4

    # Test non-binary _learn_path
    tree.root = root
    tree.data_size = 4
    tree.depth_limit = 2.0
    selected, changed, leaf = tree._learn_path(np.array([1.5, 1.5]), np.random.RandomState(1))
    assert selected is True

    # Test non-binary _refresh_bounds
    tree._refresh_bounds(root)
    assert root.min_values is not None
    assert root.max_values is not None

    # Test non-binary predict_one
    score = tree.predict_one(np.array([1.5, 1.5]))
    assert isinstance(score, float)
    assert score >= 0.0

    # Test non-binary _unlearn
    changed, leaf = tree.unlearn_one(np.array([0.0, 0.0]), np.random.RandomState(0), selected=True)
    assert isinstance(changed, bool)

    # Test non-binary score_partial through OnlineIsolationForest
    from pysad.models import OnlineIsolationForest

    model = OnlineIsolationForest(
        num_trees=2, max_leaf_samples=2, window_size=20, branching_factor=4, random_state=0
    )
    model.fit_partial(np.array([0.5, 0.5]))
    model.fit_partial(np.array([1.5, 1.5]))
    score = model.score_partial(np.array([0.5, 0.5]))
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_online_isolation_forest_subsample_learn_and_forget():
    """Test subsample < 1.0 code paths including selected tracking and _unlearn_oldest."""
    import numpy as np

    from pysad.models import OnlineIsolationForest

    # Test with subsample < 1.0 and small window to trigger eviction
    model = OnlineIsolationForest(
        num_trees=4, max_leaf_samples=2, window_size=5, subsample=0.5, random_state=0
    )

    # Fit enough points to fill window and trigger eviction
    for i in range(10):
        model.fit_partial(np.array([float(i), float(i)]))

    # Window should be at max size
    assert len(model.window_) == 5

    # Score should work
    score = model.score_partial(np.array([5.0, 5.0]))
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0

    # Test _unlearn_oldest directly
    model2 = OnlineIsolationForest(
        num_trees=2, max_leaf_samples=2, window_size=3, subsample=0.5, random_state=0
    )
    model2.fit_partial(np.array([0.0, 0.0]))
    model2.fit_partial(np.array([1.0, 1.0]))
    model2.fit_partial(np.array([2.0, 2.0]))
    # Now window is full, next fit triggers _unlearn_oldest
    model2.fit_partial(np.array([3.0, 3.0]))
    assert len(model2.window_) == 3


def test_online_isolation_forest_score_partial_max_depth_zero():
    """Test score_partial when _p_max_depth == 0 (trees have no splits yet)."""
    import numpy as np

    from pysad.models import OnlineIsolationForest

    # With very small max_leaf_samples and few points, trees may not split
    model = OnlineIsolationForest(num_trees=2, max_leaf_samples=10, window_size=20, random_state=0)
    # Fit just one point - trees won't split yet (max_depth == 0)
    model.fit_partial(np.array([0.5, 0.5]))
    # Score should use the max_d == 0 branch
    score = model.score_partial(np.array([0.5, 0.5]))
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_online_isolation_forest_getstate_setstate_non_global_rng():
    """__getstate__ and __setstate__ handle non-global RNG correctly."""
    import pickle

    import numpy as np

    from pysad.models import OnlineIsolationForest

    model = OnlineIsolationForest(num_trees=4, max_leaf_samples=4, window_size=10, random_state=42)
    model.fit_partial(np.array([0.5, 0.5]))
    # Use seeded RNG (not global)
    assert model.rng_ is not np.random.mtrand._rand
    pickled = pickle.dumps(model)
    restored = pickle.loads(pickled)
    assert restored.rng_ is not np.random.mtrand._rand
    # Scores should match
    np.testing.assert_allclose(
        model.score_partial(np.array([0.5, 0.5])),
        restored.score_partial(np.array([0.5, 0.5])),
    )


def test_online_isolation_forest_tree_left_fractions_with_tree():
    """_tree_left_fractions computes fractions for actual tree with children."""
    import numpy as np

    from pysad.models import OnlineIsolationForest

    model = OnlineIsolationForest(num_trees=2, max_leaf_samples=2, window_size=10, random_state=0)
    # Fit enough points to create splits
    model.fit_partial(np.array([0.0, 0.0]))
    model.fit_partial(np.array([1.0, 1.0]))
    model.fit_partial(np.array([2.0, 2.0]))
    model.fit_partial(np.array([3.0, 3.0]))

    # Now tree has structure, test _tree_left_fractions
    fractions = model._tree_left_fractions(0, model.trees_[0].node_count)
    assert fractions.shape == (model.trees_[0].node_count,)
    assert np.all(fractions >= 0.0)
    assert np.all(fractions <= 1.0)


def test_online_isolation_forest_packed_tree_for_shap_with_tree():
    """_packed_tree_for_shap extracts arrays for actual tree with splits."""
    import numpy as np

    from pysad.models import OnlineIsolationForest

    model = OnlineIsolationForest(num_trees=2, max_leaf_samples=2, window_size=10, random_state=0)
    # Fit enough points to create splits
    model.fit_partial(np.array([0.0, 0.0]))
    model.fit_partial(np.array([1.0, 1.0]))
    model.fit_partial(np.array([2.0, 2.0]))
    model.fit_partial(np.array([3.0, 3.0]))

    # Now tree has structure, test _packed_tree_for_shap
    features, thresholds, left, right, values, fractions = model._packed_tree_for_shap(0)
    assert features.shape == (model.trees_[0].node_count,)
    assert thresholds.shape == (model.trees_[0].node_count,)
    assert left.shape == (model.trees_[0].node_count,)
    assert right.shape == (model.trees_[0].node_count,)
    assert values.shape == (model.trees_[0].node_count,)
    assert fractions.shape == (model.trees_[0].node_count,)
    # Leaf nodes should have feature -1
    leaf_mask = left == right
    assert np.all(features[leaf_mask] == -1)


def test_online_isolation_tree_learn_all_with_none():
    """_learn_all handles X=None for recursive calls on other branches."""
    import numpy as np

    from pysad.models.online_isolation_forest import _OnlineIsolationTree

    tree = _OnlineIsolationTree(
        max_leaf_samples=2, growth_criterion="adaptive", subsample=1.0, branching_factor=2
    )
    # Build initial tree
    tree.learn_one(np.array([0.0, 0.0]), np.random.RandomState(0))
    tree.learn_one(np.array([1.0, 1.0]), np.random.RandomState(0))
    # Force depth limit increase
    tree.depth_limit = 2.0
    tree._settled_limit = 1
    # Call _learn_all with X=None (simulates depth limit increase without new point)
    new_root, changed = tree._learn_all(tree.root, None, np.random.RandomState(1))
    assert new_root is not None
    assert isinstance(changed, bool)


def test_online_isolation_forest_update_packed_leaf_changed_true():
    """_update_packed_leaf repacks tree when changed=True."""
    import numpy as np

    from pysad.models import OnlineIsolationForest

    model = OnlineIsolationForest(num_trees=2, max_leaf_samples=2, window_size=10, random_state=0)
    # Fit enough points to create a tree structure
    model.fit_partial(np.array([0.0, 0.0]))
    model.fit_partial(np.array([1.0, 1.0]))
    model.fit_partial(np.array([2.0, 2.0]))

    # Now _update_packed_leaf with changed=True should trigger repack
    # This happens when tree structure changes during learn_one
    # We can't easily trigger changed=True without internal access,
    # but we can test the changed=True branch directly
    tree = model.trees_[0]
    assert tree.root is not None
    # Call with changed=True (simulates structure change)
    result = model._update_packed_leaf(0, changed=True, leaf=None)
    assert result is True  # Should return True when repacked


def test_online_isolation_forest_learn_one_subsample_rejection():
    """Test learn_one subsample rejection path."""
    import numpy as np

    from pysad.models.online_isolation_forest import _OnlineIsolationTree

    # Create a tree with subsample=0.5
    tree = _OnlineIsolationTree(
        max_leaf_samples=4, growth_criterion="fixed", subsample=0.5, branching_factor=2
    )

    # Mock RNG to always reject (return > 0.5)
    class RejectRNG(np.random.RandomState):
        def random_sample(self, size=None):
            return 1.0  # Always reject

    rng = RejectRNG(0)
    selected, changed, leaf = tree.learn_one(np.array([0.5, 0.5]), rng)
    assert selected is False
    assert changed is False
    assert leaf is None
    assert tree.root is None  # No tree created


def test_online_isolation_forest_unlearn_non_selected():
    """Test unlearn_one with selected=False returns early."""
    import numpy as np

    from pysad.models.online_isolation_forest import _OnlineIsolationTree

    tree = _OnlineIsolationTree(
        max_leaf_samples=4, growth_criterion="fixed", subsample=1.0, branching_factor=2
    )
    # Build a tree
    tree.learn_one(np.array([0.0, 0.0]), np.random.RandomState(0))
    tree.learn_one(np.array([1.0, 1.0]), np.random.RandomState(0))

    # Unlearn with selected=False should return early without changing tree
    changed, leaf = tree.unlearn_one(np.array([0.0, 0.0]), np.random.RandomState(0), selected=False)
    assert changed is False
    assert leaf is None
    assert tree.data_size == 2  # Unchanged
