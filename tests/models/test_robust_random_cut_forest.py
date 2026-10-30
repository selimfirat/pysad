import numpy as np


def _fitted_forest(**params):
    from pysad.models import RobustRandomCutForest

    model = RobustRandomCutForest(random_state=0, **params)
    for x in np.random.RandomState(0).rand(40, 2):
        model.fit_partial(x)

    return model


def test_rrcf_trees_keep_independent_uniform_samples():
    model = _fitted_forest(num_trees=400, tree_size=10)

    samples = [frozenset(tree.leaves) for tree in model.forest]
    assert all(len(sample) == 10 for sample in samples)
    assert all(
        sample == {index for _, index in heap}
        for sample, heap in zip(samples, model.samples, strict=True)
    )
    assert len(set(samples)) > 1

    # A uniform sample of 10 out of 40 keeps each instance in a quarter of the trees.
    kept = np.array([sum(index in sample for sample in samples) for index in range(40)]) / 400
    np.testing.assert_allclose(kept, 0.25, atol=0.11)


def test_rrcf_large_time_decay_keeps_the_latest_instances():
    model = _fitted_forest(num_trees=5, tree_size=10, time_decay=50.0)

    for tree in model.forest:
        assert sorted(tree.leaves) == list(range(30, 40))


def test_rrcf_scores_before_fitting():
    from pysad.models import RobustRandomCutForest

    # With this time_decay, every new instance replaces the oldest one in each tree.
    model = RobustRandomCutForest(num_trees=3, tree_size=8, time_decay=50.0, random_state=0)

    assert model.fit_score_partial(np.zeros(1)) == 0.0
    for _ in range(19):
        model.fit_partial(np.zeros(1))

    # Scored before it replaces one of the 8 zeros, the new instance is cut off from all 8 of them.
    assert model.fit_score_partial(np.ones(1)) == 8.0


def test_rrcf_trees_hold_shingles_of_consecutive_instances():
    from pysad.models import RobustRandomCutForest

    X = np.arange(10.0).reshape(5, 2)
    model = RobustRandomCutForest(num_trees=2, shingle_size=3, random_state=0)

    scores = [model.fit_score_partial(x) for x in X]

    assert scores[:2] == [0.0, 0.0]
    for tree in model.forest:
        np.testing.assert_array_equal(
            np.array([leaf.x for _, leaf in sorted(tree.leaves.items())]),
            [[0, 1, 2, 3, 4, 5], [2, 3, 4, 5, 6, 7], [4, 5, 6, 7, 8, 9]],
        )


def _tree_snapshot(node):
    if node is None:
        return None
    return (
        id(node),
        node.dim,
        node.cut,
        node.count,
        node.bbox.tolist(),
        _tree_snapshot(node.left),
        _tree_snapshot(node.right),
    )


def _insert_codisp_forget(tree, x, rng):
    tree.insert(x, "test_point", rng)
    node = tree.leaves["test_point"]
    result = 0.0
    while node.parent is not None:
        parent = node.parent
        sibling = parent.right if node is parent.left else parent.left
        result = max(result, sibling.count / node.count)
        node = parent
    tree.forget("test_point")
    return result


def test_rrcf_score_partial_matches_insert_and_forget_without_mutating_trees():
    import copy

    from pysad.models import RobustRandomCutForest

    # For a single tree, level-by-level batched cuts match tree-by-tree sequential cuts bit-for-bit.
    single_tree_model = RobustRandomCutForest(num_trees=1, tree_size=10, random_state=0)

    # Empty trees return 0.0 without drawing from the random number generator.
    rng_state = single_tree_model.rng.get_state()[1].copy()
    assert single_tree_model.score_partial(np.zeros(2)) == 0.0
    np.testing.assert_array_equal(single_tree_model.rng.get_state()[1], rng_state)

    points = np.random.RandomState(0).rand(25, 2)
    for x in points[:15]:
        single_tree_model.fit_partial(x)

    # Test duplicate points (already in the trees), interior points, and far-away outliers.
    candidates = [points[14], points[10], points[15], np.full(2, 100.0), np.full(2, -100.0)]
    for candidate in candidates:
        ref_model = copy.deepcopy(single_tree_model)
        snapshot = _tree_snapshot(single_tree_model.forest[0].root)
        leaves = dict(single_tree_model.forest[0].leaves)

        score = single_tree_model.score_partial(candidate)
        expected = _insert_codisp_forget(ref_model.forest[0], candidate, ref_model.rng)

        assert score == expected
        np.testing.assert_array_equal(
            single_tree_model.rng.get_state()[1], ref_model.rng.get_state()[1]
        )
        assert single_tree_model.forest[0].leaves == leaves
        assert _tree_snapshot(single_tree_model.forest[0].root) == snapshot

    # For a multi-tree forest, scoring batches cuts level-by-level without mutating any tree.
    multi_model = RobustRandomCutForest(num_trees=5, tree_size=10, random_state=0)
    for x in points[:15]:
        multi_model.fit_partial(x)

    for candidate in candidates:
        snapshots = [_tree_snapshot(tree.root) for tree in multi_model.forest]
        leaves = [dict(tree.leaves) for tree in multi_model.forest]

        score = multi_model.score_partial(candidate)
        assert isinstance(score, float)
        assert 0.0 <= score <= 10.0
        for tree, snap, leaf_map in zip(multi_model.forest, snapshots, leaves, strict=True):
            assert tree.leaves == leaf_map
            assert _tree_snapshot(tree.root) == snap


def test_rrcf_cuts_on_varying_feature_at_leaf():
    from pysad.models import RobustRandomCutForest

    model = RobustRandomCutForest(num_trees=20, tree_size=2, random_state=0)
    model.fit_partial(np.array([0.0, 0.0]))
    model.fit_partial(np.array([0.0, 1.0]))

    for tree in model.forest:
        assert tree.root.dim == 1
        assert 0.0 <= tree.root.cut <= 1.0
        np.testing.assert_array_equal(tree.root.left.x, [0.0, 0.0])
        np.testing.assert_array_equal(tree.root.right.x, [0.0, 1.0])


def test_rrcf_constant_features_and_identical_observations():
    from pysad.models import RobustRandomCutForest

    model = RobustRandomCutForest(num_trees=10, tree_size=10, random_state=0)
    # Stream identical observations (duplicates)
    for _ in range(15):
        score = model.fit_score_partial(np.ones(5))
        assert isinstance(score, float)
        assert score >= 0.0

    # Score a point with constant features different from the fitted ones
    score = model.score_partial(np.full(5, 5.0))
    assert isinstance(score, float)
    assert score > 0.0


def test_rrcf_high_dimensional_and_boundary_cuts():
    from pysad.models import RobustRandomCutForest

    d = 50
    rng = np.random.RandomState(42)
    model = RobustRandomCutForest(num_trees=15, tree_size=20, random_state=42)

    for _ in range(30):
        x = rng.randn(d)
        score = model.fit_score_partial(x)
        assert isinstance(score, float)
        assert score >= 0.0

    # Extreme outlier
    outlier = np.full(d, 1e6)
    outlier_score = model.score_partial(outlier)
    normal_score = model.score_partial(np.zeros(d))
    assert outlier_score > normal_score


def test_rrcf_reservoir_eviction_with_batched_traversal():
    from pysad.models import RobustRandomCutForest

    model = RobustRandomCutForest(num_trees=10, tree_size=8, random_state=0)
    rng = np.random.RandomState(0)

    for _ in range(40):
        model.fit_partial(rng.rand(3))

    for tree in model.forest:
        assert len(tree.leaves) == 8


def test_rrcf_cut_batch_matches_single_cut():
    from pysad.models.robust_random_cut_forest import _Node, _RCTree

    rng = np.random.RandomState(123)
    d = 10
    num_nodes = 25

    nodes = []
    for _ in range(num_nodes):
        low = rng.uniform(-10, 0, size=d)
        high = low + rng.uniform(0.1, 10, size=d)
        bbox = np.stack([low, high])
        node = _Node()
        node.bbox = bbox
        nodes.append(node)

    x = rng.uniform(-15, 15, size=d)

    rng1 = np.random.RandomState(999)
    rng2 = np.random.RandomState(999)

    dims, cuts, stopped = _RCTree._cut_batch(x, nodes, rng1)

    for i, node in enumerate(nodes):
        expected_dim, expected_cut = _RCTree._cut(x, node.bbox, rng2)
        expected_stopped = (
            expected_cut <= node.bbox[0, expected_dim]
            or expected_cut >= node.bbox[-1, expected_dim]
        )
        assert dims[i] == expected_dim
        np.testing.assert_allclose(cuts[i], expected_cut)
        assert stopped[i] == expected_stopped


def test_rrcf_cut_batch_empty_nodes():
    from pysad.models.robust_random_cut_forest import _RCTree

    rng = np.random.RandomState(42)
    x = np.array([1.0, 2.0, 3.0])
    dims, cuts, stopped = _RCTree._cut_batch(x, [], rng)
    assert len(dims) == 0
    assert len(cuts) == 0
    assert len(stopped) == 0


def test_rrcf_methods_are_fully_annotated():
    import inspect
    from typing import get_type_hints

    from pysad.models.robust_random_cut_forest import RobustRandomCutForest, _Node, _RCTree

    # Regression guard for the type-annotation work: every method of the tree, node and model
    # must annotate all of its parameters and its return type, so that the mypy overrides in
    # pyproject.toml (disallow_untyped_defs = true) keep passing for this module.
    for cls in (RobustRandomCutForest, _RCTree, _Node):
        for name, raw in cls.__dict__.items():
            method = raw if isinstance(raw, staticmethod) else raw
            if not inspect.isfunction(method):
                continue

            hints = get_type_hints(method)
            parameters = inspect.signature(method).parameters
            for param_name, parameter in parameters.items():
                if param_name == "self":
                    continue
                assert parameter.annotation is not inspect.Parameter.empty, (
                    cls.__name__,
                    name,
                    param_name,
                )
                assert param_name in hints, (cls.__name__, name, param_name)
            assert "return" in hints, (cls.__name__, name)
