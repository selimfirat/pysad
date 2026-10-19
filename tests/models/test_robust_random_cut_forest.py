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

    model = RobustRandomCutForest(num_trees=5, tree_size=10, random_state=0)

    # Empty trees return 0.0 without drawing from the random number generator.
    rng_state = model.rng.get_state()[1].copy()
    assert model.score_partial(np.zeros(2)) == 0.0
    np.testing.assert_array_equal(model.rng.get_state()[1], rng_state)

    points = np.random.RandomState(0).rand(25, 2)
    for x in points[:15]:
        model.fit_partial(x)

    # Test duplicate points (already in the trees), interior points, and far-away outliers.
    candidates = [points[14], points[10], points[15], np.full(2, 100.0), np.full(2, -100.0)]
    for candidate in candidates:
        ref_model = copy.deepcopy(model)
        snapshots = [_tree_snapshot(tree.root) for tree in model.forest]
        leaves = [dict(tree.leaves) for tree in model.forest]

        score = model.score_partial(candidate)
        expected = 0.0
        for tree in ref_model.forest:
            expected += _insert_codisp_forget(tree, candidate, ref_model.rng)
        expected /= ref_model.num_trees

        assert score == expected
        np.testing.assert_array_equal(model.rng.get_state()[1], ref_model.rng.get_state()[1])
        for tree, snap, leaf_map in zip(model.forest, snapshots, leaves, strict=True):
            assert tree.leaves == leaf_map
            assert _tree_snapshot(tree.root) == snap
