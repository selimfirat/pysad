import numpy as np


def _fitted_forest(**params):
    from pysad.models import RobustRandomCutForest

    model = RobustRandomCutForest(random_state=0, **params)
    for x in np.random.RandomState(0).rand(40, 2):
        model.fit_partial(x)

    return model


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
    assert scores[2] == 0.0
    assert scores[3] > 0.0
    for slot_map in model._slot_leaves:
        np.testing.assert_array_equal(
            model._arena.bbox_low[[slot for _, slot in sorted(slot_map.items())]],
            [[0, 1, 2, 3, 4, 5], [2, 3, 4, 5, 6, 7], [4, 5, 6, 7, 8, 9]],
        )


def _arena_snapshot(model):
    arena = model._arena
    return (
        arena.free_top,
        arena.roots.copy(),
        arena.left.copy(),
        arena.right.copy(),
        arena.parent.copy(),
        arena.sibling.copy(),
        arena.dim.copy(),
        arena.cut.copy(),
        arena.count.copy(),
        arena.bbox_low.copy(),
        arena.bbox_high.copy(),
        [dict(m) for m in model._slot_leaves],
    )


def _assert_snapshot_unchanged(model, snapshot):
    (
        free_top,
        roots,
        left,
        right,
        parent,
        sibling,
        dim,
        cut,
        count,
        bbox_low,
        bbox_high,
        slot_leaves,
    ) = snapshot
    arena = model._arena
    assert arena.free_top == free_top
    np.testing.assert_array_equal(arena.roots, roots)
    np.testing.assert_array_equal(arena.left, left)
    np.testing.assert_array_equal(arena.right, right)
    np.testing.assert_array_equal(arena.parent, parent)
    np.testing.assert_array_equal(arena.sibling, sibling)
    np.testing.assert_array_equal(arena.dim, dim)
    np.testing.assert_array_equal(arena.cut, cut)
    np.testing.assert_array_equal(arena.count, count)
    np.testing.assert_array_equal(arena.bbox_low, bbox_low)
    np.testing.assert_array_equal(arena.bbox_high, bbox_high)
    assert model._slot_leaves == slot_leaves


def _reference_single_tree_codisp(arena, root, point, rng):
    s = root
    while arena.dim[s] >= 0:
        d = int(arena.dim[s])
        s = int(arena.left[s]) if point[d] <= arena.cut[s] else int(arena.right[s])
    if (arena.bbox_low[s] == point).all():
        best = 0.0
        curr = s
    else:
        curr = root
        while True:
            _, _, stop = arena.cut_one_slot(point, curr, rng)
            if stop:
                break
            d = int(arena.dim[curr])
            curr = int(arena.left[curr]) if point[d] <= arena.cut[curr] else int(arena.right[curr])
        best = float(arena.count[curr])

    while (p := int(arena.parent[curr])) >= 0:
        sib = int(arena.sibling[curr])
        best = max(best, int(arena.count[sib]) / (int(arena.count[curr]) + 1))
        curr = p
    return best


def test_rrcf_score_partial_matches_reference_codisp_without_mutating_trees():
    import copy

    from pysad.models import RobustRandomCutForest

    single_tree_model = RobustRandomCutForest(num_trees=1, tree_size=10, random_state=0)

    # Empty trees return 0.0 without drawing from the random number generator.
    rng_state = single_tree_model.rng.get_state()[1].copy()
    assert single_tree_model.score_partial(np.zeros(2)) == 0.0
    np.testing.assert_array_equal(single_tree_model.rng.get_state()[1], rng_state)

    points = np.random.RandomState(0).rand(25, 2)
    for x in points[:15]:
        single_tree_model.fit_partial(x)

    candidates = [points[14], points[10], points[15], np.full(2, 100.0), np.full(2, -100.0)]
    for candidate in candidates:
        ref_rng = copy.deepcopy(single_tree_model.rng)
        snapshot = _arena_snapshot(single_tree_model)

        score = single_tree_model.score_partial(candidate)
        expected = _reference_single_tree_codisp(
            single_tree_model._arena,
            int(single_tree_model._arena.roots[0]),
            candidate,
            ref_rng,
        )

        assert score == expected
        np.testing.assert_array_equal(single_tree_model.rng.get_state()[1], ref_rng.get_state()[1])
        _assert_snapshot_unchanged(single_tree_model, snapshot)

    multi_model = RobustRandomCutForest(num_trees=5, tree_size=10, random_state=0)
    for x in points[:15]:
        multi_model.fit_partial(x)

    for candidate in candidates:
        snapshot = _arena_snapshot(multi_model)
        score = multi_model.score_partial(candidate)
        assert isinstance(score, float)
        assert 0.0 <= score <= 10.0
        _assert_snapshot_unchanged(multi_model, snapshot)


def test_rrcf_constant_features_and_identical_observations():
    from pysad.models import RobustRandomCutForest

    model = RobustRandomCutForest(num_trees=10, tree_size=10, random_state=0)
    for _ in range(15):
        score = model.fit_score_partial(np.ones(5))
        assert isinstance(score, float)
        assert score >= 0.0

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

    outlier = np.full(d, 1e6)
    outlier_score = model.score_partial(outlier)
    normal_score = model.score_partial(np.zeros(d))
    assert outlier_score > normal_score
