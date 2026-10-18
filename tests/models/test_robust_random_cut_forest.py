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
