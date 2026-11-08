import numpy as np
import pytest


def _patterns(num_patterns, num_inputs, num_on, seed):
    rng = np.random.default_rng(seed)
    X = np.zeros((num_patterns, num_inputs), dtype=bool)
    for x in X:
        x[rng.choice(num_inputs, size=num_on, replace=False)] = True
    return X


def test_spatial_pooler_activates_num_active_columns_or_nothing_below_the_stimulus_threshold():
    from pysad.models.htm.spatial_pooler import SpatialPooler

    pooler = SpatialPooler(
        num_inputs=200, num_columns=500, num_active=20, stimulus_threshold=1.0, random_state=0
    )
    for x in _patterns(20, 200, 20, seed=0):
        assert len(pooler.compute(x)) == 20

    assert len(pooler.compute(np.zeros(200, dtype=bool))) == 0


@pytest.mark.parametrize(("num_active", "expected"), [(15, 15), (30, 22)])
def test_spatial_pooler_activates_num_active_columns_up_to_half_of_them(num_active, expected):
    from pysad.models.htm.spatial_pooler import SpatialPooler

    pooler = SpatialPooler(num_inputs=50, num_columns=44, num_active=num_active, random_state=0)
    assert len(pooler.compute(np.ones(50, dtype=bool))) == expected


def test_spatial_pooler_learning_makes_codes_noise_robust():
    from pysad.models.htm.spatial_pooler import SpatialPooler

    X = _patterns(20, 400, 40, seed=0)
    rng = np.random.default_rng(1)
    noisy = X.copy()
    for x in noisy:
        # Move 10% of the active bits of each pattern to other inputs.
        x[rng.choice(np.flatnonzero(x), size=4, replace=False)] = False
        x[rng.choice(np.flatnonzero(~x), size=4, replace=False)] = True

    def mean_shared(pooler, Y):
        return (
            np.mean(
                [
                    len(
                        np.intersect1d(
                            pooler.compute(x, learn=False), pooler.compute(y, learn=False)
                        )
                    )
                    for x, y in zip(X, Y, strict=True)
                ]
            )
            / pooler.num_active
        )

    pooler = SpatialPooler(
        num_inputs=400,
        num_columns=1000,
        num_active=20,
        connected_threshold=0.5,
        permanence_increment=0.1,
        permanence_decrement=0.02,
        random_state=0,
    )
    untrained = mean_shared(pooler, noisy)
    for _ in range(5):
        for x in X:
            pooler.compute(x)

    assert untrained < 0.7
    assert mean_shared(pooler, noisy) > 0.9
    assert mean_shared(pooler, np.roll(X, 1, axis=0)) < 0.1


def test_spatial_pooler_does_not_learn_without_learn():
    from pysad.models.htm.spatial_pooler import SpatialPooler

    X = _patterns(30, 100, 10, seed=0)
    pooler = SpatialPooler(num_inputs=100, num_columns=200, num_active=10, random_state=0)
    twin = SpatialPooler(num_inputs=100, num_columns=200, num_active=10, random_state=0)
    for x in X:
        pooler.compute(x, learn=False)

    for x in X:
        np.testing.assert_array_equal(pooler.compute(x), twin.compute(x))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_spatial_pooler_boosting_uses_more_columns(seed):
    from pysad.models.htm.spatial_pooler import SpatialPooler

    # Boosting spreads the codes of many inputs over more of the columns (Section 3.1).
    X = _patterns(100, 200, 20, seed=seed)

    def num_used(boost_strength):
        pooler = SpatialPooler(
            num_inputs=200,
            num_columns=300,
            num_active=6,
            boost_strength=boost_strength,
            random_state=seed,
        )
        for _ in range(10):
            for x in X:
                pooler.compute(x)
        return len(np.unique(np.concatenate([pooler.compute(x, learn=False) for x in X])))

    assert num_used(100.0) > num_used(0.0)


def test_spatial_pooler_connects_each_column_to_the_stimulus_threshold():
    from pysad.models.htm.spatial_pooler import SpatialPooler

    pooler = SpatialPooler(
        num_inputs=100,
        num_columns=200,
        num_active=10,
        potential_fraction=0.1,
        connected_threshold=0.9,
        stimulus_threshold=8.0,
        random_state=0,
    )
    assert pooler.connected_.sum(axis=1).min() >= 8
    for x in _patterns(30, 100, 50, seed=0):
        pooler.compute(x)
    assert pooler.connected_.sum(axis=1).min() >= 8


def test_spatial_pooler_rejects_a_potential_pool_below_the_stimulus_threshold():
    from pysad.models.htm.spatial_pooler import SpatialPooler

    with pytest.raises(ValueError, match="potential_fraction"):
        SpatialPooler(num_inputs=10, num_columns=10, potential_fraction=0.2, stimulus_threshold=3.0)
