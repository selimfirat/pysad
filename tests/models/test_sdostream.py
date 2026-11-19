def _mixed_stream(seed=0, num_instances=1000, num_features=3):
    """A standard normal stream with 30 far outliers, added after a warmup of 100."""
    import numpy as np

    rng = np.random.RandomState(seed)
    X = rng.normal(0.0, 1.0, size=(num_instances, num_features))
    y = np.zeros(num_instances, dtype=np.int64)

    outliers = rng.choice(np.arange(100, num_instances), size=30, replace=False)
    X[outliers] += 6.0 * rng.choice([-1.0, 1.0], size=(30, num_features))
    y[outliers] = 1

    return X, y


def test_sdostream_ranks_far_outliers_above_normal_points():
    """Far outliers sit far from the active observers, so they score above the normal points."""
    from sklearn.metrics import roc_auc_score

    from pysad.models import SDOstream

    X, y = _mixed_stream()

    scores = SDOstream(random_state=0).fit_score(X)

    assert scores.shape == (X.shape[0],)
    assert roc_auc_score(y, scores) > 0.9


def test_sdostream_score_grows_with_the_distance_to_a_cluster():
    """On one tight cluster, the score rises as the probe moves away from it."""
    import numpy as np

    from pysad.models import SDOstream

    X = np.random.RandomState(3).normal(0.0, 0.1, size=(400, 2))
    model = SDOstream(random_state=0).fit(X)

    near = model.score_partial(np.array([0.0, 0.0]))
    mid = model.score_partial(np.array([3.0, 3.0]))
    far = model.score_partial(np.array([8.0, 8.0]))

    assert near < mid < far


def test_sdostream_scores_before_it_fits_and_leaves_scoring_stateless():
    """fit_score_partial returns the score from before the fit, and score_partial changes nothing.

    The probe is scored for a second time through fit_score_partial. It returns the same value as
    the earlier score_partial only if scoring left the model unchanged and scored from the same
    state, which is the prequential order.
    """
    import numpy as np

    from pysad.models import SDOstream

    X = np.random.RandomState(4).rand(80, 3)
    model = SDOstream(random_state=0)
    for x in X[:-1]:
        model.fit_partial(x)

    probe = X[-1]
    before = model.score_partial(probe)
    combined = model.fit_score_partial(probe)

    assert before == combined


def test_sdostream_is_reproducible_and_seed_sensitive():
    """The same seed gives the same stream of scores, and different seeds give different scores."""
    import numpy as np

    from pysad.models import SDOstream

    X = np.random.RandomState(5).rand(120, 3)

    np.testing.assert_array_equal(
        SDOstream(random_state=1).fit_score(X), SDOstream(random_state=1).fit_score(X)
    )
    assert not np.array_equal(
        SDOstream(random_state=1).fit_score(X), SDOstream(random_state=2).fit_score(X)
    )


def test_sdostream_scores_stay_finite_on_extreme_streams():
    """A constant stream scores 0 and large values do not break the fading or the median."""
    import numpy as np

    from pysad.models import SDOstream

    constant = np.full((60, 2), 3.14)
    np.testing.assert_array_equal(SDOstream().fit_score(constant), np.zeros(60))

    large = np.random.RandomState(6).normal(1e6, 1.0, size=(200, 3))
    assert np.all(np.isfinite(SDOstream().fit_score(large)))


def test_sdostream_keeps_a_repeated_anomaly_anomalous_through_its_idle_observers():
    """A burst of one value becomes observers, and the idle split keeps them out of the score.

    Without idle observers the burst puts observers on itself, so its next score is zero. The
    idle split holds the observers with the lowest quality, and a burst observer is fresh. The
    burst then keeps the score of a point far from the normal instances.
    """
    import numpy as np

    from pysad.models import SDOstream

    rng = np.random.RandomState(0)
    X = rng.normal(scale=0.5, size=(400, 3))
    X = np.vstack([X, np.full((10, 3), 50.0)])
    far = np.full(3, 50.0)

    used = SDOstream(idle_fraction=0.0, random_state=0).fit(X)
    idle = SDOstream(idle_fraction=0.9, random_state=0).fit(X)

    assert used.score_partial(far) == 0.0
    assert idle.score_partial(far) > 20.0


def test_sdostream_time_param_sets_how_long_the_model_remembers_a_cluster():
    """A large time_param keeps the observers of a cluster that stopped, a small one forgets it.

    The stream holds one cluster and then moves to a second cluster. A small time_param replaces
    every observer inside the new cluster, so the old cluster scores high. A large time_param
    adds few observers, so the observers of the old cluster stay and it scores low.
    """
    import numpy as np

    from pysad.models import SDOstream

    rng = np.random.RandomState(0)
    X = rng.normal(scale=0.5, size=(600, 3))
    X[300:] += 50.0

    fast = SDOstream(time_param=100.0, random_state=0).fit(X)
    slow = SDOstream(time_param=100000.0, random_state=0).fit(X)

    assert fast.score_partial(np.zeros(3)) > 10.0
    assert slow.score_partial(np.zeros(3)) < 5.0


def test_sdostream_learns_a_cluster_that_appears_in_the_stream():
    """A new cluster scores high at first, and the score drops as the model moves onto it.

    The stream holds one cluster and then moves to a second cluster far away. The observers stay
    on the first cluster, so the instances of the second cluster score high. The model then adds
    observers on the second cluster, and its score falls to the level of the first cluster.
    """
    import numpy as np

    from pysad.models import SDOstream

    rng = np.random.RandomState(0)
    X = rng.normal(0.0, 1.0, size=(1200, 2))
    X[600:] += 6.0

    scores = SDOstream(time_param=100.0, random_state=0).fit_score(X)

    assert scores[600:620].mean() > 2.0
    assert scores[1100:].mean() < scores[600:620].mean() / 2


def test_sdostream_keeps_the_model_at_the_set_observer_count():
    """The model holds at most `num_observers` points, so its size does not grow with the stream.

    The stream is much longer than `num_observers`. The model then fills once and keeps that
    size, because each new observer replaces one old observer.
    """
    import numpy as np

    from pysad.models import SDOstream

    X = np.random.RandomState(0).rand(2000, 3)

    model = SDOstream(num_observers=20, random_state=0).fit(X)

    assert model.size_ == 20
    assert model.observers_.shape == (20, 3)
