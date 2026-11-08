import numpy as np


def test_warmup_and_score_does_not_learn():
    from pysad.models import MemStream

    model = MemStream(window_size=4, num_epochs=10, random_state=0)
    X = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [1.0, 2.0]])
    assert np.array_equal(model.fit_score(X), np.zeros(4))
    before = model.score_partial(np.array([3.0, 4.0]))
    model.score(np.full((20, 2), 100.0))
    assert model.score_partial(np.array([3.0, 4.0])) == before
    assert before > 0


def test_rejected_samples_do_not_change_scores():
    from pysad.models import MemStream

    model = MemStream(window_size=4, num_epochs=10, beta=0.0, random_state=0)
    model.fit(np.arange(8.0).reshape(4, 2))
    query = np.array([3.0, 5.0])
    score = model.score_partial(query)
    # A stored point has score zero and must also be rejected at beta=0.
    model.fit_partial(np.array([0.0, 1.0]))
    model.fit(np.full((100, 2), 100.0))
    assert model.score_partial(query) == score


def test_fifo_memory_and_discounted_distance():
    from pysad.models import MemStream

    model = MemStream(
        window_size=3, num_neighbors=3, gamma=0.5, num_epochs=1, beta=100.0, random_state=3
    )
    model.fit(np.array([[0.0], [1.0], [2.0]]))
    # The public score equals the paper's weighted L1 distances.
    query = np.array([4.0])
    z = np.maximum(
        (query - model.mean_) / model.std_ @ model.encoder_weights_ + model.encoder_bias_, 0.0
    )
    distances = np.abs(model.memory_ - z).sum(axis=1)
    assert np.isclose(
        model.score_partial(query), np.sort(distances) @ np.array([1.0, 0.5, 0.25]) / 1.75
    )
    # Once both data and encodings have expired, the repeated record scores zero.
    model.fit(np.repeat(query[None, :], 6, axis=0))
    assert model.score_partial(query) == 0.0


def test_constant_features_are_finite():
    from pysad.models import MemStream

    model = MemStream(window_size=4, num_epochs=10, random_state=4)
    with np.errstate(all="raise"):
        scores = model.fit_score(np.ones((100, 3)))
        assert np.isfinite(model.score_partial(np.full(3, 1e6)))
    assert np.isfinite(scores).all()


def test_multivariate_training_without_floating_point_warnings():
    from pysad.models import MemStream

    X = np.random.RandomState(0).normal(size=(20, 34))
    model = MemStream(window_size=16, num_epochs=20, random_state=0)
    with np.errstate(all="raise"):
        assert np.isfinite(model.fit_score(X)).all()
