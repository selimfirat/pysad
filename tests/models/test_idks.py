import numpy as np
import pytest

from pysad.models import IDKS

NUM_FEATURES = 3
WINDOW_SIZE = 32


def test_score_is_a_finite_float_before_and_after_the_window_is_full():
    model = IDKS(
        input_shape=NUM_FEATURES, window_size=WINDOW_SIZE, num_partitions=10, random_state=0
    )
    X = np.random.RandomState(0).normal(size=(WINDOW_SIZE + 10, NUM_FEATURES))

    before = model.score(X[:10])
    assert before.shape == (10,)
    assert before.dtype == np.float64
    assert np.all(before == 0.0)

    model.fit(X[:WINDOW_SIZE])
    after = model.score(X[WINDOW_SIZE : WINDOW_SIZE + 10])
    assert after.shape == (10,)
    assert np.isfinite(after).all()


def test_rejects_wrong_instance_shape():
    model = IDKS(
        input_shape=NUM_FEATURES, window_size=WINDOW_SIZE, num_partitions=10, random_state=0
    )
    with pytest.raises(ValueError):
        model.score_partial(np.zeros(NUM_FEATURES - 1))
    with pytest.raises(ValueError):
        model.score_partial(np.zeros((2, NUM_FEATURES)))
    with pytest.raises(ValueError):
        model.fit_partial(np.zeros(NUM_FEATURES + 1))


def test_window_size_must_be_greater_than_num_samples():
    with pytest.raises(ValueError):
        IDKS(input_shape=NUM_FEATURES, window_size=4, num_samples=8)
