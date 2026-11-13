import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from pysad.models import IDKS
from pysad.utils import fix_seed

NUM_FEATURES = 3
WINDOW_SIZE = 64


def _make_stream(num_instances=400, anomaly_shift=8.0, seed=1):
    """Normal instances around zero with a shifted anomalous block."""
    fix_seed(seed)
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(num_instances, NUM_FEATURES))
    X[300:320] += anomaly_shift
    y = np.zeros(num_instances, dtype=int)
    y[300:320] = 1
    return X, y


def test_score_is_a_finite_float_before_and_after_the_window_is_full():
    model = IDKS(input_shape=NUM_FEATURES, window_size=WINDOW_SIZE, random_state=0)
    X = np.random.RandomState(0).normal(size=(WINDOW_SIZE + 10, NUM_FEATURES))

    before = model.score(X[:10])
    assert before.shape == (10,)
    assert before.dtype == np.float64
    assert np.all(before == 0.0)

    model.fit(X[:WINDOW_SIZE])
    after = model.score(X[WINDOW_SIZE : WINDOW_SIZE + 10])
    assert after.shape == (10,)
    assert np.isfinite(after).all()


def test_fit_score_matches_the_per_instance_loop():
    X, _ = _make_stream()
    model_a = IDKS(input_shape=NUM_FEATURES, window_size=WINDOW_SIZE, random_state=0)
    scores_a = model_a.fit_score(X)

    model_b = IDKS(input_shape=NUM_FEATURES, window_size=WINDOW_SIZE, random_state=0)
    scores_b = np.array([model_b.fit_score_partial(x) for x in X])

    np.testing.assert_array_equal(scores_a, scores_b)


def test_score_partial_does_not_change_the_model():
    model = IDKS(input_shape=NUM_FEATURES, window_size=WINDOW_SIZE, random_state=0)
    X = np.random.RandomState(0).normal(size=(WINDOW_SIZE + 5, NUM_FEATURES))
    model.fit(X[:WINDOW_SIZE])

    window_before = model.window_.copy()
    samples_before = model.samples_.copy()
    radius_sq_before = model.radius_sq_.copy()
    feature_maps_before = model.feature_maps_.copy()
    sigma_phi_before = model.sigma_phi_.copy()

    first = model.score_partial(X[WINDOW_SIZE])
    second = model.score_partial(X[WINDOW_SIZE])
    assert first == second
    # The window and the partitions are unchanged by scoring.
    np.testing.assert_array_equal(model.window_, window_before)


def test_anomalies_score_higher_than_normals():
    X, y = _make_stream()
    model = IDKS(input_shape=NUM_FEATURES, window_size=WINDOW_SIZE, random_state=0)
    scores = model.fit_score(X)

    normal_scores = scores[150:200]
    anomaly_scores = scores[300:320]
    assert anomaly_scores.mean() > normal_scores.mean()
    assert roc_auc_score(y, scores) > 0.5


def test_reaches_auroc_above_chance_on_a_separated_stream():
    X, y = _make_stream(anomaly_shift=10.0)
    model = IDKS(input_shape=NUM_FEATURES, window_size=WINDOW_SIZE, random_state=0)
    scores = model.fit_score(X)

    assert roc_auc_score(y, scores) > 0.5


def test_same_seed_gives_same_scores():
    X, _ = _make_stream()
    first = IDKS(input_shape=NUM_FEATURES, window_size=WINDOW_SIZE, random_state=5).fit_score(X)
    second = IDKS(input_shape=NUM_FEATURES, window_size=WINDOW_SIZE, random_state=5).fit_score(X)
    np.testing.assert_array_equal(first, second)


def test_different_seeds_give_different_scores():
    X, _ = _make_stream()
    first = IDKS(input_shape=NUM_FEATURES, window_size=WINDOW_SIZE, random_state=5).fit_score(X)
    second = IDKS(input_shape=NUM_FEATURES, window_size=WINDOW_SIZE, random_state=6).fit_score(X)
    assert not np.array_equal(first, second)


def test_numerically_stable_on_a_constant_stream_and_duplicates():
    model = IDKS(input_shape=NUM_FEATURES, window_size=WINDOW_SIZE, random_state=0)
    X = np.tile(np.array([0.5, -0.5, 1.0]), (WINDOW_SIZE + 20, 1))
    scores = model.fit_score(X)

    assert np.isfinite(scores).all()
    assert scores.shape == (WINDOW_SIZE + 20,)


def test_rejects_wrong_instance_shape():
    model = IDKS(input_shape=NUM_FEATURES, window_size=WINDOW_SIZE, random_state=0)
    with pytest.raises(ValueError):
        model.score_partial(np.zeros(NUM_FEATURES - 1))
    with pytest.raises(ValueError):
        model.score_partial(np.zeros((2, NUM_FEATURES)))
    with pytest.raises(ValueError):
        model.fit_partial(np.zeros(NUM_FEATURES + 1))


def test_window_size_must_be_greater_than_num_samples():
    with pytest.raises(ValueError):
        IDKS(input_shape=NUM_FEATURES, window_size=4, num_samples=8)
