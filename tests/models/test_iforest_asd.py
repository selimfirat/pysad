import numpy as np
import pytest

from pysad.models import IForestASD
from pysad.models.integrations import ReferenceWindowModel
from pysad.utils import fix_seed

WINDOW_SIZE = 32
IFOREST_KWARGS = {"n_estimators": 10, "random_state": 61, "contamination": 0.1}


def _scores(model, X):
    return np.array([model.fit_score_partial(x) for x in X])


def test_drift_threshold_none_matches_reference_window():
    """None skips the drift test and retrains on every window, preserving historical behavior."""
    fix_seed(61)
    rng = np.random.RandomState(61)
    X = rng.normal(size=(WINDOW_SIZE * 3, 3))

    asd = IForestASD(window_size=WINDOW_SIZE, **IFOREST_KWARGS)
    from pysad.models.iforest_asd import _IsolationForest

    reference = ReferenceWindowModel(
        _IsolationForest,
        window_size=WINDOW_SIZE,
        sliding_size=WINDOW_SIZE,
        **IFOREST_KWARGS,
    )

    np.testing.assert_array_equal(_scores(asd, X), _scores(reference, X))
    assert asd.drift_threshold is None


def test_first_window_matches_unconditional_retrain():
    """The drift test starts after the first forest, so the first window is unchanged."""
    fix_seed(61)
    rng = np.random.RandomState(61)
    X = rng.normal(size=(WINDOW_SIZE, 3))

    plain = IForestASD(window_size=WINDOW_SIZE, **IFOREST_KWARGS)
    gated = IForestASD(window_size=WINDOW_SIZE, drift_threshold=0.0, **IFOREST_KWARGS)

    np.testing.assert_array_equal(_scores(plain, X), _scores(gated, X))


def test_stationary_stream_keeps_forest_when_threshold_is_high():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(WINDOW_SIZE * 2, 2))
    model = IForestASD(window_size=WINDOW_SIZE, drift_threshold=1.0, **IFOREST_KWARGS)

    for x in X[:WINDOW_SIZE]:
        model.fit_partial(x)
    forest_before = model.model
    reference_before = model.reference_window_X

    for x in X[WINDOW_SIZE:]:
        model.fit_partial(x)

    assert model.model is forest_before
    assert model.reference_window_X is reference_before


def test_distribution_shift_replaces_forest_when_threshold_is_low():
    rng = np.random.RandomState(0)
    first = rng.normal(loc=0.0, scale=1.0, size=(WINDOW_SIZE, 2))
    second = rng.normal(loc=5.0, scale=1.0, size=(WINDOW_SIZE, 2))
    model = IForestASD(window_size=WINDOW_SIZE, drift_threshold=0.05, **IFOREST_KWARGS)

    for x in first:
        model.fit_partial(x)
    forest_before = model.model

    for x in second:
        model.fit_partial(x)

    assert model.model is not forest_before


def test_initial_window_is_kept_when_later_window_is_stationary():
    rng = np.random.RandomState(1)
    initial = rng.normal(size=(WINDOW_SIZE, 2))
    later = rng.normal(size=(WINDOW_SIZE, 2))
    model = IForestASD(
        initial_window_X=initial,
        window_size=WINDOW_SIZE,
        drift_threshold=1.0,
        **IFOREST_KWARGS,
    )
    forest_before = model.model

    for x in later:
        model.fit_partial(x)

    assert model.model is forest_before


def test_anomaly_rate_boundary_algorithm_2():
    """Algorithm 2 rebuilds when the window anomaly rate is at least u (rate >= u)."""
    rng = np.random.RandomState(2)
    X = rng.normal(size=(WINDOW_SIZE * 2, 2))
    model = IForestASD(window_size=WINDOW_SIZE, drift_threshold=0.5, **IFOREST_KWARGS)

    for x in X[:WINDOW_SIZE]:
        model.fit_partial(x)
    forest_before = model.model

    # Exactly half the window is labeled anomalous: rate == 0.5 == u
    def predict_half(window):
        labels = np.zeros(len(window), dtype=int)
        labels[: len(window) // 2] = 1
        return labels

    model.model.predict = predict_half
    for x in X[WINDOW_SIZE:]:
        model.fit_partial(x)

    # Since rate >= u (0.5 >= 0.5), it replaces the forest
    assert model.model is not forest_before

    # When rate < u, it keeps the forest
    model2 = IForestASD(window_size=WINDOW_SIZE, drift_threshold=0.51, **IFOREST_KWARGS)
    for x in X[:WINDOW_SIZE]:
        model2.fit_partial(x)
    forest_before2 = model2.model
    model2.model.predict = predict_half
    for x in X[WINDOW_SIZE:]:
        model2.fit_partial(x)
    assert model2.model is forest_before2


@pytest.mark.parametrize("threshold", [-0.01, 1.01, -1, 2, np.nan, np.inf, True, "0.5"])
def test_drift_threshold_outside_unit_interval_raises(threshold):
    with pytest.raises(ValueError, match="drift_threshold"):
        IForestASD(window_size=8, drift_threshold=threshold)


@pytest.mark.parametrize("threshold", [0, 0.0, 1, 1.0])
def test_drift_threshold_bounds_are_accepted(threshold):
    model = IForestASD(window_size=8, drift_threshold=threshold, n_estimators=2, random_state=0)
    assert model.drift_threshold == threshold


@pytest.mark.parametrize("drift_threshold", [None, 0.05])
def test_reused_input_buffer_gives_same_scores(drift_threshold):
    rng = np.random.RandomState(4)
    X = np.concatenate(
        [rng.normal(size=(WINDOW_SIZE * 2, 2)), rng.normal(5.0, size=(WINDOW_SIZE, 2))]
    )
    expected = _scores(
        IForestASD(window_size=WINDOW_SIZE, drift_threshold=drift_threshold, **IFOREST_KWARGS),
        X.copy(),
    )

    model = IForestASD(window_size=WINDOW_SIZE, drift_threshold=drift_threshold, **IFOREST_KWARGS)
    buffer = np.empty(2)
    scores = []
    for x in X:
        buffer[:] = x
        scores.append(model.fit_score_partial(buffer))

    np.testing.assert_array_equal(scores, expected)


def test_changing_initial_window_after_construction_keeps_scores():
    from pysad.models.iforest_asd import _IsolationForest

    rng = np.random.RandomState(5)
    initial = rng.normal(size=(WINDOW_SIZE, 2))
    X = rng.normal(size=(WINDOW_SIZE, 2))

    def make(initial_window_X):
        # A slide shorter than the window keeps part of the initial window in the next fit.
        return ReferenceWindowModel(
            _IsolationForest,
            window_size=WINDOW_SIZE,
            sliding_size=WINDOW_SIZE // 4,
            initial_window_X=initial_window_X,
            **IFOREST_KWARGS,
        )

    expected = _scores(make(initial.copy()), X)
    caller_initial = initial.copy()
    model = make(caller_initial)
    caller_initial[:] = 0.0

    np.testing.assert_array_equal(_scores(model, X), expected)
