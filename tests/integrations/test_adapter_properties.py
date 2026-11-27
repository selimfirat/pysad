"""Property-based tests for the invariants of the adapters in pysad.integrations.

Each test draws a stream of instances with Hypothesis and checks one adapter invariant:

- PySADToRiverAdapter: score_one returns the wrapped model's score without learning it;
  learn_one forwards the instance to fit_partial; feature names stay fixed after the first
  call and dictionaries with the same keys in any order give the same array.
- PySADToScikitMultiflowAdapter: partial_fit learns each row in order and returns the
  adapter; predict matches predict_partial per row; predict_proba matches
  [1 - score, score] per row; score is the accuracy; the wrong output kind raises
  TypeError or ValueError.
- PySADToCapyMoaAdapter: partial_fit learns each instance's x and returns the adapter;
  score returns the model score without learning; predict matches predict_partial;
  predict_proba matches [1 - score, score]; reset is a no-op; raw arrays and scalars are
  accepted.
- PySADToScikitLearnAdapter: score_samples returns the negative model scores;
  decision_function equals score_samples plus the threshold; predict maps negative
  decisions to -1 and zero or positive decisions to 1; fit leaves the wrapped model
  unchanged.
- RiverDriftDetector: update delegates to the wrapped meter; drift_detected reflects the
  meter; clone creates a fresh meter; clone(include_attributes=True) preserves state.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from pysad.integrations import (
    PySADToCapyMoaAdapter,
    PySADToScikitLearnAdapter,
    PySADToScikitMultiflowAdapter,
)

try:  # pragma: no cover - River is a test dependency
    import river  # noqa: F401

    from pysad.integrations import PySADToRiverAdapter, RiverDriftDetector

    HAS_RIVER = True
except ImportError:
    HAS_RIVER = False

PROPERTIES = settings(
    max_examples=50,
    deadline=None,
    database=None,
    suppress_health_check=[HealthCheck.too_slow],
    derandomize=True,
)

FINITE = st.floats(-10.0, 10.0, allow_nan=False, allow_infinity=False)
PROBABILITY = st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False)
FEATURE_NAMES = ["a", "b"]


class _RecordingModel:
    """A minimal PySAD-like model that records every fit_partial and score_partial call."""

    def __init__(self, output_kind: str = "score") -> None:
        self.output_kind = output_kind
        self.fit_calls: list[tuple[np.ndarray, float | None]] = []
        self.score_calls: list[np.ndarray] = []

    def _can_predict(self) -> bool:
        return self.output_kind == "decision"

    def fit(self, X, y=None):
        return self

    def fit_partial(self, X, y=None):
        self.fit_calls.append(
            (np.asarray(X, dtype=np.float64).copy(), None if y is None else float(y))
        )
        return self

    def score(self, X):
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return np.array([float(row[0]) for row in arr], dtype=np.float64)

    def score_partial(self, X):
        arr = np.asarray(X, dtype=np.float64).reshape(-1)
        self.score_calls.append(arr.copy())
        return float(arr[0])

    def predict_partial(self, X):
        value = float(np.asarray(X, dtype=np.float64).reshape(-1)[0])
        return 1 if value > 0.0 else 0


class _RecordingMeter:
    """A minimal PySAD-like drift meter that records every update."""

    def __init__(self) -> None:
        self.updates: list[float] = []
        self.drift_detected = False

    def update(self, x) -> None:
        self.updates.append(float(x))
        self.drift_detected = abs(float(x)) > 50.0


class _Instance:
    """A minimal CapyMoa-like instance exposing its features in ``x``."""

    def __init__(self, x) -> None:
        self.x = np.asarray(x, dtype=np.float64)


def _two_d_vector(instance: dict) -> np.ndarray:
    return np.array([instance["a"], instance["b"]], dtype=np.float64)


# ---------------------------------------------------------------------------
# PySADToRiverAdapter
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_RIVER, reason="river is not installed")
@given(
    instances=st.lists(
        st.dictionaries(st.sampled_from(FEATURE_NAMES), FINITE, min_size=2, max_size=2),
        min_size=1,
        max_size=10,
    )
)
@PROPERTIES
def test_river_score_one_returns_wrapped_score_without_learning(instances):
    model = _RecordingModel()
    adapter = PySADToRiverAdapter(lambda: model, feature_names=FEATURE_NAMES)
    before = len(model.fit_calls)
    for instance in instances:
        expected = model.score_partial(_two_d_vector(instance))
        assert adapter.score_one(instance) == expected
    assert len(model.fit_calls) == before


@pytest.mark.skipif(not HAS_RIVER, reason="river is not installed")
@given(
    instances=st.lists(
        st.dictionaries(st.sampled_from(FEATURE_NAMES), FINITE, min_size=2, max_size=2),
        min_size=1,
        max_size=10,
    )
)
@PROPERTIES
def test_river_learn_one_forwards_instance_to_fit_partial(instances):
    model = _RecordingModel()
    adapter = PySADToRiverAdapter(lambda: model, feature_names=FEATURE_NAMES)
    # River's learn_one is a mutator and returns None.
    first = True
    for instance in instances:
        returned = adapter.learn_one(instance)
        if first:
            assert returned is None
            first = False
    assert len(model.fit_calls) == len(instances)
    for (xi, yi), instance in zip(model.fit_calls, instances, strict=True):
        np.testing.assert_array_equal(xi, _two_d_vector(instance))
        assert yi is None


@pytest.mark.skipif(not HAS_RIVER, reason="river is not installed")
@given(instance=st.dictionaries(st.sampled_from(FEATURE_NAMES), FINITE, min_size=2, max_size=2))
@PROPERTIES
def test_river_score_one_accepts_same_keys_in_any_order(instance):
    model = _RecordingModel()
    adapter = PySADToRiverAdapter(lambda: model, feature_names=FEATURE_NAMES)
    reversed_instance = {"b": instance["b"], "a": instance["a"]}
    assert adapter.score_one(instance) == adapter.score_one(reversed_instance)


@pytest.mark.skipif(not HAS_RIVER, reason="river is not installed")
@given(instance=st.dictionaries(st.sampled_from(FEATURE_NAMES), FINITE, min_size=2, max_size=2))
@PROPERTIES
def test_river_score_one_rejects_different_keys(instance):
    model = _RecordingModel()
    adapter = PySADToRiverAdapter(lambda: model, feature_names=FEATURE_NAMES)
    different = dict(instance)
    different["c"] = different.pop("a")
    with pytest.raises(ValueError, match="Expected exactly the same feature names"):
        adapter.score_one(different)


@pytest.mark.skipif(not HAS_RIVER, reason="river is not installed")
@given(instance=st.dictionaries(st.sampled_from(FEATURE_NAMES), FINITE, min_size=2, max_size=2))
@PROPERTIES
def test_river_score_one_infers_feature_names_from_first_call(instance):
    model = _RecordingModel()
    adapter = PySADToRiverAdapter(lambda: model, feature_names=None)
    assert adapter.feature_names_ is None
    score = adapter.score_one(instance)
    assert adapter.feature_names_ == tuple(instance.keys())
    expected = model.score_partial(np.array(list(instance.values()), dtype=np.float64))
    assert score == expected


# ---------------------------------------------------------------------------
# PySADToScikitMultiflowAdapter
# ---------------------------------------------------------------------------


@given(rows=st.lists(st.lists(FINITE, min_size=2, max_size=2), min_size=1, max_size=10))
@PROPERTIES
def test_skmultiflow_partial_fit_learns_each_row_in_order(rows):
    model = _RecordingModel()
    adapter = PySADToScikitMultiflowAdapter(model)
    X_arr = np.array(rows, dtype=np.float64)
    original = X_arr.copy()
    returned = adapter.partial_fit(X_arr)
    assert returned is adapter
    np.testing.assert_array_equal(X_arr, original)
    assert len(model.fit_calls) == len(X_arr)
    for (xi, yi), row in zip(model.fit_calls, X_arr, strict=True):
        np.testing.assert_array_equal(xi, row)
        assert yi == 0.0


@given(rows=st.lists(st.lists(FINITE, min_size=2, max_size=2), min_size=1, max_size=10))
@PROPERTIES
def test_skmultiflow_predict_matches_predict_partial_per_row(rows):
    model = _RecordingModel(output_kind="decision")
    adapter = PySADToScikitMultiflowAdapter(model)
    X_arr = np.array(rows, dtype=np.float64)
    labels = adapter.predict(X_arr)
    assert labels.dtype == np.int64
    assert labels.shape == (len(X_arr),)
    for i, row in enumerate(X_arr):
        assert labels[i] == model.predict_partial(row)


@given(rows=st.lists(st.lists(PROBABILITY, min_size=2, max_size=2), min_size=1, max_size=10))
@PROPERTIES
def test_skmultiflow_predict_proba_matches_one_minus_score_and_score(rows):
    model = _RecordingModel(output_kind="probability")
    adapter = PySADToScikitMultiflowAdapter(model)
    X_arr = np.array(rows, dtype=np.float64)
    proba = adapter.predict_proba(X_arr)
    assert proba.shape == (len(X_arr), 2)
    assert proba.dtype == np.float64
    for i, row in enumerate(X_arr):
        score = model.score_partial(row)
        np.testing.assert_allclose(proba[i], [1.0 - score, score])


@given(
    pairs=st.lists(
        st.tuples(st.lists(FINITE, min_size=2, max_size=2), st.integers(0, 1)),
        min_size=1,
        max_size=10,
    )
)
@PROPERTIES
def test_skmultiflow_score_is_accuracy(pairs):
    model = _RecordingModel(output_kind="decision")
    adapter = PySADToScikitMultiflowAdapter(model)
    X_arr = np.array([pair[0] for pair in pairs], dtype=np.float64)
    y_arr = np.array([float(pair[1]) for pair in pairs], dtype=np.float64)
    labels = adapter.predict(X_arr)
    accuracy = adapter.score(X_arr, y_arr)
    assert accuracy == float(np.mean(labels == y_arr))


@given(output_kind=st.sampled_from(["score", "probability"]))
@PROPERTIES
def test_skmultiflow_predict_requires_a_decision_rule(output_kind):
    model = _RecordingModel(output_kind=output_kind)
    adapter = PySADToScikitMultiflowAdapter(model)
    with pytest.raises(TypeError, match="without an intrinsic decision rule"):
        adapter.predict(np.array([[1.0, 2.0]]))


@given(output_kind=st.sampled_from(["score", "decision"]))
@PROPERTIES
def test_skmultiflow_predict_proba_requires_a_probability_model(output_kind):
    model = _RecordingModel(output_kind=output_kind)
    adapter = PySADToScikitMultiflowAdapter(model)
    with pytest.raises(ValueError, match="predict_proba is only available"):
        adapter.predict_proba(np.array([[1.0, 2.0]]))


# ---------------------------------------------------------------------------
# PySADToCapyMoaAdapter
# ---------------------------------------------------------------------------


@given(rows=st.lists(st.lists(FINITE, min_size=2, max_size=2), min_size=1, max_size=10))
@PROPERTIES
def test_capymoa_partial_fit_learns_each_instance_x(rows):
    model = _RecordingModel()
    adapter = PySADToCapyMoaAdapter(model)
    objs = [_Instance(np.array(row, dtype=np.float64)) for row in rows]
    for obj in objs:
        returned = adapter.partial_fit(obj)
        assert returned is adapter
    assert len(model.fit_calls) == len(objs)
    for (xi, yi), row in zip(model.fit_calls, rows, strict=True):
        np.testing.assert_array_equal(xi, np.array(row, dtype=np.float64))
        assert yi is None


@given(rows=st.lists(st.lists(FINITE, min_size=2, max_size=2), min_size=1, max_size=10))
@PROPERTIES
def test_capymoa_score_returns_model_score_without_learning(rows):
    model = _RecordingModel()
    adapter = PySADToCapyMoaAdapter(model)
    objs = [_Instance(np.array(row, dtype=np.float64)) for row in rows]
    before = len(model.fit_calls)
    for obj, row in zip(objs, rows, strict=True):
        expected = model.score_partial(np.array(row, dtype=np.float64))
        assert adapter.score(obj) == expected
    assert len(model.fit_calls) == before


@given(rows=st.lists(st.lists(FINITE, min_size=2, max_size=2), min_size=1, max_size=10))
@PROPERTIES
def test_capymoa_predict_matches_predict_partial(rows):
    model = _RecordingModel(output_kind="decision")
    adapter = PySADToCapyMoaAdapter(model)
    objs = [_Instance(np.array(row, dtype=np.float64)) for row in rows]
    for obj, row in zip(objs, rows, strict=True):
        label = adapter.predict(obj)
        assert label in (0, 1)
        assert label == model.predict_partial(np.array(row, dtype=np.float64))


@given(rows=st.lists(st.lists(PROBABILITY, min_size=2, max_size=2), min_size=1, max_size=10))
@PROPERTIES
def test_capymoa_predict_proba_matches_one_minus_score_and_score(rows):
    model = _RecordingModel(output_kind="probability")
    adapter = PySADToCapyMoaAdapter(model)
    objs = [_Instance(np.array(row, dtype=np.float64)) for row in rows]
    for obj, row in zip(objs, rows, strict=True):
        score = model.score_partial(np.array(row, dtype=np.float64))
        proba = adapter.predict_proba(obj)
        assert proba.shape == (2,)
        assert proba.dtype == np.float64
        np.testing.assert_allclose(proba, [1.0 - score, score])


@given(rows=st.lists(st.lists(FINITE, min_size=2, max_size=2), min_size=1, max_size=10))
@PROPERTIES
def test_capymoa_reset_is_a_noop(rows):
    model = _RecordingModel()
    adapter = PySADToCapyMoaAdapter(model)
    objs = [_Instance(np.array(row, dtype=np.float64)) for row in rows]
    for obj in objs:
        adapter.partial_fit(obj)
    assert len(model.fit_calls) == len(objs)
    adapter.reset()
    assert len(model.fit_calls) == len(objs)


@given(values=st.lists(FINITE, min_size=2, max_size=2))
@PROPERTIES
def test_capymoa_partial_fit_accepts_raw_array_and_scalar(values):
    model = _RecordingModel()
    adapter = PySADToCapyMoaAdapter(model)
    adapter.partial_fit(np.array(values, dtype=np.float64))
    adapter.partial_fit(np.float64(values[0]))
    assert len(model.fit_calls) == 2
    np.testing.assert_array_equal(model.fit_calls[0][0], np.array(values, dtype=np.float64))
    np.testing.assert_array_equal(model.fit_calls[1][0], np.array([values[0]], dtype=np.float64))


# ---------------------------------------------------------------------------
# PySADToScikitLearnAdapter
# ---------------------------------------------------------------------------


@given(
    rows=st.lists(st.lists(FINITE, min_size=2, max_size=2), min_size=1, max_size=10),
    threshold=st.floats(-10.0, 10.0, allow_nan=False, allow_infinity=False),
)
@PROPERTIES
def test_sklearn_score_samples_returns_negative_model_scores(rows, threshold):
    model = _RecordingModel()
    adapter = PySADToScikitLearnAdapter(model, threshold=threshold)
    X_arr = np.array(rows, dtype=np.float64)
    assert adapter.fit(X_arr) is adapter
    assert len(model.fit_calls) == 0
    scores = model.score(X_arr)
    np.testing.assert_allclose(adapter.score_samples(X_arr), -scores)


@given(
    rows=st.lists(st.lists(FINITE, min_size=2, max_size=2), min_size=1, max_size=10),
    threshold=st.floats(-10.0, 10.0, allow_nan=False, allow_infinity=False),
)
@PROPERTIES
def test_sklearn_decision_function_equals_score_samples_plus_threshold(rows, threshold):
    model = _RecordingModel()
    adapter = PySADToScikitLearnAdapter(model, threshold=threshold)
    X_arr = np.array(rows, dtype=np.float64)
    adapter.fit(X_arr)
    scores = model.score(X_arr)
    np.testing.assert_allclose(adapter.decision_function(X_arr), -scores + threshold)


def test_sklearn_decision_function_requires_threshold():
    model = _RecordingModel()
    adapter = PySADToScikitLearnAdapter(model, threshold=None)
    with pytest.raises(ValueError, match="Set threshold before calling"):
        adapter.decision_function(np.array([[1.0, 2.0]]))


@given(
    rows=st.lists(st.lists(FINITE, min_size=2, max_size=2), min_size=1, max_size=10),
    threshold=st.floats(-10.0, 10.0, allow_nan=False, allow_infinity=False),
)
@PROPERTIES
def test_sklearn_predict_maps_negative_decisions_to_minus_one(rows, threshold):
    model = _RecordingModel()
    adapter = PySADToScikitLearnAdapter(model, threshold=threshold)
    X_arr = np.array(rows, dtype=np.float64)
    adapter.fit(X_arr)
    scores = model.score(X_arr)
    decisions = -scores + threshold
    labels = adapter.predict(X_arr)
    assert labels.dtype == np.int64
    np.testing.assert_array_equal(labels, np.where(decisions < 0.0, -1, 1))


# ---------------------------------------------------------------------------
# RiverDriftDetector
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_RIVER, reason="river is not installed")
@given(values=st.lists(FINITE, min_size=1, max_size=20))
@PROPERTIES
def test_drift_update_delegates_to_meter(values):
    meter = _RecordingMeter()
    detector = RiverDriftDetector(lambda: meter)
    for value in values:
        detector.update(value)
        assert meter.updates[-1] == value
        assert detector.drift_detected == meter.drift_detected
    assert meter.updates == [float(value) for value in values]


@pytest.mark.skipif(not HAS_RIVER, reason="river is not installed")
@given(values=st.lists(FINITE, min_size=1, max_size=20))
@PROPERTIES
def test_drift_clone_creates_fresh_meter(values):
    detector = RiverDriftDetector(lambda: _RecordingMeter())
    for value in values:
        detector.update(value)
    original_updates = list(detector.meter_.updates)
    cloned = detector.clone()
    assert cloned.drift_detected is False
    assert cloned.meter_.updates == []
    assert detector.meter_.updates == original_updates


@pytest.mark.skipif(not HAS_RIVER, reason="river is not installed")
@given(values=st.lists(FINITE, min_size=1, max_size=20))
@PROPERTIES
def test_drift_clone_with_attributes_preserves_state(values):
    detector = RiverDriftDetector(lambda: _RecordingMeter())
    for value in values:
        detector.update(value)
    cloned = detector.clone(include_attributes=True)
    assert cloned.drift_detected == detector.drift_detected
    assert cloned.meter_.updates == detector.meter_.updates
