"""Tests for PySADToScikitMultiflowAdapter, the zero-dependency scikit-multiflow adapter.

The adapter never imports scikit-multiflow; it duck-types the scikit-multiflow stream
estimator interface (``partial_fit``, ``predict``, ``predict_proba``, ``score``) around
any PySAD model. The tests use small fake models instead of scikit-multiflow itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from pysad.integrations import PySADToScikitMultiflowAdapter


class _RecordingModel:
    """A minimal PySAD-like model that records every fit_partial call."""

    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, float | None]] = []

    @property
    def output_kind(self) -> str:
        return "score"

    def _can_predict(self) -> bool:
        return False

    def fit_partial(self, X: np.ndarray, y: float | None = None) -> _RecordingModel:
        self.calls.append(
            (
                np.asarray(X, dtype=np.float64).copy(),
                None if y is None else float(y),
            )
        )
        return self

    def score_partial(self, X: np.ndarray) -> float:
        return 0.0


class _DecisionModel:
    """A minimal decision-capable PySAD-like model."""

    def __init__(self) -> None:
        self.output_kind = "decision"

    def _can_predict(self) -> bool:
        return True

    def fit_partial(self, X: np.ndarray, y: float | None = None) -> _DecisionModel:
        return self

    def score_partial(self, X: np.ndarray) -> float:
        return float(np.asarray(X, dtype=np.float64).reshape(-1)[0])

    def predict_partial(self, X: np.ndarray) -> int:
        return 1 if float(np.asarray(X, dtype=np.float64).reshape(-1)[0]) > 0.0 else 0


class _ProbabilityModel:
    """A minimal probability-calibrated PySAD-like model."""

    def __init__(self) -> None:
        self.output_kind = "probability"

    def _can_predict(self) -> bool:
        return False

    def fit_partial(self, X: np.ndarray, y: float | None = None) -> _ProbabilityModel:
        return self

    def score_partial(self, X: np.ndarray) -> float:
        value = float(np.asarray(X, dtype=np.float64).reshape(-1)[0])
        return min(1.0, max(0.0, value))


def _calls_match(calls, expected):
    """Compares recorded fit_partial calls with expected (X, y) pairs."""
    assert len(calls) == len(expected)
    for (xi, yi), (expected_x, expected_y) in zip(calls, expected, strict=True):
        np.testing.assert_array_equal(xi, expected_x)
        assert yi == expected_y


def test_adapter_wraps_the_given_model():
    """The adapter keeps a reference to the wrapped model."""
    model = _RecordingModel()
    adapter = PySADToScikitMultiflowAdapter(model)
    assert adapter.model is model


def test_partial_fit_learns_each_row_in_order():
    """partial_fit calls fit_partial once per row, in order, and returns the adapter."""
    model = _RecordingModel()
    adapter = PySADToScikitMultiflowAdapter(model)
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([0.0, 1.0, 0.0])
    expected = [
        (np.array([1.0, 2.0]), 0.0),
        (np.array([3.0, 4.0]), 1.0),
        (np.array([5.0, 6.0]), 0.0),
    ]
    returned = adapter.partial_fit(X, y)
    assert returned is adapter
    _calls_match(model.calls, expected)


def test_partial_fit_defaults_y_to_zeros():
    """A missing y is recorded as 0.0 for every row."""
    model = _RecordingModel()
    adapter = PySADToScikitMultiflowAdapter(model)
    adapter.partial_fit(np.array([[1.0], [2.0], [3.0]]))
    _calls_match(
        model.calls, [(np.array([1.0]), 0.0), (np.array([2.0]), 0.0), (np.array([3.0]), 0.0)]
    )


def test_partial_fit_accepts_skmultiflow_fit_arguments():
    """classes, sample_weight and weight are accepted and ignored."""
    model = _RecordingModel()
    adapter = PySADToScikitMultiflowAdapter(model)
    adapter.partial_fit(
        np.array([[1.0]]),
        np.array([0.0]),
        classes=np.array([0.0, 1.0]),
        sample_weight=np.array([1.0]),
        weight=np.array([1.0]),
    )
    _calls_match(model.calls, [(np.array([1.0]), 0.0)])


def test_partial_fit_rejects_mismatched_row_counts():
    """X and y with different lengths raise ValueError."""
    model = _RecordingModel()
    adapter = PySADToScikitMultiflowAdapter(model)
    with pytest.raises(ValueError, match="same number of rows"):
        adapter.partial_fit(np.array([[1.0], [2.0]]), np.array([0.0]))


def test_predict_returns_binary_labels():
    """predict maps each row to 0 or 1 through predict_partial."""
    model = _DecisionModel()
    adapter = PySADToScikitMultiflowAdapter(model)
    X = np.array([[-1.0], [0.5], [2.0], [-3.0]])
    labels = adapter.predict(X)
    assert labels.dtype == np.int64
    np.testing.assert_array_equal(labels, np.array([0, 1, 1, 0]))


def test_predict_accepts_one_dimensional_X():
    """A one-dimensional X is treated as a single instance."""
    model = _DecisionModel()
    adapter = PySADToScikitMultiflowAdapter(model)
    labels = adapter.predict(np.array([-1.0, 2.0]))
    assert labels.shape == (1,)
    assert labels.dtype == np.int64
    np.testing.assert_array_equal(labels, np.array([0]))


def test_predict_requires_a_decision_rule():
    """A score-only model raises TypeError from predict."""
    model = _RecordingModel()
    adapter = PySADToScikitMultiflowAdapter(model)
    with pytest.raises(TypeError, match="without an intrinsic decision rule"):
        adapter.predict(np.array([[1.0]]))


def test_predict_proba_returns_normal_and_anomaly_probabilities():
    """predict_proba returns [P(normal), P(anomaly)] from the model score."""
    model = _ProbabilityModel()
    adapter = PySADToScikitMultiflowAdapter(model)
    X = np.array([[0.2], [0.7]])
    proba = adapter.predict_proba(X)
    assert proba.shape == (2, 2)
    np.testing.assert_allclose(proba, np.array([[0.8, 0.2], [0.3, 0.7]]))


def test_predict_proba_requires_a_probability_model():
    """A non-probability model raises ValueError from predict_proba."""
    model = _DecisionModel()
    adapter = PySADToScikitMultiflowAdapter(model)
    with pytest.raises(ValueError, match="predict_proba is only available"):
        adapter.predict_proba(np.array([[1.0]]))


def test_score_returns_accuracy():
    """score is the fraction of predicted labels equal to y."""
    model = _DecisionModel()
    adapter = PySADToScikitMultiflowAdapter(model)
    X = np.array([[-1.0], [0.5], [2.0], [-3.0]])
    y = np.array([0, 1, 1, 0])
    assert adapter.score(X, y) == 1.0
    assert adapter.score(X, np.array([1, 1, 1, 1])) == 0.5


def test_score_requires_a_decision_rule():
    """score raises TypeError when the wrapped model cannot predict labels."""
    model = _RecordingModel()
    adapter = PySADToScikitMultiflowAdapter(model)
    with pytest.raises(TypeError, match="without an intrinsic decision rule"):
        adapter.score(np.array([[1.0]]), np.array([1]))


def test_score_rejects_mismatched_row_counts():
    """X and y with different lengths raise ValueError."""
    model = _DecisionModel()
    adapter = PySADToScikitMultiflowAdapter(model)
    with pytest.raises(ValueError, match="same number of rows"):
        adapter.score(np.array([[1.0], [2.0]]), np.array([1]))


def test_predict_rejects_three_dimensional_X():
    """A three-dimensional X raises ValueError from predict."""
    model = _DecisionModel()
    adapter = PySADToScikitMultiflowAdapter(model)
    with pytest.raises(ValueError, match="one- or two-dimensional"):
        adapter.predict(np.array([[[1.0], [2.0]]]))


def test_score_rejects_three_dimensional_X():
    """A three-dimensional X raises ValueError from score."""
    model = _DecisionModel()
    adapter = PySADToScikitMultiflowAdapter(model)
    with pytest.raises(ValueError, match="one- or two-dimensional"):
        adapter.score(np.array([[[1.0], [2.0]]]), np.array([1, 0]))


def test_batch_partial_fit_matches_sequential_fit_partial():
    """One batch partial_fit call records the same calls as sequential single-row calls."""
    model1 = _RecordingModel()
    model2 = _RecordingModel()
    adapter1 = PySADToScikitMultiflowAdapter(model1)
    adapter2 = PySADToScikitMultiflowAdapter(model2)
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([0.0, 1.0, 0.0])
    adapter1.partial_fit(X, y)
    for xi, yi in zip(X, y, strict=True):
        adapter2.partial_fit(xi.reshape(1, -1), np.array([yi]))
    _calls_match(model1.calls, model2.calls)


def test_end_to_end_with_a_real_pysad_model():
    """A fitted RelativeEntropy model works end to end through the adapter."""
    from pysad.models import RelativeEntropy

    model = RelativeEntropy(min_val=0.0, max_val=10.0, window_size=8)
    adapter = PySADToScikitMultiflowAdapter(model)
    rng = np.random.RandomState(7)
    normal = rng.uniform(2.0, 4.0, size=(12, 1))
    anomalous = rng.uniform(7.0, 9.0, size=(12, 1))
    X = np.vstack([normal, anomalous])
    adapter.partial_fit(X, None)
    labels = adapter.predict(X)
    assert labels.shape == (24,)
    assert labels.dtype == np.int64
    assert set(np.unique(labels).tolist()) <= {0, 1}
    accuracy = adapter.score(X, labels)
    assert 0.0 <= accuracy <= 1.0
    with pytest.raises(ValueError, match="predict_proba is only available"):
        adapter.predict_proba(X)
