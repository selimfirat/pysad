"""Tests for PySADToCapyMoaAdapter, the zero-dependency CapyMoa adapter.

The adapter never imports CapyMoa; it duck-types CapyMoa's streaming detector interface
(``partial_fit``, ``score``, ``predict``, ``predict_proba`` and ``reset``) around any
PySAD model. The tests use small fake models and fake ``Instance`` objects instead of
CapyMoa itself, so they run without CapyMoa or Java.
"""

from __future__ import annotations

import numpy as np
import pytest

from pysad.integrations import PySADToCapyMoaAdapter


class _Instance:
    """A minimal CapyMoa-like instance exposing its features in ``x``."""

    def __init__(self, x) -> None:
        self.x = np.asarray(x, dtype=np.float64)


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
        value = float(np.asarray(X, dtype=np.float64).reshape(-1)[0])
        return 1 if value > 0.0 else 0


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
    adapter = PySADToCapyMoaAdapter(model)
    assert adapter.model is model


def test_partial_fit_learns_each_instance_in_order():
    """partial_fit calls fit_partial with each instance's x, in order, and returns the adapter."""
    model = _RecordingModel()
    adapter = PySADToCapyMoaAdapter(model)
    instances = [_Instance([1.0, 2.0]), _Instance([3.0, 4.0]), _Instance([5.0, 6.0])]
    expected = [
        (np.array([1.0, 2.0]), None),
        (np.array([3.0, 4.0]), None),
        (np.array([5.0, 6.0]), None),
    ]
    for instance in instances:
        adapter.partial_fit(instance)
    _calls_match(model.calls, expected)


def test_partial_fit_ignores_y():
    """An optional y is accepted for interface compatibility and ignored."""
    model = _RecordingModel()
    adapter = PySADToCapyMoaAdapter(model)
    adapter.partial_fit(_Instance([1.0, 2.0]), y=np.array([1]))
    _calls_match(model.calls, [(np.array([1.0, 2.0]), None)])


def test_partial_fit_accepts_a_raw_array():
    """A raw one-dimensional array is treated as the feature vector."""
    model = _RecordingModel()
    adapter = PySADToCapyMoaAdapter(model)
    adapter.partial_fit(np.array([1.0, 2.0, 3.0]))
    _calls_match(model.calls, [(np.array([1.0, 2.0, 3.0]), None)])


def test_score_returns_the_model_score():
    """score calls score_partial on the instance's x without learning."""
    model = _RecordingModel()
    adapter = PySADToCapyMoaAdapter(model)
    assert adapter.score(_Instance([1.0, 2.0])) == 0.0
    assert model.calls == []


def test_predict_returns_binary_labels():
    """predict maps each instance to 0 or 1 through predict_partial."""
    model = _DecisionModel()
    adapter = PySADToCapyMoaAdapter(model)
    instances = [_Instance([-1.0]), _Instance([0.5]), _Instance([2.0]), _Instance([-3.0])]
    labels = [adapter.predict(instance) for instance in instances]
    assert labels == [0, 1, 1, 0]


def test_predict_accepts_a_raw_array():
    """A raw one-dimensional array is treated as a single instance."""
    model = _DecisionModel()
    adapter = PySADToCapyMoaAdapter(model)
    label = adapter.predict(np.array([2.0]))
    assert label == 1


def test_predict_requires_a_decision_rule():
    """A score-only model raises TypeError from predict."""
    model = _RecordingModel()
    adapter = PySADToCapyMoaAdapter(model)
    with pytest.raises(TypeError, match="without an intrinsic decision rule"):
        adapter.predict(_Instance([1.0]))


def test_predict_proba_returns_normal_and_anomaly_probabilities():
    """predict_proba returns [P(normal), P(anomaly)] from the model score."""
    model = _ProbabilityModel()
    adapter = PySADToCapyMoaAdapter(model)
    proba = adapter.predict_proba(_Instance([0.2]))
    np.testing.assert_allclose(proba, np.array([0.8, 0.2]))
    assert proba.dtype == np.float64
    assert proba.shape == (2,)


def test_predict_proba_requires_a_probability_model():
    """A non-probability model raises ValueError from predict_proba."""
    model = _DecisionModel()
    adapter = PySADToCapyMoaAdapter(model)
    with pytest.raises(ValueError, match="predict_proba is only available"):
        adapter.predict_proba(_Instance([1.0]))


def test_reset_is_a_noop():
    """reset does nothing and leaves the wrapped model untouched."""
    model = _RecordingModel()
    adapter = PySADToCapyMoaAdapter(model)
    adapter.partial_fit(_Instance([1.0]))
    adapter.reset()
    assert len(model.calls) == 1


def test_rejects_multi_dimensional_features():
    """A feature vector with more than one dimension raises ValueError."""
    model = _DecisionModel()
    adapter = PySADToCapyMoaAdapter(model)
    with pytest.raises(ValueError, match="one-dimensional"):
        adapter.predict(_Instance([[1.0, 2.0]]))


def test_end_to_end_with_a_real_pysad_model():
    """A fitted RelativeEntropy model works end to end through the adapter."""
    from pysad.models import RelativeEntropy

    model = RelativeEntropy(min_val=0.0, max_val=10.0, window_size=8)
    adapter = PySADToCapyMoaAdapter(model)
    rng = np.random.RandomState(7)
    normal = rng.uniform(2.0, 4.0, size=12)
    anomalous = rng.uniform(7.0, 9.0, size=12)
    values = np.concatenate([normal, anomalous])
    for value in values:
        adapter.partial_fit(_Instance([value]))
    labels = [adapter.predict(_Instance([value])) for value in values]
    assert len(labels) == 24
    assert all(label in (0, 1) for label in labels)
    scores = [adapter.score(_Instance([value])) for value in values]
    assert all(isinstance(score, float) for score in scores)
    with pytest.raises(ValueError, match="predict_proba is only available"):
        adapter.predict_proba(_Instance([5.0]))
