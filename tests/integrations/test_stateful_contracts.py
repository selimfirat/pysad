"""Hypothesis stateful property tests for streaming detector contracts in pysad.integrations.

These tests use Hypothesis's RuleBasedStateMachine to model the state transitions of
streaming anomaly detectors in pysad.integrations, covering:

- Warmup/grace/probationary period behavior (models return specific values during warmup)
- Single-instance equivalence to batch (sequential fit_partial/score_partial matches batch)
- State reset (new instance with same params + same data = same scores)
- Score state invariance (score_partial does not mutate model state)
- Input immutability (fit_partial/score_partial do not mutate input arrays)

Each integration model in pysad.integrations.__all__ that implements fit_partial/score_partial
is tested against these contracts.
"""

import copy
import functools
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from hypothesis import HealthCheck, settings
from hypothesis.stateful import RuleBasedStateMachine, initialize, rule

from pysad.utils import fix_seed

SEED = 61
MAX_FEATURES = 4
# Small number of examples for CI; raise locally for more coverage.
MAX_EXAMPLES = 10
STATEFUL_SETTINGS = settings(
    max_examples=MAX_EXAMPLES,
    deadline=None,
    database=None,
    suppress_health_check=[HealthCheck.too_slow],
    derandomize=True,
)

# Values drawn for stream instances. Small pools make constant features and duplicates likely.
FINITE_VALUES = [
    0.0,
    1.0,
    -1.0,
    0.5,
    -0.5,
    2.0,
    -2.0,
    10.0,
    -10.0,
    100.0,
    -100.0,
]


def _create_one_fit_model(num_features: int = 3, seed: int = 0):
    """Create a OneFitModel with a simple batch model for testing."""
    from pyod.models.iforest import IForest

    rng = np.random.RandomState(seed)
    initial_X = rng.rand(12, num_features)
    return {
        "model_cls": IForest,
        "initial_X": initial_X,
        "n_estimators": 10,
        "random_state": seed,
    }


def _create_reference_window_model(num_features: int = 3, seed: int = 0):
    """Create a ReferenceWindowModel with a simple batch model for testing."""
    from pyod.models.iforest import IForest

    rng = np.random.RandomState(seed)
    initial_X = rng.rand(12, num_features)
    return {
        "model_cls": IForest,
        "window_size": 8,
        "sliding_size": 4,
        "initial_window_X": initial_X[:8],
        "n_estimators": 10,
        "random_state": seed,
    }


def _create_predictive_anomaly_detector(*args, **kwargs):
    """Create a PredictiveAnomalyDetector with a simple forecaster for testing."""
    try:
        from river import time_series

        return {
            "model": time_series.SNARIMAX(p=2, d=0, q=1),
            "relative": False,
            "probationary_period": 2,
        }
    except ImportError:
        pytest.skip("River not installed")


class _SimpleForecaster:
    """Simple forecaster implementing ForecasterProtocol for testing."""

    def __init__(self, window: int = 5):
        self.window = window
        self.history = []

    def forecast(self, horizon: int = 1) -> list[float]:
        if not self.history:
            return [0.0]
        return [sum(self.history[-self.window :]) / min(len(self.history), self.window)]

    def learn_one(self, y: float) -> None:
        self.history.append(y)


def _create_forecasting_anomaly_detector(*args, **kwargs):
    """Create a ForecastingAnomalyDetector with a simple forecaster for testing."""
    return {
        "forecaster": _SimpleForecaster(window=5),
        "relative": False,
        "probationary_period": 2,
    }


def _get_warmup_length(name: str, params: dict[str, Any], is_fit_then_score: bool = False) -> int:
    """Get the actual warmup length for an integration model from its parameters."""
    if name in ("PredictiveAnomalyDetector", "ForecastingAnomalyDetector"):
        warmup = params.get("probationary_period", 0)
        if is_fit_then_score:
            # In fit_then_score mode, fit_partial increments num_values_ before score_partial checks it,
            # so warmup is effectively one step shorter (like DAMP in models)
            return max(0, warmup - 1)
        return warmup
    # OneFitModel and ReferenceWindowModel don't have constant warmup periods
    return 0


def _has_constant_warmup(name: str, params: dict[str, Any]) -> bool:
    """Check if a model returns a constant score during its entire warmup period."""
    if name in ("PredictiveAnomalyDetector", "ForecastingAnomalyDetector"):
        return params.get("probationary_period", 0) > 0
    return False


def _warmup_score(name: str, params: dict[str, Any] | None = None) -> float:
    """Expected score during warmup period for a model with constant warmup."""
    if name in ("PredictiveAnomalyDetector", "ForecastingAnomalyDetector"):
        return 0.0
    return 0.0


INTEGRATION_MODELS = [
    "OneFitModel",
    "ReferenceWindowModel",
    "PredictiveAnomalyDetector",
    "ForecastingAnomalyDetector",
]

# PredictiveAnomalyDetector and ForecastingAnomalyDetector are univariate, others use 3 features by default
MODEL_NUM_FEATURES = {
    "OneFitModel": 3,
    "ReferenceWindowModel": 3,
    "PredictiveAnomalyDetector": 1,
    "ForecastingAnomalyDetector": 1,
}


class _FitScorePartialMachine(RuleBasedStateMachine):
    """State machine for testing fit_score_partial (prequential) streaming contracts."""

    # Class attributes set by _create_configured_machine
    model_name: str
    params_factory: Callable[[], dict[str, Any]]
    num_features: int
    warmup_length: int
    has_warmup: bool
    _expected_warmup_score: float

    # Instance attributes initialized in setup
    model = None
    step = 0
    scores = None
    all_X = None
    all_y = None
    in_warmup = True

    @initialize()
    def setup(self):
        fix_seed(SEED)
        params = self.params_factory()
        model_cls = getattr(
            __import__("pysad.integrations", fromlist=[self.model_name]), self.model_name
        )
        self.model = model_cls(**copy.deepcopy(params))
        self.step = 0
        self.scores = []
        self.all_X = []
        self.all_y = []
        self.in_warmup = self.has_warmup and self.warmup_length > 0

    def _draw_instance(self) -> tuple[np.ndarray, int | None]:
        rng = np.random.RandomState(SEED + self.step * 17 + hash(self.model_name) % 1000)
        xi = rng.choice(FINITE_VALUES, size=self.num_features).astype(np.float64)
        # ReferenceWindowModel with initial_window_y=None fails when y is provided
        # because reference_window_y is None and concatenation fails.
        # Pass y=None for ReferenceWindowModel to avoid this issue.
        if self.model_name == "ReferenceWindowModel":
            return xi, None
        yi = rng.randint(0, 2)
        return xi, yi

    @rule()
    def step_fit_score_partial(self):
        xi, yi = self._draw_instance()
        self.all_X.append(xi.copy())
        self.all_y.append(yi)

        score = self.model.fit_score_partial(xi, yi)
        assert isinstance(score, float), f"Expected float, got {type(score)}"
        assert np.isfinite(score), f"Non-finite score: {score}"
        self.scores.append(score)
        self.step += 1

        if self.in_warmup and self.step >= self.warmup_length:
            self.in_warmup = False

    @rule()
    def check_warmup_scores(self):
        if not self.has_warmup or not self.scores:
            return
        warmup_scores = self.scores[: self.warmup_length]
        for i, score in enumerate(warmup_scores):
            assert score == self._expected_warmup_score, (
                f"{self.model_name}: warmup score at step {i} is {score}, "
                f"expected {self._expected_warmup_score}"
            )

    @rule()
    def check_batch_equivalence(self):
        if not self.all_X:
            return
        X_arr = np.array(self.all_X)
        y_arr = np.array(self.all_y) if self.all_y else None

        fix_seed(SEED)
        params = self.params_factory()
        model_cls = getattr(
            __import__("pysad.integrations", fromlist=[self.model_name]), self.model_name
        )
        batch_model = model_cls(**copy.deepcopy(params))
        batch_scores = batch_model.fit_score(X_arr, y_arr)

        np.testing.assert_allclose(
            batch_scores,
            np.array(self.scores),
            rtol=1e-10,
            atol=1e-12,
            err_msg=f"{self.model_name}: batch fit_score != sequential fit_score_partial",
        )

    @rule()
    def check_state_reset_equivalence(self):
        if not self.all_X:
            return
        X_arr = np.array(self.all_X)
        y_arr = np.array(self.all_y) if self.all_y else None

        fix_seed(SEED)
        params = self.params_factory()
        model_cls = getattr(
            __import__("pysad.integrations", fromlist=[self.model_name]), self.model_name
        )
        reset_model = model_cls(**copy.deepcopy(params))
        reset_scores = reset_model.fit_score(X_arr, y_arr)

        np.testing.assert_allclose(
            reset_scores,
            np.array(self.scores),
            rtol=1e-10,
            atol=1e-12,
            err_msg=f"{self.model_name}: state reset scores != original sequential scores",
        )

    @rule()
    def check_input_immutability(self):
        if not self.all_X:
            return
        original_X = [x.copy() for x in self.all_X]
        original_y = self.all_y.copy() if self.all_y else None

        fix_seed(SEED)
        params = self.params_factory()
        model_cls = getattr(
            __import__("pysad.integrations", fromlist=[self.model_name]), self.model_name
        )
        test_model = model_cls(**copy.deepcopy(params))
        for xi, yi in zip(self.all_X, self.all_y, strict=True):
            test_model.fit_score_partial(xi, yi)

        for i, (orig, curr) in enumerate(zip(original_X, self.all_X, strict=False)):
            np.testing.assert_array_equal(orig, curr, err_msg=f"Input X[{i}] was mutated")
        if original_y is not None:
            np.testing.assert_array_equal(original_y, self.all_y, err_msg="Input y was mutated")

    @rule()
    def check_score_state_invariance(self):
        """Multiple score_partial calls on the same instance return the same value (idempotence)."""
        if not self.all_X:
            return
        # Pick a few instances to test (first, middle, last)
        indices = [0]
        if len(self.all_X) > 2:
            indices.append(len(self.all_X) // 2)
        indices.append(len(self.all_X) - 1)

        for idx in indices:
            xi = self.all_X[idx]
            # Score multiple times
            score1 = self.model.score_partial(xi)
            score2 = self.model.score_partial(xi)
            score3 = self.model.score_partial(xi)
            assert score1 == score2 == score3, (
                f"{self.model_name}: score_partial not idempotent on instance {idx}: "
                f"{score1}, {score2}, {score3}"
            )


class _FitThenScoreMachine(RuleBasedStateMachine):
    """State machine for testing fit_partial + score_partial streaming contracts."""

    # Class attributes set by _create_configured_machine
    model_name: str
    params_factory: Callable[[], dict[str, Any]]
    num_features: int
    warmup_length: int
    has_warmup: bool
    _expected_warmup_score: float

    # Instance attributes initialized in setup
    model = None
    step = 0
    fit_scores = None
    all_X = None
    all_y = None
    in_warmup = True

    @initialize()
    def setup(self):
        fix_seed(SEED)
        params = self.params_factory()
        model_cls = getattr(
            __import__("pysad.integrations", fromlist=[self.model_name]), self.model_name
        )
        self.model = model_cls(**copy.deepcopy(params))
        self.step = 0
        self.fit_scores = []
        self.all_X = []
        self.all_y = []
        self.in_warmup = self.has_warmup and self.warmup_length > 0

    def _draw_instance(self) -> tuple[np.ndarray, int | None]:
        rng = np.random.RandomState(SEED + self.step * 17 + hash(self.model_name) % 1000)
        xi = rng.choice(FINITE_VALUES, size=self.num_features).astype(np.float64)
        # ReferenceWindowModel with initial_window_y=None fails when y is provided
        # because reference_window_y is None and concatenation fails.
        # Pass y=None for ReferenceWindowModel to avoid this issue.
        if self.model_name == "ReferenceWindowModel":
            return xi, None
        yi = rng.randint(0, 2)
        return xi, yi

    @rule()
    def step_fit_then_score(self):
        xi, yi = self._draw_instance()
        self.all_X.append(xi.copy())
        self.all_y.append(yi)

        self.model.fit_partial(xi, yi)
        score = self.model.score_partial(xi)
        assert isinstance(score, float), f"Expected float, got {type(score)}"
        assert np.isfinite(score), f"Non-finite score: {score}"
        self.fit_scores.append(score)
        self.step += 1

        if self.in_warmup and self.step >= self.warmup_length:
            self.in_warmup = False

    @rule()
    def check_warmup_scores(self):
        if not self.has_warmup or not self.fit_scores:
            return
        warmup_scores = self.fit_scores[: self.warmup_length]
        for i, score in enumerate(warmup_scores):
            assert score == self._expected_warmup_score, (
                f"{self.model_name}: warmup score at step {i} is {score}, "
                f"expected {self._expected_warmup_score}"
            )

    @rule()
    def check_input_immutability(self):
        if not self.all_X:
            return
        original_X = [x.copy() for x in self.all_X]
        original_y = self.all_y.copy() if self.all_y else None

        fix_seed(SEED)
        params = self.params_factory()
        model_cls = getattr(
            __import__("pysad.integrations", fromlist=[self.model_name]), self.model_name
        )
        test_model = model_cls(**copy.deepcopy(params))
        for xi, yi in zip(self.all_X, self.all_y, strict=True):
            test_model.fit_partial(xi, yi)
            test_model.score_partial(xi)

        for i, (orig, curr) in enumerate(zip(original_X, self.all_X, strict=False)):
            np.testing.assert_array_equal(orig, curr, err_msg=f"Input X[{i}] was mutated")
        if original_y is not None:
            np.testing.assert_array_equal(original_y, self.all_y, err_msg="Input y was mutated")

    @rule()
    def check_score_state_invariance(self):
        """Multiple score_partial calls on the same instance return the same value (idempotence)."""
        if not self.all_X:
            return
        # Pick a few instances to test (first, middle, last)
        indices = [0]
        if len(self.all_X) > 2:
            indices.append(len(self.all_X) // 2)
        indices.append(len(self.all_X) - 1)

        for idx in indices:
            xi = self.all_X[idx]
            # Score multiple times
            score1 = self.model.score_partial(xi)
            score2 = self.model.score_partial(xi)
            score3 = self.model.score_partial(xi)
            assert score1 == score2 == score3, (
                f"{self.model_name}: score_partial not idempotent on instance {idx}: "
                f"{score1}, {score2}, {score3}"
            )


def _create_configured_machine(
    base_machine_cls,
    model_name: str,
    params_factory: Callable[[], dict[str, Any]],
    num_features: int,
    is_fit_then_score: bool = False,
):
    """Create a RuleBasedStateMachine subclass configured for a specific integration model.

    Returns the TestCase class for pytest to collect.
    """
    warmup_length = _get_warmup_length(model_name, params_factory(), is_fit_then_score)
    has_warmup = _has_constant_warmup(model_name, params_factory())
    expected_warmup_score = _warmup_score(model_name, params_factory())

    # Create class dynamically with the configuration as class attributes
    # Wrap params_factory in staticmethod to prevent functools.partial from binding
    # self in Python 3.14+ (where partial becomes a method descriptor)
    class_attrs = {
        "model_name": model_name,
        "params_factory": staticmethod(params_factory),
        "num_features": num_features,
        "warmup_length": warmup_length,
        "has_warmup": has_warmup,
        "_expected_warmup_score": expected_warmup_score,
    }

    ConfiguredMachine = type(
        f"_{model_name}{base_machine_cls.__name__}", (base_machine_cls,), class_attrs
    )

    # Apply settings
    ConfiguredMachine = STATEFUL_SETTINGS(ConfiguredMachine)

    # Get the TestCase class for pytest
    TestCase = ConfiguredMachine.TestCase
    TestCase.__name__ = f"Test{model_name}{base_machine_cls.__name__}"
    TestCase.__qualname__ = f"Test{model_name}{base_machine_cls.__name__}"

    return TestCase


# Generate test case classes for each integration model
for _name in INTEGRATION_MODELS:
    _num_features = MODEL_NUM_FEATURES.get(_name, 3)
    if _name == "OneFitModel":
        _params_factory = functools.partial(_create_one_fit_model, _num_features, SEED)
    elif _name == "ReferenceWindowModel":
        _params_factory = functools.partial(_create_reference_window_model, _num_features, SEED)
    elif _name == "PredictiveAnomalyDetector":
        _params_factory = _create_predictive_anomaly_detector
    elif _name == "ForecastingAnomalyDetector":
        _params_factory = _create_forecasting_anomaly_detector
    else:
        continue

    # FitScorePartial machine
    _fit_score_testcase = _create_configured_machine(
        _FitScorePartialMachine, _name, _params_factory, _num_features, is_fit_then_score=False
    )
    globals()[f"Test{_name}FitScorePartial"] = _fit_score_testcase

    # FitThenScore machine
    _fit_then_score_testcase = _create_configured_machine(
        _FitThenScoreMachine, _name, _params_factory, _num_features, is_fit_then_score=True
    )
    globals()[f"Test{_name}FitThenScore"] = _fit_then_score_testcase


# Additional focused tests for specific contract properties (non-stateful, simpler)


@pytest.mark.parametrize("model_name", INTEGRATION_MODELS)
def test_constant_warmup_period_behavior(model_name):
    """Test that models with constant warmup periods return expected scores during warmup."""
    num_features = MODEL_NUM_FEATURES.get(model_name, 3)
    if model_name == "OneFitModel":
        params = _create_one_fit_model(num_features, SEED)
    elif model_name == "ReferenceWindowModel":
        params = _create_reference_window_model(num_features, SEED)
    elif model_name == "PredictiveAnomalyDetector":
        params = _create_predictive_anomaly_detector()
    elif model_name == "ForecastingAnomalyDetector":
        params = _create_forecasting_anomaly_detector()
    else:
        pytest.skip(f"Unknown model: {model_name}")

    if not _has_constant_warmup(model_name, params):
        pytest.skip(f"{model_name} has no constant warmup period")

    warmup_length = _get_warmup_length(model_name, params)
    if warmup_length == 0:
        pytest.skip(f"{model_name} warmup length is 0")

    fix_seed(SEED)
    model_cls = getattr(__import__("pysad.integrations", fromlist=[model_name]), model_name)
    model = model_cls(**copy.deepcopy(params))

    expected_score = _warmup_score(model_name, params)

    rng = np.random.RandomState(SEED)
    for i in range(warmup_length + 2):
        xi = rng.choice(FINITE_VALUES, size=num_features).astype(np.float64)
        yi = rng.randint(0, 2)
        score = model.fit_score_partial(xi, yi)
        if i < warmup_length:
            assert score == expected_score, (
                f"{model_name}: warmup score at step {i} is {score}, expected {expected_score}"
            )


@pytest.mark.parametrize("model_name", INTEGRATION_MODELS)
def test_sequential_vs_batch_equivalence(model_name):
    """Test that sequential fit_score_partial matches batch fit_score."""
    num_features = MODEL_NUM_FEATURES.get(model_name, 3)
    if model_name == "OneFitModel":
        params = _create_one_fit_model(num_features, SEED)
    elif model_name == "ReferenceWindowModel":
        params = _create_reference_window_model(num_features, SEED)
    elif model_name == "PredictiveAnomalyDetector":
        params = _create_predictive_anomaly_detector()
    elif model_name == "ForecastingAnomalyDetector":
        params = _create_forecasting_anomaly_detector()
    else:
        pytest.skip(f"Unknown model: {model_name}")

    fix_seed(SEED)
    rng = np.random.RandomState(SEED + 1)
    X = rng.choice(FINITE_VALUES, size=(20, num_features)).astype(np.float64)
    # ReferenceWindowModel with initial_window_y=None fails when y is provided
    # because reference_window_y is None and concatenation fails.
    # Pass y=None for ReferenceWindowModel to avoid this issue.
    y = None if model_name == "ReferenceWindowModel" else rng.randint(0, 2, size=20)

    fix_seed(SEED)
    model_cls = getattr(__import__("pysad.integrations", fromlist=[model_name]), model_name)
    seq_model = model_cls(**copy.deepcopy(params))
    if y is None:
        seq_scores = [seq_model.fit_score_partial(xi) for xi in X]
    else:
        seq_scores = [seq_model.fit_score_partial(xi, yi) for xi, yi in zip(X, y, strict=True)]

    fix_seed(SEED)
    batch_model = model_cls(**copy.deepcopy(params))
    batch_scores = batch_model.fit_score(X, y)

    np.testing.assert_allclose(
        batch_scores,
        np.array(seq_scores),
        rtol=1e-10,
        atol=1e-12,
        err_msg=f"{model_name}: sequential fit_score_partial != batch fit_score",
    )


@pytest.mark.parametrize("model_name", INTEGRATION_MODELS)
def test_fit_then_score_vs_batch_equivalence(model_name):
    """Test that sequential fit_partial + score_partial matches batch fit().score()."""
    num_features = MODEL_NUM_FEATURES.get(model_name, 3)
    if model_name == "OneFitModel":
        params = _create_one_fit_model(num_features, SEED)
    elif model_name == "ReferenceWindowModel":
        params = _create_reference_window_model(num_features, SEED)
    elif model_name == "PredictiveAnomalyDetector":
        params = _create_predictive_anomaly_detector()
    elif model_name == "ForecastingAnomalyDetector":
        params = _create_forecasting_anomaly_detector()
    else:
        pytest.skip(f"Unknown model: {model_name}")

    fix_seed(SEED)
    rng = np.random.RandomState(SEED + 1)
    X = rng.choice(FINITE_VALUES, size=(20, num_features)).astype(np.float64)
    # ReferenceWindowModel with initial_window_y=None fails when y is provided
    # because reference_window_y is None and concatenation fails.
    # Pass y=None for ReferenceWindowModel to avoid this issue.
    y = None if model_name == "ReferenceWindowModel" else rng.randint(0, 2, size=20)

    fix_seed(SEED)
    model_cls = getattr(__import__("pysad.integrations", fromlist=[model_name]), model_name)
    seq_model = model_cls(**copy.deepcopy(params))
    if y is None:
        for xi in X:
            seq_model.fit_partial(xi)
    else:
        for xi, yi in zip(X, y, strict=True):
            seq_model.fit_partial(xi, yi)
    seq_scores = [seq_model.score_partial(xi) for xi in X]

    fix_seed(SEED)
    batch_model = model_cls(**copy.deepcopy(params))
    batch_model.fit(X, y)
    batch_scores = batch_model.score(X)

    np.testing.assert_allclose(
        batch_scores,
        np.array(seq_scores),
        rtol=1e-10,
        atol=1e-12,
        err_msg=f"{model_name}: sequential fit_partial+score_partial != batch fit().score()",
    )


@pytest.mark.parametrize("model_name", INTEGRATION_MODELS)
def test_state_reset_equivalence(model_name):
    """Test that a new model instance with same params gives same scores (state reset)."""
    num_features = MODEL_NUM_FEATURES.get(model_name, 3)
    if model_name == "OneFitModel":
        params = _create_one_fit_model(num_features, SEED)
    elif model_name == "ReferenceWindowModel":
        params = _create_reference_window_model(num_features, SEED)
    elif model_name == "PredictiveAnomalyDetector":
        params = _create_predictive_anomaly_detector()
    elif model_name == "ForecastingAnomalyDetector":
        params = _create_forecasting_anomaly_detector()
    else:
        pytest.skip(f"Unknown model: {model_name}")

    fix_seed(SEED)
    rng = np.random.RandomState(SEED + 1)
    X = rng.choice(FINITE_VALUES, size=(20, num_features)).astype(np.float64)
    # ReferenceWindowModel with initial_window_y=None fails when y is provided
    # because reference_window_y is None and concatenation fails.
    # Pass y=None for ReferenceWindowModel to avoid this issue.
    y = None if model_name == "ReferenceWindowModel" else rng.randint(0, 2, size=20)

    fix_seed(SEED)
    model_cls = getattr(__import__("pysad.integrations", fromlist=[model_name]), model_name)
    model1 = model_cls(**copy.deepcopy(params))
    if y is None:
        scores1 = [model1.fit_score_partial(xi) for xi in X]
    else:
        scores1 = [model1.fit_score_partial(xi, yi) for xi, yi in zip(X, y, strict=True)]

    fix_seed(SEED)
    model2 = model_cls(**copy.deepcopy(params))
    scores2 = model2.fit_score(X, y)

    np.testing.assert_allclose(
        scores2,
        np.array(scores1),
        rtol=1e-10,
        atol=1e-12,
        err_msg=f"{model_name}: state reset scores != original sequential scores",
    )


# Focused tests for TSFMResidualAdapter


def test_tsfm_residual_adapter_returns_float():
    """transform_partial returns a float."""
    from pysad.integrations import TSFMResidualAdapter

    adapter = TSFMResidualAdapter(window_size=10, min_std=1e-8)
    result = adapter.fit_partial(1.0).transform_partial(0.5)
    assert isinstance(result, float)


def test_tsfm_residual_adapter_normalizes_by_rolling_mean_and_std():
    """transform_partial normalizes by rolling mean and std after enough data."""
    from pysad.integrations import TSFMResidualAdapter

    adapter = TSFMResidualAdapter(window_size=5, min_std=1e-8)
    # Feed 5 values to fill the window: [1.0, 2.0, 3.0, 4.0, 5.0]
    # Mean = 3.0, Std = sqrt(2.0) = 1.414...
    for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
        adapter.fit_partial(v)
    # Next value 6.0 -> (6.0 - 3.0) / 1.414... = 2.121...
    result = adapter.transform_partial(6.0)
    expected = (6.0 - 3.0) / np.sqrt(2.0)
    assert np.isclose(result, expected)


def test_tsfm_residual_adapter_returns_zero_before_enough_data():
    """transform_partial returns 0.0 before at least 2 values are seen."""
    from pysad.integrations import TSFMResidualAdapter

    adapter = TSFMResidualAdapter(window_size=5, min_std=1e-8)
    # First value: num_seen_ = 1
    result1 = adapter.fit_partial(1.0).transform_partial(1.0)
    assert result1 == 0.0
    # Second value: num_seen_ = 2, variance = 0 (all same), min_std prevents division
    result2 = adapter.fit_partial(1.0).transform_partial(1.0)
    assert result2 == 0.0
    # Third value: now we have variance
    result3 = adapter.fit_partial(2.0).transform_partial(2.0)
    assert result3 != 0.0


def test_tsfm_residual_adapter_raises_on_invalid_window_size():
    """ValueError raised when window_size < 2."""
    from pysad.integrations import TSFMResidualAdapter

    try:
        TSFMResidualAdapter(window_size=1)
    except ValueError as e:
        assert "window_size must be at least 2" in str(e)
    else:
        raise AssertionError("Expected ValueError for window_size=1")

    try:
        TSFMResidualAdapter(window_size=0)
    except ValueError as e:
        assert "window_size must be at least 2" in str(e)
    else:
        raise AssertionError("Expected ValueError for window_size=0")


def test_tsfm_residual_adapter_raises_on_non_positive_min_std():
    """ValueError raised when min_std <= 0."""
    from pysad.integrations import TSFMResidualAdapter

    try:
        TSFMResidualAdapter(window_size=10, min_std=0.0)
    except ValueError as e:
        assert "min_std must be positive" in str(e)
    else:
        raise AssertionError("Expected ValueError for min_std=0.0")

    try:
        TSFMResidualAdapter(window_size=10, min_std=-1e-8)
    except ValueError as e:
        assert "min_std must be positive" in str(e)
    else:
        raise AssertionError("Expected ValueError for min_std=-1e-8")


def test_tsfm_residual_adapter_batch_transform_matches_sequential():
    """Batch transform equals sequential transform_partial with frozen statistics."""
    from pysad.integrations import TSFMResidualAdapter

    adapter = TSFMResidualAdapter(window_size=5, min_std=1e-8)
    for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
        adapter.fit_partial(v)
    # Frozen statistics: mean 3.0, population variance 2.0
    batch = adapter.transform(np.array([6.0, 7.0]))
    expected = np.array([(6.0 - 3.0) / np.sqrt(2.0), (7.0 - 3.0) / np.sqrt(2.0)])
    np.testing.assert_allclose(batch, expected)
    # Sequential calls use the same frozen statistics and give the same values.
    for value, expected_value in zip([6.0, 7.0], expected, strict=True):
        assert adapter.transform_partial(value) == expected_value


def test_tsfm_residual_adapter_fit_transform_partial():
    """fit_transform_partial updates the statistics with the value, then transforms it."""
    from pysad.integrations import TSFMResidualAdapter

    adapter = TSFMResidualAdapter(window_size=5, min_std=1e-8)
    # Not enough data yet: num_seen_ is 1.
    assert adapter.fit_transform_partial(1.0) == 0.0
    # Zero variance: all values equal so far.
    assert adapter.fit_transform_partial(1.0) == 0.0
    # Statistics now cover [1.0, 1.0, 2.0].
    result = adapter.fit_transform_partial(2.0)
    values = np.array([1.0, 1.0, 2.0])
    mean = values.mean()
    variance = values.var()
    expected = (2.0 - mean) / np.sqrt(variance)
    assert np.isclose(result, expected)


def test_tsfm_residual_adapter_transform_before_enough_data_returns_zeros():
    """transform returns zeros before at least two values are seen."""
    from pysad.integrations import TSFMResidualAdapter

    adapter = TSFMResidualAdapter(window_size=5, min_std=1e-8)
    adapter.fit_partial(1.0)
    np.testing.assert_array_equal(
        adapter.transform(np.array([1.0, 2.0])), np.zeros(2, dtype=np.float64)
    )


def test_tsfm_residual_adapter_transform_empty_returns_empty():
    """transform of an empty array returns an empty float64 array."""
    from pysad.integrations import TSFMResidualAdapter

    adapter = TSFMResidualAdapter(window_size=5, min_std=1e-8)
    result = adapter.transform(np.array([], dtype=np.float64))
    assert result.shape == (0,)
    assert result.dtype == np.float64


def test_tsfm_residual_adapter_transform_low_variance_returns_zeros():
    """transform returns zeros when the rolling variance is below min_std squared."""
    from pysad.integrations import TSFMResidualAdapter

    adapter = TSFMResidualAdapter(window_size=5, min_std=1.0)
    adapter.fit_partial(1.0)
    adapter.fit_partial(1.0)
    # Variance 0.0 is below min_std ** 2 = 1.0.
    np.testing.assert_array_equal(
        adapter.transform(np.array([1.0, 2.0])), np.zeros(2, dtype=np.float64)
    )
