"""Hypothesis stateful property tests for streaming detector contracts.

These tests use Hypothesis's RuleBasedStateMachine to model the state transitions of
streaming anomaly detectors, covering:

- Warmup/grace/probationary period behavior (models return specific values during warmup)
- Single-instance equivalence to batch (sequential fit_partial/score_partial matches batch)
- State reset (new instance with same params + same data = same scores)
- Score state invariance (score_partial does not mutate model state)
- Predict contracts for decision-capable models (predict_partial, fit_predict_partial, predict, fit_predict)

Each model in pysad.models.__all__ is tested against these contracts.
"""

import copy
from typing import Any

import numpy as np
import pytest
from hypothesis import HealthCheck, settings
from hypothesis.stateful import RuleBasedStateMachine, initialize, rule

import pysad.models
from pysad.utils import fix_seed
from tests.models.conftest import (
    EDGE_MODELS,
    TIMESTAMPED_MODELS,
    UNIVARIATE_MODELS,
    active_model_names,
    short_stream_params,
)

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
FINITE_VALUES = [0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 10.0, -10.0, 100.0, -100.0]


def _model_input_kind(name: str) -> str:
    return str(getattr(pysad.models, name)._effective_input_kind())


def _num_features_for(name: str, default: int = 3) -> int:
    if name in TIMESTAMPED_MODELS or name in EDGE_MODELS:
        return 2
    return 1 if name in UNIVARIATE_MODELS else default


# Models that return a CONSTANT score during their entire warmup period.
# These are models with explicit grace/probationary periods where scoring is suppressed.
CONSTANT_WARMUP_MODELS = {
    "KitNET": {"grace_feature_mapping", "grace_anomaly_detector"},
    "HTM": {"probationary_period"},
    "KNNCAD": {"probationary_period"},
    "DAMP": {"probationary_period"},
    "FluxEV": {"window_size", "period", "num_periods", "drift"},
}


def _has_constant_warmup(model_name: str, params: dict[str, Any]) -> bool:
    """Check if a model returns a constant score during its entire warmup period."""
    if model_name not in CONSTANT_WARMUP_MODELS:
        return False
    return bool(CONSTANT_WARMUP_MODELS[model_name] & set(params.keys()))


def _get_warmup_length(model_name: str, params: dict[str, Any]) -> int:
    """Get the actual warmup length for a model from its parameters."""
    if model_name == "KitNET":
        return params.get("grace_feature_mapping", 0) + params.get("grace_anomaly_detector", 0)
    if model_name == "HTM":
        return params.get("probationary_period", 0)
    if model_name == "KNNCAD":
        # KNNCAD returns 0.0 for the first (probationary_period - 1) instances.
        # The probationary_period-th instance gets the first real score.
        return max(0, params.get("probationary_period", 0) - 1)
    if model_name == "DAMP":
        return params.get("probationary_period", 0)
    if model_name == "FluxEV":
        ws = params.get("window_size", 10)
        period = params.get("period")
        num_periods = params.get("num_periods", 7)
        drift = params.get("drift", 2)
        warmup = 2 * ws
        if period is not None:
            warmup += period * (num_periods - 1) + drift
        return warmup
    return 0


def _test_safe_params(model_name: str, params: dict[str, Any]) -> dict[str, Any]:
    """Return params with test-safe overrides for specific models."""
    overrides = {
        "MStream": {"log_transform": False},
        "Inqmad": {"dim_x": 8, "initial_size": 8},  # smaller for testing
    }
    result = copy.deepcopy(params)
    if model_name in overrides:
        result.update(overrides[model_name])
    return result


def _warmup_score(model_name: str, params: dict[str, Any] | None = None) -> float:
    """Expected score during warmup period for a model with constant warmup."""
    if model_name == "HTM":
        # HTM returns anomaly likelihood (0.5) during probationary period when likelihood=True (default)
        # When likelihood=False, it returns raw anomaly score which is not constant
        likelihood = params.get("likelihood", True) if params else True
        return 0.5 if likelihood else 0.0
    if model_name in ("KitNET", "KNNCAD", "DAMP", "FluxEV"):
        return 0.0
    return 0.0


def _is_decision_capable(model_name: str) -> bool:
    """Check if a model supports predict_partial (has decision output or intrinsic decision rule)."""
    cls = getattr(pysad.models, model_name)
    return cls._output_kind == "decision" or cls._has_decision_rule


class _FitScorePartialMachine(RuleBasedStateMachine):
    """State machine for testing fit_score_partial (prequential) streaming contracts."""

    # Class attributes set by _create_configured_machine
    model_name: str
    params: dict[str, Any]
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
        model_cls = getattr(pysad.models, self.model_name)
        self.model = model_cls(**copy.deepcopy(self.params))
        self.step = 0
        self.scores = []
        self.all_X = []
        self.all_y = []
        self.in_warmup = self.has_warmup and self.warmup_length > 0

    def _draw_instance(self) -> tuple[np.ndarray, int]:
        rng = np.random.RandomState(SEED + self.step * 17 + hash(self.model_name) % 1000)
        xi = rng.choice(FINITE_VALUES, size=self.num_features).astype(np.float64)
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
        model_cls = getattr(pysad.models, self.model_name)
        batch_model = model_cls(**copy.deepcopy(self.params))
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
        model_cls = getattr(pysad.models, self.model_name)
        reset_model = model_cls(**copy.deepcopy(self.params))
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
        model_cls = getattr(pysad.models, self.model_name)
        test_model = model_cls(**copy.deepcopy(self.params))
        for xi, yi in zip(self.all_X, self.all_y, strict=True):
            test_model.fit_score_partial(xi, yi)

        for i, (orig, curr) in enumerate(zip(original_X, self.all_X, strict=False)):
            np.testing.assert_array_equal(orig, curr, err_msg=f"Input X[{i}] was mutated")
        if original_y is not None:
            np.testing.assert_array_equal(original_y, self.all_y, err_msg="Input y was mutated")

    @rule()
    def check_score_state_invariance(self):
        """Multiple score_partial calls on the same instance return the same value (idempotence)."""
        if not self.all_X or self.model_name in NON_IDEMPOTENT_SCORE_MODELS:
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
    params: dict[str, Any]
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
        model_cls = getattr(pysad.models, self.model_name)
        self.model = model_cls(**copy.deepcopy(self.params))
        self.step = 0
        self.fit_scores = []
        self.all_X = []
        self.all_y = []
        self.in_warmup = self.has_warmup and self.warmup_length > 0

    def _draw_instance(self) -> tuple[np.ndarray, int]:
        rng = np.random.RandomState(SEED + self.step * 17 + hash(self.model_name) % 1000)
        xi = rng.choice(FINITE_VALUES, size=self.num_features).astype(np.float64)
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
        model_cls = getattr(pysad.models, self.model_name)
        test_model = model_cls(**copy.deepcopy(self.params))
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
        if not self.all_X or self.model_name in NON_IDEMPOTENT_SCORE_MODELS:
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


# Known failures that are tracked as strict xfails
KNOWN_STATEFUL_FAILURES: dict[str, str] = {}

# Models to skip entirely from stateful testing
SKIPPED_STATEFUL: dict[str, str] = {}

# Models to skip from predict stateful testing (e.g., models with known predict issues)
SKIPPED_PREDICT_STATEFUL: dict[str, str] = {}

# Models with known non-idempotent score_partial (by design or bug)
# RandomModel returns random scores by design.
# RobustRandomCutForest has a bug where score_partial mutates state.
NON_IDEMPOTENT_SCORE_MODELS = frozenset({"RandomModel", "RobustRandomCutForest"})


class _PredictMachine(RuleBasedStateMachine):
    """State machine for testing predict contracts for decision-capable models."""

    # Class attributes set by _create_configured_predict_machine
    model_name: str
    params: dict[str, Any]
    num_features: int

    # Instance attributes initialized in setup
    model = None
    step = 0
    all_X = None
    all_y = None
    predictions = None

    @initialize()
    def setup(self):
        fix_seed(SEED)
        model_cls = getattr(pysad.models, self.model_name)
        self.model = model_cls(**copy.deepcopy(self.params))
        self.step = 0
        self.all_X = []
        self.all_y = []
        self.predictions = []

    def _draw_instance(self) -> tuple[np.ndarray, int]:
        rng = np.random.RandomState(SEED + self.step * 17 + hash(self.model_name) % 1000)
        xi = rng.choice(FINITE_VALUES, size=self.num_features).astype(np.float64)
        yi = rng.randint(0, 2)
        return xi, yi

    @rule()
    def step_fit_predict_partial(self):
        xi, yi = self._draw_instance()
        self.all_X.append(xi.copy())
        self.all_y.append(yi)

        pred = self.model.fit_predict_partial(xi, yi)
        assert isinstance(pred, int), f"Expected int, got {type(pred)}"
        assert pred in (0, 1), f"Expected 0 or 1, got {pred}"
        self.predictions.append(pred)
        self.step += 1

    @rule()
    def check_predict_partial_state_invariance(self):
        """Multiple predict_partial calls on the same instance return the same value and don't mutate state."""
        if not self.all_X:
            return
        indices = [0]
        if len(self.all_X) > 2:
            indices.append(len(self.all_X) // 2)
        indices.append(len(self.all_X) - 1)

        for idx in indices:
            xi = self.all_X[idx]
            pred1 = self.model.predict_partial(xi)
            pred2 = self.model.predict_partial(xi)
            pred3 = self.model.predict_partial(xi)
            assert pred1 == pred2 == pred3, (
                f"{self.model_name}: predict_partial not idempotent on instance {idx}: "
                f"{pred1}, {pred2}, {pred3}"
            )

    @rule()
    def check_fit_predict_partial_equivalence(self):
        """fit_predict_partial(xi, yi) == predict_partial(xi) then fit_partial(xi, yi)."""
        if not self.all_X:
            return
        # Test on the last instance
        idx = len(self.all_X) - 1
        xi = self.all_X[idx]
        yi = self.all_y[idx]

        # Path 1: fit_predict_partial
        fix_seed(SEED)
        model_cls = getattr(pysad.models, self.model_name)
        model1 = model_cls(**copy.deepcopy(self.params))
        for j in range(idx):
            model1.fit_predict_partial(self.all_X[j], self.all_y[j])
        pred_fit_predict = model1.fit_predict_partial(xi, yi)

        # Path 2: predict_partial then fit_partial
        fix_seed(SEED)
        model2 = model_cls(**copy.deepcopy(self.params))
        for j in range(idx):
            model2.fit_predict_partial(self.all_X[j], self.all_y[j])
        pred_predict = model2.predict_partial(xi)
        model2.fit_partial(xi, yi)

        assert pred_fit_predict == pred_predict, (
            f"{self.model_name}: fit_predict_partial ({pred_fit_predict}) != "
            f"predict_partial then fit_partial ({pred_predict}) on instance {idx}"
        )

    @rule()
    def check_batch_predict_equivalence(self):
        """Sequential predict_partial matches batch predict."""
        if not self.all_X:
            return
        X_arr = np.array(self.all_X)

        fix_seed(SEED)
        model_cls = getattr(pysad.models, self.model_name)
        seq_model = model_cls(**copy.deepcopy(self.params))
        for xi, yi in zip(self.all_X, self.all_y, strict=True):
            seq_model.fit_predict_partial(xi, yi)
        seq_preds = [seq_model.predict_partial(xi) for xi in self.all_X]

        fix_seed(SEED)
        batch_model = model_cls(**copy.deepcopy(self.params))
        batch_model.fit(X_arr, np.array(self.all_y) if self.all_y else None)
        batch_preds = batch_model.predict(X_arr)

        np.testing.assert_array_equal(
            batch_preds,
            np.array(seq_preds, dtype=np.int64),
            err_msg=f"{self.model_name}: batch predict != sequential predict_partial",
        )

    @rule()
    def check_batch_fit_predict_equivalence(self):
        """Sequential fit_predict_partial matches batch fit_predict."""
        if not self.all_X:
            return
        X_arr = np.array(self.all_X)
        y_arr = np.array(self.all_y) if self.all_y else None

        fix_seed(SEED)
        model_cls = getattr(pysad.models, self.model_name)
        seq_model = model_cls(**copy.deepcopy(self.params))
        seq_preds = [
            seq_model.fit_predict_partial(xi, yi)
            for xi, yi in zip(self.all_X, self.all_y, strict=True)
        ]

        fix_seed(SEED)
        batch_model = model_cls(**copy.deepcopy(self.params))
        batch_preds = batch_model.fit_predict(X_arr, y_arr)

        np.testing.assert_array_equal(
            batch_preds,
            np.array(seq_preds, dtype=np.int64),
            err_msg=f"{self.model_name}: batch fit_predict != sequential fit_predict_partial",
        )

    @rule()
    def check_input_immutability(self):
        if not self.all_X:
            return
        original_X = [x.copy() for x in self.all_X]
        original_y = self.all_y.copy() if self.all_y else None

        fix_seed(SEED)
        model_cls = getattr(pysad.models, self.model_name)
        test_model = model_cls(**copy.deepcopy(self.params))
        for xi, yi in zip(self.all_X, self.all_y, strict=True):
            test_model.fit_predict_partial(xi, yi)

        for i, (orig, curr) in enumerate(zip(original_X, self.all_X, strict=False)):
            np.testing.assert_array_equal(orig, curr, err_msg=f"Input X[{i}] was mutated")
        if original_y is not None:
            np.testing.assert_array_equal(original_y, self.all_y, err_msg="Input y was mutated")


def _create_configured_machine(
    base_machine_cls,
    model_name: str,
    params: dict[str, Any],
    num_features: int,
    is_fit_then_score: bool = False,
):
    """Create a RuleBasedStateMachine subclass configured for a specific model.

    Returns the TestCase class for pytest to collect.
    """
    # For fit_then_score, some models have different warmup behavior
    if is_fit_then_score:
        if model_name == "DAMP":
            # DAMP's fit_partial increments num_values_ before score_partial,
            # so warmup is effectively one step shorter
            warmup_length = max(0, _get_warmup_length(model_name, params) - 1)
        elif model_name == "HTM":
            # HTM's fit_partial and score_partial both advance the calibrator,
            # so warmup is effectively half the probationary period
            warmup_length = max(0, _get_warmup_length(model_name, params) // 2)
        else:
            warmup_length = _get_warmup_length(model_name, params)
    else:
        warmup_length = _get_warmup_length(model_name, params)

    # Model-specific _draw_instance overrides
    draw_instance_overrides = {}
    if model_name == "HTM":
        # HTM needs timestamp and value within min/max range to avoid spatial anomalies
        min_val = params.get("min_val", 0.0)
        max_val = params.get("max_val", 1.0)
        mid_value = (min_val + max_val) / 2.0

        def htm_draw_instance(self):
            rng = np.random.RandomState(SEED + self.step * 17 + hash(self.model_name) % 1000)
            timestamp = float(self.step * 3600)  # hourly timestamps
            xi = np.array([timestamp, mid_value], dtype=np.float64)
            yi = rng.randint(0, 2)
            return xi, yi

        draw_instance_overrides["_draw_instance"] = htm_draw_instance
    elif model_name == "MStream":
        # MStream needs positive values for log(1+x) transform
        def mstream_draw_instance(self):
            rng = np.random.RandomState(SEED + self.step * 17 + hash(self.model_name) % 1000)
            xi = rng.uniform(0, 10, size=self.num_features).astype(np.float64)
            yi = rng.randint(0, 2)
            return xi, yi

        draw_instance_overrides["_draw_instance"] = mstream_draw_instance
    elif model_name == "Inqmad":
        # Inqmad can have numerical issues with extreme values; use moderate range
        def inqmad_draw_instance(self):
            rng = np.random.RandomState(SEED + self.step * 17 + hash(self.model_name) % 1000)
            xi = rng.uniform(-1, 1, size=self.num_features).astype(np.float64)
            yi = rng.randint(0, 2)
            return xi, yi

        draw_instance_overrides["_draw_instance"] = inqmad_draw_instance

    # Create class dynamically with the configuration as class attributes
    class_attrs = {
        "model_name": model_name,
        "params": copy.deepcopy(params),
        "num_features": num_features,
        "warmup_length": warmup_length,
        "has_warmup": _has_constant_warmup(model_name, params),
        "_expected_warmup_score": _warmup_score(model_name, params),
    }
    class_attrs.update(draw_instance_overrides)

    ConfiguredMachine = type(
        f"_{model_name}{base_machine_cls.__name__}", (base_machine_cls,), class_attrs
    )

    # Apply settings
    ConfiguredMachine = STATEFUL_SETTINGS(ConfiguredMachine)

    # Get the TestCase class for pytest
    TestCase = ConfiguredMachine.TestCase
    TestCase.__name__ = f"Test{model_name}{base_machine_cls.__name__}"
    TestCase.__qualname__ = f"Test{model_name}{base_machine_cls.__name__}"

    # Apply xfail marks if needed
    if model_name in KNOWN_STATEFUL_FAILURES:
        reason = KNOWN_STATEFUL_FAILURES[model_name]
        TestCase = pytest.mark.xfail(reason=reason, strict=True)(TestCase)

    return TestCase


def _create_configured_predict_machine(
    base_machine_cls,
    model_name: str,
    params: dict[str, Any],
    num_features: int,
):
    """Create a RuleBasedStateMachine subclass configured for a specific decision-capable model.

    Returns the TestCase class for pytest to collect.
    """
    # Model-specific _draw_instance overrides (same as for fit/score machines)
    draw_instance_overrides = {}
    if model_name == "HTM":
        # HTM needs timestamp and value within min/max range to avoid spatial anomalies
        min_val = params.get("min_val", 0.0)
        max_val = params.get("max_val", 1.0)
        mid_value = (min_val + max_val) / 2.0

        def htm_draw_instance(self):
            rng = np.random.RandomState(SEED + self.step * 17 + hash(self.model_name) % 1000)
            timestamp = float(self.step * 3600)  # hourly timestamps
            xi = np.array([timestamp, mid_value], dtype=np.float64)
            yi = rng.randint(0, 2)
            return xi, yi

        draw_instance_overrides["_draw_instance"] = htm_draw_instance
    elif model_name == "MStream":
        # MStream needs positive values for log(1+x) transform
        def mstream_draw_instance(self):
            rng = np.random.RandomState(SEED + self.step * 17 + hash(self.model_name) % 1000)
            xi = rng.uniform(0, 10, size=self.num_features).astype(np.float64)
            yi = rng.randint(0, 2)
            return xi, yi

        draw_instance_overrides["_draw_instance"] = mstream_draw_instance
    elif model_name == "Inqmad":
        # Inqmad can have numerical issues with extreme values; use moderate range
        def inqmad_draw_instance(self):
            rng = np.random.RandomState(SEED + self.step * 17 + hash(self.model_name) % 1000)
            xi = rng.uniform(-1, 1, size=self.num_features).astype(np.float64)
            yi = rng.randint(0, 2)
            return xi, yi

        draw_instance_overrides["_draw_instance"] = inqmad_draw_instance

    # Create class dynamically with the configuration as class attributes
    class_attrs = {
        "model_name": model_name,
        "params": copy.deepcopy(params),
        "num_features": num_features,
    }
    class_attrs.update(draw_instance_overrides)

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


# Generate test case classes for each model
for _name in active_model_names():
    if _name not in SKIPPED_STATEFUL:
        _num_features = _num_features_for(_name)
        _params = short_stream_params(_name, _num_features, seed=SEED)

        # FitScorePartial machine
        _fit_score_testcase = _create_configured_machine(
            _FitScorePartialMachine, _name, _params, _num_features, is_fit_then_score=False
        )
        globals()[f"Test{_name}FitScorePartial"] = _fit_score_testcase

        # FitThenScore machine
        _fit_then_score_testcase = _create_configured_machine(
            _FitThenScoreMachine, _name, _params, _num_features, is_fit_then_score=True
        )
        globals()[f"Test{_name}FitThenScore"] = _fit_then_score_testcase

        # Predict machine for decision-capable models
        if _is_decision_capable(_name) and _name not in SKIPPED_PREDICT_STATEFUL:
            _predict_testcase = _create_configured_predict_machine(
                _PredictMachine, _name, _params, _num_features
            )
            globals()[f"Test{_name}Predict"] = _predict_testcase


# Additional focused tests for specific contract properties (non-stateful, simpler)


@pytest.mark.parametrize("model_name", active_model_names())
def test_constant_warmup_period_behavior(model_name):
    """Test that models with constant warmup periods return expected scores during warmup."""
    num_features = _num_features_for(model_name)
    params = short_stream_params(model_name, num_features, seed=SEED)

    if not _has_constant_warmup(model_name, params):
        pytest.skip(f"{model_name} has no constant warmup period")

    warmup_length = _get_warmup_length(model_name, params)
    if warmup_length == 0:
        pytest.skip(f"{model_name} warmup length is 0")

    fix_seed(SEED)
    model_cls = getattr(pysad.models, model_name)
    model = model_cls(**copy.deepcopy(params))

    expected_score = _warmup_score(model_name, params)

    rng = np.random.RandomState(SEED)
    for i in range(warmup_length + 2):
        # For HTM (timestamped), use constant value to avoid spatial anomalies during warmup
        if model_name == "HTM":
            min_val = params.get("min_val", 0.0)
            max_val = params.get("max_val", 1.0)
            # Use midpoint value and incrementing timestamp
            timestamp = float(i * 3600)  # hourly timestamps
            value = (min_val + max_val) / 2.0
            xi = np.array([timestamp, value], dtype=np.float64)
        else:
            xi = rng.choice(FINITE_VALUES, size=num_features).astype(np.float64)
        yi = rng.randint(0, 2)
        score = model.fit_score_partial(xi, yi)
        if i < warmup_length:
            assert score == expected_score, (
                f"{model_name}: warmup score at step {i} is {score}, expected {expected_score}"
            )


@pytest.mark.parametrize("model_name", active_model_names())
def test_sequential_vs_batch_equivalence(model_name):
    """Test that sequential fit_score_partial matches batch fit_score."""
    num_features = _num_features_for(model_name)
    params = _test_safe_params(model_name, short_stream_params(model_name, num_features, seed=SEED))

    fix_seed(SEED)
    rng = np.random.RandomState(SEED + 1)
    # For MStream, generate positive values
    if model_name == "MStream":
        X = rng.uniform(0, 10, size=(20, num_features)).astype(np.float64)
    else:
        X = rng.choice(FINITE_VALUES, size=(20, num_features)).astype(np.float64)
    y = rng.randint(0, 2, size=20)

    fix_seed(SEED)
    model_cls = getattr(pysad.models, model_name)
    seq_model = model_cls(**copy.deepcopy(params))
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


@pytest.mark.parametrize("model_name", active_model_names())
def test_fit_then_score_vs_batch_equivalence(model_name):
    """Test that sequential fit_partial + score_partial matches batch fit().score()."""
    num_features = _num_features_for(model_name)
    params = _test_safe_params(model_name, short_stream_params(model_name, num_features, seed=SEED))

    fix_seed(SEED)
    rng = np.random.RandomState(SEED + 1)
    if model_name == "MStream":
        X = rng.uniform(0, 10, size=(20, num_features)).astype(np.float64)
    else:
        X = rng.choice(FINITE_VALUES, size=(20, num_features)).astype(np.float64)
    y = rng.randint(0, 2, size=20)

    fix_seed(SEED)
    model_cls = getattr(pysad.models, model_name)
    seq_model = model_cls(**copy.deepcopy(params))
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


@pytest.mark.parametrize("model_name", active_model_names())
def test_state_reset_equivalence(model_name):
    """Test that a new model instance with same params gives same scores (state reset)."""
    num_features = _num_features_for(model_name)
    params = _test_safe_params(model_name, short_stream_params(model_name, num_features, seed=SEED))

    fix_seed(SEED)
    rng = np.random.RandomState(SEED + 1)
    if model_name == "MStream":
        X = rng.uniform(0, 10, size=(20, num_features)).astype(np.float64)
    else:
        X = rng.choice(FINITE_VALUES, size=(20, num_features)).astype(np.float64)
    y = rng.randint(0, 2, size=20)

    fix_seed(SEED)
    model_cls = getattr(pysad.models, model_name)
    model1 = model_cls(**copy.deepcopy(params))
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
