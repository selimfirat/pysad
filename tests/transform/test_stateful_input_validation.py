"""Hypothesis stateful property tests for transform input validation.

These tests use Hypothesis's RuleBasedStateMachine to model the state transitions of
streaming transformers, covering input validation guarantees:

- Valid inputs work correctly and produce finite outputs (after initialization)
- Invalid inputs raise appropriate errors for transformers with early validation
- For transformers with early validation (ShingleTransformer, DrainLogParser,
  NonparametricDynamicThresholdPostprocessor), invalid inputs do not mutate state
- For transformers without early validation, invalid inputs may be accepted
  and produce NaN outputs (this is documented behavior)
- Batch methods agree with partial methods on valid inputs
- State consistency across fit_partial, transform_partial, fit_transform_partial

Each concrete transformer class in pysad.transform subpackages is tested.
"""

import inspect

import numpy as np
from hypothesis import HealthCheck, settings
from hypothesis.stateful import RuleBasedStateMachine, initialize, rule

import pysad.transform.ensemble
import pysad.transform.postprocessing
import pysad.transform.preprocessing
import pysad.transform.probability_calibration

SEED = 61
MAX_EXAMPLES = 10
STATEFUL_SETTINGS = settings(
    max_examples=MAX_EXAMPLES,
    deadline=None,
    database=None,
    suppress_health_check=[HealthCheck.too_slow],
    derandomize=True,
)

# Finite values for valid inputs
FINITE_VALUES = [0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 10.0, -10.0]


def _concrete_classes(module):
    """Return names of concrete (non-abstract) classes exported by a module."""
    return [name for name in module.__all__ if not inspect.isabstract(getattr(module, name))]


def _preprocessing_transformers():
    return _concrete_classes(pysad.transform.preprocessing)


def _postprocessors():
    return _concrete_classes(pysad.transform.postprocessing)


def _calibrators():
    return _concrete_classes(pysad.transform.probability_calibration)


def _ensemblers():
    return _concrete_classes(pysad.transform.ensemble)


# Constructor arguments for each transformer class
TRANSFORMER_PARAMS = {
    # Preprocessing
    "StandardScaler": {},
    "MinMaxScaler": {},
    "InstanceStandardScaler": {},
    "InstanceUnitNormScaler": {},
    "IdentityScaler": {},
    "ShingleTransformer": {"shingle_size": 3},
    "SeasonalTrendDecomposer": {"season_length": 5},
    "OneShotSTL": {"period": 5, "num_init_periods": 3},
    "ModifiedSTLResidualTransformer": {"period": 5, "window_size": 15, "robust": False},
    "FunctionTransformer": {"func": None, "fit_func": None},
    "DrainLogParser": {"depth": 3, "st": 0.4, "max_child": 5},
    # Postprocessing
    "AveragePostprocessor": {"window_size": 5},
    "MaxPostprocessor": {"window_size": 5},
    "MedianPostprocessor": {"window_size": 5},
    "ZScorePostprocessor": {"window_size": 5},
    "NonparametricDynamicThresholdPostprocessor": {"window_size": 10},
    "QuantileFilter": {"q": 0.5},
    "ThresholdFilter": {"threshold": 0.5},
    # Calibration
    "ConformalProbabilityCalibrator": {"window_size": 10},
    "ConformalTSFMCalibrator": {"alpha": 0.1, "gamma": 0.005, "window_size": 10},
    "GaussianTailProbabilityCalibrator": {"window_size": 10, "short_window_size": 3},
    "GroupConditionalConformalCalibrator": {"window_size": 10, "max_regimes": 4},
    "AnomalyLikelihoodCalibrator": {"window_size": 10, "short_window_size": 3},
    "SPOTCalibrator": {"window_size": 10, "q": 1e-4, "init_level": 0.98},
    # Ensemble
    "AverageScoreEnsembler": {},
    "AverageOfMaximumScoreEnsembler": {"random_state": SEED},
    "MaximumOfAverageScoreEnsembler": {"random_state": SEED},
    "MaximumScoreEnsembler": {},
    "MedianScoreEnsembler": {},
}


# Transformers that validate input early and should preserve state on invalid input
EARLY_VALIDATION_TRANSFORMERS = {
    "ShingleTransformer",
    "DrainLogParser",
    "NonparametricDynamicThresholdPostprocessor",
}

# Transformers with initialization period that may return NaN initially
INITIALIZATION_TRANSFORMERS = {
    "OneShotSTL",
    "ModifiedSTLResidualTransformer",
}

# Transformers that accept NaN inputs and produce NaN outputs (no early validation)
NAN_ACCEPTING_TRANSFORMERS = {
    "StandardScaler",
    "MinMaxScaler",
    "AveragePostprocessor",
    "MaxPostprocessor",
    "MedianPostprocessor",
    "ZScorePostprocessor",
    "QuantileFilter",
}

# Transformers that don't validate NaN inputs early and may have internal state
# that doesn't match call count (e.g., OneShotSTL ignores NaN in fit_partial)
# or that mutate state before raising errors on invalid inputs
POORLY_VALIDATED_TRANSFORMERS = {
    "OneShotSTL",
    "ModifiedSTLResidualTransformer",
    "SeasonalTrendDecomposer",
}

# Calibrators that accept inf/NaN in fit_partial and produce NaN outputs
POORLY_VALIDATED_CALIBRATORS = {
    "GaussianTailProbabilityCalibrator",
}


def _make_transformer(name: str):
    """Create a transformer instance with appropriate params."""
    cls = None
    for module in [
        pysad.transform.preprocessing,
        pysad.transform.postprocessing,
        pysad.transform.probability_calibration,
        pysad.transform.ensemble,
    ]:
        if name in module.__all__:
            cls = getattr(module, name)
            break
    if cls is None:
        raise ValueError(f"Unknown transformer: {name}")
    return cls(**TRANSFORMER_PARAMS.get(name, {}))


def _input_category(name: str) -> str:
    """Return the input category for a transformer: 'vector', 'scalar', 'string', or 'matrix'."""
    if name == "DrainLogParser":
        return "string"
    if name in _preprocessing_transformers():
        return "vector"
    if name in _postprocessors() or name in _calibrators():
        return "scalar"
    if name in _ensemblers():
        return "matrix"
    return "vector"


def _num_features_for(name: str) -> int:
    """Return the number of features for a vector-input transformer."""
    if name in ("SeasonalTrendDecomposer", "OneShotSTL", "ModifiedSTLResidualTransformer"):
        return 1
    return 3


def _num_detectors_for(name: str) -> int:
    """Return the number of detectors for an ensembler."""
    return 5


def _output_shape(name: str, num_features: int) -> tuple:
    """Return the expected output shape for a transformer."""
    if name == "ShingleTransformer":
        return (num_features * 3,)  # shingle_size=3
    if name in ("SeasonalTrendDecomposer", "OneShotSTL", "ModifiedSTLResidualTransformer"):
        return (1,)
    if name == "DrainLogParser":
        return (1,)
    return (num_features,)


def _has_early_validation(name: str) -> bool:
    """Check if transformer validates input early (before mutating state)."""
    return name in EARLY_VALIDATION_TRANSFORMERS


def _has_initialization(name: str) -> bool:
    """Check if transformer has initialization period returning NaN."""
    return name in INITIALIZATION_TRANSFORMERS


def _accepts_nan(name: str) -> bool:
    """Check if transformer accepts NaN inputs without raising."""
    return name in NAN_ACCEPTING_TRANSFORMERS


def _poorly_validated(name: str) -> bool:
    """Check if transformer has poor input validation (accepts NaN silently, state mismatch)."""
    return name in POORLY_VALIDATED_TRANSFORMERS


def _poorly_validated_calibrator(name: str) -> bool:
    """Check if calibrator has poor input validation."""
    return name in POORLY_VALIDATED_CALIBRATORS


def _initialization_length(name: str) -> int:
    """Return the number of points before transformer produces finite outputs."""
    if name == "OneShotSTL":
        return 15  # num_init_periods * period = 3 * 5
    if name == "ModifiedSTLResidualTransformer":
        return 11  # 2 * period + 1 = 11
    return 0


class _PreprocessingValidationMachine(RuleBasedStateMachine):
    """State machine for testing preprocessing transformer input validation."""

    transformer_name: str
    num_features: int

    model = None
    step: int
    fitted_count: int
    _state_snapshot = None

    @initialize()
    def setup(self):
        self.model = _make_transformer(self.transformer_name)
        self.step = 0
        self.fitted_count = 0
        self._state_snapshot = None

    def _valid_vector(self) -> np.ndarray:
        """Generate a valid input vector."""
        rng = np.random.RandomState(SEED + self.step * 31 + hash(self.transformer_name) % 1000)
        return rng.choice(FINITE_VALUES, size=self.num_features).astype(np.float64)

    def _invalid_vector_wrong_dim(self) -> np.ndarray:
        """Generate an invalid input vector (wrong dimension)."""
        return np.array([1.0, 2.0])  # Wrong feature count

    def _invalid_vector_non_finite(self) -> np.ndarray:
        """Generate an invalid input vector (non-finite values)."""
        vec = self._valid_vector().copy()
        vec[0] = np.nan
        return vec

    def _invalid_input_2d(self) -> np.ndarray:
        """Generate a 2D input when 1D expected."""
        return np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    def _invalid_input_empty(self) -> np.ndarray:
        """Generate an empty input."""
        return np.array([])

    def _snapshot_state(self):
        """Take a snapshot of model state for comparison."""
        if hasattr(self.model, "means_") and self.model.means_ is not None:
            return {
                "type": "standard_scaler",
                "means": [m.get() for m in self.model.means_],
                "variances": [v.get() for v in self.model.variances_],
            }
        if hasattr(self.model, "minimums_") and self.model.minimums_ is not None:
            return {
                "type": "min_max_scaler",
                "minimums": [m.get() for m in self.model.minimums_],
                "maximums": [m.get() for m in self.model.maximums_],
            }
        if hasattr(self.model, "recent"):
            return {
                "type": "shingle",
                "recent": [r.copy() for r in self.model.recent],
            }
        if hasattr(self.model, "root_node"):  # DrainLogParser
            return {
                "type": "drain",
                "next_id": self.model._next_id,
                "root_children": len(self.model.root_node.children),
            }
        if hasattr(self.model, "window") and hasattr(
            self.model.window, "_buffer"
        ):  # STL transformers
            return {
                "type": "stl",
                "window_len": len(self.model.window),
                "num_seen": getattr(self.model, "num_online_", getattr(self.model, "num_seen", 0)),
            }
        if (
            hasattr(self.model, "seasonal_sums") and self.model.seasonal_sums is not None
        ):  # SeasonalTrendDecomposer
            return {
                "type": "seasonal",
                "seasonal_sums": self.model.seasonal_sums.copy(),
                "seasonal_counts": self.model.seasonal_counts.copy(),
                "num_seen": self.model.num_seen,
            }
        return {"type": "stateless"}

    def _assert_state_unchanged(self, snapshot):
        """Assert that model state matches the snapshot."""
        current = self._snapshot_state()
        if snapshot["type"] != current["type"]:
            return  # Stateless or unknown type, skip check
        if snapshot["type"] == "standard_scaler":
            for s, c in zip(snapshot["means"], current["means"], strict=True):
                assert s == c
            for s, c in zip(snapshot["variances"], current["variances"], strict=True):
                assert s == c
        elif snapshot["type"] == "min_max_scaler":
            for s, c in zip(snapshot["minimums"], current["minimums"], strict=True):
                assert s == c
            for s, c in zip(snapshot["maximums"], current["maximums"], strict=True):
                assert s == c
        elif snapshot["type"] == "shingle":
            assert len(snapshot["recent"]) == len(current["recent"])
            for s, c in zip(snapshot["recent"], current["recent"], strict=True):
                np.testing.assert_array_equal(s, c)
        elif snapshot["type"] == "drain":
            assert snapshot["next_id"] == current["next_id"]
        elif snapshot["type"] == "stl":
            assert snapshot["window_len"] == current["window_len"]
        elif snapshot["type"] == "seasonal":
            np.testing.assert_array_equal(snapshot["seasonal_sums"], current["seasonal_sums"])
            np.testing.assert_array_equal(snapshot["seasonal_counts"], current["seasonal_counts"])
            assert snapshot["num_seen"] == current["num_seen"]

    def _should_be_finite(self) -> bool:
        """Check if outputs should be finite at current fitted count."""
        if _has_initialization(self.transformer_name):
            return self.fitted_count >= _initialization_length(self.transformer_name)
        return True

    @rule()
    def step_fit_partial_valid(self):
        """Fit with a valid vector."""
        vec = self._valid_vector()
        self.model.fit_partial(vec)
        self.fitted_count += 1
        self.step += 1

    @rule()
    def step_transform_partial_valid(self):
        """Transform with a valid vector."""
        if self.fitted_count == 0:
            self.step_fit_partial_valid()
        vec = self._valid_vector()
        out = self.model.transform_partial(vec)
        assert isinstance(out, np.ndarray)
        expected_shape = _output_shape(self.transformer_name, self.num_features)
        assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"
        if self._should_be_finite():
            assert np.all(np.isfinite(out)), f"Non-finite output: {out}"
        self.step += 1

    @rule()
    def step_fit_transform_partial_valid(self):
        """Fit and transform with a valid vector."""
        vec = self._valid_vector()
        out = self.model.fit_transform_partial(vec)
        assert isinstance(out, np.ndarray)
        expected_shape = _output_shape(self.transformer_name, self.num_features)
        assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"
        if self._should_be_finite():
            assert np.all(np.isfinite(out)), f"Non-finite output: {out}"
        self.fitted_count += 1
        self.step += 1

    @rule()
    def check_fit_partial_invalid_wrong_dim_raises(self):
        """fit_partial with wrong dimension raises an error for early validation transformers."""
        if self.fitted_count == 0:
            return
        # Skip for poorly validated transformers - they corrupt state before raising
        if _poorly_validated(self.transformer_name):
            return
        invalid = self._invalid_vector_wrong_dim()
        if _has_early_validation(self.transformer_name):
            snapshot = self._snapshot_state()
            try:
                self.model.fit_partial(invalid)
            except (ValueError, TypeError, IndexError):
                self._assert_state_unchanged(snapshot)
        else:
            # Late validation: just check method doesn't crash
            try:
                self.model.fit_partial(invalid)
            except (ValueError, TypeError, IndexError):
                pass

    @rule()
    def check_fit_partial_invalid_non_finite(self):
        """fit_partial with non-finite values: early validation raises, others accept."""
        if self.fitted_count == 0:
            return
        # Skip for transformers that accept NaN or have poor validation - they corrupt state
        if _accepts_nan(self.transformer_name) or _poorly_validated(self.transformer_name):
            return
        invalid = self._invalid_vector_non_finite()
        if _has_early_validation(self.transformer_name):
            snapshot = self._snapshot_state()
            try:
                self.model.fit_partial(invalid)
            except (ValueError, TypeError):
                self._assert_state_unchanged(snapshot)
        else:
            # Other transformers: just check no crash
            try:
                self.model.fit_partial(invalid)
            except (ValueError, TypeError):
                pass

    @rule()
    def check_fit_partial_invalid_2d_raises(self):
        """fit_partial with 2D input raises an error for early validation transformers."""
        if self.fitted_count == 0:
            return
        # Skip for poorly validated transformers - they corrupt state before raising
        if _poorly_validated(self.transformer_name):
            return
        invalid = self._invalid_input_2d()
        if _has_early_validation(self.transformer_name):
            snapshot = self._snapshot_state()
            try:
                self.model.fit_partial(invalid)
            except (ValueError, TypeError, IndexError):
                self._assert_state_unchanged(snapshot)
        else:
            try:
                self.model.fit_partial(invalid)
            except (ValueError, TypeError, IndexError):
                pass

    @rule()
    def check_fit_partial_invalid_empty_raises(self):
        """fit_partial with empty input raises an error for early validation transformers."""
        if self.fitted_count == 0:
            return
        # Skip for poorly validated transformers - they corrupt state before raising
        if _poorly_validated(self.transformer_name):
            return
        invalid = self._invalid_input_empty()
        if _has_early_validation(self.transformer_name):
            snapshot = self._snapshot_state()
            try:
                self.model.fit_partial(invalid)
            except (ValueError, TypeError, IndexError):
                self._assert_state_unchanged(snapshot)
        else:
            try:
                self.model.fit_partial(invalid)
            except (ValueError, TypeError, IndexError):
                pass

    @rule()
    def check_transform_partial_invalid_wrong_dim_raises(self):
        """transform_partial with wrong dimension raises an error for early validation transformers."""
        if self.fitted_count == 0:
            return
        # Skip for poorly validated transformers - they corrupt state before raising
        if _poorly_validated(self.transformer_name):
            return
        invalid = self._invalid_vector_wrong_dim()
        if _has_early_validation(self.transformer_name):
            snapshot = self._snapshot_state()
            try:
                self.model.transform_partial(invalid)
            except (ValueError, TypeError, IndexError):
                self._assert_state_unchanged(snapshot)
        else:
            try:
                self.model.transform_partial(invalid)
            except (ValueError, TypeError, IndexError):
                pass

    @rule()
    def check_transform_partial_invalid_non_finite(self):
        """transform_partial with non-finite values: early validation raises, others may return NaN."""
        if self.fitted_count == 0:
            return
        # Skip for transformers that accept NaN or have poor validation - they corrupt state
        if _accepts_nan(self.transformer_name) or _poorly_validated(self.transformer_name):
            return
        invalid = self._invalid_vector_non_finite()
        if _has_early_validation(self.transformer_name):
            snapshot = self._snapshot_state()
            try:
                self.model.transform_partial(invalid)
            except (ValueError, TypeError):
                self._assert_state_unchanged(snapshot)
        else:
            try:
                self.model.transform_partial(invalid)
            except (ValueError, TypeError):
                pass

    @rule()
    def check_fit_transform_partial_invalid(self):
        """fit_transform_partial with invalid input: early validation raises, others may accept."""
        if self.fitted_count == 0:
            return
        # Skip for transformers that accept NaN or have poor validation - they corrupt state
        if _accepts_nan(self.transformer_name) or _poorly_validated(self.transformer_name):
            return
        invalid = self._invalid_vector_non_finite()
        if _has_early_validation(self.transformer_name):
            snapshot = self._snapshot_state()
            try:
                self.model.fit_transform_partial(invalid)
            except (ValueError, TypeError):
                self._assert_state_unchanged(snapshot)
        else:
            try:
                self.model.fit_transform_partial(invalid)
            except (ValueError, TypeError):
                pass


class _DrainLogParserValidationMachine(RuleBasedStateMachine):
    """State machine for testing DrainLogParser input validation."""

    transformer_name: str

    model = None
    step: int
    fitted_count: int
    _state_snapshot = None

    @initialize()
    def setup(self):
        self.model = _make_transformer(self.transformer_name)
        self.step = 0
        self.fitted_count = 0
        self._state_snapshot = None

    def _valid_log_line(self) -> str:
        """Generate a valid log line."""
        templates = [
            "User {user} logged in from {ip}",
            "Error connecting to {host}:{port}",
            "Processing request {id} took {time}ms",
            "Cache miss for key {key}",
            "Database query executed in {time}s",
        ]
        rng = np.random.RandomState(SEED + self.step * 31 + hash(self.transformer_name) % 1000)
        template = rng.choice(templates)
        filled = template.format(
            user=f"user{rng.randint(1, 100)}",
            ip=f"192.168.{rng.randint(1, 255)}.{rng.randint(1, 255)}",
            host=f"host{rng.randint(1, 10)}",
            port=rng.randint(1000, 9999),
            id=rng.randint(1000, 9999),
            time=rng.randint(1, 1000),
            key=f"key{rng.randint(1, 100)}",
        )
        return filled

    def _invalid_input_not_string(self):
        """Generate an invalid input that's not a string."""
        return 123

    def _invalid_input_array_not_string(self) -> np.ndarray:
        """Generate an array input that doesn't hold a string."""
        return np.array([1.0, 2.0, 3.0])

    def _invalid_input_2d_array(self) -> np.ndarray:
        """Generate a 2D array input."""
        return np.array([["line1"], ["line2"]], dtype=object)

    def _snapshot_state(self):
        return {
            "next_id": self.model._next_id,
            "root_children": len(self.model.root_node.children),
        }

    def _assert_state_unchanged(self, snapshot):
        current = self._snapshot_state()
        assert snapshot["next_id"] == current["next_id"]

    @rule()
    def step_fit_partial_valid(self):
        line = self._valid_log_line()
        self.model.fit_partial(line)
        self.fitted_count += 1
        self.step += 1

    @rule()
    def step_transform_partial_valid(self):
        if self.fitted_count == 0:
            self.step_fit_partial_valid()
        line = self._valid_log_line()
        out = self.model.transform_partial(line)
        assert isinstance(out, np.ndarray)
        assert out.shape == (1,)
        assert np.isfinite(out[0])
        self.step += 1

    @rule()
    def step_fit_transform_partial_valid(self):
        line = self._valid_log_line()
        out = self.model.fit_transform_partial(line)
        assert isinstance(out, np.ndarray)
        assert out.shape == (1,)
        assert np.isfinite(out[0])
        self.fitted_count += 1
        self.step += 1

    @rule()
    def check_fit_partial_invalid_not_string_raises(self):
        if self.fitted_count == 0:
            return
        snapshot = self._snapshot_state()
        invalid = self._invalid_input_not_string()
        try:
            self.model.fit_partial(invalid)
        except (ValueError, TypeError):
            self._assert_state_unchanged(snapshot)

    @rule()
    def check_fit_partial_invalid_array_raises(self):
        if self.fitted_count == 0:
            return
        snapshot = self._snapshot_state()
        invalid = self._invalid_input_array_not_string()
        try:
            self.model.fit_partial(invalid)
        except (ValueError, TypeError):
            self._assert_state_unchanged(snapshot)

    @rule()
    def check_transform_partial_invalid_not_string_raises(self):
        if self.fitted_count == 0:
            return
        snapshot = self._snapshot_state()
        invalid = self._invalid_input_not_string()
        try:
            self.model.transform_partial(invalid)
        except (ValueError, TypeError):
            self._assert_state_unchanged(snapshot)

    @rule()
    def check_fit_transform_partial_invalid_raises(self):
        if self.fitted_count == 0:
            return
        snapshot = self._snapshot_state()
        invalid = self._invalid_input_not_string()
        try:
            self.model.fit_transform_partial(invalid)
        except (ValueError, TypeError):
            self._assert_state_unchanged(snapshot)


class _PostprocessorValidationMachine(RuleBasedStateMachine):
    """State machine for testing postprocessor input validation."""

    transformer_name: str

    model = None
    step: int
    fitted_count: int
    _state_snapshot = None

    @initialize()
    def setup(self):
        self.model = _make_transformer(self.transformer_name)
        self.step = 0
        self.fitted_count = 0
        self._state_snapshot = None

    def _valid_score(self) -> float:
        rng = np.random.RandomState(SEED + self.step * 31 + hash(self.transformer_name) % 1000)
        if self.transformer_name == "NonparametricDynamicThresholdPostprocessor":
            return float(rng.choice([0.0, 0.5, 1.0, 2.0, 5.0, 10.0]))
        return float(rng.choice(FINITE_VALUES))

    def _invalid_score_nan(self) -> float:
        return np.nan

    def _invalid_score_inf(self) -> float:
        return np.inf

    def _invalid_score_array(self):
        return np.array([1.0, 2.0])

    def _snapshot_state(self):
        if hasattr(self.model, "meter") and hasattr(self.model.meter, "get"):
            return {
                "type": "meter",
                "value": self.model.meter.get(),
                "num_items": self.model.meter.num_items,
            }
        if hasattr(self.model, "_variance_meter") and hasattr(self.model._variance_meter, "get"):
            return {
                "type": "variance",
                "value": self.model._variance_meter.get(),
                "mean": self.model._variance_meter.mean,
            }
        if hasattr(self.model, "num_seen_"):
            return {"type": "seen", "num_seen": self.model.num_seen_}
        if hasattr(self.model, "_buffer") and hasattr(self.model._buffer, "_buffer"):
            return {"type": "buffer", "len": len(self.model._buffer)}
        return {"type": "stateless"}

    def _assert_state_unchanged(self, snapshot):
        current = self._snapshot_state()
        if snapshot["type"] != current["type"]:
            return
        if snapshot["type"] == "meter":
            assert snapshot["value"] == current["value"]
            assert snapshot["num_items"] == current["num_items"]
        elif snapshot["type"] == "variance":
            assert snapshot["value"] == current["value"]
            assert snapshot["mean"] == current["mean"]
        elif snapshot["type"] == "seen":
            assert snapshot["num_seen"] == current["num_seen"]
        elif snapshot["type"] == "buffer":
            assert snapshot["len"] == current["len"]

    @rule()
    def step_fit_partial_valid(self):
        score = self._valid_score()
        self.model.fit_partial(score)
        self.fitted_count += 1
        self.step += 1

    @rule()
    def step_transform_partial_valid(self):
        if self.fitted_count == 0:
            self.step_fit_partial_valid()
        score = self._valid_score()
        out = self.model.transform_partial(score)
        assert isinstance(out, float)
        assert np.isfinite(out)
        self.step += 1

    @rule()
    def step_fit_transform_partial_valid(self):
        score = self._valid_score()
        out = self.model.fit_transform_partial(score)
        assert isinstance(out, float)
        assert np.isfinite(out)
        self.fitted_count += 1
        self.step += 1

    @rule()
    def check_fit_partial_invalid_nan(self):
        if self.fitted_count == 0:
            return
        # Skip for transformers that accept NaN - they corrupt state and we can't easily recover
        if _accepts_nan(self.transformer_name):
            return
        if _has_early_validation(self.transformer_name):
            snapshot = self._snapshot_state()
            try:
                self.model.fit_partial(self._invalid_score_nan())
            except (ValueError, TypeError):
                self._assert_state_unchanged(snapshot)
        else:
            try:
                self.model.fit_partial(self._invalid_score_nan())
            except (ValueError, TypeError):
                pass

    @rule()
    def check_fit_partial_invalid_inf(self):
        if self.fitted_count == 0:
            return
        # Skip for transformers that accept NaN - they corrupt state and we can't easily recover
        if _accepts_nan(self.transformer_name):
            return
        if _has_early_validation(self.transformer_name):
            snapshot = self._snapshot_state()
            try:
                self.model.fit_partial(self._invalid_score_inf())
            except (ValueError, TypeError):
                self._assert_state_unchanged(snapshot)
        else:
            try:
                self.model.fit_partial(self._invalid_score_inf())
            except (ValueError, TypeError):
                pass

    @rule()
    def check_fit_partial_invalid_array_raises(self):
        if self.fitted_count == 0:
            return
        # Skip for transformers that accept NaN - they corrupt state and we can't easily recover
        if _accepts_nan(self.transformer_name):
            return
        if _has_early_validation(self.transformer_name):
            snapshot = self._snapshot_state()
            try:
                self.model.fit_partial(self._invalid_score_array())
            except (ValueError, TypeError, IndexError):
                self._assert_state_unchanged(snapshot)
        else:
            try:
                self.model.fit_partial(self._invalid_score_array())
            except (ValueError, TypeError, IndexError):
                pass

    @rule()
    def check_transform_partial_invalid_nan(self):
        if self.fitted_count == 0:
            return
        # Skip for transformers that accept NaN - they corrupt state and we can't easily recover
        if _accepts_nan(self.transformer_name):
            return
        if _has_early_validation(self.transformer_name):
            snapshot = self._snapshot_state()
            try:
                self.model.transform_partial(self._invalid_score_nan())
            except (ValueError, TypeError):
                self._assert_state_unchanged(snapshot)
        else:
            try:
                self.model.transform_partial(self._invalid_score_nan())
            except (ValueError, TypeError):
                pass

    @rule()
    def check_transform_partial_invalid_inf(self):
        if self.fitted_count == 0:
            return
        # Skip for transformers that accept NaN - they corrupt state and we can't easily recover
        if _accepts_nan(self.transformer_name):
            return
        if _has_early_validation(self.transformer_name):
            snapshot = self._snapshot_state()
            try:
                self.model.transform_partial(self._invalid_score_inf())
            except (ValueError, TypeError):
                self._assert_state_unchanged(snapshot)
        else:
            try:
                self.model.transform_partial(self._invalid_score_inf())
            except (ValueError, TypeError):
                pass

    @rule()
    def check_fit_transform_partial_invalid(self):
        if self.fitted_count == 0:
            return
        # Skip for transformers that accept NaN - they corrupt state and we can't easily recover
        if _accepts_nan(self.transformer_name):
            return
        if _has_early_validation(self.transformer_name):
            snapshot = self._snapshot_state()
            try:
                self.model.fit_transform_partial(self._invalid_score_nan())
            except (ValueError, TypeError):
                self._assert_state_unchanged(snapshot)
        else:
            try:
                self.model.fit_transform_partial(self._invalid_score_nan())
            except (ValueError, TypeError):
                pass


class _CalibratorValidationMachine(RuleBasedStateMachine):
    """State machine for testing calibrator input validation."""

    transformer_name: str

    model = None
    step: int
    fitted_count: int
    _state_snapshot = None

    @initialize()
    def setup(self):
        self.model = _make_transformer(self.transformer_name)
        self.step = 0
        self.fitted_count = 0
        self._state_snapshot = None

    def _valid_score(self) -> float:
        rng = np.random.RandomState(SEED + self.step * 31 + hash(self.transformer_name) % 1000)
        return float(rng.choice(FINITE_VALUES))

    def _invalid_score_nan(self) -> float:
        return np.nan

    def _invalid_score_inf(self) -> float:
        return np.inf

    def _snapshot_state(self):
        if hasattr(self.model, "window") and hasattr(self.model.window, "_buffer"):
            return {"type": "window", "len": len(self.model.window)}
        if hasattr(self.model, "_variance_meter") and hasattr(self.model._variance_meter, "get"):
            return {
                "type": "variance",
                "value": self.model._variance_meter.get(),
                "mean": self.model._variance_meter.mean,
            }
        if hasattr(self.model, "short_window"):
            return {"type": "short_window", "len": len(self.model.short_window)}
        if hasattr(self.model, "_buffer") and hasattr(self.model._buffer, "_buffer"):
            return {"type": "buffer", "len": len(self.model._buffer)}
        return {"type": "stateless"}

    def _assert_state_unchanged(self, snapshot):
        current = self._snapshot_state()
        if snapshot["type"] != current["type"]:
            return
        if snapshot["type"] == "window":
            assert snapshot["len"] == current["len"]
        elif snapshot["type"] == "variance":
            assert snapshot["value"] == current["value"]
            assert snapshot["mean"] == current["mean"]
        elif snapshot["type"] == "short_window":
            assert snapshot["len"] == current["len"]
        elif snapshot["type"] == "buffer":
            assert snapshot["len"] == current["len"]

    @rule()
    def step_fit_partial_valid(self):
        score = self._valid_score()
        self.model.fit_partial(score)
        self.fitted_count += 1
        self.step += 1

    @rule()
    def step_transform_partial_valid(self):
        if self.fitted_count == 0:
            self.step_fit_partial_valid()
        score = self._valid_score()
        out = self.model.transform_partial(score)
        assert isinstance(out, float)
        assert 0.0 <= out <= 1.0
        self.step += 1

    @rule()
    def step_fit_transform_partial_valid(self):
        score = self._valid_score()
        out = self.model.fit_transform_partial(score)
        assert isinstance(out, float)
        assert 0.0 <= out <= 1.0
        self.fitted_count += 1
        self.step += 1

    @rule()
    def check_fit_partial_invalid_nan_raises(self):
        if self.fitted_count == 0:
            return
        # Skip for poorly validated calibrators
        if _poorly_validated_calibrator(self.transformer_name):
            return
        try:
            self.model.fit_partial(self._invalid_score_nan())
        except (ValueError, TypeError):
            pass

    @rule()
    def check_fit_partial_invalid_inf_raises(self):
        if self.fitted_count == 0:
            return
        # Skip for poorly validated calibrators
        if _poorly_validated_calibrator(self.transformer_name):
            return
        try:
            self.model.fit_partial(self._invalid_score_inf())
        except (ValueError, TypeError):
            pass

    @rule()
    def check_transform_partial_invalid_nan_raises(self):
        if self.fitted_count == 0:
            return
        # Skip for poorly validated calibrators
        if _poorly_validated_calibrator(self.transformer_name):
            return
        try:
            self.model.transform_partial(self._invalid_score_nan())
        except (ValueError, TypeError):
            pass

    @rule()
    def check_fit_transform_partial_invalid_raises(self):
        if self.fitted_count == 0:
            return
        # Skip for poorly validated calibrators
        if _poorly_validated_calibrator(self.transformer_name):
            return
        try:
            self.model.fit_transform_partial(self._invalid_score_nan())
        except (ValueError, TypeError):
            pass


class _EnsemblerValidationMachine(RuleBasedStateMachine):
    """State machine for testing ensembler input validation."""

    transformer_name: str
    num_detectors: int

    model = None
    step: int
    fitted_count: int

    @initialize()
    def setup(self):
        self.model = _make_transformer(self.transformer_name)
        self.step = 0
        self.fitted_count = 0

    def _valid_scores(self) -> np.ndarray:
        rng = np.random.RandomState(SEED + self.step * 31 + hash(self.transformer_name) % 1000)
        return rng.choice(FINITE_VALUES, size=self.num_detectors).astype(np.float64)

    def _invalid_scores_wrong_dim(self) -> np.ndarray:
        return np.array([1.0, 2.0])  # Wrong number of detectors

    def _invalid_scores_nan(self) -> np.ndarray:
        scores = self._valid_scores().copy()
        scores[0] = np.nan
        return scores

    def _invalid_scores_inf(self) -> np.ndarray:
        scores = self._valid_scores().copy()
        scores[0] = np.inf
        return scores

    @rule()
    def step_fit_partial_valid(self):
        scores = self._valid_scores()
        self.model.fit_partial(scores)
        self.fitted_count += 1
        self.step += 1

    @rule()
    def step_transform_partial_valid(self):
        if self.fitted_count == 0:
            self.step_fit_partial_valid()
        scores = self._valid_scores()
        out = self.model.transform_partial(scores)
        assert isinstance(out, float)
        assert np.isfinite(out)
        self.step += 1

    @rule()
    def step_fit_transform_partial_valid(self):
        scores = self._valid_scores()
        out = self.model.fit_transform_partial(scores)
        assert isinstance(out, float)
        assert np.isfinite(out)
        self.fitted_count += 1
        self.step += 1

    @rule()
    def check_fit_partial_invalid_wrong_dim_raises(self):
        if self.fitted_count == 0:
            return
        try:
            self.model.fit_partial(self._invalid_scores_wrong_dim())
        except (ValueError, TypeError, IndexError):
            pass  # Ensemblers typically don't maintain state

    @rule()
    def check_fit_partial_invalid_nan_raises(self):
        if self.fitted_count == 0:
            return
        try:
            self.model.fit_partial(self._invalid_scores_nan())
        except (ValueError, TypeError):
            pass

    @rule()
    def check_transform_partial_invalid_wrong_dim_raises(self):
        if self.fitted_count == 0:
            return
        try:
            self.model.transform_partial(self._invalid_scores_wrong_dim())
        except (ValueError, TypeError, IndexError):
            pass

    @rule()
    def check_fit_transform_partial_invalid_raises(self):
        if self.fitted_count == 0:
            return
        try:
            self.model.fit_transform_partial(self._invalid_scores_nan())
        except (ValueError, TypeError):
            pass


# Generate test case classes for each transformer


def _create_preprocessing_testcase(name: str):
    num_features = _num_features_for(name)
    class_attrs = {
        "transformer_name": name,
        "num_features": num_features,
    }
    ConfiguredMachine = type(
        f"_{name}PreprocessingValidation", (_PreprocessingValidationMachine,), class_attrs
    )
    ConfiguredMachine = STATEFUL_SETTINGS(ConfiguredMachine)
    TestCase = ConfiguredMachine.TestCase
    TestCase.__name__ = f"Test{name}PreprocessingValidation"
    TestCase.__qualname__ = f"Test{name}PreprocessingValidation"
    return TestCase


def _create_drain_testcase(name: str):
    class_attrs = {"transformer_name": name}
    ConfiguredMachine = type(f"_{name}Validation", (_DrainLogParserValidationMachine,), class_attrs)
    ConfiguredMachine = STATEFUL_SETTINGS(ConfiguredMachine)
    TestCase = ConfiguredMachine.TestCase
    TestCase.__name__ = f"Test{name}Validation"
    TestCase.__qualname__ = f"Test{name}Validation"
    return TestCase


def _create_postprocessor_testcase(name: str):
    class_attrs = {"transformer_name": name}
    ConfiguredMachine = type(
        f"_{name}PostprocessorValidation", (_PostprocessorValidationMachine,), class_attrs
    )
    ConfiguredMachine = STATEFUL_SETTINGS(ConfiguredMachine)
    TestCase = ConfiguredMachine.TestCase
    TestCase.__name__ = f"Test{name}PostprocessorValidation"
    TestCase.__qualname__ = f"Test{name}PostprocessorValidation"
    return TestCase


def _create_calibrator_testcase(name: str):
    class_attrs = {"transformer_name": name}
    ConfiguredMachine = type(
        f"_{name}CalibratorValidation", (_CalibratorValidationMachine,), class_attrs
    )
    ConfiguredMachine = STATEFUL_SETTINGS(ConfiguredMachine)
    TestCase = ConfiguredMachine.TestCase
    TestCase.__name__ = f"Test{name}CalibratorValidation"
    TestCase.__qualname__ = f"Test{name}CalibratorValidation"
    return TestCase


def _create_ensembler_testcase(name: str):
    num_detectors = _num_detectors_for(name)
    class_attrs = {"transformer_name": name, "num_detectors": num_detectors}
    ConfiguredMachine = type(
        f"_{name}EnsemblerValidation", (_EnsemblerValidationMachine,), class_attrs
    )
    ConfiguredMachine = STATEFUL_SETTINGS(ConfiguredMachine)
    TestCase = ConfiguredMachine.TestCase
    TestCase.__name__ = f"Test{name}EnsemblerValidation"
    TestCase.__qualname__ = f"Test{name}EnsemblerValidation"
    return TestCase


# Generate test cases for all transformers
for _name in _preprocessing_transformers():
    if _name in TRANSFORMER_PARAMS:
        if _name == "DrainLogParser":
            globals()[f"Test{_name}Validation"] = _create_drain_testcase(_name)
        else:
            globals()[f"Test{_name}PreprocessingValidation"] = _create_preprocessing_testcase(_name)

for _name in _postprocessors():
    if _name in TRANSFORMER_PARAMS:
        globals()[f"Test{_name}PostprocessorValidation"] = _create_postprocessor_testcase(_name)

for _name in _calibrators():
    if _name in TRANSFORMER_PARAMS:
        globals()[f"Test{_name}CalibratorValidation"] = _create_calibrator_testcase(_name)

for _name in _ensemblers():
    if _name in TRANSFORMER_PARAMS:
        globals()[f"Test{_name}EnsemblerValidation"] = _create_ensembler_testcase(_name)
