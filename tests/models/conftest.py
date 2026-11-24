import functools
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

import pysad
import pysad.models
from pysad.transform.ensemble import AverageScoreEnsembler
from pysad.transform.postprocessing import ZScorePostprocessor
from pysad.transform.preprocessing import ShingleTransformer
from pysad.transform.probability_calibration import ConformalProbabilityCalibrator

ROOT = Path(__file__).resolve().parents[2]
_SHARED_MODEL_TEST_FILES = frozenset(
    {
        ".github/workflows/coverage.yml",
        "tests/models/conftest.py",
        "tests/models/test_discrimination.py",
        "tests/models/test_models.py",
        "tests/models/test_nan_input.py",
        "tests/models/test_pickling.py",
        "tests/models/test_properties.py",
        "tests/models/test_random_state.py",
        "tests/models/test_score_types.py",
    }
)


def _model_input_kind(name: str) -> str:
    return str(getattr(pysad.models, name)._effective_input_kind())


# Univariate models validate their input through check_univariate or set _input_kind = "univariate".
UNIVARIATE_MODELS = frozenset(
    name for name in pysad.models.__all__ if _model_input_kind(name) == "univariate"
)

# These models take each instance as [timestamp, value].
TIMESTAMPED_MODELS = frozenset(
    name for name in pysad.models.__all__ if _model_input_kind(name) == "timestamped"
)

# These models take each instance as an edge [source, destination].
EDGE_MODELS = frozenset(name for name in pysad.models.__all__ if _model_input_kind(name) == "edge")


def num_features_for(name: str, default: int = 3) -> int:
    """Returns the number of input features expected by model `name`."""
    if name in TIMESTAMPED_MODELS or name in EDGE_MODELS:
        return 2
    return 1 if name in UNIVARIATE_MODELS else default


def _objects_pysad_files(objects: list[Any]) -> set[str]:
    """Returns the repository-relative pysad source files that `objects` depend on."""
    files: set[str] = set()
    todo = [obj.__module__ for obj in objects]
    pysad_root = Path(pysad.__file__).parents[1]
    while todo:
        module = sys.modules.get(todo.pop())
        if module is None or module.__file__ is None:
            continue
        path = Path(module.__file__).relative_to(pysad_root).as_posix()
        if path in files:
            continue
        files.add(path)
        if path.endswith("/__init__.py"):
            continue
        for value in vars(module).values():
            mod_name = (
                value.__name__
                if isinstance(value, ModuleType)
                else getattr(value, "__module__", "")
            )
            if isinstance(mod_name, str) and mod_name.split(".")[0] == "pysad":
                todo.append(mod_name)
    return files


def _model_pysad_files(name: str) -> set[str]:
    """Returns the repository-relative pysad source files that model `name` depends on."""
    objects: list[Any] = [getattr(pysad.models, name)]
    if name == "ModelEnsemble":
        objects += [
            pysad.models.LODA,
            pysad.models.xStream,
            ZScorePostprocessor,
            AverageScoreEnsembler,
        ]
    if name == "StreamPipeline":
        objects += [
            pysad.models.LODA,
            ShingleTransformer,
            ZScorePostprocessor,
            ConformalProbabilityCalibrator,
        ]
    return _objects_pysad_files(objects)


@functools.cache
def _changed_files_since(base: str) -> frozenset[str]:
    return frozenset(
        subprocess.run(
            ["git", "-C", str(ROOT), "diff", "--name-only", base],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.split()
    )


@functools.cache
def _deps_changed_since(base: str) -> bool:
    if "pyproject.toml" not in _changed_files_since(base):
        return False
    before = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{base}:pyproject.toml"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout
    after = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    return before.split("\ndocs = [", 1)[0] != after.split("\ndocs = [", 1)[0]


@functools.cache
def _models_changed_since(base: str) -> tuple[str, ...]:
    changed = _changed_files_since(base)
    if (_SHARED_MODEL_TEST_FILES & changed) or _deps_changed_since(base):
        return tuple(pysad.models.__all__)
    return tuple(name for name in pysad.models.__all__ if _model_pysad_files(name) & changed)


def is_component_active(obj: Any) -> bool:
    """Returns whether non-model component `obj` should run under PYSAD_TESTS_SINCE."""
    base = os.environ.get("PYSAD_TESTS_SINCE")
    if not base:
        return True
    changed = _changed_files_since(base)
    if (_SHARED_MODEL_TEST_FILES & changed) or _deps_changed_since(base):
        return True
    return bool(_objects_pysad_files([obj]) & changed)


def active_model_names() -> tuple[str, ...]:
    """Returns the model names to run in shared model suites (filtered when PYSAD_TESTS_SINCE is set)."""
    base = os.environ.get("PYSAD_TESTS_SINCE")
    if not base:
        return tuple(pysad.models.__all__)
    return _models_changed_since(base)


def _default_required_values(
    num_features: int, X: np.ndarray | None = None, seed: int = 0
) -> dict[str, Any]:
    if X is not None:
        arr = np.atleast_2d(np.asarray(X, dtype=np.float64))
        feature_mins = arr.min(axis=0)
        feature_maxes = arr.max(axis=0)
        min_val = float(arr[:, -1].min())
        max_val = float(arr[:, -1].max())
        initial_x = arr[: min(100, len(arr))]
    else:
        feature_mins = np.zeros(num_features, dtype=np.float64)
        feature_maxes = np.ones(num_features, dtype=np.float64)
        min_val = 0.0
        max_val = 1.0
        initial_x = np.random.RandomState(seed).rand(12, num_features)
    return {
        "feature_mins": feature_mins,
        "feature_maxes": feature_maxes,
        "min_val": min_val,
        "max_val": max_val,
        "period": 2,
        "input_shape": num_features,
        "initial_X": initial_x,
        "window_length": 3,
        "subsequence_length": 2,
        "probationary_period": 10,
        "window_size": 8,
    }


def _auto_short_stream_params(
    cls: type[Any], num_features: int, *, X: np.ndarray | None = None, seed: int = 0
) -> dict[str, Any]:
    """Derives required constructor arguments for a newly added detector from its signature."""
    custom = getattr(cls, "_short_stream_params", None)
    if callable(custom):
        return dict(custom(num_features))
    if isinstance(custom, dict):
        return dict(custom)
    defaults = _default_required_values(num_features, X=X, seed=seed)
    return {name: defaults[name] for name in cls._required_init_params() if name in defaults}


def short_stream_params(
    name: str, num_features: int = 3, *, X: np.ndarray | None = None, seed: int = 0
) -> dict[str, Any]:
    """Returns constructor arguments for `name` sized to `num_features` and short test streams."""
    defaults = _default_required_values(num_features, X=X, seed=seed)
    mins = defaults["feature_mins"]
    maxes = defaults["feature_maxes"]
    min_val = defaults["min_val"]
    max_val = defaults["max_val"]
    initial_x = defaults["initial_X"]
    params: dict[str, dict[str, Any]] = {
        "AnyOut": {"max_height": 2, "decay": 0.1, "snapshot_interval": 5},
        "ApproxStorm": {
            "window_size": 20,
            "max_radius": 0.5,
            "num_neighbors": 3,
            "sample_fraction": 0.25,
        },
        "CPOD": {"window_size": 20, "max_radius": 0.5},
        "D3": {"window_size": 20, "sample_fraction": 0.5, "max_radius": 0.3},
        "DAMP": {"window_length": 3, "probationary_period": 4, "history_size": 20},
        "DenStream": {
            "decaying_factor": 0.05,
            "epsilon": 1.5,
            "beta": 0.5,
            "mu": 3,
            "init_samples": 3,
        },
        "DILOF": {"num_neighbors": 3, "window_size": 12},
        "DriftAdaptiveDetector": {
            "model_factory": functools.partial(
                pysad.models.LODA, num_bins=5, num_random_cuts=10, random_state=0
            )
        },
        "ExactStorm": {"window_size": 20, "max_radius": 0.5},
        "EXPoSE": {"num_random_features": 50},
        "FeedbackHST": {
            "feature_mins": mins,
            "feature_maxes": maxes,
            "window_size": 10,
            "num_trees": 5,
            "max_depth": 5,
            "feedback_budget": 10,
        },
        "FluxEV": {
            "window_size": 3,
            "period": 4,
            "num_periods": 2,
            "drift": 1,
            "calibration_size": 20,
        },
        "HalfSpaceTrees": {
            "feature_mins": mins,
            "feature_maxes": maxes,
            "window_size": 10,
            "num_trees": 5,
            "max_depth": 5,
        },
        "HTM": {
            "min_val": min_val,
            "max_val": max_val,
            "probationary_period": 15,
            "num_columns": 64,
            "num_cells_per_column": 4,
        },
        "HoltWinters": {"period": 2, "window_size": 3, "threshold": 1},
        "IForestASD": {"window_size": 20, "n_estimators": 5},
        "IDKS": {
            "input_shape": num_features,
            "window_size": 8,
            "num_samples": 4,
            "num_partitions": 10,
        },
        "IncrementalLOF": {"num_neighbors": 3, "window_size": 12},
        "Inqmad": {"input_shape": num_features, "dim_x": 16, "initial_size": 15},
        "KELOS": {"window_size": 20, "radius": 0.5, "num_neighbors": 3},
        "KitNet": {"grace_feature_mapping": 10, "grace_anomaly_detector": 10},
        "KNNCAD": {"probationary_period": 48},
        "LODA": {"num_bins": 5, "num_random_cuts": 10},
        "LocalOutlierProbability": {
            "initial_X": initial_x,
            "num_neighbors": 10,
        },
        "MatrixProfile": {"window_length": 4, "history_size": 20},
        "MCOD": {"window_size": 20, "max_radius": 0.5, "num_neighbors": 2},
        "MemStream": {"window_size": 8, "num_epochs": 20},
        "MIDAS": {"num_buckets": 16, "tick_size": 5},
        "MiLOF": {"num_neighbors": 3, "window_size": 16, "num_summaries": 2},
        "NETS": {"window_size": 20, "max_radius": 0.5},
        "OeSNNUAD": {"window_size": 10, "num_output_neurons": 5},
        "OnlineIsolationForest": {"num_trees": 5, "max_leaf_samples": 4, "window_size": 10},
        "OnlineKMeans": {"num_clusters": 2},
        "OnlineOneClassSVM": {"window_size": 10},
        "PCBIsolationForest": {"num_trees": 5, "window_size": 20, "stat_size": 6},
        "RelativeEntropy": {
            "min_val": min_val,
            "max_val": max_val,
            "window_size": 5,
            "num_bins": 5,
        },
        "RobustRandomCutForest": {"num_trees": 4, "shingle_size": 2, "tree_size": 16},
        "RSForest": {"window_size": 20, "num_trees": 5, "max_depth": 6},
        "RSHash": {
            "feature_mins": mins,
            "feature_maxes": maxes,
            "num_components": 10,
        },
        "SAND": {
            "subsequence_length": 2,
            "centroid_length": 4,
            "num_clusters": 2,
            "batch_size": 10,
        },
        "SeasonalESD": {"period": 2, "window_size": 8, "max_anomalies": 0.25},
        "SeasonalHybridESD": {"period": 2, "window_size": 8, "max_anomalies": 0.25},
        "SpectralResidual": {"window_size": 8, "num_average_points": 3},
        "ThreshLEAP": {"window_size": 20, "max_radius": 0.5, "num_neighbors": 2},
        "ModelEnsemble": {
            "models": [
                pysad.models.LODA(num_bins=5, num_random_cuts=10, random_state=0)
                | ZScorePostprocessor(window_size=10),
                pysad.models.xStream(
                    num_components=10, n_chains=10, depth=5, window_size=10, random_state=1
                )
                | ZScorePostprocessor(window_size=10),
            ],
        },
        "StreamPipeline": {
            "model": pysad.models.LODA(num_bins=5, num_random_cuts=10),
            "preprocessors": [ShingleTransformer(shingle_size=2)],
            "postprocessors": [ZScorePostprocessor(window_size=10)],
        },
        "xStream": {"num_components": 10, "n_chains": 10, "depth": 5, "window_size": 10},
        "ZeroShotEmbeddingDetector": {
            "semantic_anchors": np.random.RandomState(seed).rand(10, num_features)
        },
    }
    if name in params:
        return params[name]
    return _auto_short_stream_params(getattr(pysad.models, name), num_features, X=X, seed=seed)


def required_params(
    name: str, num_features: int = 3, *, X: np.ndarray | None = None, seed: int = 0
) -> dict[str, Any]:
    """Returns only the required constructor parameters for `name` (empty for default-ready models)."""
    cls = getattr(pysad.models, name)
    required = set(cls._required_init_params())
    if not required:
        return {}
    candidates = short_stream_params(name, num_features, X=X, seed=seed)
    defaults = _default_required_values(num_features, X=X, seed=seed)
    return {
        param: candidates[param] if param in candidates else defaults[param]
        for param in cls._required_init_params()
        if param in candidates or param in defaults
    }


@pytest.fixture
def test_path():  # adapted from https://github.com/scikit-multiflow/scikit-multiflow/blob/master/tests/anomaly_detection/test_half_space_trees.py
    return os.path.dirname(os.path.abspath(__file__))
