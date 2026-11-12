import functools
import inspect
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
    cls = getattr(pysad.models, name)
    kind = getattr(cls, "_input_kind", None)
    if kind is not None and kind != "multivariate":
        return str(kind)
    if hasattr(sys.modules[cls.__module__], "check_univariate"):
        return "univariate"
    return str(kind or "multivariate")


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


@functools.cache
def _models_changed_since(base: str) -> tuple[str, ...]:
    changed = set(
        subprocess.run(
            ["git", "-C", str(ROOT), "diff", "--name-only", base],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.split()
    )
    if _SHARED_MODEL_TEST_FILES & changed:
        return tuple(pysad.models.__all__)
    if "pyproject.toml" in changed:
        before = subprocess.run(
            ["git", "-C", str(ROOT), "show", f"{base}:pyproject.toml"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout
        after = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        if before.split("\ndocs = [", 1)[0] != after.split("\ndocs = [", 1)[0]:
            return tuple(pysad.models.__all__)
    return tuple(name for name in pysad.models.__all__ if _model_pysad_files(name) & changed)


def active_model_names() -> tuple[str, ...]:
    """Returns the model names to run in shared model suites (filtered when PYSAD_TESTS_SINCE is set)."""
    base = os.environ.get("PYSAD_TESTS_SINCE")
    if not base:
        return tuple(pysad.models.__all__)
    return _models_changed_since(base)


def _auto_short_stream_params(cls: type[Any], num_features: int) -> dict[str, Any]:
    """Derives required constructor arguments for a newly added detector from its signature."""
    custom = getattr(cls, "_short_stream_params", None)
    if callable(custom):
        return dict(custom(num_features))
    if isinstance(custom, dict):
        return dict(custom)
    sig = inspect.signature(cls.__init__)
    defaults: dict[str, Any] = {
        "feature_mins": np.zeros(num_features),
        "feature_maxes": np.ones(num_features),
        "min_val": 0.0,
        "max_val": 1.0,
        "period": 2,
        "input_shape": num_features,
        "initial_X": np.random.RandomState(0).rand(12, num_features),
        "window_length": 3,
        "subsequence_length": 2,
        "probationary_period": 10,
        "window_size": 8,
    }
    return {
        p.name: defaults[p.name]
        for p in sig.parameters.values()
        if p.name != "self"
        and p.default is inspect.Parameter.empty
        and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        and p.name in defaults
    }


def short_stream_params(name: str, num_features: int = 3) -> dict[str, Any]:
    """Returns constructor arguments for `name` sized to `num_features` and short test streams."""
    ones, zeros = [1.0] * num_features, [0.0] * num_features
    params: dict[str, dict[str, Any]] = {
        "HalfSpaceTrees": {
            "feature_mins": np.array(zeros),
            "feature_maxes": np.array(ones),
            "window_size": 10,
            "num_trees": 5,
            "max_depth": 5,
        },
        "HTM": {"min_val": 0.0, "max_val": 1.0, "num_columns": 256},
        "HoltWinters": {"period": 2},
        "IForestASD": {"window_size": 20, "n_estimators": 10},
        "Inqmad": {"input_shape": num_features, "dim_x": 16},
        "MemStream": {"window_size": 4, "num_epochs": 20},
        "KitNet": {"grace_feature_mapping": 10, "grace_anomaly_detector": 10},
        "KNNCAD": {"probationary_period": 48},
        "MatrixProfile": {"window_length": 3},
        "DAMP": {"window_length": 3, "probationary_period": 4},
        "SpectralResidual": {"window_size": 8, "num_average_points": 3},
        "LocalOutlierProbability": {
            "initial_X": np.random.RandomState(0).rand(12, num_features),
            "num_neighbors": 10,
        },
        "OeSNNUAD": {"window_size": 10, "num_output_neurons": 5},
        "PCBIsolationForest": {"num_trees": 10, "window_size": 32, "stat_size": 10},
        "OnlineIsolationForest": {"num_trees": 5, "max_leaf_samples": 4, "window_size": 10},
        "RelativeEntropy": {"min_val": 0.0, "max_val": 1.0, "window_size": 5, "num_bins": 5},
        "RobustRandomCutForest": {"num_trees": 3, "shingle_size": 2, "tree_size": 16},
        "RSForest": {"window_size": 5, "num_trees": 5, "max_depth": 5},
        "RSHash": {
            "feature_mins": np.array(zeros),
            "feature_maxes": np.array(ones),
            "num_components": 10,
        },
        "SAND": {"subsequence_length": 2, "centroid_length": 4, "num_clusters": 2, "batch_size": 6},
        "SeasonalESD": {"period": 2, "window_size": 8, "max_anomalies": 0.25},
        "SeasonalHybridESD": {"period": 2, "window_size": 8, "max_anomalies": 0.25},
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
            "model": pysad.models.LODA(),
            "preprocessors": [ShingleTransformer(shingle_size=2)],
            "postprocessors": [ZScorePostprocessor(window_size=10)],
        },
        "xStream": {"num_components": 10, "n_chains": 10, "depth": 5, "window_size": 10},
    }
    if name in params:
        return params[name]
    return _auto_short_stream_params(getattr(pysad.models, name), num_features)


@pytest.fixture
def test_path():  # adapted from https://github.com/scikit-multiflow/scikit-multiflow/blob/master/tests/anomaly_detection/test_half_space_trees.py
    return os.path.dirname(os.path.abspath(__file__))
