from __future__ import annotations

import importlib
import importlib.util
import inspect
from functools import partial
from typing import Any

import pytest
from pyod.models.iforest import IForest

from pysad.evaluation import PrecisionMetric
from pysad.statistics import AverageMeter, RunningStatistic
from pysad.transform.probability_calibration import (
    ConformalProbabilityCalibrator,
    GaussianTailProbabilityCalibrator,
)

SUBPACKAGES: tuple[str, ...] = (
    "pysad.core",
    "pysad.evaluation",
    "pysad.integrations",
    "pysad.models",
    "pysad.statistics",
    "pysad.transform.ensemble",
    "pysad.transform.postprocessing",
    "pysad.transform.preprocessing",
    "pysad.transform.probability_calibration",
    "pysad.transform.projection",
    "pysad.utils",
)

REQUIRED_ARGS: dict[str, dict[str, Any]] = {
    "ApproxStorm": {"sample_fraction": 0.5},
    "HalfSpaceTrees": {"feature_mins": [0.0], "feature_maxes": [1.0]},
    "HoltWinters": {"period": 2},
    "IDK": {"input_shape": 1},
    "ModifiedSTLResidualTransformer": {"period": 2},
    "ReferenceWindowModel": {"model_cls": IForest, "sliding_size": 5},
    "RelativeEntropy": {"min_val": 0.0, "max_val": 1.0},
    "RollingMetric": {"metric_cls": PrecisionMetric},
    "RunningStatistic": {"statistic_cls": AverageMeter},
    "SeasonalESD": {"period": 2},
    "SeasonalHybridESD": {"period": 2},
    "WindowedMetric": {"metric_cls": PrecisionMetric},
}


def _discover_window_size_components() -> list[Any]:
    components: list[Any] = []
    seen: set[type] = set()
    for pkg in SUBPACKAGES:
        mod = importlib.import_module(pkg)
        for name in getattr(mod, "__all__", ()):
            if name.startswith("_"):
                continue
            if name == "RiverAnomalyDetector" and importlib.util.find_spec("river") is None:
                continue
            obj = getattr(mod, name)
            if not (inspect.isclass(obj) and not inspect.isabstract(obj)) or obj in seen:
                continue
            if "window_size" in inspect.signature(obj.__init__).parameters:
                seen.add(obj)
                factory = partial(obj, **REQUIRED_ARGS.get(name, {}))
                components.append(pytest.param(factory, id=name))
    return components


WINDOW_SIZE_COMPONENTS = _discover_window_size_components()


def test_discovers_all_windowed_components() -> None:
    assert len(WINDOW_SIZE_COMPONENTS) >= 42


@pytest.mark.parametrize("window_size", [0, -1])
@pytest.mark.parametrize("component", WINDOW_SIZE_COMPONENTS)
def test_nonpositive_window_size_is_rejected(component: Any, window_size: int) -> None:
    with pytest.raises(ValueError, match="window_size"):
        component(window_size=window_size)


@pytest.mark.parametrize("short_window_size", [0, -1])
def test_nonpositive_short_window_size_is_rejected(short_window_size: int) -> None:
    with pytest.raises(ValueError, match=r"short_window_size must be a positive integer\."):
        GaussianTailProbabilityCalibrator(short_window_size=short_window_size)


def test_running_statistic_with_single_item_window() -> None:
    statistic = RunningStatistic(AverageMeter, window_size=1)
    for value in [1.0, 3.0, 2.0]:
        assert statistic.update(value).get() == value


def test_conformal_calibrator_with_single_item_window() -> None:
    calibrator = ConformalProbabilityCalibrator(window_size=1)
    for value in [1.0, 3.0, 2.0]:
        assert calibrator.fit_transform_partial(value) == 0.0
        assert calibrator.window.get() == [value]
