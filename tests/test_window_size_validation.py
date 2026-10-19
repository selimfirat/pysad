from functools import partial

import pytest

from pysad.statistics import AverageMeter, RunningStatistic
from pysad.transform.postprocessing import (
    RunningAveragePostprocessor,
    RunningMaxPostprocessor,
    RunningMedianPostprocessor,
    RunningZScorePostprocessor,
)
from pysad.transform.probability_calibration import (
    ConformalProbabilityCalibrator,
    GaussianTailProbabilityCalibrator,
)
from pysad.utils import Window


@pytest.mark.parametrize("window_size", [0, -1])
@pytest.mark.parametrize(
    "component",
    [
        pytest.param(partial(RunningStatistic, AverageMeter), id="RunningStatistic"),
        Window,
        ConformalProbabilityCalibrator,
        GaussianTailProbabilityCalibrator,
        RunningAveragePostprocessor,
        RunningMaxPostprocessor,
        RunningMedianPostprocessor,
        RunningZScorePostprocessor,
    ],
)
def test_nonpositive_window_size_is_rejected(component, window_size):
    with pytest.raises(ValueError, match=r"window_size must be a positive integer\."):
        component(window_size=window_size)


@pytest.mark.parametrize("short_window_size", [0, -1])
def test_nonpositive_short_window_size_is_rejected(short_window_size):
    with pytest.raises(ValueError, match=r"short_window_size must be a positive integer\."):
        GaussianTailProbabilityCalibrator(short_window_size=short_window_size)


def test_running_statistic_with_single_item_window():
    statistic = RunningStatistic(AverageMeter, window_size=1)
    for value in [1.0, 3.0, 2.0]:
        assert statistic.update(value).get() == value


def test_conformal_calibrator_with_single_item_window():
    calibrator = ConformalProbabilityCalibrator(window_size=1)
    for value in [1.0, 3.0, 2.0]:
        assert calibrator.fit_transform_partial(value) == 0.0
        assert calibrator.window.get() == [value]
