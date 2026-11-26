"""
The :mod:`pysad.statistics` module contains methods to keep track of statistics on streaming data.
"""

from .basic import (
    AbsStatistic,
    AverageMeter,
    CountMeter,
    MaxMeter,
    MinMeter,
    SumMeter,
    SumSquaresMeter,
    VarianceMeter,
)
from .cusum import CUSUMMeter
from .drift import (
    ADWINMeter,
    DDMMeter,
    EDDMMeter,
    FHDDMMeter,
    FHDDMSAddMeter,
    FHDDMSMeter,
    HDDMAMeter,
    HDDMWMeter,
    KSWINMeter,
    PageHinkleyMeter,
)
from .exponential_moving_average_meter import ExponentialMovingAverageMeter
from .exponential_moving_variance_meter import ExponentialMovingVarianceMeter
from .geometric_moving_average_meter import GeometricMovingAverageMeter
from .kll import KLLMeter
from .online_periodicity_meter import OnlinePeriodicityMeter
from .quantiles import DDSketchMeter, MedianMeter, QuantileMeter
from .relative_error_quantile_meter import RelativeErrorQuantileMeter
from .running_statistic import RunningStatistic
from .sketch import CountMinSketchStatistic, HyperLogLogStatistic

__all__ = [
    "AbsStatistic",
    "ADWINMeter",
    "AverageMeter",
    "CountMeter",
    "CountMinSketchStatistic",
    "CUSUMMeter",
    "DDMMeter",
    "DDSketchMeter",
    "EDDMMeter",
    "ExponentialMovingAverageMeter",
    "ExponentialMovingVarianceMeter",
    "FHDDMMeter",
    "FHDDMSAddMeter",
    "FHDDMSMeter",
    "GeometricMovingAverageMeter",
    "HDDMAMeter",
    "HDDMWMeter",
    "HyperLogLogStatistic",
    "KLLMeter",
    "KSWINMeter",
    "MaxMeter",
    "MedianMeter",
    "MinMeter",
    "OnlinePeriodicityMeter",
    "PageHinkleyMeter",
    "QuantileMeter",
    "RelativeErrorQuantileMeter",
    "RunningStatistic",
    "SumMeter",
    "SumSquaresMeter",
    "VarianceMeter",
]
