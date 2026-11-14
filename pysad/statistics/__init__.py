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
from .drift import (
    ADWINMeter,
    DDMMeter,
    EDDMMeter,
    FHDDMMeter,
    HDDMAMeter,
    HDDMWMeter,
    KSWINMeter,
    PageHinkleyMeter,
)
from .exponential_moving_average_meter import ExponentialMovingAverageMeter
from .kll import KLLMeter
from .online_periodicity_meter import OnlinePeriodicityMeter
from .quantiles import DDSketchMeter, MedianMeter, QuantileMeter
from .running_statistic import RunningStatistic

__all__ = [
    "AbsStatistic",
    "ADWINMeter",
    "AverageMeter",
    "CountMeter",
    "DDMMeter",
    "DDSketchMeter",
    "EDDMMeter",
    "ExponentialMovingAverageMeter",
    "FHDDMMeter",
    "HDDMAMeter",
    "HDDMWMeter",
    "KLLMeter",
    "KSWINMeter",
    "MaxMeter",
    "MedianMeter",
    "MinMeter",
    "OnlinePeriodicityMeter",
    "PageHinkleyMeter",
    "QuantileMeter",
    "RunningStatistic",
    "SumMeter",
    "SumSquaresMeter",
    "VarianceMeter",
]
