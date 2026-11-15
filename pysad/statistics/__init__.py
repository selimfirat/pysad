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
    FHDDMSAddMeter,
    FHDDMSMeter,
    HDDMAMeter,
    HDDMWMeter,
    KSWINMeter,
    PageHinkleyMeter,
)
from .exponential_moving_average_meter import ExponentialMovingAverageMeter
from .online_periodicity_meter import OnlinePeriodicityMeter
from .quantiles import DDSketchMeter, MedianMeter, QuantileMeter
from .relative_error_quantile_meter import RelativeErrorQuantileMeter
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
    "FHDDMSAddMeter",
    "FHDDMSMeter",
    "HDDMAMeter",
    "HDDMWMeter",
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
