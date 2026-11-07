"""
The :mod:`pysad.statistics` module contains methods to keep track of statistics on streaming data.
"""

from .abs_statistic import AbsStatistic
from .adwin_meter import ADWINMeter
from .average_meter import AverageMeter
from .count_meter import CountMeter
from .ddsketch_meter import DDSketchMeter
from .eddm_meter import EDDMMeter
from .exponential_moving_average_meter import ExponentialMovingAverageMeter
from .hddm_a_meter import HDDMAMeter
from .hddm_w_meter import HDDMWMeter
from .kswin_meter import KSWINMeter
from .max_meter import MaxMeter
from .median_meter import MedianMeter
from .min_meter import MinMeter
from .page_hinkley_meter import PageHinkleyMeter
from .quantile_meter import QuantileMeter
from .running_statistic import RunningStatistic
from .sum_meter import SumMeter
from .sum_squares_meter import SumSquaresMeter
from .variance_meter import VarianceMeter

__all__ = [
    "ADWINMeter",
    "AbsStatistic",
    "AverageMeter",
    "CountMeter",
    "DDSketchMeter",
    "EDDMMeter",
    "ExponentialMovingAverageMeter",
    "HDDMAMeter",
    "HDDMWMeter",
    "KSWINMeter",
    "MaxMeter",
    "MedianMeter",
    "MinMeter",
    "PageHinkleyMeter",
    "QuantileMeter",
    "RunningStatistic",
    "SumMeter",
    "SumSquaresMeter",
    "VarianceMeter",
]
