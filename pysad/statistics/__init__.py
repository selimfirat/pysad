"""
The :mod:`pysad.statistics` module contains methods to keep track of statistics on streaming data.
"""

from .abs_statistic import AbsStatistic
from .adwin_meter import ADWINMeter
from .average_meter import AverageMeter
from .count_meter import CountMeter
from .ddm_meter import DDMMeter
from .ddsketch_meter import DDSketchMeter
from .eddm_meter import EDDMMeter
from .exponential_moving_average_meter import ExponentialMovingAverageMeter
from .fhddm_meter import FHDDMMeter
from .hddm_a_meter import HDDMAMeter
from .hddm_w_meter import HDDMWMeter
from .kswin_meter import KSWINMeter
from .max_meter import MaxMeter
from .median_meter import MedianMeter
from .min_meter import MinMeter
from .online_periodicity_meter import OnlinePeriodicityMeter
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
    "DDMMeter",
    "DDSketchMeter",
    "EDDMMeter",
    "ExponentialMovingAverageMeter",
    "FHDDMMeter",
    "HDDMAMeter",
    "HDDMWMeter",
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
