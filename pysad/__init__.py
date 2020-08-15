"""
An open-source python framework for anomaly detection on streaming multivariate data.
"""
from .version import __version__
from . import core, evaluation, models, statistics, transform, utils

__all__ = ['__version__', 'core', "evaluation", "models", "statistics", "transform", "utils"]
