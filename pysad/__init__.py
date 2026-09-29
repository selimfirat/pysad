"""
An open-source python framework for anomaly detection on streaming multivariate data.
"""

from . import core, evaluation, models, statistics, transform, utils
from .version import __version__

__all__ = ["__version__", "core", "evaluation", "models", "statistics", "transform", "utils"]
