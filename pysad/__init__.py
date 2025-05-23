"""
An open-source python framework for anomaly detection on streaming multivariate data.
"""
import warnings

# Filter out NumPy deprecation warnings to avoid noise in user applications
warnings.filterwarnings("ignore", message="numpy.core is deprecated")
warnings.filterwarnings("ignore", message="Conversion of an array with ndim > 0 to a scalar is deprecated")
warnings.filterwarnings("ignore", message="Calling nonzero on 0d arrays is deprecated")

from .version import __version__
from . import core, evaluation, models, statistics, transform, utils

__all__ = ['__version__', 'core', "evaluation", "models", "statistics", "transform", "utils"]
