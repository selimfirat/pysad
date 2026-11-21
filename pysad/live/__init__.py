"""
The :mod:`pysad.live` module provides a live streaming anomaly detection CLI
and web dashboard for real-time visualization of anomaly scores, conformal
ribbons, and alert spikes.
"""

from .cli import main

__all__ = ["main"]
