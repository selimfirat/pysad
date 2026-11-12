"""
The :mod:`pysad.transform.stream_generators` module provides adversarial stream generators that wrap a normal stream to benchmark detector resilience.
"""

from .slow_poison_stream import SlowPoisonStream
from .stealth_burst_stream import StealthBurstStream

__all__ = ["SlowPoisonStream", "StealthBurstStream"]
