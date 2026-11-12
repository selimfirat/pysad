from __future__ import annotations

import copy
import math

import numpy as np
from sklearn.utils import check_random_state

# The time-of-day encoder of NAB's numenta detector, NuPIC's DateEncoder(timeOfDay=(21, 9.49)): a
# periodic scalar encoder of the hour of the day with 21 active bits and a radius of 9.49 hours.
TIME_OF_DAY_WIDTH = 21
TIME_OF_DAY_SIZE = math.ceil(TIME_OF_DAY_WIDTH * (24.0 / 9.49))  # 54 bits
_TIME_OF_DAY_OFFSETS = np.arange(TIME_OF_DAY_WIDTH) - TIME_OF_DAY_WIDTH // 2


def encode_time_of_day(timestamp: float) -> np.ndarray:
    """Returns the indices of the active bits that encode the time of day of `timestamp`, as NuPIC's DateEncoder.

    Args:
        timestamp (float): Seconds since the Unix epoch. The time of day is that of UTC.

    Returns:
        np.int64 array of shape (TIME_OF_DAY_WIDTH,): Indices in [0, TIME_OF_DAY_SIZE).
    """
    # NuPIC uses the hour and the minute, and drops the seconds.
    hours = (timestamp % 86400.0) // 60.0 / 60.0
    center = int(hours * TIME_OF_DAY_SIZE / 24.0)
    return (center + _TIME_OF_DAY_OFFSETS) % TIME_OF_DAY_SIZE


class RandomDistributedScalarEncoder:
    """Encodes a value into a fixed number of active bits, so that close values share bits, as NuPIC's RandomDistributedScalarEncoder.

    This is an internal part of the HTM detector. It ports the encoder of NuPIC (https://github.com/numenta/nupic, MIT license), which NAB's numenta detector uses. The first value that the encoder learns sets the offset of the middle bucket. A value goes to the bucket ``max_buckets // 2 + round((value - offset) / resolution)``, clipped to ``[0, max_buckets)``. Each bucket has ``width`` active bits out of ``size``. Two buckets fewer than ``width`` apart share ``width`` minus their distance bits, and two buckets farther apart share at most 2 bits. The encoder creates a bucket the first time that a value falls in it, from the next bucket toward the middle, with one bit replaced at random until the overlaps hold.

    The random draws come from NumPy, so the bits differ from those of NuPIC with the same seed.

    Args:
        resolution (float): The width of a bucket in the units of the values.
        width (int): The number of active bits of a bucket (Default=21).
        size (int): The number of bits (Default=400).
        max_buckets (int): The number of buckets (Default=1000).
        random_state (int, np.random.RandomState or None): Seed or random number generator for the bits of the buckets. None draws from NumPy's global random state (Default=None).
    """

    def __init__(
        self,
        resolution: float,
        width: int = 21,
        size: int = 400,
        max_buckets: int = 1000,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        self.resolution = resolution
        self.width = width
        self.size = size
        self.max_buckets = max_buckets
        self._rng = check_random_state(random_state)
        self.offset_: float | None = None
        middle = max_buckets // 2
        self.buckets_ = np.zeros((max_buckets, width), dtype=np.int64)
        self.buckets_[middle] = self._rng.permutation(size)[:width]
        self.min_index_ = self.max_index_ = middle

    def encode(self, value: float, learn: bool = True) -> np.ndarray:
        """Returns the indices of the active bits of `value`, of shape (width,).

        If `learn` is False, the state does not change: a value that sets the offset or needs a new bucket is encoded on a copy of the encoder.
        """
        offset = value if self.offset_ is None else self.offset_
        # NuPIC rounds half away from zero, as Python 2 does.
        steps = (value - offset) / self.resolution
        index = self.max_buckets // 2 + int(math.copysign(math.floor(abs(steps) + 0.5), steps))
        index = min(max(index, 0), self.max_buckets - 1)
        known = self.min_index_ <= index <= self.max_index_
        if not learn and (self.offset_ is None or not known):
            return copy.deepcopy(self).encode(value)

        self.offset_ = offset
        while index > self.max_index_:
            self.buckets_[self.max_index_ + 1] = self._new_bucket(
                self.max_index_, self.max_index_ + 1
            )
            self.max_index_ += 1
        while index < self.min_index_:
            self.buckets_[self.min_index_ - 1] = self._new_bucket(
                self.min_index_, self.min_index_ - 1
            )
            self.min_index_ -= 1
        return self.buckets_[index].copy()

    def _new_bucket(self, index: int, new_index: int) -> np.ndarray:
        """Returns the bits of `new_index`, which differ in one bit from those of the next bucket `index`."""
        old = self.buckets_[index]
        bits = old.copy()
        # The existing buckets: (num_buckets, width).
        existing = self.buckets_[self.min_index_ : self.max_index_ + 1]
        distances = np.abs(np.arange(self.min_index_, self.max_index_ + 1) - new_index)
        near = distances < self.width
        mask = np.zeros(self.size, dtype=bool)
        while True:
            bit = self._rng.randint(self.size)
            if bit in old:
                continue
            bits[new_index % self.width] = bit
            mask[:] = False
            mask[bits] = True
            overlaps = mask[existing].sum(axis=1)
            if np.where(near, overlaps == self.width - distances, overlaps <= 2).all():
                return bits


# RandomDistributedScalarEncoder and encode_time_of_day are adapted from the encoders of NuPIC,
# which are distributed under the following license:
#
# MIT License
#
# Copyright 2013-2024 Numenta Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
