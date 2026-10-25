from __future__ import annotations

import heapq
from bisect import bisect_left, insort

from pysad.core.base_statistic import UnivariateStatistic


class MedianMeter(UnivariateStatistic):
    """The statistic that keeps track of the median.

    Uses a sorted list with :func:`bisect.insort` when tracking 1,000 items or fewer,
    and switches to two heaps with lazy deletion for windows exceeding 1,000 items
    to bound per-instance latency to :math:`O(\\log W)`.

    Args:
        high_threshold (int): The maximum number of items to track via sorted list
            before transitioning to two heaps (Default=1000).
        low_threshold (int): The threshold below which two heaps are converted
            back to a sorted list (Default=800).

    Attributes:
        num_items (int): The number of items that are used to update the statistic.
        lst (list[float]): The list of values that are used to update the statistic
            when in list mode.
    """

    def __init__(self, high_threshold: int = 1000, low_threshold: int = 800) -> None:
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.num_items = 0
        self.is_heap = False

        self.lst: list[float] = []
        self.small: list[float] = []
        self.large: list[float] = []
        self.small_size = 0
        self.large_size = 0
        self.delayed: dict[float, int] = {}
        self.counts: dict[float, int] = {}

    def _prune(self, heap: list[float], is_small: bool) -> None:
        sign = -1.0 if is_small else 1.0
        while heap:
            val = sign * heap[0]
            cnt = self.delayed.get(val, 0)
            if cnt > 0:
                if cnt == 1:
                    del self.delayed[val]
                else:
                    self.delayed[val] = cnt - 1
                heapq.heappop(heap)
            else:
                break

    def _balance(self) -> None:
        if self.small_size > self.large_size + 1:
            self._prune(self.small, is_small=True)
            val = -heapq.heappop(self.small)
            heapq.heappush(self.large, val)
            self.small_size -= 1
            self.large_size += 1
            self._prune(self.small, is_small=True)
        elif self.small_size < self.large_size:
            self._prune(self.large, is_small=False)
            val = heapq.heappop(self.large)
            heapq.heappush(self.small, -val)
            self.large_size -= 1
            self.small_size += 1
            self._prune(self.large, is_small=False)

    def _to_heaps(self) -> None:
        for x in self.lst:
            self.counts[x] = self.counts.get(x, 0) + 1
        mid = (len(self.lst) + 1) // 2
        self.small = [-x for x in self.lst[:mid]]
        self.small_size = len(self.small)
        heapq.heapify(self.small)
        self.large = list(self.lst[mid:])
        self.large_size = len(self.large)
        heapq.heapify(self.large)
        self.lst.clear()
        self.is_heap = True
        self._balance()

    def _to_list(self) -> None:
        lst: list[float] = []
        for val, c in self.counts.items():
            lst.extend([val] * c)
        lst.sort()
        self.lst = lst
        self.small.clear()
        self.large.clear()
        self.delayed.clear()
        self.counts.clear()
        self.small_size = 0
        self.large_size = 0
        self.is_heap = False

    def update(self, num: float) -> MedianMeter:
        """Updates the statistic with the value for a timestep.

        Args:
            num (float): The incoming value, for which the statistic is used.

        Returns:
            object: self.
        """
        if not self.is_heap:
            if self.num_items < self.high_threshold:
                insort(self.lst, num)
                self.num_items += 1
                return self
            self._to_heaps()

        self.counts[num] = self.counts.get(num, 0) + 1
        self._prune(self.small, is_small=True)
        if not self.small or num <= -self.small[0]:
            heapq.heappush(self.small, -num)
            self.small_size += 1
        else:
            heapq.heappush(self.large, num)
            self.large_size += 1

        self._balance()
        self.num_items += 1

        return self

    def remove(self, num: float) -> MedianMeter:
        """Updates the statistic by removing a particular value.

        Args:
            num (float): The value to be removed.

        Returns:
            object: self.
        """
        if not self.is_heap:
            idx = bisect_left(self.lst, num)
            if idx < self.num_items and self.lst[idx] == num:
                del self.lst[idx]
                self.num_items -= 1
                return self
            raise ValueError(f"{num} not in MedianMeter")

        cnt = self.counts.get(num, 0)
        if cnt <= 0:
            raise ValueError(f"{num} not in MedianMeter")
        if cnt == 1:
            del self.counts[num]
        else:
            self.counts[num] = cnt - 1

        self._prune(self.small, is_small=True)
        if self.small and num <= -self.small[0]:
            self.small_size -= 1
            self.delayed[num] = self.delayed.get(num, 0) + 1
            if num == -self.small[0]:
                self._prune(self.small, is_small=True)
        else:
            self.large_size -= 1
            self.delayed[num] = self.delayed.get(num, 0) + 1
            self._prune(self.large, is_small=False)

        self._balance()
        self.num_items -= 1

        if self.num_items < self.low_threshold:
            self._to_list()

        return self

    def get(self) -> float:
        """Method to obtain the tracked statistic.

        Returns:
            float: The statistic.
        """
        if self.num_items == 0:
            return 0.0

        if not self.is_heap:
            if self.num_items % 2 == 0:
                return (self.lst[self.num_items // 2] + self.lst[self.num_items // 2 - 1]) / 2
            return self.lst[self.num_items // 2]

        self._prune(self.small, is_small=True)
        if self.small_size == self.large_size:
            self._prune(self.large, is_small=False)
            return (-self.small[0] + self.large[0]) / 2.0
        return float(-self.small[0])
