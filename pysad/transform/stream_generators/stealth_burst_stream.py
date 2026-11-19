from __future__ import annotations

import numbers

import numpy as np

from pysad.transform.stream_generators._base import _AdversarialStream


class StealthBurstStream(_AdversarialStream):
    """Adds stealthy bursts of anomalies to a stream.

    Each transform_partial call receives a normal base point and returns either that point
    unchanged or a burst point shifted along a fixed attack direction. After the call,
    is_anomaly is True iff the returned point is an injected anomaly.

    Anomalies occur in short bursts, separated by normal points. The first burst starts after
    burst_every normal points, so a detector can learn the normal distribution first. Each burst
    lasts burst_length points and shifts them by magnitude standard deviations along the attack
    direction. Bursts are brief and low-magnitude, so per-point detectors can miss them, which is
    the stealth-burst effect this generator benchmarks.

    Args:
        random_state (int, np.random.RandomState or None): Seed or random number generator for
            the attack direction. None draws from NumPy's global random state, which
            pysad.utils.fix_seed seeds (Default=None).
        burst_every (int): Number of points between the starts of consecutive bursts, at least 1
            (Default=100).
        burst_length (int): Number of points in each burst, at least 1 (Default=10).
        magnitude (float): Length of the burst offset in standard deviations of the normal
            distribution, at least 0 (Default=3.0).

    Raises:
        TypeError: If burst_every or burst_length is not an int, or magnitude is not a real
            number.
        ValueError: If burst_every or burst_length is less than 1, or magnitude is negative.
    """

    burst_every: int
    burst_length: int
    magnitude: float
    _t: int

    def __init__(
        self,
        *,
        random_state: int | np.random.RandomState | None = None,
        burst_every: int = 100,
        burst_length: int = 10,
        magnitude: float = 3.0,
    ) -> None:
        if isinstance(burst_every, bool) or not isinstance(burst_every, numbers.Integral):
            raise TypeError(f"burst_every must be an int, got {burst_every!r}.")
        if burst_every < 1:
            raise ValueError(f"burst_every must be at least 1, got {burst_every}.")
        if isinstance(burst_length, bool) or not isinstance(burst_length, numbers.Integral):
            raise TypeError(f"burst_length must be an int, got {burst_length!r}.")
        if burst_length < 1:
            raise ValueError(f"burst_length must be at least 1, got {burst_length}.")
        if isinstance(magnitude, bool) or not isinstance(magnitude, numbers.Real):
            raise TypeError(f"magnitude must be a real number, got {magnitude!r}.")
        if magnitude < 0.0:
            raise ValueError(f"magnitude must be at least 0, got {magnitude}.")
        super().__init__(random_state=random_state)
        self.burst_every = int(burst_every)
        self.burst_length = int(burst_length)
        self.magnitude = float(magnitude)
        self._t = 0

    def transform_partial(self, X: np.ndarray) -> np.ndarray:
        """Returns the next base point, either unchanged or part of a burst.

        Args:
            X (np.float64 array of shape (num_features,)): Input feature vector.

        Returns:
            np.float64 array of shape (num_features,): The base point unchanged, or the base
                point shifted by magnitude standard deviations along the attack direction during
                a burst.

        Raises:
            ValueError: If the point is not a 1-D point, is empty, holds non-finite values, or
                has another number of features than the earlier points.
        """
        x = self._validate_point(X)
        direction = self._ensure_direction(x.shape[0])
        self._t += 1
        index = self._t - 1
        in_burst = (
            index >= self.burst_every
            and (index - self.burst_every) % self.burst_every < self.burst_length
        )
        if in_burst:
            out = x + self.magnitude * self._scale() * direction
            self._is_anomaly = True
        else:
            out = x.copy()
            self._is_anomaly = False
        return out
