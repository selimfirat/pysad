from __future__ import annotations

import numbers

import numpy as np

from pysad.transform.stream_generators._base import _AdversarialStream


class SlowPoisonStream(_AdversarialStream):
    """Poisons a stream slowly by injecting anomalies at a gradually growing rate.

    Each transform_partial call receives a normal base point and returns either that point
    unchanged or a poisoned point shifted along a fixed attack direction. After the call,
    is_anomaly is True iff the returned point is an injected anomaly.

    The injection rate ramps up from zero to poison_rate over warmup points. Each poisoned point
    is shifted by magnitude standard deviations of the normal distribution along the attack
    direction. As the detector fits the poisoned points, its model drifts toward them and stops
    flagging them, which is the slow-poisoning effect this generator benchmarks.

    Args:
        random_state (int, np.random.RandomState or None): Seed or random number generator for
            the attack direction and the injection decisions. None draws from NumPy's global
            random state, which pysad.utils.fix_seed seeds (Default=None).
        poison_rate (float): Fraction of points to inject as anomalies, between 0 and 1
            (Default=0.1).
        magnitude (float): Length of the poison offset in standard deviations of the normal
            distribution, at least 0 (Default=5.0).
        warmup (int): Number of points over which the injection rate ramps from zero to
            poison_rate, at least 0 (Default=100).

    Raises:
        TypeError: If poison_rate or magnitude is not a real number, or warmup is not an int.
        ValueError: If poison_rate is outside [0, 1], or magnitude or warmup is negative.
    """

    poison_rate: float
    magnitude: float
    warmup: int
    _t: int

    def __init__(
        self,
        *,
        random_state: int | np.random.RandomState | None = None,
        poison_rate: float = 0.1,
        magnitude: float = 5.0,
        warmup: int = 100,
    ) -> None:
        if isinstance(poison_rate, bool) or not isinstance(poison_rate, numbers.Real):
            raise TypeError(f"poison_rate must be a real number, got {poison_rate!r}.")
        if not 0.0 <= poison_rate <= 1.0:
            raise ValueError(f"poison_rate must be in [0, 1], got {poison_rate}.")
        if isinstance(magnitude, bool) or not isinstance(magnitude, numbers.Real):
            raise TypeError(f"magnitude must be a real number, got {magnitude!r}.")
        if magnitude < 0.0:
            raise ValueError(f"magnitude must be at least 0, got {magnitude}.")
        if isinstance(warmup, bool) or not isinstance(warmup, numbers.Integral):
            raise TypeError(f"warmup must be an int, got {warmup!r}.")
        if warmup < 0:
            raise ValueError(f"warmup must be at least 0, got {warmup}.")
        super().__init__(random_state=random_state)
        self.poison_rate = float(poison_rate)
        self.magnitude = float(magnitude)
        self.warmup = int(warmup)
        self._t = 0

    def transform_partial(self, X: np.ndarray) -> np.ndarray:
        """Returns the next base point, either unchanged or poisoned.

        Args:
            X (np.float64 array of shape (num_features,)): Input feature vector.

        Returns:
            np.float64 array of shape (num_features,): The base point unchanged, or the base
                point shifted by magnitude standard deviations along the attack direction.

        Raises:
            ValueError: If the point is not a 1-D point, is empty, holds non-finite values, or
                has another number of features than the earlier points.
        """
        x = self._validate_point(X)
        direction = self._ensure_direction(x.shape[0])
        self._t += 1
        ramp = 1.0 if self.warmup <= 0 else min(1.0, self._t / self.warmup)
        rate = self.poison_rate * ramp
        if self._rng.random() < rate:
            out = x + self.magnitude * self._scale() * direction
            self._is_anomaly = True
        else:
            out = x.copy()
            self._is_anomaly = False
        return out
