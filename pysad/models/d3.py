from __future__ import annotations

from collections import deque

import numpy as np
from sklearn.utils import check_random_state

from pysad.core.base_model import BaseModel


class D3(BaseModel):
    """Distance-based outliers from a kernel density estimate of a sliding window, the D3 algorithm of one sensor :cite:`subramaniam2006online`.

    D3 models the values in the window of the last ``window_size`` values (W) with a kernel density estimate. The kernels sit on a uniform random sample R of the window, which the chain-sample algorithm of :cite:`babcock2002sampling` maintains, as in Section 5 of the paper. The sample has ``round(sample_fraction * window_size)`` values, drawn with replacement by independent chains. Each chain holds one sample value. When the i-th value arrives, a chain selects it with probability 1 / min(i, W) and draws the index of its successor uniformly from the next W values. A chain stores its successor when it arrives, and the successor replaces the sample value when that value leaves the window.

    The kernel is the Epanechnikov product kernel of Eq. 2, with the bandwidth ``B_i = sqrt(5) sigma_i |R|^(-1 / (d + 4))`` of Scott's rule in each of the d dimensions, where sigma_i is the standard deviation of the window values in dimension i. As in Eq. 4 and Eq. 5, the estimated number of window values within ``max_radius`` (r) of an instance p is ``N(p, r) = |W| P(p, r)``, where P(p, r) is the integral of the estimate over the box [p - r, p + r], in closed form. The paper reports p as an outlier if N(p, r) is less than a threshold t. This class returns the score -N(p, r), so that fewer neighbors give a higher score, and the outliers of the paper are the instances with a score above -t. In d dimensions, this class takes the interval of Eq. 4 as the box, so the neighborhood uses the maximum norm.

    Only the D3 process of a leaf sensor is ported. The sensor hierarchy, the propagation of samples to parent nodes and the MGDD algorithm are not. The deviations from the paper:

    - The paper estimates sigma with the variance histogram of :cite:`babcock2003maintaining`, with a bounded relative error. This class keeps the window values and computes the exact (population) standard deviation that the histogram estimates, so it needs O(W d) memory instead of O(d / epsilon^2 log W).
    - The paper adds a value to the model before it checks the value. This class scores each instance before ``fit_partial`` adds it, the order of :meth:`fit_score_partial`. Until the window is full, ``|W|`` is the number of values that the window holds so far.
    - The paper maps the values to [0, 1]^d. The default r = 0.01 is the radius of the experiments of the paper (Section 10), for values scaled to [0, 1]. The default ``sample_fraction`` is the ``|R| = 0.05 |W|`` of these experiments.
    - If sigma_i is 0, the kernel in dimension i is a point mass. An instance scored before any fit gets the score 0 of an instance with no neighbors.

    Args:
        window_size (int): The number of the most recent values in the window (Default=10000).
        sample_fraction (float): The size of the kernel sample, as a fraction of the window size (Default=0.05).
        max_radius (float): The radius r of the neighborhood (Default=0.01).
        random_state (int, np.random.RandomState or None): Seed or random number generator for the chain sample. None draws from NumPy's global random state, which `pysad.utils.fix_seed` seeds (Default=None).
    """

    def __init__(
        self,
        window_size: int = 10000,
        sample_fraction: float = 0.05,
        max_radius: float = 0.01,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be a positive integer.")
        self.window_size = window_size
        self.sample_fraction = sample_fraction
        self.max_radius = max_radius
        self.random_state = random_state
        # None is resolved on every draw, so that a pickled model keeps drawing from the global state.
        self._rng = None if random_state is None else check_random_state(random_state)
        self.num_samples_ = max(1, round(sample_fraction * window_size))
        self.num_seen_ = 0
        # The stored values of each chain as (index, value), oldest first. The first one is the sample value.
        self.chains_: list[deque[tuple[int, np.ndarray]]] = [
            deque() for _ in range(self.num_samples_)
        ]
        # The index of the value that each chain waits for, and the index of its sample value.
        self.successors_ = np.zeros(self.num_samples_, dtype=np.int64)
        self.sample_indices_ = np.zeros(self.num_samples_, dtype=np.int64)
        # The arrays get their number of features d from the first instance.
        self.sample_ = np.empty((self.num_samples_, 0))
        self.window_ = np.empty((window_size, 0))  # a ring buffer of the window values
        self.mean_ = np.empty(0)
        self.m2_ = np.empty(0)  # the sum of squared deviations from the mean, per dimension

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> D3:
        """Adds the next value to the chain sample and to the window standard deviation.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: self.
        """
        X = np.array(X, dtype=np.float64)
        if self.num_seen_ == 0:
            self.sample_ = np.zeros((self.num_samples_, X.size))
            self.window_ = np.zeros((self.window_size, X.size))
            self.mean_ = np.zeros(X.size)
            self.m2_ = np.zeros(X.size)
        self.num_seen_ += 1
        i, W = self.num_seen_, self.window_size
        rng = self._rng if self._rng is not None else check_random_state(None)
        self._update_moments(X)

        # The chain sample of Babcock et al. (2002), with the value of index i.
        for j in np.flatnonzero(self.successors_ == i):
            self.chains_[j].append((i, X))
            self.successors_[j] = i + rng.randint(1, W + 1)
        for j in np.flatnonzero(rng.random_sample(self.num_samples_) < 1.0 / min(i, W)):
            self.chains_[j] = deque([(i, X)])
            self.successors_[j] = i + rng.randint(1, W + 1)
            self.sample_indices_[j], self.sample_[j] = i, X
        # The successor of an expired sample value arrived, since it is at most W values later.
        for j in np.flatnonzero(self.sample_indices_ <= i - W):
            self.chains_[j].popleft()
            self.sample_indices_[j], self.sample_[j] = self.chains_[j][0]

        return self

    def _update_moments(self, X: np.ndarray) -> None:
        """Adds X to the window mean and sum of squared deviations, and removes the expired value."""
        slot = (self.num_seen_ - 1) % self.window_size
        if self.num_seen_ <= self.window_size:
            delta = X - self.mean_
            self.mean_ += delta / self.num_seen_
            self.m2_ += delta * (X - self.mean_)
        else:
            expired = self.window_[slot]
            mean = self.mean_ + (X - expired) / self.window_size
            self.m2_ += (X - expired) * (X - mean + expired - self.mean_)
            np.maximum(self.m2_, 0.0, out=self.m2_)
            self.mean_ = mean
        self.window_[slot] = X

    def score_partial(self, X: np.ndarray) -> float:
        """Scores the next instance by the negative estimated number of window values within ``max_radius`` of it.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The anomalousness score of the input instance.
        """
        if self.num_seen_ == 0:
            return 0.0
        X = np.asarray(X, dtype=np.float64)
        count = min(self.num_seen_, self.window_size)
        std = np.sqrt(self.m2_ / count)
        bandwidth = np.sqrt(5.0) * std * self.num_samples_ ** (-1.0 / (X.size + 4))

        # The box bounds in kernel units, (num_samples, d). A zero bandwidth gives the point mass, -1 or 1.
        lower, upper = X - self.max_radius - self.sample_, X + self.max_radius - self.sample_
        lower = np.divide(lower, bandwidth, out=np.sign(lower), where=bandwidth > 0).clip(-1.0, 1.0)
        upper = np.divide(upper, bandwidth, out=np.sign(upper), where=bandwidth > 0).clip(-1.0, 1.0)
        # The integral of the Epanechnikov kernel 3/4 (1 - u^2) from lower to upper.
        mass = 0.75 * (upper - lower) - 0.25 * (upper**3 - lower**3)

        return -float(count * mass.prod(axis=1).mean())
