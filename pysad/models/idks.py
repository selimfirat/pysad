from __future__ import annotations

import numpy as np
from sklearn.utils import check_random_state

from pysad.core.base_model import BaseModel
from pysad.utils.validation import _int_at_least


class IDK(BaseModel):
    r"""Scores each instance by its similarity to the data distribution of a sliding window (IDK-S) :cite:`xu2026idks`.

    IDK-S is the streaming form of the Isolation Distributional Kernel (IDK). It represents a window
    of recent instances as a set of hypersphere partitions. Each partition picks ``num_samples``
    random samples from the window, and each sample is the center of a hypersphere whose radius is its
    distance to the nearest other sample of the same partition. An instance falls into the hypersphere
    of the closest sample when it is closer than that sample's radius. The feature map of an instance
    marks, for each partition, the sample it falls into. The normal score of an instance is the
    average, over the partitions, of the fraction of the window that falls into the same sample as the
    instance. The anomaly score is the negated normal score, so higher scores mean more anomalous
    instances.

    The paper shows that replacing the samples tied to obsolete instances with new samples from the new
    instance is statistically equivalent to retraining IDK on the current window. This model keeps only
    one fixed window and the partitions of that window, so one call stays bounded as the stream grows.

    Deviations from the paper:

    - The paper scores a new instance with the updated window that already contains it. This model
      scores the instance before it is added to the window (prequential evaluation), so the instance's
      own state never contributes to its score.
    - The paper slides the window by an update step size larger than one. This model slides by one
      instance per call, which matches the one-instance PySAD interface.

    Args:
        input_shape (int): Number of features. Must be an int >= 1 (a NumPy integer is accepted, but
            not a bool): `TypeError` is raised for other types and `ValueError` for values below 1.
            Every instance must be an array of shape (input_shape,) or (1, input_shape), or
            `ValueError` is raised before the model changes.
        window_size (int): Number of recent instances in the sliding window, the paper's :math:`\omega`.
            Must be an int > num_samples (a NumPy integer is accepted, but not a bool)
            (Default=100).
        num_partitions (int): Number of partitions, the paper's :math:`t`. Must be an int >= 1
            (a NumPy integer is accepted, but not a bool) (Default=50).
        num_samples (int): Number of samples per partition, the paper's :math:`\psi`. Must be an int
            between 2 and window_size - 1 (a NumPy integer is accepted, but not a bool)
            (Default=4).
        random_state (int, np.random.RandomState or None): Seed or random number generator for the
            samples. None draws from NumPy's global random state, which `pysad.utils.fix_seed` seeds
            (Default=None).
    """

    def __init__(
        self,
        input_shape: int,
        window_size: int = 100,
        num_partitions: int = 50,
        num_samples: int = 4,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        input_shape = _int_at_least(input_shape, "input_shape", 1)
        self.window_size = _int_at_least(window_size, "window_size", 1)
        self.num_partitions = _int_at_least(num_partitions, "num_partitions", 1)
        self.num_samples = _int_at_least(num_samples, "num_samples", 2)
        if self.window_size <= self.num_samples:
            raise ValueError(
                "window_size must be greater than num_samples, "
                f"got window_size={self.window_size} and num_samples={self.num_samples}."
            )
        self.rng = check_random_state(random_state)
        # The sliding window of the most recent instances, filled in order.
        self.window_ = np.empty((self.window_size, input_shape), dtype=np.float64)
        self.window_index_ = 0
        self.window_count_ = 0
        self.initialized_ = False
        # samples_[p, s] is the center of sample s of partition p. Filled by _initialize.
        self.samples_ = np.empty(
            (self.num_partitions, self.num_samples, input_shape), dtype=np.float64
        )
        # sample_window_index_[p, s] is the position in window_ that sample s of partition p was drawn from.
        self.sample_window_index_ = np.full((self.num_partitions, self.num_samples), -1, dtype=int)
        # radius_sq_[p, s] is the squared radius of sample s of partition p. Filled by _initialize.
        self.radius_sq_ = np.empty((self.num_partitions, self.num_samples), dtype=np.float64)
        # feature_maps_[i] is the feature map of window_[i]. Filled by _initialize.
        self.num_detectors = self.num_partitions * self.num_samples
        self.feature_maps_ = np.empty((self.window_size, self.num_detectors), dtype=np.float64)
        # sigma_phi_[j] is the sum of feature map entry j over the window. Filled by _initialize.
        self.sigma_phi_ = np.zeros(self.num_detectors, dtype=np.float64)

    def _instance(self, X: np.ndarray) -> np.ndarray:
        """``X`` as a (1, num_features) row, or raises if it is not one instance of the fitted feature count."""
        num_features = self.window_.shape[1]
        X = np.asarray(X, dtype=np.float64)
        if X.shape not in ((num_features,), (1, num_features)):
            raise ValueError(
                f"IDK expects one instance of shape ({num_features},) or (1, {num_features}), "
                f"got an array of shape {X.shape}."
            )

        return X.reshape(1, num_features)

    def _compute_radii(self, part: int) -> None:
        """The squared radius of every sample of partition ``p``: its distance to the nearest other sample."""
        samples = self.samples_[part]
        diff = samples[:, None, :] - samples[None, :, :]
        d2 = (diff**2).sum(axis=-1)
        np.fill_diagonal(d2, np.inf)
        self.radius_sq_[part] = d2.min(axis=1)

    def _single_feature_map(self, x: np.ndarray) -> np.ndarray:
        """The feature map of one instance against the current partitions, as a flat (num_detectors,) array."""
        d2 = ((self.samples_ - x[None, None, :]) ** 2).sum(axis=-1)
        nearest = d2.argmin(axis=1)
        nearest_d2 = d2[np.arange(self.num_partitions), nearest]
        within = nearest_d2 < self.radius_sq_[np.arange(self.num_partitions), nearest]
        fmap = np.zeros((self.num_partitions, self.num_samples), dtype=np.float64)
        fmap[np.arange(self.num_partitions), nearest] = within

        return fmap.ravel()

    def _partition_feature_map(self, window: np.ndarray, part: int) -> np.ndarray:
        """The feature map of every row of ``window`` against partition ``part``, as a (num_instances, num_samples) array."""
        d2 = ((window[:, None, :] - self.samples_[part][None, :, :]) ** 2).sum(axis=-1)
        nearest = d2.argmin(axis=1)
        nearest_d2 = d2[np.arange(window.shape[0]), nearest]
        within = nearest_d2 < self.radius_sq_[part, nearest]
        fmap = np.zeros((window.shape[0], self.num_samples), dtype=np.float64)
        fmap[np.arange(window.shape[0]), nearest] = within

        return fmap

    def _initialize(self) -> None:
        """Builds the partitions from the first full window and the feature maps of all its instances."""
        num_samples = self.num_samples
        num_partitions = self.num_partitions
        for part in range(num_partitions):
            choice = self.rng.choice(self.window_size, size=num_samples, replace=False)
            self.samples_[part] = self.window_[choice]
            self.sample_window_index_[part] = choice
            self._compute_radii(part)
        blocks = [self._partition_feature_map(self.window_, part) for part in range(num_partitions)]
        self.feature_maps_[:] = np.concatenate(blocks, axis=1)
        self.sigma_phi_[:] = self.feature_maps_.sum(axis=0)
        self.initialized_ = True
        self.window_index_ = 0

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> IDK:
        """Updates the model with the next instance: buffers it until the first window is full, then slides the window by one.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: The model itself.

        Raises:
            ValueError: If `X` is not of shape (input_shape,) or (1, input_shape). The model is left unchanged.
        """
        X = self._instance(X)
        x = X.reshape(-1)
        if not self.initialized_:
            self.window_[self.window_index_] = x
            self.window_index_ += 1
            self.window_count_ += 1
            if self.window_count_ == self.window_size:
                self._initialize()

            return self

        write_pos = self.window_index_
        fmap_old = self.feature_maps_[write_pos].copy()
        x_fmap = self._single_feature_map(x)
        self.window_[write_pos] = x
        affected = [
            int(part)
            for part in np.flatnonzero(np.any(self.sample_window_index_ == write_pos, axis=1))
        ]
        affected_set = set(affected)
        for part in affected:
            sample_idxs = np.flatnonzero(self.sample_window_index_[part] == write_pos)
            for s in sample_idxs:
                self.samples_[part, s] = x
                self.sample_window_index_[part, s] = write_pos
            self._compute_radii(part)
            block = slice(part * self.num_samples, (part + 1) * self.num_samples)
            old_entries = self.feature_maps_[:, block].copy()
            new_entries = self._partition_feature_map(self.window_, part)
            self.sigma_phi_[block] += new_entries.sum(axis=0) - old_entries.sum(axis=0)
            self.feature_maps_[:, block] = new_entries
        for part in range(self.num_partitions):
            if part in affected_set:
                continue
            block = slice(part * self.num_samples, (part + 1) * self.num_samples)
            self.sigma_phi_[block] += x_fmap[block] - fmap_old[block]
            self.feature_maps_[write_pos, block] = x_fmap[block]
        self.window_index_ = (write_pos + 1) % self.window_size

        return self

    def score_partial(self, X: np.ndarray) -> float:
        """Scores the anomalousness of the next instance against the current window, without changing the model.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score. Higher scores represent
                more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The negated normal score of the input instance, as a Python float. Before the first
            window is full, the model has no partition yet and the score is 0.0.

        Raises:
            ValueError: If `X` is not of shape (input_shape,) or (1, input_shape).
        """
        X = self._instance(X)
        if not self.initialized_:
            return 0.0

        x = X.reshape(-1)
        fmap = self._single_feature_map(x)
        normal_score = float((fmap * self.sigma_phi_).sum()) / (
            self.num_partitions * self.window_size
        )

        return -normal_score
