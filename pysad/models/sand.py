from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.utils import check_random_state

from pysad.core.base_model import BaseModel
from pysad.utils.validation import _finite_float, _int_at_least, check_univariate

# The iteration limit of k-Shape, as in its paper and in tslearn.
KSHAPE_MAX_ITER = 100


def _znorm(X: np.ndarray) -> np.ndarray:
    """Returns the rows of X with mean 0 and standard deviation 1, and the constant rows as zeros."""
    mean = X.mean(axis=-1, keepdims=True)
    std = X.std(axis=-1, keepdims=True)
    constant = std <= 1e-10 * np.abs(mean)
    return np.where(constant, 0.0, (X - mean) / np.where(constant, 1.0, std))


def _spectrum(X: np.ndarray) -> np.ndarray:
    """Returns the conjugate FFT of the rows of X, zero padded to hold every shift."""
    return np.fft.rfft(X, 1 << (2 * X.shape[1] - 1).bit_length(), axis=1).conj()


def _sbd(
    X: np.ndarray, c: np.ndarray, spectrum: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Returns the shape-based distance of each row of X to c, and the shift that aligns the row to c.

    Args:
        X (np.float64 array of shape (n, m)): The sequences.
        c (np.float64 array of shape (m,)): The reference sequence.
        spectrum (np.complex128 array or None): ``_spectrum(X)``, if known (Default=None).

    Returns:
        tuple: The distances and the shifts, both of shape (n,).
    """
    n, m = X.shape
    c_norm = np.linalg.norm(c)
    if c_norm == 0.0:  # k-Shape does not align to an all-zero centroid
        return np.ones(n), np.zeros(n, dtype=np.intp)

    if spectrum is None:
        spectrum = _spectrum(X)
    size = 2 * (spectrum.shape[1] - 1)
    # cc[i, s] = sum_t c[t + s] X[i, t] for the shifts s = -(m - 1), ..., m - 1, of shape (n, 2m - 1).
    cc = np.fft.irfft(np.fft.rfft(c, size) * spectrum, size, axis=1)
    cc = np.concatenate((cc[:, -(m - 1) :], cc[:, :m]), axis=1)
    den = c_norm * np.linalg.norm(X, axis=1)
    ncc = np.divide(cc, den[:, None], out=np.zeros_like(cc), where=den[:, None] > 0)
    best = ncc.argmax(axis=1)
    return 1.0 - ncc[np.arange(n), best], best - (m - 1)


def _align(X: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Shifts each row of X right by its shift, with zero padding."""
    m = X.shape[1]
    source = np.arange(m) - shifts[:, None]  # (n, m) index in X of each aligned value
    valid = (source >= 0) & (source < m)
    return np.where(valid, np.take_along_axis(X, source.clip(0, m - 1), axis=1), 0.0)


def _shape(S: np.ndarray, aligned: np.ndarray) -> np.ndarray:
    """Returns the k-Shape centroid of the sequences whose sum of outer products is S.

    The centroid is the eigenvector of ``Q^T S Q`` with the largest eigenvalue, where ``Q = I - O / m``, with the sign that brings it closer to the aligned sequences.

    Args:
        S (np.float64 array of shape (m, m)): The sum of ``a a^T`` over the aligned sequences ``a``.
        aligned (np.float64 array of shape (n, m)): Aligned sequences to choose the sign with.

    Returns:
        np.float64 array of shape (m,): The z-normalized centroid.
    """
    # Q^T S Q subtracts the row means and the column means of S and adds back its mean.
    M = S - S.mean(axis=0) - S.mean(axis=1, keepdims=True) + S.mean()
    mu = np.linalg.eigh(M)[1][:, -1]
    if np.linalg.norm(aligned + mu, axis=1).sum() < np.linalg.norm(aligned - mu, axis=1).sum():
        mu = -mu
    return _znorm(mu)


def _kshape(X: np.ndarray, k: int, rng: np.random.RandomState) -> tuple[np.ndarray, np.ndarray]:
    """Clusters the z-normalized rows of X with k-Shape, from random labels.

    Args:
        X (np.float64 array of shape (n, m)): The z-normalized sequences.
        k (int): The number of clusters.
        rng (np.random.RandomState): The generator of the initial labels.

    Returns:
        tuple: The labels, of shape (n,), and the centroids, of shape (k, m). A cluster that ends empty keeps its last centroid.
    """
    labels = np.asarray(rng.randint(k, size=X.shape[0]))
    centroids = np.zeros((k, X.shape[1]))
    spectrum = _spectrum(X)
    for _ in range(KSHAPE_MAX_ITER):
        for j in range(k):
            member = labels == j
            if member.any():
                members = X[member]
                aligned = _align(members, _sbd(members, centroids[j], spectrum[member])[1])
                centroids[j] = _shape(aligned.T @ aligned, aligned)
        distances = np.stack([_sbd(X, c, spectrum)[0] for c in centroids], axis=1)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

    return labels, centroids


class SAND(BaseModel):
    """Scores each value of a univariate stream by the distance of its latest subsequence to a weighted set of subsequence clusters that SAND keeps up to date :cite:`boniol2021sand`.

    SAND (Streaming Subsequence Anomaly Detection) splits the stream into batches of ``batch_size`` values. It clusters the overlapping subsequences of ``centroid_length`` values in each batch with k-Shape :cite:`paparrizos2015kshape`. The centroids of the clusters form the set Θ. A new cluster merges into a cluster of Θ if their shape-based distance (SBD) is below the mean SBD of that cluster to its centroid. Otherwise it joins Θ. A merge adds the sequences of the new cluster to the matrix ``S`` of k-Shape and takes the centroid from ``S`` (Eq. 3 of the paper), so the model keeps no raw subsequences. Each cluster has a weight ``|C|^2 / sum_j SBD(C, C_j)``, which favors large clusters far from the others. The weight decays while the cluster gets no new subsequences and follows the rate ``alpha`` (Eq. 4).

    The distance ``d`` of a subsequence of ``subsequence_length`` values is the weighted sum, over the clusters, of its smallest z-normalized Euclidean distance to a subsequence of the centroid (Eq. 5). The score of a value is ``(d - mu) / sigma`` for the subsequence that ends with the value, where ``mu`` and ``sigma`` follow the mean and the standard deviation of ``d`` over the batches at the rate ``alpha`` (Eq. 6). Higher scores are more anomalous. Until the first batch completes, the score is ``0.0``. The model keeps the last ``batch_size + subsequence_length - 1`` values, and one ``centroid_length`` by ``centroid_length`` matrix per cluster. By default, as in the paper, no cluster leaves Θ, so Θ grows by up to ``num_clusters`` clusters per batch, and the memory and the cost of a score grow with it. Section 3.3.3 of the paper suggests that one can remove the clusters whose weight is about zero. If ``min_weight`` is set, the model removes the clusters whose weight falls below it after each batch, but always keeps the heaviest cluster, and normalizes the remaining weights to sum to 1. Then Θ holds at most ``1 / min_weight`` clusters between batches, so the memory and the cost of a score stay bounded, but a removed behavior scores as anomalous if it returns.

    The implementation differs from the paper as follows:

    - The paper clusters each batch before it scores the subsequences of the batch, with the mean and the standard deviation of the batch. A stream gives one value at a time, so the model scores a value with the Θ, ``mu`` and ``sigma`` of the last complete batch. It updates them when a batch completes.
    - The paper starts ``mu`` and ``sigma`` at ``0`` (Algorithm 1), which divides the first scores by zero. The model takes the mean and the standard deviation of the first batch, as the reference implementation does. If ``sigma`` is ``0``, the score is ``d - mu``.
    - The paper normalizes the weights to sum to 1. The model also normalizes the current weights ``|C|^2 / sum_j SBD(C, C_j)`` to sum to 1 before it mixes them with the previous weights at the rate ``alpha``, as Eq. 2 does for the first batch, so that both terms have the same scale.
    - The paper writes the intra-cluster distance as a sum, but updates it as a mean. The model keeps the mean, as the reference implementation does.
    - The reference implementation gives each value the mean distance of the subsequences that contain it, which needs the next ``subsequence_length - 1`` values. The paper scores subsequences, and the model scores the subsequence that ends with each value. :class:`~pysad.transform.postprocessing.AveragePostprocessor` with ``window_size=subsequence_length`` gives the mean of the reference ``subsequence_length - 1`` values later.
    - A new cluster that is close enough to several clusters of Θ merges into the closest one.
    - k-Shape starts from random labels and stops after 100 iterations, as in its paper. It drops the clusters that end empty, as the reference implementation does.

    The reference implementation in TSB-UAD differs from the paper in ways that the model does not copy. By default, it clusters every 10th subsequence of a batch instead of all of them. It scales ``|C|`` and the sum of distances to ``[1, 2]`` before it computes the weight ``|C|^2 / sum_j SBD(C, C_j)``. It divides by ``1 + max(0, A - batch_size)`` instead of ``max(1, A - batch_size)`` in Eq. 4. It compares a new cluster only with the closest cluster of Θ. With these differences, the scores of the model and of the reference implementation are not equal on the same stream, but their AUROC is similar.

    Args:
        subsequence_length (int): The length ``ℓ`` of the subsequences to score, at least 2. The paper sets it to the length of the anomalies.
        centroid_length (int or None): The length ``ℓΘ`` of the centroids, greater than ``subsequence_length``. None uses ``4 * subsequence_length``, as in the experiments of the paper (Default=None).
        num_clusters (int): The number of clusters ``k`` of k-Shape on each batch (Default=6).
        batch_size (int): The number of values in a batch. A batch must hold at least ``num_clusters`` subsequences of ``centroid_length`` values (Default=5000).
        alpha (float): The rate in ``[0, 1]`` at which the weights, ``mu`` and ``sigma`` follow each new batch (Default=0.5).
        min_weight (float or None): The weight in ``[0, 1)`` below which a cluster leaves Θ after a batch. None keeps every cluster, as in the paper (Default=None).
        random_state (int, np.random.RandomState or None): Seed or random number generator for the initial labels of k-Shape. None draws from NumPy's global random state, which `pysad.utils.fix_seed` seeds (Default=None).
    """

    def __init__(
        self,
        subsequence_length: int,
        centroid_length: int | None = None,
        num_clusters: int = 6,
        batch_size: int = 5000,
        alpha: float = 0.5,
        min_weight: float | None = None,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        self.subsequence_length = _int_at_least(subsequence_length, "subsequence_length", 2)
        if centroid_length is None:
            centroid_length = 4 * self.subsequence_length
        self.centroid_length = _int_at_least(
            centroid_length, "centroid_length", self.subsequence_length + 1
        )
        self.num_clusters = _int_at_least(num_clusters, "num_clusters", 1)
        self.batch_size = _int_at_least(
            batch_size, "batch_size", self.centroid_length + self.num_clusters - 1
        )
        self.alpha = _finite_float(alpha, "alpha")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}.")
        self.min_weight = None if min_weight is None else _finite_float(min_weight, "min_weight")
        if self.min_weight is not None and not 0.0 <= self.min_weight < 1.0:
            raise ValueError(f"min_weight must be in [0, 1), got {self.min_weight}.")
        self.random_state = random_state
        self._rng: np.random.RandomState | None = (
            None if random_state is None else check_random_state(random_state)
        )

        # The last subsequence_length - 1 values of the previous batch, then the current batch.
        self.buffer_ = np.zeros(self.subsequence_length - 1 + self.batch_size)
        self.count_ = 0  # values of the current batch in buffer_
        self.time_ = 0  # values seen
        self.centroids_: np.ndarray | None = None  # (num_centroids, centroid_length)
        self.matrices_ = np.empty((0, self.centroid_length, self.centroid_length))
        self.distances_ = np.empty(0)  # mean SBD of each cluster to its centroid
        self.sizes_ = np.empty(0)  # subsequences in each cluster
        self.last_ = np.empty(0)  # start of the latest subsequence of each cluster
        self.weights_ = np.empty(0)
        # The z-normalized windows of subsequence_length values of each centroid, and their squared norms.
        self.windows_ = np.empty(
            (0, self.centroid_length - self.subsequence_length + 1, self.subsequence_length)
        )
        self.window_sq_ = np.empty((0, 1, self.centroid_length - self.subsequence_length + 1))
        self.mean_ = 0.0
        self.std_ = 0.0

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> SAND:
        """Updates the model with the next value, and Θ when the value completes a batch.

        Args:
            X (np.float64 array of shape (1,)): The instance to fit. This model is univariate.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: The model itself.
        """
        self.buffer_[self.subsequence_length - 1 + self.count_] = check_univariate(X, "SAND")
        self.count_ += 1
        self.time_ += 1
        if self.count_ == self.batch_size:
            self._update(self.buffer_[self.subsequence_length - 1 :])
            self.buffer_[: self.subsequence_length - 1] = self.buffer_[
                -(self.subsequence_length - 1) :
            ]
            self.count_ = 0

        return self

    def score_partial(self, X: np.ndarray) -> float:
        """Scores the subsequence that ends with the next value.

        Args:
            X (np.float64 array of shape (1,)): The instance to score. This model is univariate.

        Returns:
            float: The anomalousness score of the input instance.
        """
        value = check_univariate(X, "SAND")
        if self.centroids_ is None:
            return 0.0

        previous = self.buffer_[self.count_ : self.count_ + self.subsequence_length - 1]
        d = float(self._distances(np.append(previous, value)[None])[0])
        return (d - self.mean_) / self.std_ if self.std_ > 0.0 else d - self.mean_

    def _distances(self, subsequences: np.ndarray) -> np.ndarray:
        """Returns the distance ``d`` (Eq. 5) of each subsequence to Θ.

        Args:
            subsequences (np.float64 array of shape (n, subsequence_length)): The subsequences.

        Returns:
            np.float64 array of shape (n,): The weighted sums of the distances to the centroids.
        """
        Q = _znorm(subsequences)
        q_sq = (Q * Q).sum(axis=1, keepdims=True)
        d = np.zeros(len(Q))
        for weight, P, p_sq in zip(self.weights_, self.windows_, self.window_sq_, strict=True):
            # (n, centroid_length - subsequence_length + 1) squared distances to the windows of a centroid.
            sq = q_sq + p_sq - 2.0 * (Q @ P.T)
            d += weight * np.sqrt(np.maximum(sq.min(axis=1), 0.0))
        return d

    def _update(self, batch: np.ndarray) -> None:
        """Clusters a complete batch and updates Θ, its weights, ``mu`` and ``sigma`` (Algorithms 2 to 4)."""
        rng = self._rng if self._rng is not None else check_random_state(None)
        Z = _znorm(sliding_window_view(batch, self.centroid_length))
        labels, kshape_centroids = _kshape(Z, self.num_clusters, rng)
        start = self.time_ - self.batch_size  # stream index of the first value of the batch
        num_old = 0 if self.centroids_ is None else len(self.centroids_)
        centroids = list(self.centroids_) if self.centroids_ is not None else []
        matrices = list(self.matrices_)
        distances, sizes, last = self.distances_.tolist(), self.sizes_.tolist(), self.last_.tolist()

        # Algorithm 2 merges each new cluster into Θ as soon as it matches, so a later cluster of the
        # batch meets the updated centroid. Algorithm 3 updates τ and the size after the batch.
        merges: dict[int, list[tuple[int, float, int]]] = {}
        for j in range(self.num_clusters):
            indices = np.flatnonzero(labels == j)
            if not len(indices):
                continue
            members, centroid = Z[indices], kshape_centroids[j]
            distance = _sbd(members, centroid)[0].mean()
            latest = start + indices[-1]

            if num_old:
                to_old = _sbd(np.array(centroids[:num_old]), centroid)[0]
                close = to_old < self.distances_
                if close.any():
                    match = int(np.flatnonzero(close)[to_old[close].argmin(axis=0)])
                    aligned = _align(members, _sbd(members, centroids[match])[1])
                    matrices[match] = matrices[match] + aligned.T @ aligned
                    centroids[match] = _shape(matrices[match], aligned)
                    merges.setdefault(match, []).append((len(indices), distance, latest))
                    continue

            aligned = _align(members, _sbd(members, centroid)[1])
            centroids.append(centroid)
            matrices.append(aligned.T @ aligned)
            distances.append(distance)
            sizes.append(float(len(indices)))
            last.append(float(latest))

        # Algorithm 3 updates τ and the size of each matched cluster with all the new clusters merged into it.
        for match, merged in merges.items():
            total = sizes[match] + sum(n for n, _, _ in merged)
            distances[match] = (
                sizes[match] * distances[match] + sum(n * d for n, d, _ in merged)
            ) / total
            sizes[match] = total
            last[match] = max(last[match], *(t for _, _, t in merged))

        self.centroids_ = np.array(centroids)
        self.matrices_ = np.array(matrices)
        self.distances_ = np.array(distances)
        self.sizes_ = np.array(sizes)
        self.last_ = np.array(last)

        # Algorithm 3: the current weights, normalized as in Eq. 2, and their time decay.
        centrality = np.array([_sbd(self.centroids_, c)[0].sum() for c in self.centroids_])
        current = self.sizes_**2 / np.maximum(centrality, np.finfo(np.float64).eps)
        current /= current.sum()
        decay = np.maximum(1.0, self.time_ - self.last_ - self.batch_size)
        weights = current.copy()
        weights[:num_old] = (1.0 - self.alpha) * self.weights_ + self.alpha * current[
            :num_old
        ] / decay[:num_old]
        self.weights_ = weights / weights.sum()
        if self.min_weight is not None:
            # Section 3.3.3: remove the clusters whose weight is about zero, but keep the heaviest.
            keep = self.weights_ >= self.min_weight
            keep[self.weights_.argmax()] = True
            self.centroids_, self.matrices_ = self.centroids_[keep], self.matrices_[keep]
            self.distances_, self.sizes_ = self.distances_[keep], self.sizes_[keep]
            self.last_ = self.last_[keep]
            self.weights_ = self.weights_[keep] / self.weights_[keep].sum()

        windows = _znorm(sliding_window_view(self.centroids_, self.subsequence_length, axis=1))
        self.windows_ = windows
        self.window_sq_ = (windows * windows).sum(axis=2)[:, None, :]

        # Algorithm 4: mu and sigma follow the distances of the batch.
        d = self._distances(sliding_window_view(batch, self.subsequence_length))
        if num_old == 0:
            self.mean_, self.std_ = float(d.mean()), float(d.std())
        else:
            self.mean_ = self.alpha * float(d.mean()) + (1.0 - self.alpha) * self.mean_
            self.std_ = self.alpha * float(d.std()) + (1.0 - self.alpha) * self.std_
