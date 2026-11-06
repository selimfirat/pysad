import numpy as np
from sklearn.utils import check_random_state


def generate_stream(
    num_instances: int = 2000,
    num_features: int = 2,
    num_points: int = 10,
    num_bursts: int = 2,
    burst_length: int = 20,
    num_drifts: int = 1,
    drift_length: int = 100,
    magnitude: float = 5.0,
    random_state: int | np.random.RandomState | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates a labeled synthetic stream with point, burst and drift anomalies.

    Normal instances are independent draws from the standard normal distribution. An anomaly adds an offset to its instance along a random unit direction. The stream has three types of anomalies:

    - A point anomaly is one instance with an offset of length ``magnitude``. Each point anomaly has its own direction.
    - A burst is ``burst_length`` consecutive instances that share one offset of length ``magnitude``.
    - A drift is ``drift_length`` consecutive instances whose offset grows linearly from zero to ``magnitude``. The stream then returns to the normal distribution.

    The anomalies do not overlap. The first tenth of the stream holds only normal instances, so that a detector can learn the normal distribution first.

    Args:
        num_instances (int): The number of instances in the stream (Default=2000).
        num_features (int): The number of features of each instance (Default=2).
        num_points (int): The number of point anomalies (Default=10).
        num_bursts (int): The number of bursts (Default=2).
        burst_length (int): The number of instances in each burst (Default=20).
        num_drifts (int): The number of drifts (Default=1).
        drift_length (int): The number of instances in each drift (Default=100).
        magnitude (float): The length of the anomaly offset, in standard deviations of the normal distribution (Default=5.0).
        random_state (int, np.random.RandomState or None): Seed or random number generator for the stream. None draws from NumPy's global random state, which `pysad.utils.fix_seed` seeds (Default=None).

    Returns:
        tuple[np.ndarray, np.ndarray]: The instances, a np.float64 array of shape (num_instances, num_features), and the labels, an int array of shape (num_instances,) with 1 for an anomaly and 0 for a normal instance.

    Raises:
        ValueError: If the anomalies do not fit into the stream after its first tenth.
    """
    rng = check_random_state(random_state)
    X = rng.standard_normal((num_instances, num_features))
    y = np.zeros(num_instances, dtype=int)
    start = num_instances // 10

    def place(length: int) -> slice:
        occupied = np.concatenate(([0], y.cumsum()))
        # counts[s] is the number of anomalies in [s, s + length).
        counts = occupied[length:] - occupied[:-length]
        starts = np.flatnonzero(counts[start:] == 0) + start
        if not starts.size:
            raise ValueError(
                "The anomalies do not fit into the stream. Use a longer stream or fewer anomalies."
            )
        s = rng.choice(starts)
        y[s : s + length] = 1
        return slice(s, s + length)

    def direction() -> np.ndarray:
        d = rng.standard_normal(num_features)
        return d / np.linalg.norm(d)

    # Place the longest anomalies first, so that the short ones fill the gaps.
    for _ in range(num_drifts):
        segment = place(drift_length)
        # (drift_length, 1) * (num_features,) -> (drift_length, num_features)
        X[segment] += np.linspace(0.0, magnitude, drift_length)[:, None] * direction()
    for _ in range(num_bursts):
        X[place(burst_length)] += magnitude * direction()
    for _ in range(num_points):
        X[place(1)] += magnitude * direction()

    return X, y
