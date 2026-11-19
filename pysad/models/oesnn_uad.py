from __future__ import annotations

import numpy as np
from sklearn.utils import check_random_state

from pysad.core.base_model import BaseModel
from pysad.utils.validation import _finite_float, _int_at_least, check_univariate


class OeSNNUAD(BaseModel):
    r"""Online evolving spiking neural network for unsupervised anomaly detection (OeSNN-UAD) :cite:`maciag2021unsupervised`.

    The model follows Sections 5 and 6 and Algorithms 1 to 7 of the paper.
    It keeps the last ``window_size`` values :math:`\mathcal{W}`, which include the new value :math:`x_t`.
    Gaussian receptive fields over :math:`[\min \mathcal{W}, \max \mathcal{W}]` encode :math:`x_t` into a firing order of ``num_input_neurons`` input neurons.
    Each output neuron has synapse weights and an output value.
    The first output neuron whose post-synaptic potential exceeds :math:`\gamma = C (1 - mod^{2 NI}) / (1 - mod^2)` fires, and its output value is the prediction :math:`y_t`.
    The error is :math:`e_t = |x_t - y_t|`.
    The value is an anomaly if :math:`e_t - \bar{e} > \varepsilon s_e`, where :math:`\bar{e}` and :math:`s_e` are the mean and standard deviation of the last ``window_size - 1`` errors of values that were not anomalies.

    Each value then creates a candidate output neuron with weights :math:`mod^{order}` and an output value drawn from :math:`\mathcal{N}(\bar{x}_{\mathcal{W}}, s_{\mathcal{W}}^2)`.
    If the value is not an anomaly, the output value moves toward :math:`x_t` by the factor :math:`\xi`.
    The candidate merges with the output neuron of the closest weights if their Euclidean distance is at most ``sim``.
    Otherwise it joins the repository, or it replaces the output neuron with the oldest update time if the repository holds ``num_output_neurons`` neurons.
    As in the paper, the first ``window_size`` values only fill the window, and they are not anomalies.
    Their errors come from predictions drawn from the normal distribution of the first window.
    The memory holds the window, ``num_output_neurons`` neurons and ``window_size - 1`` errors.
    One call costs O(``window_size`` + ``num_output_neurons`` × ``num_input_neurons``).

    The model scores each value as :math:`(e_t - \bar{e}) / s_e` with the model before it fits that value.
    A higher score is more anomalous, and the paper flags a value when its score exceeds :math:`\varepsilon`.
    Only learning depends on that decision, so ``epsilon`` does not change the score of a value directly.

    Deviations from the paper:

    - The paper outputs a binary label. The model returns the score above.
    - If no output neuron fires, the paper sets :math:`e_t = \infty` and flags the value. The model learns from that decision. For a finite score, it predicts with the output neuron of the largest post-synaptic potential instead.
    - The score is 0 during the first ``window_size`` values, while the repository is empty, or when :math:`s_e = 0`. The model learns from the decision of the paper in these cases.
    - Algorithm 4 flags a value when :math:`e_t - \bar{e} \ge \varepsilon s_e`, and the text when it is greater. The model follows the text, so a constant stream, whose errors are all 0, has no anomalies.
    - The firing order depends only on :math:`|x_t - \mu_j|`, the distance to the center of each receptive field (Proposition 1). The model measures it on the window range scaled to [0, 1], so the order does not depend on the units. Equal distances, for example when :math:`x_t` is the minimum of the window, fire in input neuron order. When all values of the window are equal, the fields have width 0, and the model encodes :math:`x_t` as the minimum.
    - The paper does not specify the standard deviations. The model uses the population standard deviation.
    - The code of the authors learns before it predicts. The model predicts first, as the paper does.

    Args:
        window_size (int): The number of recent values :math:`\mathcal{W}_{size}` that set the receptive fields, the distribution of the output values and the number of errors. The paper tunes it per series in [100, 600] on NAB. It must be at least 2 (Default=100).
        num_input_neurons (int): The number of input neurons :math:`NI_{size}`. It must be at least 3 (Default=10).
        num_output_neurons (int): The maximal number of output neurons :math:`NO_{size}`. It must be at least 1 (Default=50).
        sim (float): The largest Euclidean distance between the weights of a candidate and an output neuron that merges them (Default=0.17).
        mod (float): The modulation factor of the synapse weights, in (0, 1) (Default=0.6).
        c (float): The fraction :math:`C` of the maximal post-synaptic potential that fires an output neuron, in (0, 1] (Default=0.6).
        xi (float): The factor :math:`\xi` that moves the output value of a candidate toward a value that is not an anomaly, in [0, 1] (Default=0.9).
        epsilon (float): The anomaly classification factor :math:`\varepsilon`. The paper tunes it per series in [2, 7] on NAB (Default=4.0).
        random_state (int, np.random.RandomState or None): Seed or random number generator for the output values of the candidates. None draws from NumPy's global random state, which `pysad.utils.fix_seed` seeds (Default=None).

    Raises:
        ValueError: If ``mod`` is not in (0, 1), ``c`` is not in (0, 1], ``xi`` is not in [0, 1], or an int parameter is below its minimum.
    """

    def __init__(
        self,
        window_size: int = 100,
        num_input_neurons: int = 10,
        num_output_neurons: int = 50,
        sim: float = 0.17,
        mod: float = 0.6,
        c: float = 0.6,
        xi: float = 0.9,
        epsilon: float = 4.0,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        self.window_size = _int_at_least(window_size, "window_size", 2)
        self.num_input_neurons = _int_at_least(num_input_neurons, "num_input_neurons", 3)
        self.num_output_neurons = _int_at_least(num_output_neurons, "num_output_neurons", 1)
        self.sim = _finite_float(sim, "sim")
        self.mod = _finite_float(mod, "mod")
        if not 0.0 < self.mod < 1.0:
            raise ValueError(f"mod must be in (0, 1), got {self.mod}.")
        self.c = _finite_float(c, "c")
        if not 0.0 < self.c <= 1.0:
            raise ValueError(f"c must be in (0, 1], got {self.c}.")
        self.xi = _finite_float(xi, "xi")
        if not 0.0 <= self.xi <= 1.0:
            raise ValueError(f"xi must be in [0, 1], got {self.xi}.")
        self.epsilon = _finite_float(epsilon, "epsilon")
        self.random_state = random_state
        self._rng: np.random.RandomState | None = (
            None if random_state is None else check_random_state(random_state)
        )

        # The firing threshold of every output neuron (Corollary 1).
        self.gamma_ = (
            self.c * (1.0 - self.mod ** (2 * self.num_input_neurons)) / (1.0 - self.mod**2)
        )
        # The weights of a candidate in firing order (Equation 9).
        self.powers_ = self.mod ** np.arange(self.num_input_neurons)
        # The receptive field centers on the window range scaled to [0, 1] (Equation 5).
        self.centers_ = (2.0 * np.arange(self.num_input_neurons) - 3.0) / (
            2.0 * (self.num_input_neurons - 2)
        )
        self.count_ = 0
        self.window_ = np.zeros(self.window_size)
        self.position_ = 0  # the slot of the oldest value of a full window
        self.errors_ = np.zeros(self.window_size - 1)
        # Whether each error belongs to a value that was not an anomaly.
        self.normal_ = np.zeros(self.window_size - 1, dtype=bool)
        self.error_position_ = 0
        self.num_neurons_ = 0
        self.weights_ = np.zeros((self.num_output_neurons, self.num_input_neurons))
        self.outputs_ = np.zeros(self.num_output_neurons)
        self.update_times_ = np.zeros(self.num_output_neurons)
        self.update_counts_ = np.zeros(self.num_output_neurons, dtype=np.int64)

    def _encode(self, window: np.ndarray, x: float) -> tuple[np.ndarray, np.ndarray]:
        """The candidate weights mod^order of each input neuron, and the input neurons in firing order."""
        low, scale = window.min(), window.max() - window.min()
        # On [0, 1], the minimum and the maximum of the window are exact, so tied fields stay tied.
        position = (x - low) / scale if scale > 0.0 else 0.0
        firing = np.argsort(np.abs(position - self.centers_), kind="stable")
        weights = np.empty(self.num_input_neurons)
        weights[firing] = self.powers_
        return weights, firing

    def _predict(self, firing: np.ndarray) -> tuple[int | None, int]:
        """The output neuron that fires first (Algorithm 3), or None, and the one of the largest potential."""
        # (num_neurons, num_input_neurons): the potentials after each input neuron fires.
        potentials = np.cumsum(self.weights_[: self.num_neurons_, firing] * self.powers_, axis=1)
        largest = int(potentials[:, -1].argmax())
        fired = (potentials > self.gamma_).any(axis=0)
        if not fired.any():
            return None, largest
        return int(potentials[:, fired.argmax()].argmax()), largest

    def _z(self, error: float) -> float:
        """The error in standard deviations above the mean error of the recent values that were not anomalies."""
        normal_errors = self.errors_[self.normal_]
        if normal_errors.size == 0:
            return 0.0
        std = normal_errors.std()
        return (error - normal_errors.mean()) / std if std > 0.0 else 0.0

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> OeSNNUAD:
        """Classifies the next value as the paper does, then learns from it.

        Args:
            X (np.float64 array of shape (1,)): The value to fit. This model is univariate.
            y (int): The model ignores this label (Default=None).

        Returns:
            object: The model itself.
        """
        x = check_univariate(X, "OeSNNUAD")
        rng = self._rng if self._rng is not None else check_random_state(None)
        self.count_ += 1
        self.window_[self.position_] = x
        self.position_ = (self.position_ + 1) % self.window_size
        if self.count_ < self.window_size:
            return self

        mean, std = self.window_.mean(), self.window_.std()
        if self.count_ == self.window_size:
            # The errors of the predictions drawn for the last window_size - 1 values of the first window.
            self.errors_[:] = np.abs(self.window_[1:] - rng.normal(mean, std, self.window_size - 1))
            self.normal_[:] = True
            return self

        weights, firing = self._encode(self.window_, x)
        fired = self._predict(firing)[0] if self.num_neurons_ else None
        if fired is None:
            error, anomaly = np.inf, True
        else:
            error = abs(x - self.outputs_[fired])
            normal_errors = self.errors_[self.normal_]
            anomaly = (
                normal_errors.size > 0
                and error - normal_errors.mean() > self.epsilon * normal_errors.std()
            )
        self.errors_[self.error_position_] = error
        self.normal_[self.error_position_] = not anomaly
        self.error_position_ = (self.error_position_ + 1) % (self.window_size - 1)

        output = rng.normal(mean, std)
        if not anomaly:
            output += (x - output) * self.xi

        n = self.num_neurons_
        distances = np.linalg.norm(self.weights_[:n] - weights, axis=1)
        if n and distances.min() <= self.sim:
            i = int(distances.argmin())
            m = self.update_counts_[i]
            self.weights_[i] = (weights + m * self.weights_[i]) / (m + 1)
            self.outputs_[i] = (output + m * self.outputs_[i]) / (m + 1)
            self.update_times_[i] = (self.count_ + m * self.update_times_[i]) / (m + 1)
            self.update_counts_[i] = m + 1
            return self

        if n < self.num_output_neurons:
            i = n
            self.num_neurons_ += 1
        else:
            i = int(self.update_times_.argmin())
        self.weights_[i] = weights
        self.outputs_[i] = output
        self.update_times_[i] = self.count_
        self.update_counts_[i] = 1
        return self

    def score_partial(self, X: np.ndarray) -> float:
        """Scores the value with the current model, without changing it.

        Args:
            X (np.float64 array of shape (1,)): The value to score. This model is univariate.

        Returns:
            float: The prediction error in standard deviations above the mean recent error of the values that were not anomalies. It is 0 during the first ``window_size`` values, while the repository is empty, or when that standard deviation is 0.
        """
        x = check_univariate(X, "OeSNNUAD")
        if self.count_ < self.window_size or not self.num_neurons_:
            return 0.0

        # The window after the value replaces the oldest value.
        window = self.window_.copy()
        window[self.position_] = x
        fired, largest = self._predict(self._encode(window, x)[1])
        return self._z(abs(x - self.outputs_[largest if fired is None else fired]))
