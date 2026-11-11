from __future__ import annotations

import math
from collections import deque

import numpy as np
from sklearn.utils import check_random_state

from pysad.core.base_model import BaseModel
from pysad.statistics import VarianceMeter
from pysad.utils.validation import _finite_float, _int_at_least, check_univariate

# The paper sets the error to +infinity when no output neuron fires. The model returns the largest
# finite float, so such a value ranks above every value for which an output neuron fires.
_NO_FIRING_SCORE = float(np.finfo(np.float64).max)


class OeSNNUAD(BaseModel):
    r"""Scores each value of a univariate stream with an online evolving spiking neural network :cite:`maciag2021unsupervised`.

    The model keeps a window of the last ``window_size`` values. Each of the ``num_input_neurons`` input neurons has one Gaussian receptive field (GRF). The centre of GRF ``j`` is

    .. math::

        \mu_j = I_{min} + \frac{2j-3}{2} \cdot \frac{I_{max} - I_{min}}{NI_{size} - 2},

    where ``I_min`` and ``I_max`` are the smallest and the largest values of the window. The GRF excitation of a value ``x_t`` falls with its distance from the centre. The model ranks the input neurons by excitation, and that rank is the firing order ``order(j)``. A candidate output neuron for ``x_t`` takes the synaptic weight ``mod ** order(j)`` from input neuron ``j``. An output neuron with the weights ``w_j`` fires when its post-synaptic potential

    .. math::

        PSP = \sum_j w_j \cdot mod^{order(j)}

    is above the threshold ``gamma = c * sum_k mod ** (2 * k)``. Every output neuron has the same threshold.

    The model finds the first output neuron that fires. Its output value is the prediction ``y_t`` of ``x_t``. The model flags a value in two cases. It flags the value when no output neuron fires. It also flags the value when its error ``e_t = |x_t - y_t|`` is at least ``epsilon`` times the standard deviation of the errors above their mean.

    The model gives each candidate output neuron an output value from the normal distribution of the window. When the model does not flag the value, it moves that value towards ``x_t`` by the factor ``xi``. The model then merges the candidate neuron with the output neuron that has the smallest Euclidean distance between the weight vectors, when that distance is at most ``sim``. Otherwise it adds the candidate neuron to the repository. When the repository is full, the candidate neuron replaces the output neuron with the smallest update time.

    The first ``window_size`` values only fill the window. The paper gives them random predictions from the normal distribution of the window, and their errors start the error statistics. The model scores 0.0 until the window is full.

    The cost of one call is O(window_size + num_input_neurons * num_output_neurons), and the memory is fixed.

    Deviations from the paper:

    - The paper makes a binary decision. The model returns the excess of the error above the mean recent error, in standard deviations, that is ``(e_t - mean_e) / s_e``. The paper flags a value when this quantity is at least ``epsilon``, so the scores keep the order of the paper's decision. When no output neuron fires, the paper sets the error to +infinity. The model returns the largest finite float, which ranks that value above every other value.
    - Algorithm 4 of the paper writes ">=" where Section 5.2 writes "greater than". The model flags a value when its score is at least ``epsilon``, as Algorithm 4 does.
    - The paper does not say which standard deviation it means, the population one or the sample one. As in the authors' C++ code, the model uses the sample standard deviation, with ``n - 1`` in the denominator.
    - The GRF width parameter ``beta`` and the firing time scale ``TS`` of the paper change the firing times but not the firing order, which is a property that the paper proves. The model leaves both out, because they do not change any result.
    - The paper does not break ties between input neurons with the same excitation. The model keeps the index order of such neurons, so a constant window gives the firing order 0, 1, ..., ``num_input_neurons - 1``.

    References:

    - The authors' C++ implementation at https://github.com/piotr-maciag/OeSNN-UAD has no license, so the model uses it only for comparison. The code differs from the paper in four places: it scales the similarity threshold by the largest possible distance between two weight vectors, the merged neuron takes the weights of the candidate neuron, it corrects the output value of the neuron that fired instead of the candidate neuron, and it adds the candidate neuron to the repository before it looks for the neuron that fires. The model follows the paper in all four places.
    - The Python implementation at https://github.com/OESNN-AD-Python/oesnn-ad-package has an MIT license. It follows the paper, and the model follows it where it follows the paper. The model differs from it in three places: it uses the population standard deviation, it uses ``|x_t|`` as the error when no output neuron fires, and it returns the last neuron that fires instead of the neuron with the greatest post-synaptic potential.

    Args:
        window_size (int): The number of recent values in the window, ``W_size``. Must be at least 2. The paper tunes it for each stream, and its best values are usually between 100 and 600 on the Numenta Anomaly Benchmark.
        num_input_neurons (int): The number of input neurons ``NI_size``, and of Gaussian receptive fields. Must be at least 3 (Default=10).
        num_output_neurons (int): The largest number of output neurons in the repository ``NO_size``. Must be at least 1 (Default=50).
        sim (float): The largest Euclidean distance between the weights of a candidate output neuron and of an output neuron for a merge. Must not be negative (Default=0.17).
        mod (float): The modulation factor of the synaptic weights, in (0, 1) (Default=0.6).
        c (float): The fraction of the largest post-synaptic potential that a neuron needs to fire, in (0, 1] (Default=0.6).
        xi (float): The factor that moves the output value of a candidate neuron towards the value, in [0, 1] (Default=0.9).
        epsilon (float): The anomaly classification factor. The paper tunes it for each stream, and its best values are usually 3, 4 or 5 (Default=4.0).
        random_state (int, np.random.RandomState or None): Seed or random number generator for the output values. None draws from NumPy's global random state, which `pysad.utils.fix_seed` seeds (Default=None).

    Raises:
        ValueError: If a parameter is outside the range of the paper.
        TypeError: If an integer parameter is not an int, or a float parameter is not a real number.
    """

    def __init__(
        self,
        window_size: int,
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
        self.c = _finite_float(c, "c")
        self.xi = _finite_float(xi, "xi")
        self.epsilon = _finite_float(epsilon, "epsilon")
        if self.sim < 0.0:
            raise ValueError(f"sim must not be negative, got {self.sim}.")
        if not 0.0 < self.mod < 1.0:
            raise ValueError(f"mod must be in (0, 1), got {self.mod}.")
        if not 0.0 < self.c <= 1.0:
            raise ValueError(f"c must be in (0, 1], got {self.c}.")
        if not 0.0 <= self.xi <= 1.0:
            raise ValueError(f"xi must be in [0, 1], got {self.xi}.")
        if self.epsilon < 0.0:
            raise ValueError(f"epsilon must not be negative, got {self.epsilon}.")

        self.random_state = random_state
        self._rng: np.random.RandomState | None = (
            None if random_state is None else check_random_state(random_state)
        )
        # Theorem 1 of the paper: every output neuron has the same threshold, so the model computes
        # it once. The model needs num_input_neurons - 2 at least, because the GRF centres divide
        # by that difference.
        self.threshold_ = self.c * float(
            np.sum(self.mod ** (2.0 * np.arange(self.num_input_neurons)))
        )
        self._mod_powers = self.mod ** np.arange(self.num_input_neurons, dtype=np.float64)

        self.values_ = np.zeros(self.window_size)
        self.num_values_ = 0
        self.num_fitted_ = 0
        self.weights_ = np.zeros((self.num_output_neurons, self.num_input_neurons))
        self.output_values_ = np.zeros(self.num_output_neurons)
        self.update_times_ = np.zeros(self.num_output_neurons)
        self.update_counts_ = np.zeros(self.num_output_neurons, dtype=np.int64)
        self.num_neurons_ = 0
        self._window = np.empty(self.window_size)
        self._errors: deque[float] = deque()
        self._error_is_anomaly: deque[bool] = deque()
        self._error_stats = VarianceMeter()

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> OeSNNUAD:
        """Fits the model with the next value: it fills the window with the first values, and then it detects an anomaly in the value and learns from it.

        Args:
            X (float or np.float64 array): The instance to fit, holding exactly one value (e.g. a scalar or an array of shape (1,)).
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: The model itself.

        Raises:
            ValueError: If `X` does not hold exactly one value.
        """
        value = check_univariate(X, "OeSNNUAD")
        if self.num_values_ < self.window_size:
            self._append(value)
            if self.num_values_ == self.window_size:
                self._fill_error_stats()
            return self

        weights, prediction = self._encode_and_predict(value)
        if prediction is None:
            error, is_anomaly = math.inf, True
        else:
            error = abs(value - prediction)
            is_anomaly = self._score_error(error) >= self.epsilon

        self.num_fitted_ += 1
        self._append(value)
        self._learn(weights, value, is_anomaly)
        self._remember_error(error, is_anomaly)

        return self

    def score_partial(self, X: np.ndarray) -> float:
        """Scores the value with the current neurons and the current error statistics. This method does not change the model.

        Args:
            X (float or np.float64 array): The instance to score, holding exactly one value (e.g. a scalar or an array of shape (1,)). Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The excess of the prediction error of the value above the mean recent error, in standard deviations. It is 0.0 until the window is full, when no recent error comes from a normal value, or when the recent errors have no spread. It is the largest finite float when no output neuron fires.

        Raises:
            ValueError: If `X` does not hold exactly one value.
        """
        value = check_univariate(X, "OeSNNUAD")
        if self.num_values_ < self.window_size:
            return 0.0

        prediction = self._encode_and_predict(value)[1]
        if prediction is None:
            return _NO_FIRING_SCORE

        return self._score_error(abs(value - prediction))

    def _append(self, value: float) -> None:
        """Adds the value to the window and drops the oldest value when the window is full."""
        if self.num_values_ < self.window_size:
            self.values_[self.num_values_] = value
            self.num_values_ += 1
            return

        self.values_[:-1] = self.values_[1:]
        self.values_[-1] = value

    def _window_with(self, value: float) -> np.ndarray:
        """Returns the window of the recent values with ``value`` added, as the paper updates the window before it encodes the value."""
        if self.num_values_ < self.window_size:
            size = self.num_values_ + 1
            self._window[: size - 1] = self.values_[: self.num_values_]
        else:
            size = self.window_size
            self._window[: size - 1] = self.values_[1:]

        self._window[size - 1] = value
        return self._window[:size]

    @staticmethod
    def _sample_std(values: np.ndarray) -> float:
        """Returns the sample standard deviation of the values, with n - 1 in the denominator, or 0.0 for one value."""
        if values.size > 1:
            return float(values.std(ddof=1))
        return 0.0

    def _generator(self) -> np.random.RandomState:
        """Returns the random generator of the model, which is the global NumPy state when random_state is None."""
        return self._rng if self._rng is not None else check_random_state(None)

    def _fill_error_stats(self) -> None:
        """Gives each warm-up value a random prediction from the normal distribution of the window, and keeps its error."""
        values = self.values_[: self.num_values_]
        predictions = self._generator().normal(
            float(values.mean()), self._sample_std(values), size=self.window_size
        )
        for value, prediction in zip(values, predictions, strict=True):
            self._remember_error(abs(float(value) - float(prediction)), False)

    def _encode_and_predict(self, value: float) -> tuple[np.ndarray, float | None]:
        """Encodes the value with the window that will contain it.

        Args:
            value (float): The value to encode.

        Returns:
            tuple: The synaptic weights of the candidate output neuron, which are ``mod ** order(j)`` for each input neuron ``j``, and the output value of the first output neuron that fires, or None when no output neuron fires.
        """
        window = self._window_with(value)
        span = float(window.max()) - float(window.min())
        # Eq. (2) of the paper: the GRF centres spread over the window, one step of span / (NI - 2)
        # apart, with half a step outside the window at each end.
        centres = float(window.min()) + (
            2.0 * np.arange(self.num_input_neurons, dtype=np.float64) - 3.0
        ) * span / (2.0 * (self.num_input_neurons - 2))
        # Eq. (1) and Eq. (4) of the paper, with a common GRF width and a common firing time scale:
        # the excitation falls with the distance from the centre, so the firing order of the input
        # neurons is the order of those distances.
        in_order = np.argsort(np.abs(value - centres), kind="stable")
        order = np.empty(self.num_input_neurons, dtype=np.int64)
        order[in_order] = np.arange(self.num_input_neurons)
        weights = self._mod_powers[order]

        # Algorithm 3 of the paper: the model grows the post-synaptic potentials in firing order and
        # stops at the first step where a neuron fires, then takes the greatest potential.
        num_neurons = self.num_neurons_
        potentials = np.zeros(num_neurons)
        for k in range(self.num_input_neurons):
            potentials += self._mod_powers[k] * self.weights_[:num_neurons, in_order[k]]
            fired = np.flatnonzero(potentials > self.threshold_)
            if fired.size:
                best = int(fired[np.argmax(potentials[fired])])
                return weights, float(self.output_values_[best])

        return weights, None

    def _score_error(self, error: float) -> float:
        """Returns the excess of the error above the mean of the recent errors of normal values, in standard deviations."""
        count = self._error_stats.count
        if count == 0:
            return 0.0

        spread = math.sqrt(self._error_stats.m2 / (count - 1)) if count > 1 else 0.0
        if spread > 0.0:
            return (error - self._error_stats.mean) / spread

        return error - self._error_stats.mean

    def _learn(self, weights: np.ndarray, value: float, is_anomaly: bool) -> None:
        """Adds the candidate output neuron of the value to the repository, as Eq. (12) of the paper updates it."""
        mean = float(self.values_[: self.num_values_].mean())
        std = self._sample_std(self.values_[: self.num_values_])
        output_value = float(self._generator().normal(mean, std))
        if not is_anomaly:
            output_value += (value - output_value) * self.xi

        if not self.num_neurons_:
            self._set_neuron(0, weights, output_value)
            self.num_neurons_ = 1
            return

        distances = np.linalg.norm(self.weights_[: self.num_neurons_] - weights, axis=1)
        nearest = int(np.argmin(distances))
        if distances[nearest] <= self.sim:
            count = int(self.update_counts_[nearest])
            self.weights_[nearest] = (self.weights_[nearest] * count + weights) / (count + 1.0)
            self.output_values_[nearest] = (self.output_values_[nearest] * count + output_value) / (
                count + 1.0
            )
            self.update_times_[nearest] = (
                self.update_times_[nearest] * count + self.num_fitted_
            ) / (count + 1.0)
            self.update_counts_[nearest] = count + 1
            return

        if self.num_neurons_ < self.num_output_neurons:
            target = self.num_neurons_
            self.num_neurons_ += 1
        else:
            target = int(np.argmin(self.update_times_))

        self._set_neuron(target, weights, output_value)

    def _set_neuron(self, index: int, weights: np.ndarray, output_value: float) -> None:
        """Puts a fresh candidate output neuron into the repository at ``index``."""
        self.weights_[index] = weights
        self.output_values_[index] = output_value
        self.update_times_[index] = float(self.num_fitted_)
        self.update_counts_[index] = 1

    def _remember_error(self, error: float, is_anomaly: bool) -> None:
        """Keeps the error of the newest value and its anomaly flag, and forgets the oldest error when the window of errors is full."""
        if len(self._errors) >= self.window_size - 1:
            if not self._error_is_anomaly[0]:
                self._error_stats.remove(self._errors[0])
            self._errors.popleft()
            self._error_is_anomaly.popleft()

        self._errors.append(error)
        self._error_is_anomaly.append(is_anomaly)
        if not is_anomaly:
            self._error_stats.update(error)
