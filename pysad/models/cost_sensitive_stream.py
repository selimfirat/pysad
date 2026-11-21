from __future__ import annotations

import numpy as np

from pysad.core.base_model import BaseModel
from pysad.utils.validation import _finite_float


class CostSensitiveStreamModel(BaseModel):
    """Scores each instance with the decision margin of a cost-sensitive online classifier :cite:`wang2013cost`.

    The model implements the Cost-Sensitive Online Gradient Descent II algorithm from the paper. The paper treats the positive class as rare. Its analysis assumes that each instance has norm at most one. It uses the weighted hinge loss

    ``loss = cost_ratio * max(0, 1 - y * dot(w, x))`` for a positive instance, and ``loss = max(0, 1 - y * dot(w, x))`` for a negative instance.

    The online gradient descent update changes the weight vector only when the signed margin is below one. For an active update, the update is ``w += learning_rate * cost_weight * y * x``. Here, ``y`` is one for the positive class and minus one for the negative class. The score is ``dot(w, x)``. Higher scores mean that the model considers an instance more anomalous.

    The model maps label one to the paper's positive class and label zero to its negative class. A missing label means that the instance is normal. This lets the model learn from a stream without anomaly feedback. It also means that an unlabeled anomaly can affect the decision boundary.

    The paper returns a hard class label. This class returns the signed margin so that callers can rank instances. The paper tunes the learning rate and cost ratio for each dataset. This class exposes both values. The default cost ratio is 1,000 because missing an anomaly costs much more than a false alarm.

    Reference implementation:
        Searches of Papers with Code, GitHub, CRAN, and PyPI found no authors' implementation or compatible public implementation. This implementation follows the paper. This implementation copies no reference code.

    The model keeps one weight vector. Its memory and update time do not grow with the stream length.

    Args:
        cost_ratio (float): The cost of a missed anomaly divided by the cost of a false alarm. This is the weight ``rho`` for the positive class in the paper. It must be positive (Default=1000.0).
        learning_rate (float): The online gradient descent step size, ``eta`` in the paper. It must be positive (Default=0.01).
    """

    def __init__(self, cost_ratio: float = 1000.0, learning_rate: float = 0.01) -> None:
        self.cost_ratio = _finite_float(cost_ratio, "cost_ratio")
        self.learning_rate = _finite_float(learning_rate, "learning_rate")
        if self.cost_ratio <= 0.0:
            raise ValueError(f"cost_ratio must be positive, got {self.cost_ratio}.")
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}.")

        self.weights_: np.ndarray | None = None

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> CostSensitiveStreamModel:
        """Updates the model with the next instance and its label.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int or None): The label of the instance. One marks an anomaly, and zero marks a normal instance. None treats the instance as normal (Default=None).

        Raises:
            ValueError: If `y` is not 0, 1, or None.

        Returns:
            object: The model itself.
        """
        if y is None or y == 0:
            signed_label = -1.0
            cost_weight = 1.0
        elif y == 1:
            signed_label = 1.0
            cost_weight = self.cost_ratio
        else:
            raise ValueError(f"y must be 0, 1, or None, got {y!r}.")

        x = np.asarray(X, dtype=np.float64)
        if self.weights_ is None:
            self.weights_ = np.zeros(x.shape[0], dtype=np.float64)

        margin = signed_label * float(self.weights_ @ x)
        if margin < 1.0:
            self.weights_ += self.learning_rate * cost_weight * signed_label * x

        return self

    def score_partial(self, X: np.ndarray) -> float:
        """Scores the next instance without changing the model.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score.

        Returns:
            float: The signed decision margin. Higher scores mean that the model considers the instance more anomalous.
        """
        if self.weights_ is None:
            return 0.0

        x = np.asarray(X, dtype=np.float64)
        return float(self.weights_ @ x)
