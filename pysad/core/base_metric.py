from __future__ import annotations

from abc import abstractmethod, ABC


class BaseMetric(ABC):
    """Abstract base class for metrics.
    """

    def __init__(self) -> None:
        self.score: float | None = None

    @abstractmethod
    def update(self, y_true: int, y_pred: float) -> BaseMetric | None:
        """Updates the metric with given true and predicted value for a timestep.

        Args:
            y_true (int): Ground truth class. Either 1 or 0.
            y_pred (float): Predicted class or anomaly score. Higher values correspond to more anomalousness and lower values correspond to more normalness.
        """
        pass

    @abstractmethod
    def get(self) -> float | None:
        """Gets the current value of the score. Note that some methods such as AUPR and AUROC gives exception when used with only one class exist in the list of previous y_trues.

        Returns:
            float: The current score.
        """
        return self.score
