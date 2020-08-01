from abc import abstractmethod, ABC


class BaseMetric(ABC):
    """Abstract base class for metrics.

    """

    def __init__(self):
        self.score = None

    @abstractmethod
    def update(self, y_true, y_pred):
        """Updates the metric with given true and predicted value for a timestep.

        Args:
            y_true: int
                Ground truth class. Either 1 or 0.
            y_pred: int
                Predicted class. Either 1 or 0.
        """
        pass

    @abstractmethod
    def get(self):
        """Gets the current value of the score.

        Returns:
            score: float
                The current score.
        """
        return self.score
