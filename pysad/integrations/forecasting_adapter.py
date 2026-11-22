from __future__ import annotations

from typing import Any, ClassVar, Protocol, runtime_checkable

from numpy.typing import NDArray

from pysad.core.base_model import BaseModel
from pysad.utils.validation import _int_at_least, check_univariate


@runtime_checkable
class ForecasterProtocol(Protocol):
    """Protocol for a duck-typed streaming forecaster.

    A forecaster implements one-step-ahead forecasting and online learning.
    Any object with these methods can be used with :class:`ForecastingAnomalyDetector`
    without depending on a specific forecasting library.

    Examples:
        A simple forecaster using a moving average:

        >>> class MovingAverageForecaster:
        ...     def __init__(self, window: int = 10):
        ...         self.window = window
        ...         self.history = []
        ...     def forecast(self, horizon: int = 1) -> list[float]:
        ...         if not self.history:
        ...             return [0.0]
        ...         return [sum(self.history[-self.window:]) / min(len(self.history), self.window)]
        ...     def learn_one(self, y: float) -> None:
        ...         self.history.append(y)
    """

    def forecast(self, horizon: int = 1) -> list[float] | NDArray[Any]:
        """Returns the forecast for the next ``horizon`` steps.

        Args:
            horizon (int): Number of steps to forecast (Default=1).

        Returns:
            list[float] or array-like: Forecasted values. The first element is the
            one-step-ahead forecast.
        """
        ...

    def learn_one(self, y: float) -> None:
        """Updates the forecaster with the observed value ``y``.

        Args:
            y (float): The observed value to learn from.
        """
        ...


class ForecastingAnomalyDetector(BaseModel):
    """Scores each value of a univariate stream by its deviation from the one-step
    forecast of a duck-typed forecaster.

    This is a zero-dependency version of :class:`~pysad.integrations.PredictiveAnomalyDetector`
    that works with any forecaster implementing :class:`ForecasterProtocol`. The forecaster
    is not required to come from River or any specific library.

    The anomaly score of ``x_t`` is the absolute prediction error ``|x_t - u_t|`` by default,
    or the absolute relative error ``|x_t - u_t| / |u_t|`` with ``relative=True``. A forecast
    of exactly 0 raises ``ZeroDivisionError`` when ``relative=True``.

    The first ``probationary_period`` values receive a score of 0.0, allowing the forecaster
    to accumulate enough history before producing meaningful forecasts.

    The model differs from :class:`~pysad.integrations.PredictiveAnomalyDetector` as follows:

    - It accepts any forecaster with ``forecast(horizon)`` and ``learn_one(y)`` methods,
      not only River forecasters.
    - It does not depend on River or any external forecasting library.

    Args:
        forecaster (object): A forecaster implementing :class:`ForecasterProtocol`.
            The detector trains this object in place.
        relative (bool): Whether to divide the absolute prediction error by the absolute
            forecast. A forecast of exactly 0 then raises ``ZeroDivisionError``
            (Default=False).
        probationary_period (int): The number of first values that get the score 0
            (Default=0).

    Raises:
        TypeError: If ``probationary_period`` is not an int.
        ValueError: If ``probationary_period`` is negative.
    """

    _is_detector: ClassVar[bool] = False

    def __init__(
        self,
        forecaster: Any,
        relative: bool = False,
        probationary_period: int = 0,
    ) -> None:
        if not isinstance(forecaster, ForecasterProtocol):
            # Allow any object with the required methods (duck typing)
            required = ("forecast", "learn_one")
            for method in required:
                if not hasattr(forecaster, method) or not callable(getattr(forecaster, method)):
                    raise TypeError(
                        f"forecaster must have a callable '{method}' method, "
                        f"got {type(forecaster).__name__}"
                    )
        self.forecaster = forecaster
        self.relative = relative
        self.probationary_period = _int_at_least(
            value=probationary_period,
            name="probationary_period",
            minimum=0,
        )
        self.num_values_ = 0

    def fit_partial(self, X: NDArray[Any], y: int | None = None) -> ForecastingAnomalyDetector:
        """Updates the forecaster with the next value.

        Args:
            X (float or np.float64 array): The instance to fit, holding exactly one value
                (e.g. a scalar or an array of shape (1,)).
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: The model itself.

        Raises:
            ValueError: If ``X`` does not hold exactly one value.
        """
        value = check_univariate(X, "ForecastingAnomalyDetector")
        self.forecaster.learn_one(y=value)
        self.num_values_ += 1
        return self

    def score_partial(self, X: NDArray[Any]) -> float:
        """Scores the next value by its deviation from the forecast of the forecaster.

        Args:
            X (float or np.float64 array): The instance to score, holding exactly one value
                (e.g. a scalar or an array of shape (1,)). Higher scores represent more
                anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The anomalousness score of the input instance.

        Raises:
            ValueError: If ``X`` does not hold exactly one value.
            ZeroDivisionError: If ``relative=True`` and the forecast is exactly 0.
        """
        value = check_univariate(X, "ForecastingAnomalyDetector")
        if self.num_values_ < self.probationary_period:
            return 0.0

        forecast = self.forecaster.forecast(horizon=1)[0]
        error = abs(value - forecast)

        return float(error / abs(forecast) if self.relative else error)
