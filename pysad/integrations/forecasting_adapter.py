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


def _is_forecaster_object(forecaster: Any) -> bool:
    """Returns True if ``forecaster`` exposes the ``ForecasterProtocol`` methods.

    The check uses duck typing so any object with callable ``forecast`` and ``learn_one``
    methods qualifies, without importing River or another forecasting library.
    """
    return (
        hasattr(forecaster, "forecast")
        and hasattr(forecaster, "learn_one")
        and callable(forecaster.forecast)
        and callable(forecaster.learn_one)
    )


class ForecastingAnomalyDetector(BaseModel):
    """Scores each value of a univariate stream by its deviation from the one-step
    forecast of a duck-typed forecaster or a user callable.

    This is a zero-dependency version of :class:`~pysad.integrations.PredictiveAnomalyDetector`.
    It works with any forecaster implementing :class:`ForecasterProtocol`, and it also
    accepts a plain user callable. The forecaster is not required to come from River or
    any specific forecasting library such as Chronos, TimesFM, or StatsForecast.

    The anomaly score of ``x_t`` is the absolute prediction error ``|x_t - u_t|`` by default,
    or the absolute relative error ``|x_t - u_t| / |u_t|`` with ``relative=True``. A forecast
    of exactly 0 raises ``ZeroDivisionError`` when ``relative=True``.

    The first ``probationary_period`` values receive a score of 0.0, allowing the forecaster
    to accumulate enough history before producing meaningful forecasts.

    A user callable must have the signature ``forecast(history, horizon=1) -> float``, where
    ``history`` is the list of values observed so far. The detector keeps this history and
    calls the callable to forecast the next value before it learns from the current one.
    This makes it easy to wrap Chronos, TimesFM, or StatsForecast:

    .. code-block:: python

        from pysad.integrations import ForecastingAnomalyDetector

        def chronos_forecast(history, horizon=1):
            # history is the list of observed values; close over a fitted Chronos pipeline.
            return float(predict_next(history))

        detector = ForecastingAnomalyDetector(forecaster=chronos_forecast)

    The model differs from :class:`~pysad.integrations.PredictiveAnomalyDetector` as follows:

    - It accepts any forecaster with ``forecast(horizon)`` and ``learn_one(y)`` methods,
      not only River forecasters.
    - It accepts a user callable ``forecast(history, horizon=1) -> float``.
    - It does not depend on River or any external forecasting library.

    Args:
        forecaster (object or callable): A forecaster implementing :class:`ForecasterProtocol`,
            or a callable ``forecast(history, horizon=1) -> float``. The detector trains the
            forecaster in place and keeps the observed history for a callable.
        relative (bool): Whether to divide the absolute prediction error by the absolute
            forecast. A forecast of exactly 0 then raises ``ZeroDivisionError``
            (Default=False).
        probationary_period (int): The number of first values that get the score 0
            (Default=0).

    Raises:
        TypeError: If ``probationary_period`` is not an int, or if ``forecaster`` is neither
            a forecaster object with ``forecast`` and ``learn_one`` methods nor a callable.
        ValueError: If ``probationary_period`` is negative.
    """

    _is_detector: ClassVar[bool] = False

    def __init__(
        self,
        forecaster: Any,
        relative: bool = False,
        probationary_period: int = 0,
    ) -> None:
        if _is_forecaster_object(forecaster):
            self._use_callable = False
            self._forecast_fn = None
        elif callable(forecaster):
            self._use_callable = True
            self._forecast_fn = forecaster
        else:
            raise TypeError(
                "forecaster must implement forecast(horizon) and learn_one(y), "
                "or be a callable forecast(history, horizon=1) -> float; "
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
        # Observed history, used to forecast the next value when a user callable is given.
        self._history: list[float] = []

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
        if self._use_callable:
            self._history.append(value)
        else:
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

        if self._use_callable:
            forecast_fn = self._forecast_fn
            if forecast_fn is None:
                raise RuntimeError("Forecasting callable is not initialized.")
            forecast = forecast_fn(self._history, horizon=1)
        else:
            forecast = self.forecaster.forecast(horizon=1)[0]

        forecast = float(forecast)
        error = abs(value - forecast)

        return float(error / abs(forecast) if self.relative else error)
