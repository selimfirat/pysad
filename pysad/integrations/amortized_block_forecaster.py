from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

import numpy as np

from pysad.core.base_model import BaseModel
from pysad.utils.validation import _int_at_least, check_univariate

__bibtex__ = """
@inproceedings{ansari2024chronos,
  title={Chronos: Learning the Language of Time Series},
  author={Ansari, Abdul Fatir and Stella, Lorenzo and Turkmen, Caner and Zhang, Xiyuan and Mercado, Pedro and Shen, Huibin and Shchur, Oleksandr and Rangapuram, Syama Sundar and Pineda Arango, Sebastian and Kapoor, Shubham and Zschiegner, Jasper and Maddix, Danielle C. and Mahoney, Michael W. and Torkkola, Kari and Gordon Wilson, Andrew and Bohlke-Schneider, Michael and Wang, Yuyang},
  booktitle={Proceedings of the 41st International Conference on Machine Learning},
  year={2024},
  url={https://openreview.net/forum?id=gerNCVqqtR}
}

@inproceedings{das2024timesfm,
  title={A decoder-only foundation model for time-series forecasting},
  author={Das, Abhimanyu and Kong, Weihao and Sen, Rajat and Zhou, Yichen},
  booktitle={Proceedings of the 41st International Conference on Machine Learning},
  year={2024},
  url={https://arxiv.org/abs/2310.10688}
}
"""


class AmortizedBlockForecaster(BaseModel):
    """Scores each value of a univariate stream by its deviation from the forecast of a foundation model forecaster,
    using amortized block forecasting :cite:`ansari2024chronos,das2024timesfm`.

    Foundation models like Chronos :cite:`ansari2024chronos` and TimesFM :cite:`das2024timesfm` are pretrained
    time-series transformers that can forecast multiple future steps (a horizon) in a single forward pass.
    This detector amortizes the cost of the expensive foundation model inference by forecasting a block of
    ``horizon`` steps at once, then scoring each incoming value against the corresponding forecast in the block.
    When the block is exhausted, the model re-forecasts the next block using the updated context window.

    The forecaster is a callable that takes a context array of shape ``(context_length,)`` and a horizon ``int``,
    and returns a forecast array of shape ``(horizon,)``. Users can wrap Chronos, TimesFM, or any compatible
    forecaster with a simple adapter function.

    The anomaly score of ``x_t`` is the absolute prediction error ``|x_t - u_t|`` by default, or the absolute
    relative error ``|x_t - u_t| / |u_t|`` with ``relative=True``. The first ``probationary_period`` values
    get the score 0.0 while the context window fills.

    The implementation differs from the foundation model papers as follows:

    - The papers describe the pretrained models for forecasting, not streaming anomaly detection. This detector
      uses the models in a predictive anomaly detection framework :cite:`laptev2015generic`.
    - Foundation models do not update their weights online. This detector only updates the sliding context window
      with observed values; the forecaster itself is stateless between calls.
    - The model forecasts a block of ``horizon`` steps at once for efficiency, rather than one step at a time.

    Args:
        forecaster (Callable[[np.ndarray, int], np.ndarray]): A callable that takes a context array of shape
            ``(context_length,)`` and a horizon ``int``, and returns a forecast array of shape ``(horizon,)``.
        context_length (int): The number of past values to use as context for forecasting. Must be at least 1.
        horizon (int): The number of future steps to forecast in each block. Must be at least 1.
        relative (bool): Whether to divide the absolute prediction error by the absolute forecast.
            A forecast of exactly 0 then raises ``ZeroDivisionError`` (Default=False).
        probationary_period (int): The number of first values that get the score 0.0 while the context
            window fills (Default=0).

    Raises:
        ValueError: If ``context_length`` or ``horizon`` is less than 1, or ``probationary_period`` is negative.
        TypeError: If ``forecaster`` is not callable, or ``probationary_period`` is not an int.
    """

    _is_detector: ClassVar[bool] = True

    def __init__(
        self,
        forecaster: Callable[[np.ndarray, int], np.ndarray],
        context_length: int,
        horizon: int,
        relative: bool = False,
        probationary_period: int = 0,
    ) -> None:
        if not callable(forecaster):
            raise TypeError(f"forecaster must be callable, got {type(forecaster).__name__}.")
        self.forecaster = forecaster
        self.context_length = _int_at_least(context_length, "context_length", 1)
        self.horizon = _int_at_least(horizon, "horizon", 1)
        self.relative = relative
        self.probationary_period = _int_at_least(
            value=probationary_period,
            name="probationary_period",
            minimum=0,
        )

        self._context: np.ndarray = np.zeros(self.context_length, dtype=np.float64)
        self._context_filled = 0
        self._forecast_buffer: np.ndarray | None = None
        self._buffer_index = 0
        self.num_values_ = 0

    def _refill_buffer(self) -> None:
        """Calls the forecaster to fill the forecast buffer with the next horizon predictions."""
        self._forecast_buffer = self.forecaster(self._context, self.horizon)
        if self._forecast_buffer.shape != (self.horizon,):
            raise ValueError(
                f"forecaster returned shape {self._forecast_buffer.shape}, expected ({self.horizon},)."
            )
        self._buffer_index = 0

    def _is_active(self) -> bool:
        """Returns True if the model is in the active scoring phase (past probationary period and context full)."""
        return (
            self.num_values_ >= self.probationary_period
            and self._context_filled == self.context_length
        )

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> AmortizedBlockForecaster:
        """Updates the context window with the next value and advances the forecast buffer.

        Args:
            X (np.float64 array of shape (1,)): The next univariate value.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: The model itself.
        """
        x = check_univariate(X, "AmortizedBlockForecaster")

        is_active_now = self.num_values_ >= self.probationary_period

        if self._context_filled < self.context_length:
            self._context[self._context_filled] = x
            self._context_filled += 1
            if self._context_filled == self.context_length and is_active_now:
                self._refill_buffer()
        else:
            self._context = np.roll(self._context, -1)
            self._context[-1] = x
            if is_active_now:
                if self._forecast_buffer is None:
                    self._refill_buffer()
                else:
                    self._buffer_index += 1
                    if self._buffer_index >= self.horizon:
                        self._refill_buffer()

        self.num_values_ += 1

        # After increment, if we just finished the probationary period and context is full,
        # refill the buffer so the next score_partial uses a current forecast.
        if (
            self.num_values_ == self.probationary_period
            and self._context_filled == self.context_length
        ):
            self._refill_buffer()

        return self

    def score_partial(self, X: np.ndarray) -> float:
        """Scores the next value by its deviation from the forecast in the buffer (read-only).

        Args:
            X (np.float64 array of shape (1,)): The next univariate value. Higher scores represent
                more anomalous values.

        Returns:
            float: The anomalousness score of the input value, or 0.0 during the probationary period
            or while the context window is filling.
        """
        value = check_univariate(X, "AmortizedBlockForecaster")

        if not self._is_active():
            return 0.0

        if self._forecast_buffer is None or self._buffer_index >= self.horizon:
            return 0.0

        forecast = self._forecast_buffer[self._buffer_index]
        error = abs(value - forecast)
        if self.relative:
            if forecast == 0.0:
                raise ZeroDivisionError("Forecast is zero; cannot compute relative error.")
            return float(error / abs(forecast))
        return float(error)

    def fit_score_partial(self, X: np.ndarray, y: int | None = None) -> float:
        """Scores the next instance with the model from earlier instances, then updates the model with it.

        This override ensures atomic score-then-fit semantics: the score uses the current buffer position,
        and the fit advances the buffer after updating the context.

        Args:
            X (np.float64 array of shape (1,)): The instance to fit and score.
            y (int or None): The label of the instance (ignored).

        Returns:
            float: The anomalousness score of the input instance.
        """
        from pysad.core.base_model import _has_nan, _to_float_score

        if _has_nan(X):
            raise ValueError(f"{type(self).__name__} does not accept NaN values.")

        score = self.score_partial(X)
        self.fit_partial(X, y)
        return score if type(score) is float else _to_float_score(score)
