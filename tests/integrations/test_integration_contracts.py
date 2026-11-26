"""Integration contract tests for duck-typed forecasting callables and stream
state preservation.

These tests exercise the integration components as stream contracts:

- A duck-typed forecasting callable receives the growing observed history and
  scores prequentially: the forecast is computed before the model learns from the
  current value. Sequential ``fit_score_partial`` matches batch ``fit_score``.
- Stream state preservation: ``score_partial`` does not mutate the model, and
  re-scoring the same stream does not drift the model state.

The River adapter is already covered by ``tests/models/test_pysad_to_river_adapter.py``,
so no duplicate tests are added here.
"""

from __future__ import annotations

import numpy as np
import pytest


def _mean_callable(history, horizon=1):
    """Predict the mean of the values observed so far.

    Args:
        history (list[float]): The values observed so far.
        horizon (int): Number of steps to forecast (Default=1).

    Returns:
        float: The mean of ``history``, or 0.0 when empty.
    """
    if not history:
        return 0.0
    return sum(history) / len(history)


# ---------------------------------------------------------------------------
# Duck-typed forecasting callables
# ---------------------------------------------------------------------------


def test_callable_forecaster_receives_the_observed_history():
    """The callable sees the values observed so far, before the current value."""
    from pysad.integrations import ForecastingAnomalyDetector

    seen: list[list[float]] = []

    def forecast(history, horizon=1):
        seen.append(list(history))
        return _mean_callable(history, horizon)

    model = ForecastingAnomalyDetector(forecaster=forecast)
    for value in [1.0, 3.0, 2.0]:
        model.fit_score_partial(np.array([value]))

    assert seen == [[], [1.0], [1.0, 3.0]]


def test_callable_forecaster_batch_matches_sequential():
    """Batch ``fit_score`` equals sequential prequential ``fit_score_partial``."""
    from pysad.integrations import ForecastingAnomalyDetector

    values = np.array([1.0, 3.0, 2.0, 10.0, -5.0, 7.0])

    sequential = ForecastingAnomalyDetector(forecaster=_mean_callable)
    sequential_scores = [sequential.fit_score_partial(np.array([value])) for value in values]

    batched = ForecastingAnomalyDetector(forecaster=_mean_callable)
    batch_scores = batched.fit_score(values.reshape(-1, 1))

    np.testing.assert_allclose(batch_scores, np.array(sequential_scores), rtol=1e-10, atol=1e-12)


def test_callable_forecaster_score_is_idempotent():
    """Repeated ``score_partial`` calls do not change the model or the score."""
    from pysad.integrations import ForecastingAnomalyDetector

    model = ForecastingAnomalyDetector(forecaster=_mean_callable)
    for value in [1.0, 3.0, 2.0]:
        model.fit_partial(np.array([value]))

    xi = np.array([5.0])
    first = model.score_partial(xi)
    assert model.score_partial(xi) == first
    assert model.score_partial(xi) == first


# ---------------------------------------------------------------------------
# Stream state preservation
# ---------------------------------------------------------------------------


def test_scoring_a_stream_twice_gives_the_same_scores():
    """Re-scoring the same fitted stream does not drift the model state."""
    from pysad.integrations import ForecastingAnomalyDetector

    values = [1.0, 3.0, 2.0, 10.0, -5.0]

    model = ForecastingAnomalyDetector(forecaster=_mean_callable)
    for value in values:
        model.fit_partial(np.array([value]))

    first_pass = [model.score_partial(np.array([value])) for value in values]
    second_pass = [model.score_partial(np.array([value])) for value in values]

    np.testing.assert_array_equal(first_pass, second_pass)


# ---------------------------------------------------------------------------
# ForecastingAnomalyDetector constructor and scoring modes
# ---------------------------------------------------------------------------


def test_invalid_forecaster_raises_type_error():
    """A non-callable object without forecast and learn_one raises TypeError."""
    from pysad.integrations import ForecastingAnomalyDetector

    with pytest.raises(TypeError, match="forecaster must implement"):
        ForecastingAnomalyDetector(forecaster=object())


def test_object_forecaster_forecast_returning_list_is_unwrapped():
    """An object forecaster whose forecast returns a list is unwrapped to its first value."""
    from pysad.integrations import ForecastingAnomalyDetector

    class ListForecaster:
        def forecast(self, horizon: int = 1) -> list[float]:
            return [7.0]

        def learn_one(self, y: float) -> None:
            pass

    model = ForecastingAnomalyDetector(forecaster=ListForecaster())
    # value 1.0, forecast 7.0 -> |1.0 - 7.0| = 6.0
    assert model.fit_score_partial(np.array([1.0])) == 6.0


def test_relative_error_divides_by_absolute_forecast():
    """relative=True returns the absolute error divided by the absolute forecast."""
    from pysad.integrations import ForecastingAnomalyDetector

    def forecast(history, horizon=1):
        return 5.0

    model = ForecastingAnomalyDetector(forecaster=forecast, relative=True)
    # value 1.0, forecast 5.0 -> |1.0 - 5.0| / |5.0| = 0.8
    assert model.fit_score_partial(np.array([1.0])) == 0.8
    # value 9.0, forecast 5.0 -> |9.0 - 5.0| / |5.0| = 0.8
    assert model.fit_score_partial(np.array([9.0])) == 0.8


def test_relative_error_with_zero_forecast_raises_zero_division_error():
    """relative=True with a forecast of exactly 0 raises ZeroDivisionError."""
    from pysad.integrations import ForecastingAnomalyDetector

    def forecast(history, horizon=1):
        return 0.0

    model = ForecastingAnomalyDetector(forecaster=forecast, relative=True)
    with pytest.raises(ZeroDivisionError):
        model.score_partial(np.array([1.0]))


def test_probationary_period_returns_zero():
    """The first probationary_period values score 0.0 before the forecaster has history."""
    from pysad.integrations import ForecastingAnomalyDetector

    def forecast(history, horizon=1):
        return 5.0

    model = ForecastingAnomalyDetector(forecaster=forecast, probationary_period=2)
    assert model.fit_score_partial(np.array([1.0])) == 0.0
    assert model.fit_score_partial(np.array([9.0])) == 0.0
    # After probation, the forecast is used: value 1.0, forecast 5.0 -> |1.0 - 5.0| = 4.0
    assert model.fit_score_partial(np.array([1.0])) == 4.0
