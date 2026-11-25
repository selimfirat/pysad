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
