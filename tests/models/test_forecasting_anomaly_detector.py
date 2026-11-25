import numpy as np
import pytest


def _mean_forecaster():
    """User callable: predicts the mean of the observed history."""

    def forecast(history, horizon=1):
        if not history:
            return 0.0
        return sum(history) / len(history)

    return forecast


def test_callable_scores_are_absolute_errors_of_the_forecast():
    """With a user callable, each score is |value - forecast(history so far)| before learning."""
    from pysad.integrations import ForecastingAnomalyDetector

    values = [1.0, 2.0, 4.0, 7.0, 11.0]
    model = ForecastingAnomalyDetector(forecaster=_mean_forecaster())

    expected = []
    history = []
    for value in values:
        forecast = sum(history) / len(history) if history else 0.0
        expected.append(abs(value - forecast))
        history.append(value)

    scores = [model.fit_score_partial(np.array([value])) for value in values]

    assert all(type(score) is float for score in scores)
    np.testing.assert_allclose(scores, expected, rtol=0, atol=1e-12)


def test_callable_relative_scores_divide_by_the_forecast():
    """With relative=True, each score is |value - forecast| / |forecast|."""
    from pysad.integrations import ForecastingAnomalyDetector

    values = [2.0, 4.0, 8.0]

    def forecast(history, horizon=1):
        return history[-1] if history else 1.0

    model = ForecastingAnomalyDetector(forecaster=forecast, relative=True)

    expected = []
    history = []
    for value in values:
        forecast_value = history[-1] if history else 1.0
        expected.append(abs(value - forecast_value) / abs(forecast_value))
        history.append(value)

    scores = [model.fit_score_partial(np.array([value])) for value in values]

    np.testing.assert_allclose(scores, expected, rtol=0, atol=1e-12)


def test_callable_probationary_period_returns_zero():
    """The first probationary_period values score 0.0 while history builds up."""
    from pysad.integrations import ForecastingAnomalyDetector

    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    model = ForecastingAnomalyDetector(forecaster=_mean_forecaster(), probationary_period=2)

    scores = [model.fit_score_partial(np.array([value])) for value in values]

    assert scores[0] == 0.0
    assert scores[1] == 0.0
    assert scores[2] == abs(3.0 - (1.0 + 2.0) / 2.0)


def test_callable_score_partial_does_not_change_state():
    """Scoring without fitting never updates the history, so repeated scores are identical."""
    from pysad.integrations import ForecastingAnomalyDetector

    values = np.random.default_rng(0).normal(size=20)
    scored = ForecastingAnomalyDetector(forecaster=_mean_forecaster())
    unscored = ForecastingAnomalyDetector(forecaster=_mean_forecaster())

    for x in values[:10]:
        scored.score_partial(np.array([x]))
        scored.fit_partial(np.array([x]))
        unscored.fit_partial(np.array([x]))

    np.testing.assert_array_equal(scored.fit_score(values[10:]), unscored.fit_score(values[10:]))


def test_callable_ranks_spikes_above_chance():
    """A moving-average callable flags injected spikes with AUROC above chance."""
    from sklearn.metrics import roc_auc_score

    from pysad.integrations import ForecastingAnomalyDetector

    rng = np.random.default_rng(7)
    points = 800
    stream = 10.0 + 0.1 * rng.normal(size=points)
    labels = np.zeros(points, dtype=np.int64)
    spikes = rng.choice(np.arange(100, points), size=20, replace=False)
    stream[spikes] += 5.0
    labels[spikes] = 1

    def forecast(history, horizon=1):
        if not history:
            return 0.0
        return sum(history[-20:]) / min(len(history), 20)

    model = ForecastingAnomalyDetector(forecaster=forecast, probationary_period=40)
    scores = model.fit_score(stream.reshape(-1, 1))

    assert roc_auc_score(labels[100:], scores[100:]) > 0.9


def test_forecaster_object_still_works():
    """A forecaster object with forecast(horizon) and learn_one(y) still works."""
    from pysad.integrations import ForecastingAnomalyDetector

    class SimpleForecaster:
        def __init__(self):
            self.history = []

        def forecast(self, horizon=1):
            if not self.history:
                return [0.0]
            return [sum(self.history) / len(self.history)]

        def learn_one(self, y):
            self.history.append(y)

    values = [1.0, 2.0, 4.0]
    model = ForecastingAnomalyDetector(forecaster=SimpleForecaster())

    expected = []
    history = []
    for value in values:
        f = sum(history) / len(history) if history else 0.0
        expected.append(abs(value - f))
        history.append(value)

    scores = [model.fit_score_partial(np.array([value])) for value in values]
    np.testing.assert_allclose(scores, expected, rtol=0, atol=1e-12)


def test_rejects_invalid_forecaster():
    """A value that is neither a forecaster object nor a callable raises TypeError."""
    from pysad.integrations import ForecastingAnomalyDetector

    with pytest.raises(TypeError, match="forecaster"):
        ForecastingAnomalyDetector(forecaster=42)
    with pytest.raises(TypeError, match="forecaster"):
        ForecastingAnomalyDetector(forecaster="not a forecaster")
