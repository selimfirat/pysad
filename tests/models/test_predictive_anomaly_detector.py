import numpy as np
import pytest

# River needs Python 3.11 or newer, so the test group installs it only there.
pytest.importorskip("river")


def _forecasters():
    from river import time_series, tree

    return {
        "SNARIMAX": lambda: time_series.SNARIMAX(p=3, d=1, q=1),
        "SNARIMAX-HoeffdingTree": lambda: time_series.SNARIMAX(
            p=3,
            d=0,
            q=0,
            regressor=tree.HoeffdingTreeRegressor(),
        ),
        "HoltWinters": lambda: time_series.HoltWinters(alpha=0.3, beta=0.1),
    }


FORECASTERS = ["SNARIMAX", "SNARIMAX-HoeffdingTree", "HoltWinters"]


@pytest.mark.parametrize("name", FORECASTERS)
def test_scores_are_the_errors_of_the_one_step_forecasts(name):
    """Each score is the absolute error of the forecast that the forecaster makes before it learns the value."""
    from pysad.integrations import PredictiveAnomalyDetector

    values = np.sin(np.arange(60) / 3) + np.random.default_rng(0).normal(scale=0.1, size=60)
    forecaster = _forecasters()[name]()

    expected = [0.0, 0.0]
    for value in values[:2]:
        forecaster.learn_one(y=value)
    for value in values[2:]:
        expected.append(abs(value - forecaster.forecast(horizon=1)[0]))
        forecaster.learn_one(y=value)

    model = PredictiveAnomalyDetector(_forecasters()[name](), probationary_period=2)
    scores = [model.fit_score_partial(np.array([value])) for value in values]

    assert all(type(score) is float for score in scores)
    np.testing.assert_allclose(scores, expected, rtol=0, atol=1e-12)


def test_relative_scores_divide_the_errors_by_the_forecasts():
    """With relative=True, each score is the absolute relative error of Eq. (1) of EGADS."""
    from river import time_series

    from pysad.integrations import PredictiveAnomalyDetector

    values = 5.0 + np.random.default_rng(0).normal(size=40)
    forecaster = time_series.HoltWinters(alpha=0.3, beta=0.1)

    expected = [0.0, 0.0]
    for value in values[:2]:
        forecaster.learn_one(y=value)
    for value in values[2:]:
        forecast = forecaster.forecast(horizon=1)[0]
        expected.append(abs(value - forecast) / abs(forecast))
        forecaster.learn_one(y=value)

    model = PredictiveAnomalyDetector(
        time_series.HoltWinters(alpha=0.3, beta=0.1),
        relative=True,
        probationary_period=2,
    )

    np.testing.assert_allclose(model.fit_score(values.reshape(-1, 1)), expected, rtol=0, atol=1e-12)


def test_score_partial_does_not_change_the_scores():
    from river import time_series

    from pysad.integrations import PredictiveAnomalyDetector

    values = np.random.default_rng(0).normal(size=(80, 1))
    scored = PredictiveAnomalyDetector(time_series.SNARIMAX(p=2, d=0, q=1))
    unscored = PredictiveAnomalyDetector(time_series.SNARIMAX(p=2, d=0, q=1))
    for x in values[:50]:
        scored.score_partial(x)
        scored.fit_partial(x)
        unscored.fit_partial(x)

    np.testing.assert_array_equal(scored.fit_score(values[50:]), unscored.fit_score(values[50:]))


def test_pickle_round_trip():
    import pickle

    from river import time_series

    from pysad.integrations import PredictiveAnomalyDetector

    values = np.random.default_rng(0).normal(size=(80, 1))
    model = PredictiveAnomalyDetector(time_series.SNARIMAX(p=2, d=0, q=1))
    model.fit(values[:50])

    restored = pickle.loads(pickle.dumps(model))

    np.testing.assert_array_equal(restored.fit_score(values[50:]), model.fit_score(values[50:]))


def test_ranks_spikes_in_a_seasonal_series():
    from river import time_series
    from sklearn.metrics import roc_auc_score

    from pysad.integrations import PredictiveAnomalyDetector

    rng = np.random.default_rng(0)
    t = np.arange(1500)
    values = np.sin(2 * np.pi * t / 24) + 0.1 * rng.normal(size=t.size)
    labels = np.zeros(t.size, dtype=np.int64)
    spikes = rng.choice(np.arange(200, t.size), size=30, replace=False)
    values[spikes] += rng.choice([-1.0, 1.0], size=spikes.size)
    labels[spikes] = 1

    model = PredictiveAnomalyDetector(time_series.SNARIMAX(p=24, d=0, q=0), probationary_period=48)
    scores = model.fit_score(values.reshape(-1, 1))

    assert roc_auc_score(labels[200:], scores[200:]) > 0.9


def test_rejects_multivariate_instances():
    from river import time_series

    from pysad.integrations import PredictiveAnomalyDetector

    model = PredictiveAnomalyDetector(time_series.SNARIMAX(p=1, d=0, q=0))

    with pytest.raises(ValueError, match="univariate"):
        model.fit_partial(np.zeros(2))
    with pytest.raises(ValueError, match="univariate"):
        model.score_partial(np.zeros(2))
