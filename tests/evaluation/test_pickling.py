import pickle

import numpy as np
import pytest

import pysad.evaluation as evaluation

SEED = 61
NUM_FIT = 60
NUM_NEXT = 20
WINDOW_SIZE = 16

# BaseSKLearnMetric is abstract. WindowedMetric takes a metric class as a constructor argument,
# rather than being default-constructible itself; it is covered separately below.
PICKLE_SKIP_METRICS = {"BaseSKLearnMetric", "WindowedMetric"}
WINDOWED_METRICS = ["AUROCMetric", "AUPRMetric", "PrecisionMetric", "RecallMetric"]

# Precision and recall are defined on predicted classes, so they get a threshold to turn the
# scores below into 0/1 predictions.
METRIC_KWARGS = {
    "PrecisionMetric": {"threshold": 0.5},
    "RecallMetric": {"threshold": 0.5},
}


def _stream():
    rng = np.random.RandomState(SEED)
    num_instances = NUM_FIT + NUM_NEXT
    # Alternating labels keep both classes in every window, which AUROCMetric needs. The scores of
    # the two classes overlap, so that the metrics are not constant.
    y_true = np.arange(num_instances) % 2
    y_pred = 0.3 * y_true + 0.7 * rng.rand(num_instances)

    return y_true, y_pred


def _fitted_metric(metric, y_true, y_pred):
    for yt, yp in zip(y_true[:NUM_FIT], y_pred[:NUM_FIT], strict=True):
        metric.update(yt, yp)

    return metric


def _next_outputs(metric, y_true, y_pred):
    outputs = []
    for yt, yp in zip(y_true[NUM_FIT:], y_pred[NUM_FIT:], strict=True):
        metric.update(yt, yp)
        outputs.append(metric.get())

    return np.asarray(outputs, dtype=np.float64)


@pytest.mark.parametrize(
    "metric_name", [name for name in evaluation.__all__ if name not in PICKLE_SKIP_METRICS]
)
def test_pickle_round_trip(metric_name):
    y_true, y_pred = _stream()
    metric_cls = getattr(evaluation, metric_name)
    metric = _fitted_metric(metric_cls(**METRIC_KWARGS.get(metric_name, {})), y_true, y_pred)

    restored = pickle.loads(pickle.dumps(metric))

    assert type(restored) is type(metric)
    np.testing.assert_allclose(restored.get(), metric.get())
    actual = _next_outputs(restored, y_true, y_pred)
    expected = _next_outputs(metric, y_true, y_pred)
    # A constant metric would pass even if the state was lost.
    assert np.unique(expected).size > 1
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("metric_name", WINDOWED_METRICS)
def test_windowed_metric_pickle_round_trip(metric_name):
    y_true, y_pred = _stream()
    metric_cls = getattr(evaluation, metric_name)
    metric = evaluation.WindowedMetric(
        metric_cls, WINDOW_SIZE, **METRIC_KWARGS.get(metric_name, {})
    )
    # The initial segment ends in the middle of a window, and the remaining instances cross one
    # more window boundary, so the window counters and the window score meter are all in use.
    assert NUM_FIT % WINDOW_SIZE != 0
    assert (NUM_FIT // WINDOW_SIZE + 1) * WINDOW_SIZE < NUM_FIT + NUM_NEXT
    _fitted_metric(metric, y_true, y_pred)

    restored = pickle.loads(pickle.dumps(metric))

    assert type(restored) is type(metric)
    assert restored.metric_cls is metric_cls
    np.testing.assert_allclose(restored.get(), metric.get())
    actual = _next_outputs(restored, y_true, y_pred)
    expected = _next_outputs(metric, y_true, y_pred)
    # A constant metric would pass even if the state was lost.
    assert np.unique(expected).size > 1
    np.testing.assert_allclose(actual, expected)
