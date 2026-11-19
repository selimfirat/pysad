import pytest


@pytest.mark.parametrize(
    "y_true, y_pred",
    [
        ([0, 0, 1, 1, 0, 1], [0.1, 0.4, 0.8, 0.3, 0.9, 0.6]),
        ([0, 0, 0], [0.0, 0.5, 1.0]),
        ([1, 1, 1], [0.0, 0.5, 1.0]),
        ([0, 1, 0, 1], [0.0, 1.0, 0.0, 1.0]),
        ([0, 1, 0, 1], [1.0, 0.0, 1.0, 0.0]),
    ],
)
def test_brier_score_matches_sklearn_at_each_prefix(y_true, y_pred):
    from sklearn.metrics import brier_score_loss

    from pysad.evaluation import BrierScoreMetric

    metric = BrierScoreMetric()
    for end, (label, probability) in enumerate(zip(y_true, y_pred, strict=True), start=1):
        metric.update(label, probability)
        expected = brier_score_loss(y_true[:end], y_pred[:end], pos_label=1)
        assert metric.get() == pytest.approx(expected)
        assert isinstance(metric.get(), float)


def test_brier_score_empty():
    from pysad.evaluation import BrierScoreMetric

    metric = BrierScoreMetric()
    assert metric.get() == 0.0
    assert metric.update_batch([], []) is metric
    assert metric.get() == 0.0


@pytest.mark.parametrize("as_array", [False, True])
def test_brier_score_batch_continues_stream(as_array):
    import numpy as np

    from pysad.evaluation import BrierScoreMetric

    metric = BrierScoreMetric()
    metric.update(1, 0.5)
    labels, probabilities = [0, 1, 0], [0.0, 1.0, 0.5]
    if as_array:
        labels = np.asarray(labels, dtype=np.uint8)
        probabilities = np.asarray(probabilities)
    else:
        labels, probabilities = iter(labels), iter(probabilities)
    assert metric.update_batch(labels, probabilities) is metric
    assert metric.get() == pytest.approx(0.125)


@pytest.mark.parametrize(
    "label, probability",
    [
        (-1, 0.5),
        (2, 0.5),
        (0.5, 0.5),
        (float("nan"), 0.5),
        (float("inf"), 0.5),
        (0, -0.01),
        (1, 1.01),
        (0, float("nan")),
        (1, float("inf")),
        (0, -float("inf")),
    ],
)
def test_brier_score_rejects_invalid_inputs_without_changing_score(label, probability):
    from pysad.evaluation import BrierScoreMetric

    metric = BrierScoreMetric()
    metric.update(1, 0.5)
    with pytest.raises(ValueError):
        metric.update(label, probability)
    assert metric.get() == 0.25
    metric.update(0, 0.0)
    assert metric.get() == 0.125
