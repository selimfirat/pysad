import warnings

import pytest


@pytest.mark.parametrize(
    "y_true, y_pred, threshold",
    [
        ([0, 0, 0, 1, 1, 0, 1], [0, 1, 0, 1, 0, 0, 1], None),
        ([1, 1, 1, 0, 0, 1, 0], [0, 1, 0, 1, 0, 0, 1], None),
        ([0, 0, 0], [0, 1, 0], None),
        ([1, 1, 1], [1, 0, 1], None),
        ([0, 0], [1, 1], None),
        ([1, 1], [0, 0], None),
        ([0, 1, 0, 1, 0], [0.1, 0.5, 0.5, 0.2, 0.9], 0.5),
        ([1, 0, 1, 0], [0.0, -1.0, -0.1, 1.0], 0.0),
    ],
)
def test_balanced_accuracy_matches_sklearn_at_each_prefix(y_true, y_pred, threshold):
    from sklearn.metrics import balanced_accuracy_score

    from pysad.evaluation import BalancedAccuracyMetric

    metric = BalancedAccuracyMetric(threshold=threshold)
    predictions = y_pred if threshold is None else [int(p >= threshold) for p in y_pred]
    for end, (label, prediction) in enumerate(zip(y_true, y_pred, strict=True), start=1):
        metric.update(label, prediction)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            expected = balanced_accuracy_score(y_true[:end], predictions[:end])
        assert metric.get() == pytest.approx(expected)


@pytest.mark.parametrize("threshold", [None, 0.5])
def test_balanced_accuracy_empty(threshold):
    from pysad.evaluation import BalancedAccuracyMetric

    assert BalancedAccuracyMetric(threshold=threshold).get() == 0.0


@pytest.mark.parametrize(
    "label, prediction, threshold",
    [(2, 1, None), (-1, 0, 0.5), (0, 0.2, None), (1, 2, None)],
)
def test_balanced_accuracy_rejects_nonbinary_labels(label, prediction, threshold):
    from pysad.evaluation import BalancedAccuracyMetric

    metric = BalancedAccuracyMetric(threshold=threshold)
    metric.update(0, 0)
    with pytest.raises(ValueError):
        metric.update(label, prediction)
    assert metric.get() == 1.0
