import pytest


@pytest.mark.parametrize("ignore_nonempty_last", [True, False])
@pytest.mark.parametrize("chunks", [[1] * 10, [10], [6, 4], [4, 4, 2], [1, 5, 4]])
def test_window_scores_across_update_boundaries(ignore_nonempty_last, chunks):
    """Average separate windows and optionally include the final partial window."""
    from pysad.evaluation import PrecisionMetric, WindowedMetric

    labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    scores = [0.9, 0.8, 0.7, 0.2, 0.6, 0.2, 0.3, 0.4, 0.9, 0.8]
    expected = [None, 0.5, 2 / 3, 2 / 3, 2 / 3, 2 / 3, 2 / 3, 5 / 6, 5 / 6, 5 / 6]
    if not ignore_nonempty_last:
        expected[5:7] = [5 / 6, 5 / 6]
        expected[-1] = 13 / 18

    metric = WindowedMetric(
        metric_cls=PrecisionMetric,
        window_size=4,
        ignore_nonempty_last=ignore_nonempty_last,
        threshold=0.5,
    )
    assert metric.get() is None
    assert metric.update_batch([], []) is metric
    assert metric.get() is None
    start = 0
    for size in chunks:
        end = start + size
        if size == 1:
            assert metric.update(labels[start], scores[start]) is metric
        else:
            assert metric.update_batch(labels[start:end], scores[start:end]) is metric
        assert metric.get() == pytest.approx(expected[end - 1])
        start = end
    assert metric.update_batch([], []) is metric
    assert metric.get() == pytest.approx(expected[-1])


@pytest.mark.parametrize(
    "metric_name", ["AUPRMetric", "AUROCMetric", "PrecisionMetric", "RecallMetric"]
)
@pytest.mark.parametrize("correct", [True, False])
def test_windowed_metrics_with_binary_predictions(metric_name, correct):
    import pysad.evaluation as evaluation

    labels = [1, 0, 1, 0] * 2
    predictions = labels if correct else [1 - label for label in labels]
    metric = evaluation.WindowedMetric(getattr(evaluation, metric_name), window_size=4)
    metric.update_batch(labels, predictions)
    expected = 1.0 if correct else 0.5 if metric_name == "AUPRMetric" else 0.0
    assert metric.get() == pytest.approx(expected)


@pytest.mark.parametrize("ignore_nonempty_last", [True, False])
def test_single_class_windows_do_not_contribute(ignore_nonempty_last):
    from pysad.evaluation import AUROCMetric, WindowedMetric

    metric = WindowedMetric(
        metric_cls=AUROCMetric,
        window_size=4,
        ignore_nonempty_last=ignore_nonempty_last,
    )
    metric.update_batch([0, 0, 0, 0], [0.1, 0.2, 0.3, 0.4])
    assert metric.get() is None
    metric.update_batch([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2])
    assert metric.get() == 1.0
    metric.update(1, 0.7)
    assert metric.get() == 1.0


def test_windowed_metric_averages_only_valid_windows():
    """Single-class windows are skipped; the average covers valid windows only."""
    from pysad.evaluation import AUROCMetric, WindowedMetric

    metric = WindowedMetric(
        metric_cls=AUROCMetric,
        window_size=2,
    )
    # Window 1: both classes, AUROC = 1.0.
    metric.update_batch([1, 0], [0.9, 0.1])
    # Window 2: both classes, AUROC = 0.0.
    metric.update_batch([1, 0], [0.1, 0.9])
    # Window 3: single class only, skipped.
    metric.update_batch([0, 0], [0.3, 0.4])
    # Window 4: both classes, AUROC = 1.0.
    metric.update_batch([1, 0], [0.8, 0.2])

    assert metric.get() == pytest.approx(2 / 3)


def test_windowed_metric_log_loss():
    """Test WindowedMetric with LogLossMetric across consecutive windows."""
    import numpy as np

    from pysad.evaluation import LogLossMetric, WindowedMetric

    # 2 windows of size 2, ignore_nonempty_last=False
    metric = WindowedMetric(LogLossMetric, window_size=2, ignore_nonempty_last=False)

    y_true = [1, 0, 0, 1]
    y_pred = [0.8, 0.2, 0.3, 0.7]

    # Window 1: (1, 0.8), (0, 0.2) -> losses: -ln(0.8), -ln(0.8) -> mean = -ln(0.8)
    # Window 2: (0, 0.3), (1, 0.7) -> losses: -ln(0.7), -ln(0.7) -> mean = -ln(0.7)
    # Average across windows: (-ln(0.8) + -ln(0.7)) / 2
    for yt, yp in zip(y_true, y_pred, strict=True):
        metric.update(yt, yp)

    expected = (-np.log(0.8) - np.log(0.7)) / 2.0
    assert np.isclose(metric.get(), expected)

    # Batch update
    batch_metric = WindowedMetric(LogLossMetric, window_size=2, ignore_nonempty_last=False)
    batch_metric.update_batch(y_true, y_pred)
    assert np.isclose(batch_metric.get(), expected)
