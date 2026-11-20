def helper_test_all_metrics(metric_classes, y_true, y_pred, ignore_nonempty_last):
    import numpy as np

    from pysad.evaluation import WindowedMetric

    for metric_cls, val in metric_classes.items():
        metric = WindowedMetric(metric_cls, 25, ignore_nonempty_last)

        for i, (yt, yp) in enumerate(zip(y_true, y_pred, strict=True)):
            metric.update(yt, yp)
            if i > 0:
                assert np.isclose(metric.get(), val)


def test_all_correct():
    import numpy as np

    from pysad.evaluation import AUPRMetric, AUROCMetric, PrecisionMetric, RecallMetric

    metric_classes = [PrecisionMetric, RecallMetric, AUPRMetric, AUROCMetric]
    metric_classes = {metric_cls: 1.0 for metric_cls in metric_classes}
    y_true = np.random.randint(0, 2, size=(25,), dtype=np.int32)
    y_true[0] = 1
    y_true[1] = 0
    y_pred = y_true.copy()

    for ignore_nonempty_last in [True, False]:
        helper_test_all_metrics(metric_classes, y_true, y_pred, ignore_nonempty_last)


def test_none_correct():
    import numpy as np

    from pysad.evaluation import AUROCMetric, PrecisionMetric, RecallMetric

    metric_classes = {
        PrecisionMetric: 0.0,
        # AUPRMetric: 0.5
        AUROCMetric: 0.0,
        RecallMetric: 0.0,
    }
    y_true = np.random.randint(0, 2, size=(25,), dtype=np.int32)
    y_true[0] = 1
    y_true[1] = 0
    y_pred = 1 - y_true.copy()

    for ignore_nonempty_last in [True, False]:
        helper_test_all_metrics(metric_classes, y_true, y_pred, ignore_nonempty_last)


def test_windowed_metric_window_size():
    """Test WindowedMetric with different window sizes."""
    from pysad.evaluation import AUROCMetric, WindowedMetric

    # Test small window
    metric = WindowedMetric(AUROCMetric, window_size=3)

    # Add data beyond window size
    for i in range(5):
        y_true = i % 2
        y_pred = 0.5 + (i % 2) * 0.3
        metric.update(y_true, y_pred)

    score = metric.get()
    assert isinstance(score, (int, float))


def test_windowed_metric_ignore_nonempty_last():
    """Test WindowedMetric ignore_nonempty_last parameter."""
    from pysad.evaluation import (  # Use AUROC which accepts continuous scores
        AUROCMetric,
        WindowedMetric,
    )

    # Test with ignore_nonempty_last=True
    metric_ignore = WindowedMetric(AUROCMetric, window_size=5, ignore_nonempty_last=True)

    # Test with ignore_nonempty_last=False
    metric_no_ignore = WindowedMetric(AUROCMetric, window_size=5, ignore_nonempty_last=False)

    # Add same data to both with mixed classes and continuous scores
    test_data = [(1, 0.8), (0, 0.2), (1, 0.9)]
    for y_true, y_pred in test_data:
        metric_ignore.update(y_true, y_pred)
        metric_no_ignore.update(y_true, y_pred)

    # Both should work but may have different behavior
    score_ignore = metric_ignore.get()
    score_no_ignore = metric_no_ignore.get()

    assert isinstance(score_ignore, (int, float))
    assert isinstance(score_no_ignore, (int, float))


def test_windowed_metric_empty_window():
    """Test WindowedMetric behavior with empty window."""
    import pytest

    from pysad.evaluation import (
        AUPRMetric,
        AUROCMetric,
        PrecisionMetric,
        RecallMetric,
        WindowedMetric,
    )

    for metric_cls in (PrecisionMetric, RecallMetric):
        assert metric_cls().get() == 0.0
        assert WindowedMetric(metric_cls, window_size=10).get() == 0.0

    for metric_cls in (AUPRMetric, AUROCMetric):
        with pytest.raises(ValueError):
            metric_cls().get()
        with pytest.raises(ValueError):
            WindowedMetric(metric_cls, window_size=10).get()


def test_windowed_metric_various_metrics():
    """Test WindowedMetric with different base metrics."""
    import numpy as np

    from pysad.evaluation import (
        AUPRMetric,
        AUROCMetric,
        PrecisionMetric,
        RecallMetric,
        WindowedMetric,
    )

    # Separate metrics by data type they expect
    continuous_metrics = [
        WindowedMetric(AUPRMetric, window_size=10),
        WindowedMetric(AUROCMetric, window_size=10),
    ]

    binary_metrics = [
        WindowedMetric(PrecisionMetric, window_size=10),
        WindowedMetric(RecallMetric, window_size=10),
    ]

    # Add diverse data with continuous scores
    np.random.seed(42)
    for _ in range(15):
        y_true = np.random.randint(0, 2)
        y_pred_continuous = np.random.random()
        y_pred_binary = np.random.randint(0, 2)

        # Test continuous metrics with continuous predictions
        for metric in continuous_metrics:
            metric.update(y_true, y_pred_continuous)

        # Test binary metrics with binary predictions
        for metric in binary_metrics:
            metric.update(y_true, y_pred_binary)

    # All metrics should return valid scores
    for metric in continuous_metrics + binary_metrics:
        score = metric.get()
        assert isinstance(score, (int, float))
        assert 0.0 <= score <= 1.0


def test_windowed_metric_passes_kwargs_to_every_window():
    """Test WindowedMetric passes keyword arguments to the metric of every window, not only the first."""
    import numpy as np

    from pysad.evaluation import PrecisionMetric, WindowedMetric

    y_true = [1, 0, 1, 0, 1, 0, 1, 0]
    scores = [0.9, 0.1, 0.8, 0.7, 0.6, 0.2, 0.3, 0.4]
    metric = WindowedMetric(PrecisionMetric, window_size=4, threshold=0.5)
    for yt, score in zip(y_true, scores, strict=True):
        metric.update(yt, score)

    # First window predicts [1, 0, 1, 1] (precision 2/3), second predicts [1, 0, 0, 0] (precision 1).
    assert metric.metric.threshold == 0.5
    assert np.isclose(metric.get(), (2.0 / 3.0 + 1.0) / 2.0)


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


def test_windowed_metric_update_batch_across_windows_precision():
    """Test WindowedMetric.update_batch across several windows with PrecisionMetric.

    Covers partially filled starts, exact boundaries, independent window results,
    and final state consistency between batch and incremental updates.
    """
    import numpy as np

    from pysad.evaluation import PrecisionMetric, WindowedMetric

    # Window size 4. We feed 14 items total -> 3 full windows (12 items) + 2 partial items.
    # y_true pattern: [1, 0, 1, 0] repeated, last two are [1, 0]
    # y_pred pattern: [1, 1, 1, 0] repeated, last two are [1, 0]
    # With threshold=0.5, predictions >= 0.5 are 1.
    window_size = 4
    y_true = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    y_pred = [0.9, 0.8, 0.7, 0.2, 0.9, 0.8, 0.7, 0.2, 0.9, 0.8, 0.7, 0.2, 0.9, 0.2]

    # Manual window calculation with PrecisionMetric (threshold=0.5):
    # Window 0 (indices 0-3): TP=2 (indices 0,2), FP=1 (index 1), precision = 2/3
    # Window 1 (indices 4-7): TP=2 (indices 4,6), FP=1 (index 5), precision = 2/3
    # Window 2 (indices 8-11): TP=2 (indices 8,10), FP=1 (index 9), precision = 2/3
    # Partial window 3 (indices 12-13): TP=1 (index 12), FP=0 (index 13), precision = 1.0

    # Test 1: Batch update all at once
    metric_batch = WindowedMetric(
        PrecisionMetric, window_size=window_size, ignore_nonempty_last=True, threshold=0.5
    )
    metric_batch.update_batch(y_true, y_pred)

    # With ignore_nonempty_last=True, only full windows count (3 windows of 2/3)
    assert np.isclose(metric_batch.get(), 2.0 / 3.0)
    assert metric_batch.step == 14
    assert metric_batch.num_windows == 4  # 3 completed + 1 current (partial)

    # Test 2: Batch update in two chunks (partially filled start in second chunk)
    metric_chunked = WindowedMetric(
        PrecisionMetric, window_size=window_size, ignore_nonempty_last=True, threshold=0.5
    )
    metric_chunked.update_batch(y_true[:6], y_pred[:6])  # 1.5 windows
    metric_chunked.update_batch(y_true[6:], y_pred[6:])  # 2.5 windows

    assert np.isclose(metric_chunked.get(), 2.0 / 3.0)
    assert metric_chunked.step == 14
    assert metric_chunked.num_windows == 4

    # Test 3: Batch update matching exact window boundaries
    metric_boundary = WindowedMetric(
        PrecisionMetric, window_size=window_size, ignore_nonempty_last=True, threshold=0.5
    )
    metric_boundary.update_batch(y_true[:4], y_pred[:4])  # exactly 1 window
    metric_boundary.update_batch(y_true[4:8], y_pred[4:8])  # exactly 1 window
    metric_boundary.update_batch(y_true[8:12], y_pred[8:12])  # exactly 1 window
    metric_boundary.update_batch(y_true[12:], y_pred[12:])  # partial window

    assert np.isclose(metric_boundary.get(), 2.0 / 3.0)
    assert metric_boundary.step == 14
    assert metric_boundary.num_windows == 4

    # Test 4: Compare batch vs incremental (update) for exact equivalence
    metric_incremental = WindowedMetric(
        PrecisionMetric, window_size=window_size, ignore_nonempty_last=True, threshold=0.5
    )
    for yt, yp in zip(y_true, y_pred, strict=True):
        metric_incremental.update(yt, yp)

    assert np.isclose(metric_batch.get(), metric_incremental.get())
    assert metric_batch.step == metric_incremental.step
    assert metric_batch.num_windows == metric_incremental.num_windows
    assert np.isclose(metric_batch.score_meter.get(), metric_incremental.score_meter.get())

    # Test 5: With ignore_nonempty_last=False, partial window counts
    metric_include_partial = WindowedMetric(
        PrecisionMetric, window_size=window_size, ignore_nonempty_last=False, threshold=0.5
    )
    metric_include_partial.update_batch(y_true, y_pred)

    # Average of 3 full windows (2/3 each) + 1 partial window (1.0) = (2.0 + 1.0) / 4 = 0.75
    expected_with_partial = (3 * (2.0 / 3.0) + 1.0) / 4.0
    assert np.isclose(metric_include_partial.get(), expected_with_partial)
    assert metric_include_partial.step == 14
    assert metric_include_partial.num_windows == 4

    # Test 6: Verify independent window results (each window's metric state is isolated)
    # After 1st batch (6 items = 1.5 windows), score_meter should have 1 full window score
    metric_check = WindowedMetric(
        PrecisionMetric, window_size=window_size, ignore_nonempty_last=True, threshold=0.5
    )
    metric_check.update_batch(y_true[:6], y_pred[:6])
    # First window precision = 2/3
    assert np.isclose(metric_check.score_meter.get(), 2.0 / 3.0)
    # Current (partial) window: indices 4,5 -> TP=1 (index 4), FP=1 (index 5), precision = 1/2
    assert np.isclose(metric_check.metric.get(), 0.5)
    assert metric_check.num_windows == 2

    # After 2nd batch (remaining 8 items = 2.5 windows), 2 more full windows complete
    metric_check.update_batch(y_true[6:], y_pred[6:])
    # score_meter should average 3 full windows (all 2/3)
    assert np.isclose(metric_check.score_meter.get(), 2.0 / 3.0)
    # Current partial window: indices 12,13 -> TP=1, FP=0, precision = 1.0
    assert np.isclose(metric_check.metric.get(), 1.0)
    assert metric_check.num_windows == 4

    # Test 7: Empty batch should not change state
    metric_empty = WindowedMetric(
        PrecisionMetric, window_size=window_size, ignore_nonempty_last=True, threshold=0.5
    )
    metric_empty.update_batch([], [])
    assert metric_empty.step == 0
    assert metric_empty.num_windows == 1
    # score_meter has no completed windows yet, so num_items is 0
    assert metric_empty.score_meter.num_items == 0
    # get() would raise ZeroDivisionError, so we check num_items instead

    # Test 8: Single item batch (partial window)
    metric_single = WindowedMetric(
        PrecisionMetric, window_size=window_size, ignore_nonempty_last=True, threshold=0.5
    )
    metric_single.update_batch([1], [0.9])
    assert metric_single.step == 1
    assert metric_single.num_windows == 1
    assert np.isclose(metric_single.metric.get(), 1.0)  # TP=1, FP=0
    assert metric_single.get() == 1.0  # Only current window counts when no completed windows
