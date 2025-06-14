def helper_test_all_metrics(metric_classes, y_true, y_pred, ignore_nonempty_last):
    import numpy as np
    from pysad.evaluation import WindowedMetric

    for metric_cls, val in metric_classes.items():
        metric = WindowedMetric(metric_cls, 25, ignore_nonempty_last)

        for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
            metric.update(yt, yp)
            if i > 0:
                assert np.isclose(metric.get(), val)


def test_all_correct():
    from pysad.evaluation import PrecisionMetric, AUPRMetric, AUROCMetric, RecallMetric
    import numpy as np
    metric_classes = [
        PrecisionMetric,
        RecallMetric,
        AUPRMetric,
        AUROCMetric
    ]
    metric_classes = { metric_cls: 1.0 for metric_cls in metric_classes }
    y_true = np.random.randint(0, 2, size=(25,), dtype=np.int32)
    y_true[0] = 1
    y_true[1] = 0
    y_pred = y_true.copy()

    for ignore_nonempty_last in [True, False]:
        helper_test_all_metrics(metric_classes, y_true, y_pred, ignore_nonempty_last)


def test_none_correct():
    from pysad.evaluation import PrecisionMetric, AUPRMetric, AUROCMetric, RecallMetric
    import numpy as np

    metric_classes = {
        PrecisionMetric: 0.0,
        #AUPRMetric: 0.5
        AUROCMetric: 0.0,
        RecallMetric: 0.0
    }
    y_true = np.random.randint(0, 2, size=(25,), dtype=np.int32)
    y_true[0] = 1
    y_true[1] = 0
    y_pred = 1 - y_true.copy()

    for ignore_nonempty_last in [True, False]:
        helper_test_all_metrics(metric_classes, y_true, y_pred, ignore_nonempty_last)


def test_windowed_metric_window_size():
    """Test WindowedMetric with different window sizes."""
    from pysad.evaluation import WindowedMetric, AUROCMetric
    
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
    from pysad.evaluation import WindowedMetric, AUROCMetric  # Use AUROC which accepts continuous scores
    
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
    from pysad.evaluation import WindowedMetric, RecallMetric
    import warnings
    
    metric = WindowedMetric(RecallMetric, window_size=10)
    
    # Get score before any updates
    # Suppress the sklearn warning since we're testing edge cases  
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Recall is ill-defined and being set to 0.0")
        score = metric.get()
        assert score == 0.0


def test_windowed_metric_various_metrics():
    """Test WindowedMetric with different base metrics."""
    from pysad.evaluation import WindowedMetric, AUPRMetric, AUROCMetric, PrecisionMetric, RecallMetric
    import numpy as np
    
    # Separate metrics by data type they expect
    continuous_metrics = [
        WindowedMetric(AUPRMetric, window_size=10),
        WindowedMetric(AUROCMetric, window_size=10)
    ]
    
    binary_metrics = [
        WindowedMetric(PrecisionMetric, window_size=10),
        WindowedMetric(RecallMetric, window_size=10)
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
