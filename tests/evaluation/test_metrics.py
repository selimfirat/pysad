def helper_test_all_metrics(metric_classes, y_true, y_pred):
    import numpy as np

    for metric_cls, val in metric_classes.items():
        metric = metric_cls()

        for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
            metric.update(yt, yp)
            if i > 0:
                assert np.isclose(metric.get(), val)


def test_all_correct():
    from pysad.evaluation import PrecisionMetric, AUPRMetric, AUROCMetric, RecallMetric
    import numpy as np
    from pysad.utils import fix_seed
    fix_seed(61)

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

    helper_test_all_metrics(metric_classes, y_true, y_pred)


def test_none_correct():
    from pysad.evaluation import PrecisionMetric, AUPRMetric, AUROCMetric, RecallMetric
    import numpy as np
    from pysad.utils import fix_seed
    fix_seed(61)

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

    helper_test_all_metrics(metric_classes, y_true, y_pred)


def test_base_sklearn_metric():
    """Test BaseSKLearnMetric functionality."""
    from pysad.evaluation import BaseSKLearnMetric
    from sklearn.metrics import accuracy_score
    
    class TestSKLearnMetric(BaseSKLearnMetric):
        def _evaluate(self, y_true, y_pred):
            return accuracy_score(y_true, y_pred)
    
    metric = TestSKLearnMetric()
    
    # Test with data
    metric.update(1, 1)  # Correct
    metric.update(0, 0)  # Correct  
    metric.update(1, 0)  # Incorrect
    
    # Should have 2/3 accuracy
    accuracy = metric.get()
    assert abs(accuracy - 2.0/3.0) < 1e-10


def test_metric_error_handling():
    """Test metric error handling with invalid data."""
    from pysad.evaluation import AUROCMetric
    
    metric = AUROCMetric()
    
    # Test with both classes present
    metric.update(1, 0.5)
    metric.update(0, 0.7)
    metric.update(1, 0.3)
    
    # AUROC should work with both classes
    score = metric.get()
    assert isinstance(score, (int, float))
    assert 0.0 <= score <= 1.0


def test_precision_metric_edge_cases():
    """Test PrecisionMetric with edge cases."""
    from pysad.evaluation import PrecisionMetric
    import warnings
    
    metric = PrecisionMetric()
    
    # Test with no positive predictions (binary predictions)
    metric.update(0, 0)  # True negative
    metric.update(1, 0)  # False negative
    metric.update(0, 0)  # True negative
    
    # Precision should handle division by zero gracefully
    # Suppress the sklearn warning since we're testing edge cases
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Precision is ill-defined and being set to 0.0")
        precision = metric.get()
        assert precision == 0.0


def test_precision_metric_normal_case():
    """Test PrecisionMetric with normal binary predictions."""
    from pysad.evaluation import PrecisionMetric
    
    metric = PrecisionMetric()
    
    # Test with mixed predictions
    metric.update(1, 1)  # True positive
    metric.update(0, 1)  # False positive
    metric.update(1, 1)  # True positive
    
    # Precision should be 2/3
    precision = metric.get()
    assert abs(precision - 2.0/3.0) < 1e-10


def test_recall_metric_edge_cases():
    """Test RecallMetric with edge cases."""
    from pysad.evaluation import RecallMetric
    import warnings
    
    metric = RecallMetric()
    
    # Test with no positive labels (binary predictions)
    metric.update(0, 1)  # False positive
    metric.update(0, 0)  # True negative
    metric.update(0, 1)  # False positive
    
    # Recall should handle division by zero gracefully 
    # Suppress the sklearn warning since we're testing edge cases
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Recall is ill-defined and being set to 0.0")
        recall = metric.get()
        assert recall == 0.0


def test_recall_metric_normal_case():
    """Test RecallMetric with normal binary predictions."""
    from pysad.evaluation import RecallMetric
    
    metric = RecallMetric()
    
    # Test with mixed predictions
    metric.update(1, 1)  # True positive
    metric.update(1, 0)  # False negative
    metric.update(1, 1)  # True positive
    
    # Recall should be 2/3
    recall = metric.get()
    assert abs(recall - 2.0/3.0) < 1e-10


def test_aupr_metric_comprehensive():
    """Test AUPRMetric with various scenarios."""
    from pysad.evaluation import AUPRMetric
    
    metric = AUPRMetric()
    
    # Test perfect prediction
    metric.update(1, 0.9)
    metric.update(0, 0.1)
    metric.update(1, 0.8)
    metric.update(0, 0.2)
    
    aupr = metric.get()
    assert 0.0 <= aupr <= 1.0


def test_auroc_metric_comprehensive():
    """Test AUROCMetric with various scenarios."""
    from pysad.evaluation import AUROCMetric
    import numpy as np
    
    metric = AUROCMetric()
    
    # Test random prediction (should be around 0.5)
    np.random.seed(42)
    for _ in range(20):
        y_true = np.random.randint(0, 2)
        y_pred = np.random.random()
        metric.update(y_true, y_pred)
    
    auroc = metric.get()
    assert 0.0 <= auroc <= 1.0


def test_auroc_single_class_error():
    """Test AUROCMetric with single class throws expected error."""
    from pysad.evaluation import AUROCMetric
    import pytest
    
    metric = AUROCMetric()
    
    # Test with all same labels (should throw ValueError)
    metric.update(1, 0.5)
    metric.update(1, 0.7)
    metric.update(1, 0.3)
    
    # AUROC should throw ValueError with single class
    with pytest.raises(ValueError, match="Only one class present"):
        metric.get()
