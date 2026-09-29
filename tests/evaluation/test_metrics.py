def helper_test_all_metrics(metric_classes, y_true, y_pred):
    import numpy as np

    for metric_cls, val in metric_classes.items():
        metric = metric_cls()

        for i, (yt, yp) in enumerate(zip(y_true, y_pred, strict=True)):
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


def test_precision_recall_threshold_on_scores():
    """Test PrecisionMetric and RecallMetric turn scores at or above the threshold into anomalies."""
    from pysad.evaluation import PrecisionMetric, RecallMetric
    import numpy as np

    y_true = [1, 0, 0, 1, 1, 0]
    scores = [0.9, 0.2, 0.5, 0.4, 0.8, 0.6]
    # threshold=0.5 predicts [1, 0, 1, 0, 1, 1]: 2 true positives, 2 false positives, 1 false negative.
    # The 0.5 score sits on the threshold and counts as an anomaly.
    precision = PrecisionMetric(threshold=0.5)
    recall = RecallMetric(threshold=0.5)
    for yt, score in zip(y_true, scores, strict=True):
        precision.update(yt, score)
        recall.update(yt, score)

    assert np.isclose(precision.get(), 2.0 / 4.0)
    assert np.isclose(recall.get(), 2.0 / 3.0)


def test_precision_recall_threshold_on_model_scores():
    """Test PrecisionMetric and RecallMetric with a threshold on scores from a streaming model."""
    from pysad.evaluation import PrecisionMetric, RecallMetric
    from pysad.models import StandardAbsoluteDeviation
    from pysad.utils import fix_seed
    from sklearn.metrics import precision_score, recall_score
    import numpy as np
    fix_seed(61)

    X = np.random.normal(size=(200, 1))
    y_true = np.zeros(200, dtype=np.int32)
    y_true[50::25] = 1
    X[y_true == 1] += 6.0

    model = StandardAbsoluteDeviation()
    precision = PrecisionMetric(threshold=3.0)
    recall = RecallMetric(threshold=3.0)
    scores = []
    for x, yt in zip(X, y_true, strict=True):
        score = model.fit_score_partial(x)
        scores.append(score)
        precision.update(yt, score)
        recall.update(yt, score)

    y_pred = (np.asarray(scores) >= 3.0).astype(int)
    assert np.isclose(precision.get(), precision_score(y_true, y_pred))
    assert np.isclose(recall.get(), recall_score(y_true, y_pred))
    assert recall.get() > 0.0


def test_precision_recall_without_threshold_on_binary_predictions():
    """Test PrecisionMetric and RecallMetric keep their 0/1 behaviour when threshold is None."""
    from pysad.evaluation import PrecisionMetric, RecallMetric
    from sklearn.metrics import precision_score, recall_score
    import numpy as np

    y_true = [1, 0, 1, 1, 0, 0, 1]
    y_pred = [1, 1, 0, 1, 0, 1, 1]

    for metric_cls, sklearn_metric in [(PrecisionMetric, precision_score), (RecallMetric, recall_score)]:
        for metric in [metric_cls(), metric_cls(threshold=None)]:
            for yt, yp in zip(y_true, y_pred, strict=True):
                metric.update(yt, yp)
            assert np.isclose(metric.get(), sklearn_metric(y_true, y_pred))


def test_precision_recall_without_threshold_reject_scores():
    """Test PrecisionMetric and RecallMetric do not pick a threshold for scores on their own."""
    from pysad.evaluation import PrecisionMetric, RecallMetric
    import pytest

    for metric_cls in [PrecisionMetric, RecallMetric]:
        metric = metric_cls()
        metric.update(1, 0.9)
        metric.update(0, 0.2)
        with pytest.raises(ValueError):
            metric.get()
