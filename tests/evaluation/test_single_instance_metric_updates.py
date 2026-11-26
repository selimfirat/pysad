import math

import pytest


def _single_instance_scores():
    import pysad.evaluation as evaluation

    return [
        ("PrecisionMetric", evaluation.PrecisionMetric(threshold=0.5), 1.0),
        ("RecallMetric", evaluation.RecallMetric(threshold=0.5), 1.0),
        ("F1Metric", evaluation.F1Metric(threshold=0.5), 1.0),
        ("BalancedAccuracyMetric", evaluation.BalancedAccuracyMetric(threshold=0.5), 1.0),
        ("MatthewsCorrelationMetric", evaluation.MatthewsCorrelationMetric(threshold=0.5), 0.0),
        ("BrierScoreMetric", evaluation.BrierScoreMetric(), 0.04),
        ("ExpectedCalibrationErrorMetric", evaluation.ExpectedCalibrationErrorMetric(), 0.2),
        ("LogLossMetric", evaluation.LogLossMetric(), -math.log(0.8)),
        ("AUPRMetric", evaluation.AUPRMetric(), 1.0),
    ]


@pytest.mark.parametrize(("name", "metric", "expected"), _single_instance_scores())
def test_single_instance_get_returns_value(name, metric, expected):
    """One valid update produces a score instead of raising or staying empty."""
    metric.update(1, 0.8)
    assert metric.get() == pytest.approx(expected)
    assert isinstance(metric.get(), float)


def test_auroc_single_instance_raises():
    """AUROC needs both classes, so a single instance raises on get()."""
    import pysad.evaluation as evaluation

    metric = evaluation.AUROCMetric()
    metric.update(1, 0.8)
    with pytest.raises(ValueError, match="Only one class present"):
        metric.get()


@pytest.mark.parametrize(("name", "metric", "_expected"), _single_instance_scores())
def test_single_instance_update_returns_none_except_logloss(name, metric, _expected):
    """Single-instance updates return None for all metrics except LogLossMetric."""
    returned = metric.update(1, 0.8)
    if name == "LogLossMetric":
        assert returned is metric
    else:
        assert returned is None


def _validating_metrics():
    import pysad.evaluation as evaluation

    return [
        ("PrecisionMetric", evaluation.PrecisionMetric(threshold=0.5)),
        ("RecallMetric", evaluation.RecallMetric(threshold=0.5)),
        ("F1Metric", evaluation.F1Metric(threshold=0.5)),
        ("BalancedAccuracyMetric", evaluation.BalancedAccuracyMetric(threshold=0.5)),
        ("MatthewsCorrelationMetric", evaluation.MatthewsCorrelationMetric(threshold=0.5)),
        ("BrierScoreMetric", evaluation.BrierScoreMetric()),
        ("ExpectedCalibrationErrorMetric", evaluation.ExpectedCalibrationErrorMetric()),
        ("LogLossMetric", evaluation.LogLossMetric()),
    ]


@pytest.mark.parametrize(("name", "metric"), _validating_metrics())
def test_single_instance_invalid_input_keeps_score(name, metric):
    """An invalid single-instance update raises and leaves the score unchanged."""
    metric.update(1, 0.8)
    before = metric.get()

    with pytest.raises(ValueError):
        metric.update(2, 0.5)
    assert metric.get() == pytest.approx(before)

    if name == "LogLossMetric":
        with pytest.raises(ValueError, match="probability between 0 and 1"):
            metric.update(1, 1.5)
        assert metric.get() == pytest.approx(before)


def test_threshold_boundary_and_nan_single_instance():
    """A score exactly at the threshold counts as positive; NaN counts as negative without error."""
    import pysad.evaluation as evaluation

    metric = evaluation.PrecisionMetric(threshold=0.5)
    metric.update(1, 0.5)
    assert metric.get() == pytest.approx(1.0)

    metric.update(0, 0.5)
    assert metric.get() == pytest.approx(0.5)

    metric.update(1, float("nan"))
    # NaN is below the threshold, so it counts as a negative prediction.
    tp, fp = 1, 1
    assert metric.get() == pytest.approx(tp / (tp + fp))

    metric.update(1, float("-inf"))
    assert metric.get() == pytest.approx(0.5)

    # Without a threshold, NaN and non-binary scores are rejected immediately.
    plain = evaluation.PrecisionMetric()
    with pytest.raises(ValueError, match="y_pred must be 0 or 1"):
        plain.update(1, float("nan"))
