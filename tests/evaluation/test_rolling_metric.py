import pytest


@pytest.mark.parametrize(
    ("metric_name", "expected"),
    [
        ("PrecisionMetric", [1.0, 0.5, 0.5, 0.0, 1.0, 0.5]),
        ("RecallMetric", [1.0, 1.0, 1.0, 0.0, 0.5, 0.5]),
        ("AUROCMetric", [None, 0.0, 0.5, 0.5, 1.0, 0.5]),
        ("AUPRMetric", [1.0, 0.5, 0.5, 0.5, 1.0, 5.0 / 6.0]),
    ],
)
def test_recent_pairs(metric_name, expected):
    import pysad.evaluation as evaluation

    metric_cls = getattr(evaluation, metric_name)
    kwargs = {"threshold": 0.5} if metric_name in {"PrecisionMetric", "RecallMetric"} else {}
    metric = evaluation.RollingMetric(metric_cls, window_size=3, **kwargs)
    pairs = [(1, 0.8), (0, 0.9), (0, 0.1), (1, 0.2), (1, 0.7), (0, 0.6)]

    for pair, score in zip(pairs, expected, strict=True):
        assert metric.update(*pair) is metric
        if score is None:
            with pytest.raises(ValueError):
                metric.get()
        else:
            assert metric.get() == pytest.approx(score)
            assert metric.get() == pytest.approx(score)


@pytest.mark.parametrize(
    "metric_name", ["PrecisionMetric", "RecallMetric", "AUROCMetric", "AUPRMetric"]
)
def test_empty_window(metric_name):
    import pysad.evaluation as evaluation

    metric = evaluation.RollingMetric(getattr(evaluation, metric_name), window_size=3)
    assert metric.get() is None


def test_single_pair_window():
    from pysad.evaluation import RecallMetric, RollingMetric

    metric = RollingMetric(RecallMetric, window_size=1, threshold=0.5)
    metric.update(1, 0.9)
    assert metric.get() == 1.0
    metric.update(1, 0.1)
    assert metric.get() == 0.0


def test_eviction_can_leave_one_class():
    from pysad.evaluation import AUROCMetric, RollingMetric

    metric = RollingMetric(AUROCMetric, window_size=2)
    metric.update(0, 0.1).update(1, 0.9)
    assert metric.get() == 1.0
    metric.update(1, 0.8)
    with pytest.raises(ValueError):
        metric.get()
    metric.update(0, 0.2)
    assert metric.get() == 1.0
