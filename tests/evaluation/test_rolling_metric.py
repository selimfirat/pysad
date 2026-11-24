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
    "metric_name",
    ["PrecisionMetric", "RecallMetric", "AUROCMetric", "AUPRMetric", "BrierScoreMetric"],
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


def test_rolling_confusion_metrics_match_batch_and_recover_from_invalid_pair():
    import numpy as np

    from pysad.evaluation import (
        BalancedAccuracyMetric,
        F1Metric,
        MatthewsCorrelationMetric,
        PrecisionMetric,
        RecallMetric,
        RollingMetric,
    )

    rng = np.random.default_rng(0)
    yt = rng.integers(0, 2, size=60)
    yp = rng.random(size=60)

    for cls in (
        PrecisionMetric,
        RecallMetric,
        F1Metric,
        BalancedAccuracyMetric,
        MatthewsCorrelationMetric,
    ):
        rolling_stream = RollingMetric(cls, window_size=12, threshold=0.5)
        for t in range(len(yt)):
            rolling_stream.update(int(yt[t]), float(yp[t]))
            ref = cls(threshold=0.5)
            start = max(0, t + 1 - 12)
            ref.update_batch(yt[start : t + 1], yp[start : t + 1])
            assert rolling_stream.get() == pytest.approx(ref.get())

        rolling_batch = RollingMetric(cls, window_size=12, threshold=0.5)
        rolling_batch.update_batch(yt, yp)
        assert rolling_batch.get() == pytest.approx(rolling_stream.get())

    bad = RollingMetric(PrecisionMetric, window_size=2)
    bad.update(2, 1)
    with pytest.raises(ValueError, match="y_true"):
        bad.get()
    bad.update(1, 1).update(1, 0)
    assert bad.get() == pytest.approx(1.0)


@pytest.mark.parametrize("window_size", [1, 3, 8])
@pytest.mark.parametrize("batch_size", [1, 2, 3, 10])
def test_rolling_brier_matches_independent_windows(window_size, batch_size):
    from pysad.evaluation import BrierScoreMetric, RollingMetric

    pairs = [(1, 0.8), (0, 0.9), (0, 0.1), (1, 0.2), (1, 0.7), (0, 0.6)] * 4
    metric = RollingMetric(BrierScoreMetric, window_size=window_size)
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        if batch_size == 1:
            assert metric.update(*batch[0]) is metric
        else:
            assert metric.update_batch((y for y, _ in batch), (p for _, p in batch)) is metric
        end = start + len(batch)
        window = pairs[max(0, end - window_size) : end]
        expected = sum((p - y) ** 2 for y, p in window) / len(window)
        assert metric.get() == pytest.approx(expected)
        assert metric.get() == pytest.approx(expected)
        assert isinstance(metric.get(), float)

    assert metric.update_batch([], []) is metric
    assert metric.get() == pytest.approx(expected)
    with pytest.raises(ValueError):
        metric.update_batch([0, 1], [0.5])
    assert metric.get() == pytest.approx(expected)


@pytest.mark.parametrize(
    "pair",
    [
        (2, 0.5),
        (0.5, 0.5),
        (float("nan"), 0.5),
        (0, -0.1),
        (1, 1.1),
        (0, float("nan")),
        (1, float("inf")),
        (0, -float("inf")),
    ],
)
@pytest.mark.parametrize("as_batch", [False, True])
def test_rolling_brier_invalid_pair_expires(pair, as_batch):
    from pysad.evaluation import BrierScoreMetric, RollingMetric

    metric = RollingMetric(BrierScoreMetric, window_size=2)
    if as_batch:
        metric.update_batch([1, pair[0]], [0.5, pair[1]])
    else:
        metric.update(1, 0.5).update(*pair)
    with pytest.raises(ValueError):
        metric.get()
    metric.update(0, 0.0)
    with pytest.raises(ValueError):
        metric.get()
    metric.update(1, 0.5)
    assert metric.get() == pytest.approx(0.125)
    metric.update_batch([0, 1], [1.0, 0.0])
    assert metric.get() == pytest.approx(1.0)


def test_rolling_brier_subclass_keeps_custom_result():
    from pysad.evaluation import BrierScoreMetric, RollingMetric

    class DoubledBrier(BrierScoreMetric):
        def get(self):
            return 2 * super().get()

    metric = RollingMetric(DoubledBrier, window_size=2)
    metric.update_batch([0, 1, 0], [0.2, 0.5, 0.5])
    assert metric.get() == pytest.approx(0.5)


def test_rolling_precision_mixed_batch_and_single_updates():
    from pysad.evaluation import PrecisionMetric, RollingMetric

    metric = RollingMetric(PrecisionMetric, window_size=4, threshold=0.5)

    # 1. Start with single updates
    metric.update(1, 0.8)  # TP -> [TP], prec = 1/1 = 1.0
    assert metric.get() == pytest.approx(1.0)
    metric.update(0, 0.7)  # FP -> [TP, FP], prec = 1/2 = 0.5
    assert metric.get() == pytest.approx(0.5)
    metric.update(0, 0.2)  # TN -> [TP, FP, TN], prec = 1/2 = 0.5
    assert metric.get() == pytest.approx(0.5)

    # 2. Add a short batch (fills window to capacity and evicts oldest TP)
    metric.update_batch([1, 1], [0.9, 0.1])  # (1, 0.9) TP, (1, 0.1) FN -> evicts (1, 0.8) TP
    # Window: (0, 0.7) FP, (0, 0.2) TN, (1, 0.9) TP, (1, 0.1) FN -> TP=1, FP=1 -> prec = 1/2 = 0.5
    assert metric.get() == pytest.approx(0.5)

    # 3. Replace the window with a larger batch (length 5 >= window_size 4)
    metric.update_batch([0, 1, 1, 1, 0], [0.1, 0.8, 0.9, 0.7, 0.2])
    # Retained tail 4: (1, 0.8) TP, (1, 0.9) TP, (1, 0.7) TP, (0, 0.2) TN
    # TP=3, FP=0 -> prec = 3/3 = 1.0
    assert metric.get() == pytest.approx(1.0)

    # 4. Resume single updates and verify proper eviction
    metric.update(0, 0.6)  # FP -> evicts oldest (1, 0.8) TP
    # Window: (1, 0.9) TP, (1, 0.7) TP, (0, 0.2) TN, (0, 0.6) FP -> TP=2, FP=1 -> prec = 2/3
    assert metric.get() == pytest.approx(2.0 / 3.0)

    metric.update(0, 0.1)  # TN -> evicts oldest (1, 0.9) TP
    # Window: (1, 0.7) TP, (0, 0.2) TN, (0, 0.6) FP, (0, 0.1) TN -> TP=1, FP=1 -> prec = 1/2 = 0.5
    assert metric.get() == pytest.approx(0.5)

    metric.update(0, 0.2)  # TN -> evicts oldest (1, 0.7) TP
    # Window: (0, 0.2) TN, (0, 0.6) FP, (0, 0.1) TN, (0, 0.2) TN -> TP=0, FP=1 -> prec = 0/1 = 0.0
    assert metric.get() == pytest.approx(0.0)


def test_mixed_batch_and_single_updates():
    from pysad.evaluation import PrecisionMetric, RollingMetric

    metric = RollingMetric(
        PrecisionMetric,
        window_size=3,
        threshold=0.5,
    )

    # Stage 1: Start with individual updates.
    metric.update(1, 0.9)
    assert metric.get() == pytest.approx(1.0)

    metric.update(0, 0.8)
    assert metric.get() == pytest.approx(0.5)

    # Stage 2: Add a short batch.
    metric.update_batch(
        [1, 1],
        [0.2, 0.7],
    )
    assert metric.get() == pytest.approx(0.5)

    # Stage 3: Replace the window with a larger batch.
    metric.update_batch(
        [0, 1, 0, 1],
        [0.9, 0.8, 0.1, 0.9],
    )
    assert metric.get() == pytest.approx(1.0)

    # Stage 4: Resume individual updates.
    metric.update(0, 0.7)
    assert metric.get() == pytest.approx(0.5)

    metric.update(1, 0.1)
    assert metric.get() == pytest.approx(0.5)
