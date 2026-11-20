import numpy as np
import pytest


@pytest.mark.parametrize(
    "num_bins, labels, probabilities, expected",
    [
        (
            2,
            [1, 0, 1, 0, 1, 0],
            [0.25, 0.25, 0.75, 0.75, 0.5, 1.0],
            [0.75, 0.25, 0.25, 0.25, 0.1, 0.25],
        ),
        (1, [1, 0, 1, 0], [0.25, 0.75, 1.0, 0.0], [0.75, 0.0, 0.0, 0.0]),
        (4, [0, 1, 0, 1], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]),
        (4, [1, 0, 1, 0], [0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]),
    ],
)
def test_ece_hand_calculated_prefixes(num_bins, labels, probabilities, expected):
    from pysad.evaluation import ExpectedCalibrationErrorMetric

    metric = ExpectedCalibrationErrorMetric(num_bins=num_bins)
    for label, probability, score in zip(labels, probabilities, expected, strict=True):
        metric.update(label, probability)
        assert metric.get() == pytest.approx(score)
        assert isinstance(metric.get(), float)


@pytest.mark.parametrize(
    "edge, side, expected",
    [
        (0.1, -np.inf, 0.5),
        (0.3, -np.inf, 0.5),
        (0.7, -np.inf, 0.5),
        (0.9, -np.inf, 0.5),
        (0.1, np.inf, 0.4),
        (0.3, np.inf, 0.2),
        (0.7, np.inf, 0.2),
        (0.9, np.inf, 0.4),
    ],
)
def test_ece_internal_edges_start_the_next_bin(edge, side, expected):
    from pysad.evaluation import ExpectedCalibrationErrorMetric

    metric = ExpectedCalibrationErrorMetric()
    metric.update(1, edge)
    metric.update(0, np.nextafter(edge, side))
    assert metric.get() == pytest.approx(expected)


def test_ece_empty():
    from pysad.evaluation import ExpectedCalibrationErrorMetric

    metric = ExpectedCalibrationErrorMetric()
    assert metric.get() == 0.0
    assert metric.update_batch([], []) is metric
    assert metric.get() == 0.0


@pytest.mark.parametrize("as_array", [False, True])
def test_ece_batch_continues_stream(as_array):
    from pysad.evaluation import ExpectedCalibrationErrorMetric

    metric = ExpectedCalibrationErrorMetric(num_bins=2)
    metric.update(1, 0.25)
    labels, probabilities = [0, 1, 0], [0.25, 0.75, 0.75]
    if as_array:
        labels = np.asarray(labels, dtype=np.uint8)
        probabilities = np.asarray(probabilities)
    else:
        labels, probabilities = iter(labels), iter(probabilities)
    assert metric.update_batch(labels, probabilities) is metric
    assert metric.get() == pytest.approx(0.25)


@pytest.mark.parametrize("num_bins", [0, -1, 1.5])
def test_ece_rejects_invalid_bin_counts(num_bins):
    from pysad.evaluation import ExpectedCalibrationErrorMetric

    with pytest.raises((ValueError, TypeError)):
        ExpectedCalibrationErrorMetric(num_bins=num_bins)


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
def test_ece_invalid_pair_leaves_stream_unchanged(label, probability):
    from pysad.evaluation import ExpectedCalibrationErrorMetric

    metric = ExpectedCalibrationErrorMetric(num_bins=2)
    metric.update(1, 0.75)
    with pytest.raises(ValueError):
        metric.update(label, probability)
    assert metric.get() == 0.25
    metric.update(1, 0.25)
    assert metric.get() == 0.5
