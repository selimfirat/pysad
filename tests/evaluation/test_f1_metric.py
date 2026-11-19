import pytest


@pytest.mark.parametrize("threshold", [None, 0.0, 0.5, 1.0])
def test_f1_matches_sklearn_at_every_prefix(threshold):
    from sklearn.metrics import f1_score

    from pysad.evaluation import F1Metric

    labels = [0, 1, 0, 1, 1, 0, 1]
    scores = [0, 1, 1, 0, 1, 0, 1] if threshold is None else [-1, 1, 0.5, 0, 2, 0.9, 1]
    predictions = scores if threshold is None else [int(s >= threshold) for s in scores]
    metric = F1Metric(threshold=threshold)
    assert metric.get() == 0.0
    for end, (label, score) in enumerate(zip(labels, scores, strict=True), start=1):
        metric.update(label, score)
        expected = f1_score(labels[:end], predictions[:end], zero_division=0)
        assert metric.get() == pytest.approx(expected)
        assert metric.get() == pytest.approx(expected)


@pytest.mark.parametrize(
    "labels, predictions", [([0, 0], [0, 0]), ([1, 1], [0, 0]), ([0, 0], [1, 1]), ([1, 1], [1, 1])]
)
def test_f1_single_class_streams(labels, predictions):
    from sklearn.metrics import f1_score

    from pysad.evaluation import F1Metric

    metric = F1Metric()
    for label, prediction in zip(labels, predictions, strict=True):
        metric.update(label, prediction)
    assert metric.get() == f1_score(labels, predictions, zero_division=0)


@pytest.mark.parametrize("label, prediction", [(2, 1), (-1, 0), (0, 0.9), (1, -1)])
def test_f1_rejects_nonbinary_inputs_without_changing_score(label, prediction):
    from pysad.evaluation import F1Metric

    metric = F1Metric()
    metric.update(1, 1)
    with pytest.raises(ValueError):
        metric.update(label, prediction)
    assert metric.get() == 1.0


def test_f1_with_evaluate_stream():
    import numpy as np
    from sklearn.metrics import f1_score

    from pysad.evaluation import F1Metric, evaluate_stream
    from pysad.models import StandardAbsoluteDeviation

    X = np.array([[0.0], [1.0], [2.0], [10.0], [1.0], [-10.0]])
    y = np.array([0, 0, 0, 1, 0, 1])
    model = StandardAbsoluteDeviation()
    scores = np.array([model.fit_score_partial(x) for x in X])
    result = evaluate_stream(
        model=StandardAbsoluteDeviation(),
        X=X,
        y=y,
        metrics={"F1": F1Metric(threshold=3.0)},
    )
    assert result["F1"] == pytest.approx(f1_score(y, scores >= 3.0, zero_division=0))
    assert result["F1"] > 0.0
