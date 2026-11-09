import numpy as np
import pytest


@pytest.mark.parametrize(
    "as_iterable",
    [np.asarray, list, lambda values: (value for value in values)],
    ids=["array", "list", "generator"],
)
def test_evaluate_stream_matches_prequential_loop(as_iterable):
    from pysad.evaluation import AUPRMetric, AUROCMetric, evaluate_stream
    from pysad.models import LODA

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = np.zeros(200, dtype=int)
    y[::20] = 1
    X[y == 1] += 5

    results = evaluate_stream(
        model=LODA(random_state=0),
        X=as_iterable(X),
        y=as_iterable(y),
        metrics={"AUROC": AUROCMetric(), "AUPR": AUPRMetric()},
    )

    model = LODA(random_state=0)
    auroc, aupr = AUROCMetric(), AUPRMetric()
    for x, label in zip(X, y, strict=True):
        score = model.score_partial(x)
        model.fit_partial(x)
        auroc.update(label, score)
        aupr.update(label, score)

    assert results == {"AUROC": auroc.get(), "AUPR": aupr.get()}
    assert results["AUROC"] > 0.5


@pytest.mark.parametrize(
    "as_iterable",
    [np.asarray, list, lambda values: (value for value in values)],
    ids=["array", "list", "generator"],
)
@pytest.mark.parametrize("num_instances, num_labels", [(3, 2), (2, 3), (0, 1), (1, 0)])
def test_evaluate_stream_rejects_length_mismatch(as_iterable, num_instances, num_labels):
    from pysad.evaluation import evaluate_stream
    from pysad.models import NullModel

    X = np.zeros((num_instances, 2))
    y = np.zeros(num_labels, dtype=int)
    with pytest.raises(ValueError):
        evaluate_stream(
            model=NullModel(),
            X=as_iterable(X),
            y=as_iterable(y),
            metrics={},
        )


def test_evaluate_stream_scores_generators_before_learning():
    from pysad.evaluation import RecallMetric, evaluate_stream
    from pysad.models import StandardAbsoluteDeviation

    model = StandardAbsoluteDeviation().fit(np.array([[0.0], [2.0]]))
    results = evaluate_stream(
        model=model,
        X=(np.array([value]) for value in [4.0, 1.0, 10.0]),
        y=(label for label in [1, 0, 1]),
        metrics={"recall": RecallMetric(threshold=2.0)},
    )

    # Both anomalies exceed two standard deviations before the model learns from them.
    assert results == {"recall": 1.0}
    # The final mean includes the two initial values and all three streamed values.
    assert model.score_partial(np.array([3.4])) == pytest.approx(0.0)
