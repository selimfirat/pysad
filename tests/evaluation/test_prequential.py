def test_evaluate_stream_matches_prequential_loop():
    import numpy as np

    from pysad.evaluation import AUPRMetric, AUROCMetric, evaluate_stream
    from pysad.models import LODA

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = np.zeros(200, dtype=int)
    y[::20] = 1
    X[y == 1] += 5

    results = evaluate_stream(
        model=LODA(random_state=0),
        X=X,
        y=y,
        metrics={"AUROC": AUROCMetric(), "AUPR": AUPRMetric()},
    )

    model = LODA(random_state=0)
    auroc, aupr = AUROCMetric(), AUPRMetric()
    for x, label in zip(X, y, strict=True):
        score = model.fit_score_partial(x)
        auroc.update(label, score)
        aupr.update(label, score)

    assert results == {"AUROC": auroc.get(), "AUPR": aupr.get()}
    assert results["AUROC"] > 0.5


def test_evaluate_stream_rejects_length_mismatch():
    import numpy as np
    import pytest

    from pysad.evaluation import AUROCMetric, evaluate_stream
    from pysad.models import NullModel

    with pytest.raises(ValueError):
        evaluate_stream(
            model=NullModel(),
            X=np.zeros((3, 2)),
            y=np.array([0, 1]),
            metrics={"AUROC": AUROCMetric()},
        )
