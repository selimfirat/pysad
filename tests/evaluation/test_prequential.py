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


def test_evaluate_stream_with_postprocessor_matches_manual_loop():
    from pysad.evaluation import AUROCMetric, evaluate_stream
    from pysad.models import LODA
    from pysad.transform.probability_calibration import ConformalProbabilityCalibrator

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = np.zeros(200, dtype=int)
    y[::20] = 1
    X[y == 1] += 5

    results = evaluate_stream(
        model=LODA(random_state=0),
        X=X,
        y=y,
        metrics={"AUROC": AUROCMetric()},
        postprocessor=ConformalProbabilityCalibrator(window_size=50),
    )

    model = LODA(random_state=0)
    calibrator = ConformalProbabilityCalibrator(window_size=50)
    auroc = AUROCMetric()
    for x, label in zip(X, y, strict=True):
        score = model.fit_score_partial(x)
        calibrated = calibrator.fit_transform_partial(score)
        auroc.update(label, calibrated)

    assert results == {"AUROC": auroc.get()}
    assert results["AUROC"] > 0.5


def test_evaluation_config_defaults_and_fields():
    from pysad.evaluation import AUROCMetric
    from pysad.evaluation.config import EvaluationConfig
    from pysad.models import LODA

    model = LODA(random_state=0)
    metrics = {"AUROC": AUROCMetric()}
    config = EvaluationConfig(model=model, metrics=metrics)

    assert config.model is model
    assert config.postprocessor is None
    assert config.metrics is metrics


@pytest.mark.parametrize(
    "as_iterable",
    [np.asarray, list, lambda values: (value for value in values)],
    ids=["array", "list", "generator"],
)
def test_evaluate_postprocessor_matches_manual_loop(as_iterable):
    from pysad.evaluation import (
        AUROCMetric,
        BrierScoreMetric,
        ExpectedCalibrationErrorMetric,
        evaluate_postprocessor,
    )
    from pysad.models import LODA
    from pysad.transform.probability_calibration import ConformalProbabilityCalibrator

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = np.zeros(200, dtype=int)
    y[::20] = 1
    X[y == 1] += 5

    model = LODA(random_state=0)
    raw_scores = np.array([float(model.fit_score_partial(x)) for x in X])

    calibrator = ConformalProbabilityCalibrator(window_size=50)
    results = evaluate_postprocessor(
        scores=as_iterable(raw_scores),
        y=as_iterable(y),
        metrics={
            "AUROC": AUROCMetric(),
            "Brier": BrierScoreMetric(),
            "ECE": ExpectedCalibrationErrorMetric(),
        },
        postprocessor=calibrator,
    )

    model = LODA(random_state=0)
    calibrator = ConformalProbabilityCalibrator(window_size=50)
    auroc, brier, ece = AUROCMetric(), BrierScoreMetric(), ExpectedCalibrationErrorMetric()
    for x, label in zip(X, y, strict=True):
        score = model.fit_score_partial(x)
        calibrated = calibrator.fit_transform_partial(score)
        auroc.update(label, calibrated)
        brier.update(label, calibrated)
        ece.update(label, calibrated)

    assert results == {
        "AUROC": auroc.get(),
        "Brier": brier.get(),
        "ECE": ece.get(),
    }
    assert results["AUROC"] > 0.5
    assert 0.0 <= results["Brier"] <= 1.0
    assert 0.0 <= results["ECE"] <= 1.0


def test_evaluate_postprocessor_rejects_length_mismatch():
    from pysad.evaluation import AUROCMetric, evaluate_postprocessor
    from pysad.transform.probability_calibration import ConformalProbabilityCalibrator

    scores = np.zeros(5)
    y = np.zeros(3, dtype=int)
    with pytest.raises(ValueError):
        evaluate_postprocessor(
            scores=scores,
            y=y,
            metrics={"AUROC": AUROCMetric()},
            postprocessor=ConformalProbabilityCalibrator(),
        )


def test_evaluate_stream_scores_before_fitting():
    """evaluate_stream scores each instance before the model learns from it.

    A recording wrapper logs the order of score_partial and fit_partial calls. The
    default fit_score_partial is prequential (score then fit), so the events must
    alternate score(i), fit(i) for every instance i. A change that fits before scoring
    would reorder or duplicate these events and fail this assertion.
    """
    from typing import Any

    from numpy.typing import NDArray

    from pysad.core.base_model import BaseModel
    from pysad.evaluation import AUROCMetric, evaluate_stream
    from pysad.models import LODA

    class _OrderRecordingModel(BaseModel):
        """Records score_partial/fit_partial call order, delegating to a wrapped model."""

        def __init__(self, model: BaseModel) -> None:
            self._model = model
            self.events: list[tuple[str, int]] = []
            self._n = 0

        def score_partial(self, X: NDArray[Any], y: int | None = None) -> float:
            self.events.append(("score", self._n))
            return self._model.score_partial(X)

        def fit_partial(self, X: NDArray[Any], y: int | None = None) -> BaseModel:
            self.events.append(("fit", self._n))
            self._model.fit_partial(X, y)
            self._n += 1
            return self

    rng = np.random.default_rng(0)
    X = [np.array([value]) for value in rng.normal(size=60)]
    y = [i % 2 for i in range(len(X))]

    recorder = _OrderRecordingModel(LODA(random_state=0))
    results = evaluate_stream(
        model=recorder,
        X=X,
        y=y,
        metrics={"AUROC": AUROCMetric()},
    )

    assert isinstance(results, dict)
    assert results["AUROC"] is not None

    expected: list[tuple[str, int]] = []
    for i in range(len(X)):
        expected.append(("score", i))
        expected.append(("fit", i))
    assert recorder.events == expected


def test_fit_score_partial_equals_score_then_fit():
    """fit_score_partial matches scoring then fitting the instance sequentially.

    Two identically seeded models process the same stream: one with fit_score_partial,
    the other with score_partial followed by fit_partial. Their scores and their
    behavior on new instances must agree (public behavior only).
    """
    from pysad.models import LODA

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    X_new = rng.normal(size=(10, 3))

    fit_score_model = LODA(random_state=0)
    sequential_model = LODA(random_state=0)

    fit_score_scores = [fit_score_model.fit_score_partial(x) for x in X]
    sequential_scores = []
    for x in X:
        sequential_scores.append(sequential_model.score_partial(x))
        sequential_model.fit_partial(x)

    assert fit_score_scores == pytest.approx(sequential_scores)

    assert [fit_score_model.score_partial(x) for x in X_new] == pytest.approx(
        [sequential_model.score_partial(x) for x in X_new]
    )
