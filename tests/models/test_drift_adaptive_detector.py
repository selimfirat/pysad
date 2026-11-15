import pickle

import numpy as np
import pytest


def _bounded_score(score):
    return float(score > 2.0)


@pytest.mark.parametrize("meter", ["page_hinkley", "adwin"])
def test_drift_resets_model_and_fits_alerting_instance(meter):
    from pysad.models import DriftAdaptiveDetector, StandardAbsoluteDeviation
    from pysad.statistics import ADWINMeter, PageHinkleyMeter

    model = DriftAdaptiveDetector(
        model_factory=StandardAbsoluteDeviation,
        drift_meter=ADWINMeter(delta=0.1) if meter == "adwin" else PageHinkleyMeter(),
        threshold=0.5,
        score_transform=_bounded_score,
    )
    for _ in range(200):
        model.fit_partial(np.array([0.0]))
    assert model.num_resets_ == 0
    for _ in range(200):
        score = model.fit_score_partial(np.array([10.0]))
        if model.drift_detected_:
            assert score > 2.0
            assert model.score_partial(np.array([10.0])) == 0.0
            assert model.score_partial(np.array([11.0])) == pytest.approx(1e10)
            break
    else:
        pytest.fail("The shifted stream did not trigger a reset.")
    assert model.num_resets_ == 1
    for _ in range(200):
        model.fit_partial(np.array([10.0]))
    assert model.num_resets_ == 1
    assert not model.drift_detected_


@pytest.mark.parametrize("meter", ["page_hinkley", "adwin"])
def test_scoring_and_pickling_preserve_future_drift_behavior(meter):
    from pysad.models import DriftAdaptiveDetector, StandardAbsoluteDeviation
    from pysad.statistics import ADWINMeter, PageHinkleyMeter

    model = DriftAdaptiveDetector(
        model_factory=StandardAbsoluteDeviation,
        drift_meter=ADWINMeter(delta=0.1) if meter == "adwin" else PageHinkleyMeter(),
        threshold=0.5,
        score_transform=_bounded_score,
    )
    model.fit(np.zeros((200, 1)))
    restored = pickle.loads(pickle.dumps(model))
    for value in [10.0] * 200 + [0.0] * 200:
        x = np.array([value])
        model.score_partial(x)
        model.score_partial(x)
        assert model.fit_score_partial(x) == restored.fit_score_partial(x)
        assert model.drift_detected_ == restored.drift_detected_
        assert model.num_resets_ == restored.num_resets_
    assert model.num_resets_ >= 2


def test_without_drift_wrapper_matches_detector_for_batch_and_partial_calls():
    from pysad.models import DriftAdaptiveDetector, StandardAbsoluteDeviation

    stream = np.array([[0.0], [2.0], [1.0], [3.0]])
    expected = StandardAbsoluteDeviation().fit_score(stream)
    batch = DriftAdaptiveDetector(StandardAbsoluteDeviation, threshold=1e20)
    partial = DriftAdaptiveDetector(StandardAbsoluteDeviation, threshold=1e20)
    np.testing.assert_array_equal(batch.fit_score(stream), expected)
    for x, score in zip(stream, expected, strict=True):
        assert partial.score_partial(x) == score
        assert partial.fit_partial(x) is partial
    assert batch.num_resets_ == partial.num_resets_ == 0


def test_alerting_instance_is_learned_once():
    from pysad.models import DriftAdaptiveDetector, StandardAbsoluteDeviation

    model = DriftAdaptiveDetector(StandardAbsoluteDeviation, threshold=0.5)
    model.fit(np.zeros((20, 1)))
    model.fit_partial(np.array([10.0]))
    assert model.drift_detected_
    model.fit_partial(np.array([12.0]))
    assert not model.drift_detected_
    assert model.score_partial(np.array([11.0])) == 0.0


def test_adwin_rejects_unbounded_scores_before_learning():
    from pysad.models import DriftAdaptiveDetector, StandardAbsoluteDeviation
    from pysad.statistics import ADWINMeter

    model = DriftAdaptiveDetector(StandardAbsoluteDeviation, drift_meter=ADWINMeter())
    model.fit_partial(np.array([0.0]))
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        model.fit_partial(np.array([10.0]))
    assert model.score_partial(np.array([10.0])) == pytest.approx(1e11)
    assert model.num_resets_ == 0
