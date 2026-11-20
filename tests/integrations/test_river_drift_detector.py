import pickle

import pytest

pytest.importorskip("river", minversion="0.26.1")


def _page_hinkley_factory():
    from pysad.statistics import PageHinkleyMeter

    return PageHinkleyMeter(threshold=5.0, min_instances=1)


def _cusum_factory():
    from pysad.statistics import CUSUMMeter

    return CUSUMMeter(mean=10.0, delta=0.5, threshold=3.0)


def test_river_drift_detector_wraps_page_hinkley_meter():
    from pysad.integrations import RiverDriftDetector

    detector = RiverDriftDetector(_page_hinkley_factory)
    assert detector.meter_ is not None
    assert hasattr(detector.meter_, "drift_detected")

    # No drift initially
    detector.update(1.0)
    assert not detector.drift_detected

    # Large shift triggers drift
    detector.update(100.0)
    assert detector.drift_detected


def test_river_drift_detector_wraps_cusum_meter():
    from pysad.integrations import RiverDriftDetector

    detector = RiverDriftDetector(_cusum_factory)
    assert detector.meter_ is not None
    assert hasattr(detector.meter_, "drift_detected")

    # No drift initially
    detector.update(10.0)
    assert not detector.drift_detected

    # Shift triggers drift
    detector.update(20.0)
    detector.update(20.0)
    detector.update(20.0)
    assert detector.drift_detected


def test_river_drift_detector_clone_creates_fresh_meter():
    from pysad.integrations import RiverDriftDetector

    detector = RiverDriftDetector(_page_hinkley_factory)
    detector.update(1.0)
    detector.update(100.0)
    assert detector.drift_detected

    cloned = detector.clone()
    assert cloned.drift_detected is False
    assert cloned.meter_.num_items == 0


def test_river_drift_detector_clone_with_attributes_preserves_state():
    from pysad.integrations import RiverDriftDetector

    detector = RiverDriftDetector(_page_hinkley_factory)
    detector.update(1.0)
    detector.update(100.0)
    assert detector.drift_detected

    cloned = detector.clone(include_attributes=True)
    assert cloned.drift_detected == detector.drift_detected
    assert cloned.meter_.num_items == detector.meter_.num_items


def test_river_drift_detector_pickling():
    from pysad.integrations import RiverDriftDetector

    detector = RiverDriftDetector(_page_hinkley_factory)
    detector.update(1.0)
    detector.update(100.0)
    assert detector.drift_detected

    restored = pickle.loads(pickle.dumps(detector))
    assert restored.drift_detected == detector.drift_detected
    assert restored.meter_.num_items == detector.meter_.num_items


def test_river_drift_detector_is_river_drift_detector():
    from river.base import DriftDetector

    from pysad.integrations import RiverDriftDetector
    from pysad.statistics import PageHinkleyMeter

    detector = RiverDriftDetector(PageHinkleyMeter)
    assert isinstance(detector, DriftDetector)
