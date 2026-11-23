import functools

import numpy as np
import pytest


def test_seasonal_esd_detects_latest_window_anomaly(monkeypatch):
    import numpy as np

    from pysad.models import SeasonalESD

    monkeypatch.setattr(
        "pysad.models.seasonal_esd.ModifiedSTLResidualTransformer.transform_window",
        lambda self, values: np.asarray(values, dtype=float),
    )

    X = np.array([[0.0], [0.1], [-0.1], [0.0], [8.0]])
    model = SeasonalESD(period=2, window_size=5)
    scores = model.fit_score(X)

    assert scores[-1] > 0.0


def test_seasonal_hybrid_esd_detects_latest_window_anomaly(monkeypatch):
    import numpy as np

    from pysad.models import SeasonalHybridESD

    monkeypatch.setattr(
        "pysad.models.seasonal_esd.ModifiedSTLResidualTransformer.transform_window",
        lambda self, values: np.asarray(values, dtype=float),
    )

    X = np.array([[0.0], [0.1], [-0.1], [0.0], [8.0]])
    model = SeasonalHybridESD(period=2, window_size=5)
    scores = model.fit_score(X)

    assert scores[-1] > 0.0


def test_seasonal_esd_uses_window_for_latest_candidate(monkeypatch):
    import numpy as np

    from pysad.models import SeasonalESD

    seen_window_sizes = []

    def fake_transform(self, values):
        seen_window_sizes.append(len(values))
        return np.asarray(values, dtype=float)

    monkeypatch.setattr(
        "pysad.models.seasonal_esd.ModifiedSTLResidualTransformer.transform_window", fake_transform
    )

    model = SeasonalESD(period=2, window_size=5)
    model.fit(np.array([[0.0], [0.1], [-0.1], [0.0]]))
    model.score_partial(np.array([8.0]))

    assert seen_window_sizes == [5]


def test_seasonal_esd_scores_repeated_latest_candidate(monkeypatch):
    import numpy as np

    from pysad.models import SeasonalESD

    seen_windows = []

    def fake_transform(self, values):
        seen_windows.append(np.asarray(values, dtype=float).copy())
        return np.asarray(values, dtype=float)

    monkeypatch.setattr(
        "pysad.models.seasonal_esd.ModifiedSTLResidualTransformer.transform_window", fake_transform
    )

    model = SeasonalESD(period=2, window_size=5)
    model.fit(np.array([[0.0], [1.0], [2.0], [3.0]]))
    model.score_partial(np.array([3.0]))

    assert np.array_equal(seen_windows[0], np.array([0.0, 1.0, 2.0, 3.0, 3.0]))


def test_seasonal_esd_fit_score_does_not_double_append(monkeypatch):
    import numpy as np

    from pysad.models import SeasonalESD

    seen_windows = []

    def fake_transform(self, values):
        seen_windows.append(np.asarray(values, dtype=float).copy())
        return np.asarray(values, dtype=float)

    monkeypatch.setattr(
        "pysad.models.seasonal_esd.ModifiedSTLResidualTransformer.transform_window", fake_transform
    )

    model = SeasonalESD(period=2, window_size=5)
    model.fit(np.array([[0.0], [1.0], [2.0], [3.0]]))
    model.fit_score_partial(np.array([3.0]))

    assert np.array_equal(seen_windows[0], np.array([0.0, 1.0, 2.0, 3.0, 3.0]))


def test_seasonal_esd_validates_configuration():
    from numpy.testing import assert_raises

    from pysad.models import SeasonalESD

    with assert_raises(ValueError):
        SeasonalESD(period=2, window_size=5, max_anomalies=0)

    with assert_raises(ValueError):
        SeasonalESD(period=2, window_size=5, max_anomalies=0.5)

    with assert_raises(ValueError):
        SeasonalESD(period=2, window_size=5, direction="up")


def test_seasonal_esd_rejects_short_windows():
    from numpy.testing import assert_raises

    from pysad.models import SeasonalESD

    with assert_raises(ValueError):
        SeasonalESD(period=10, window_size=20)


def test_seasonal_esd_rejects_invalid_alpha():
    import numpy as np
    from numpy.testing import assert_raises

    from pysad.models import SeasonalESD

    for alpha in (0.0, 1.0, -0.1, np.nan, np.inf):
        with assert_raises(ValueError):
            SeasonalESD(period=2, window_size=5, alpha=alpha)


@pytest.mark.parametrize(
    ("direction", "expected"),
    [("pos", [100, 170]), ("neg", [130]), ("both", [100, 130, 170])],
)
def test_seasonal_hybrid_esd_matches_r_anomaly_detection(direction, expected):
    import numpy as np

    from pysad.models import SeasonalHybridESD

    t = np.arange(24 * 8)
    X = 10 + 3 * np.sin(2 * np.pi * t / 24) + 0.3 * np.sin(0.7 * t**2)
    X[[100, 130, 170]] += [4, -4, 5]

    model = SeasonalHybridESD(period=24, window_size=97, direction=direction)
    scores = model.fit_score(X.reshape(-1, 1))

    # The values i for which AnomalyDetectionVec(X[i - 96 : i + 1], period = 24, direction) of
    # Twitter's AnomalyDetection R package (commit 1f5deaa) flags the last value.
    assert np.flatnonzero(scores > 0).tolist() == expected


@functools.cache
def _unit_invariance_series_and_expected(model_name):
    import numpy as np

    import pysad.models

    t = np.arange(12 * 6)
    X = 10 + 3 * np.sin(2 * np.pi * t / 12) + 0.3 * np.sin(0.7 * t**2)
    X[[52, 60, 68]] += [4, -4, 5]
    model = getattr(pysad.models, model_name)(period=12, window_size=49)
    expected = np.flatnonzero(model.fit_score(X.reshape(-1, 1)) > 0).tolist()
    return X, expected


@pytest.mark.parametrize("model_name", ["SeasonalESD", "SeasonalHybridESD"])
@pytest.mark.parametrize(("scale", "offset"), [(1e-12, 0.0), (1e-6, 0.0), (1e6, 0.0), (1e-4, 1e6)])
def test_seasonal_esd_flags_do_not_depend_on_units(model_name, scale, offset):
    import numpy as np

    import pysad.models

    X, expected = _unit_invariance_series_and_expected(model_name)
    assert expected
    model = getattr(pysad.models, model_name)(period=12, window_size=49)
    flags = np.flatnonzero(model.fit_score((offset + scale * X).reshape(-1, 1)) > 0).tolist()
    assert flags == expected


def test_seasonal_esd_auto_period_estimates_period():
    from pysad.models import SeasonalESD

    # Signal with period 24
    t = np.arange(300)
    values = np.sin(2 * np.pi * t / 24) + 0.1 * np.random.default_rng(42).normal(size=t.size)

    model = SeasonalESD(window_size=100, auto_period=True, min_period=4, max_period=50)
    assert model.auto_period is True
    assert model.period == 4  # Initial period is min_period

    # Feed enough data for period meter to initialize
    for v in values[:100]:
        model.fit_partial(np.array([v]))

    # Period should be updated to something close to 24 (25 due to frequency resolution)
    assert model.period in (24, 25)
    assert model.residual_transformer.period == model.period


def test_seasonal_hybrid_esd_auto_period_estimates_period():
    from pysad.models import SeasonalHybridESD

    t = np.arange(300)
    values = np.sin(2 * np.pi * t / 24) + 0.1 * np.random.default_rng(42).normal(size=t.size)

    model = SeasonalHybridESD(window_size=100, auto_period=True, min_period=4, max_period=50)
    assert model.auto_period is True
    assert model.period == 4

    for v in values[:100]:
        model.fit_partial(np.array([v]))

    assert model.period in (24, 25)
    assert model.residual_transformer.period == model.period


def test_seasonal_esd_auto_period_requires_window_size_validation():
    import pytest

    from pysad.models import SeasonalESD

    # window_size must be > 2 * min_period
    with pytest.raises(ValueError, match="window_size must be greater than 2 \\* min_period"):
        SeasonalESD(window_size=5, auto_period=True, min_period=4)


def test_seasonal_esd_auto_period_rejects_invalid_period_window_size():
    import pytest

    from pysad.models import SeasonalESD

    with pytest.raises(ValueError, match="period_window_size must be at least 4"):
        SeasonalESD(window_size=100, auto_period=True, period_window_size=3)


def test_seasonal_esd_auto_period_rejects_max_period_less_than_min_period():
    import pytest

    from pysad.models import SeasonalESD

    with pytest.raises(ValueError, match="max_period must be at least min_period"):
        SeasonalESD(window_size=100, auto_period=True, min_period=10, max_period=5)


def test_seasonal_esd_auto_period_false_requires_period():
    import pytest

    from pysad.models import SeasonalESD

    with pytest.raises(ValueError, match="period is required when auto_period=False"):
        SeasonalESD(window_size=100, auto_period=False)


def test_seasonal_esd_auto_period_detects_anomaly_after_period_estimate():
    from pysad.models import SeasonalESD

    # Create a signal with period 12 and add a spike
    # Need enough data for period meter to initialize (period_window_size = window_size = 50)
    # Use zero-mean signal so period meter isn't dominated by DC component
    t = np.arange(200)
    values = 3 * np.sin(2 * np.pi * t / 12)
    values[150] += 5.0  # Anomaly at index 150 (after period meter initializes)

    model = SeasonalESD(window_size=50, auto_period=True, min_period=4, max_period=30, alpha=0.01)

    scores = []
    for v in values:
        scores.append(model.fit_score_partial(np.array([v])))

    # The model should detect the anomaly after the period is estimated
    # Period meter needs 50 samples to initialize, so anomaly at 150 should be detectable
    assert model.period in (12, 13)  # 12 or 13 due to resolution
    assert scores[150] > 0.0


def test_seasonal_esd_auto_period_with_custom_period_window_size():
    from pysad.models import SeasonalESD

    t = np.arange(200)
    values = np.sin(2 * np.pi * t / 24) + 0.1 * np.random.default_rng(42).normal(size=t.size)

    # Use a smaller period_window_size
    model = SeasonalESD(
        window_size=100, auto_period=True, period_window_size=60, min_period=4, max_period=50
    )

    for v in values[:60]:
        model.fit_partial(np.array([v]))

    # Period meter should have initialized with window_size=60
    assert model._period_meter.window_size == 60
    # With window_size=60, the frequency resolution gives periods of 60/k
    # For period 24, closest bins are k=2 (period=30) and k=3 (period=20)
    # The signal has period 24, so power at bin 2 (freq=2/60=1/30) and bin 3 (freq=3/60=1/20)
    # Bin 2 (period 30) is closer to 24 than bin 3 (period 20)
    assert model.period in (20, 30)
