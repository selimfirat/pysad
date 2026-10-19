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
