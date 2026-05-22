
def test_seasonal_esd_detects_latest_window_anomaly(monkeypatch):
    import numpy as np
    from pysad.models import SeasonalESD

    monkeypatch.setattr(
        "pysad.models.seasonal_esd.ModifiedSTLResidualTransformer.transform_window",
        lambda self, values: np.asarray(values, dtype=float),
    )

    X = np.array([[0.0], [0.1], [-0.1], [0.0], [8.0]])
    model = SeasonalESD(period=2, window_size=5, max_anomalies=1)
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
    model = SeasonalHybridESD(period=2, window_size=5, max_anomalies=1)
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
        "pysad.models.seasonal_esd.ModifiedSTLResidualTransformer.transform_window",
        fake_transform,
    )

    model = SeasonalESD(period=2, window_size=5, max_anomalies=1)
    model.fit(np.array([[0.0], [0.1], [-0.1], [0.0]]))
    model.score_partial(np.array([8.0]))

    assert seen_window_sizes == [5]


def test_seasonal_esd_validates_configuration():
    from numpy.testing import assert_raises
    from pysad.models import SeasonalESD

    with assert_raises(ValueError):
        SeasonalESD(period=2, window_size=5, max_anomalies=0)

    with assert_raises(ValueError):
        SeasonalESD(period=2, window_size=5, max_anomalies=3)
