
def test_seasonal_trend_decomposer_removes_repeating_pattern():
    import numpy as np
    from pysad.transform.preprocessing import SeasonalTrendDecomposer

    pattern = np.array([1.0, 2.0, 3.0])
    X = np.tile(pattern, 8).reshape(-1, 1)

    decomposer = SeasonalTrendDecomposer(season_length=3)
    residuals = decomposer.fit_transform(X)

    assert residuals.shape == X.shape
    assert np.all(np.isfinite(residuals))
    assert np.all(np.isclose(residuals[-6:], 0.0))


def test_seasonal_trend_decomposer_transform_uses_fitted_state():
    import numpy as np
    from pysad.transform.preprocessing import SeasonalTrendDecomposer

    X = np.tile(np.array([1.0, 2.0, 3.0]), 8).reshape(-1, 1)
    decomposer = SeasonalTrendDecomposer(season_length=3).fit(X)

    residual = decomposer.transform_partial(np.array([1.0]))

    assert np.all(np.isclose(residual, 0.0))


def test_seasonal_trend_decomposer_validates_windows():
    from numpy.testing import assert_raises
    from pysad.transform.preprocessing import SeasonalTrendDecomposer

    with assert_raises(ValueError):
        SeasonalTrendDecomposer(season_length=0)

    with assert_raises(ValueError):
        SeasonalTrendDecomposer(season_length=3, trend_window=0)
