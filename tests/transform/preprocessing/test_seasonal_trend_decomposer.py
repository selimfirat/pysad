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


def test_seasonal_trend_decomposer_batch_matches_partial_multivariate():
    import numpy as np

    from pysad.transform.preprocessing import SeasonalTrendDecomposer

    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))

    batch_dec = SeasonalTrendDecomposer(season_length=6, trend_window=10)
    partial_dec = SeasonalTrendDecomposer(season_length=6, trend_window=10)

    batch_out = batch_dec.fit_transform(X)
    partial_out = np.vstack([partial_dec.fit_transform_partial(row) for row in X])

    np.testing.assert_allclose(batch_out, partial_out, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        batch_dec.transform_partial(X[0]),
        partial_dec.transform_partial(X[0]),
        rtol=1e-12,
        atol=1e-12,
    )
