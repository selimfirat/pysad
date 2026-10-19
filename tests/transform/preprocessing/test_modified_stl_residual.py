import pytest

# The seasonal component of R 4.6.1 stl(ts(X, frequency = period), s.window = "periodic", robust)
# at each position in the period, for the series of the test below.
R_SEASONAL = {
    (2, 5, True): [-15.4100257033775, 15.4100257033771],
    (7, 15, True): [
        -9.28436298267912,
        24.2219591269983,
        -5.18974322537318,
        -6.5884712620321,
        -9.88567724124638,
        -11.469368999218,
        18.1956654799793,
    ],
    (7, 52, False): [
        -2.31571660710876,
        7.70297101944383,
        0.496558131024637,
        0.127333427503637,
        -2.77553016851077,
        0.50116720743287,
        -3.73678651330254,
    ],
    (24, 169, True): [
        0.142532555309292,
        0.937841329932238,
        1.71487776884288,
        1.95296468497989,
        2.9577826379729,
        2.47349820298868,
        2.64109410769756,
        3.09595798638028,
        2.95592624677037,
        2.19222703214535,
        1.17849159820055,
        0.464643552960937,
        0.514752577683983,
        -1.25234779009747,
        -1.41204247076849,
        -1.8392597462091,
        -2.60824916199147,
        -2.95105403551293,
        -3.1586274419593,
        -2.70568425623777,
        -2.72716375397597,
        -2.08036051422489,
        -1.63293273779853,
        -0.854868388539159,
    ],
}


@pytest.mark.parametrize(("period", "n", "robust"), list(R_SEASONAL))
def test_modified_stl_residual_matches_r_periodic_stl(period, n, robust):
    import numpy as np

    from pysad.transform.preprocessing import ModifiedSTLResidualTransformer

    t = np.arange(n)
    X = 10 + 3 * np.sin(2 * np.pi * t / period) + 0.05 * t + np.sin(0.7 * t**2)
    X[[1, n // 2, n - 2]] += 30

    residuals = ModifiedSTLResidualTransformer(period=period, robust=robust).transform_window(X)

    expected = X - np.resize(R_SEASONAL[(period, n, robust)], n) - np.median(X)
    assert np.allclose(residuals, expected, rtol=0, atol=1e-9)


def test_modified_stl_residual_uses_periodic_seasonal_component():
    import numpy as np

    from pysad.transform.preprocessing import ModifiedSTLResidualTransformer

    # A level shift in the last 8 values of a noiseless seasonal series. Periodic STL keeps the
    # seasonal component equal in every period, so the residual is the shift and 0 elsewhere.
    t = np.arange(24 * 14)
    X = 10 + 3 * np.sin(2 * np.pi * t / 24) + np.cos(4 * np.pi * t / 24)
    X[-8:] += 2.0

    residuals = ModifiedSTLResidualTransformer(period=24).transform_window(X)

    expected = np.zeros(t.size)
    expected[-8:] = 2.0
    assert np.allclose(residuals - np.median(residuals), expected, atol=1e-3)


def test_modified_stl_residual_validates_inputs():
    from numpy.testing import assert_raises

    from pysad.transform.preprocessing import ModifiedSTLResidualTransformer

    with assert_raises(ValueError):
        ModifiedSTLResidualTransformer(period=1)

    with assert_raises(ValueError):
        ModifiedSTLResidualTransformer(period=3, window_size=6)

    with assert_raises(ValueError):
        ModifiedSTLResidualTransformer(period=3).transform_window([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


def test_modified_stl_residual_fit_handles_base_transformer_tuple():
    import numpy as np

    from pysad.transform.preprocessing import ModifiedSTLResidualTransformer

    transformer = ModifiedSTLResidualTransformer(period=2, window_size=5)
    transformer.fit(np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]))

    assert transformer.window.get() == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_modified_stl_residual_transform_partial_includes_repeated_candidate(monkeypatch):
    import numpy as np

    from pysad.transform.preprocessing import ModifiedSTLResidualTransformer

    seen_windows = []

    def fake_transform_window(self, X):
        seen_windows.append(np.asarray(X, dtype=float).copy())
        return np.asarray(X, dtype=float)

    monkeypatch.setattr(
        "pysad.transform.preprocessing.modified_stl_residual."
        "ModifiedSTLResidualTransformer.transform_window",
        fake_transform_window,
    )

    transformer = ModifiedSTLResidualTransformer(period=2, window_size=5)
    transformer.fit(np.array([[1.0], [2.0], [3.0], [4.0]]))
    residual = transformer.transform_partial(np.array([4.0]))

    assert np.array_equal(residual, np.array([4.0]))
    assert np.array_equal(seen_windows[0], np.array([1.0, 2.0, 3.0, 4.0, 4.0]))


def test_modified_stl_residual_fit_transform_partial_does_not_double_append(monkeypatch):
    import numpy as np

    from pysad.transform.preprocessing import ModifiedSTLResidualTransformer

    seen_windows = []

    def fake_transform_window(self, X):
        seen_windows.append(np.asarray(X, dtype=float).copy())
        return np.asarray(X, dtype=float)

    monkeypatch.setattr(
        "pysad.transform.preprocessing.modified_stl_residual."
        "ModifiedSTLResidualTransformer.transform_window",
        fake_transform_window,
    )

    transformer = ModifiedSTLResidualTransformer(period=2, window_size=5)
    transformer.fit(np.array([[1.0], [2.0], [3.0], [4.0]]))
    residual = transformer.fit_transform_partial(np.array([4.0]))

    assert np.array_equal(residual, np.array([4.0]))
    assert np.array_equal(seen_windows[0], np.array([1.0, 2.0, 3.0, 4.0, 4.0]))


def test_modified_stl_residual_fit_transform_cold_starts_until_enough_values(monkeypatch):
    import numpy as np

    from pysad.transform.preprocessing import ModifiedSTLResidualTransformer

    seen_windows = []

    def fake_transform_window(self, X):
        seen_windows.append(np.asarray(X, dtype=float).copy())
        return np.asarray(X, dtype=float)

    monkeypatch.setattr(
        "pysad.transform.preprocessing.modified_stl_residual."
        "ModifiedSTLResidualTransformer.transform_window",
        fake_transform_window,
    )

    transformer = ModifiedSTLResidualTransformer(period=2, window_size=5)
    residuals = transformer.fit_transform(np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]))

    assert np.array_equal(residuals.ravel(), np.array([0.0, 0.0, 0.0, 0.0, 5.0]))
    assert len(seen_windows) == 1
    assert np.array_equal(seen_windows[0], np.array([1.0, 2.0, 3.0, 4.0, 5.0]))


def test_modified_stl_residual_partial_output_feeds_univariate_detector():
    import numpy as np

    from pysad.models import MedianAbsoluteDeviation
    from pysad.transform.preprocessing import ModifiedSTLResidualTransformer

    t = np.arange(40)
    X = 10 + 3 * np.sin(2 * np.pi * t / 4)
    transformer = ModifiedSTLResidualTransformer(period=4, window_size=16)
    model = MedianAbsoluteDeviation()

    for i, value in enumerate(X):
        candidate = transformer.transform_partial(np.array([value]))
        residual = transformer.fit_transform_partial(np.array([value]))

        for output in (candidate, residual):
            assert isinstance(output, np.ndarray)
            assert output.dtype == np.float64
            assert output.shape == (1,)

        assert np.array_equal(candidate, residual)
        if i < 8:
            assert np.array_equal(residual, np.zeros(1))
        else:
            expected = transformer.transform_window(X[max(0, i - 15) : i + 1])[-1]
            assert residual[0] == expected

        assert np.isfinite(model.fit_score_partial(residual))
