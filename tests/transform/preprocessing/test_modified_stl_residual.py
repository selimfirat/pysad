
def test_modified_stl_residual_transformer_uses_paper_residual(monkeypatch):
    import numpy as np
    import sys
    import types
    from pysad.transform.preprocessing import ModifiedSTLResidualTransformer

    class DummySTL:
        def __init__(self, X, period, robust, **kwargs):
            self.X = X
            self.period = period
            self.robust = robust
            self.kwargs = kwargs

        def fit(self):
            class Result:
                seasonal = np.array([0.0, 1.0, 0.0, 1.0])

            return Result()

    statsmodels_module = types.ModuleType("statsmodels")
    tsa_module = types.ModuleType("statsmodels.tsa")
    seasonal_module = types.ModuleType("statsmodels.tsa.seasonal")
    seasonal_module.STL = DummySTL
    monkeypatch.setitem(sys.modules, "statsmodels", statsmodels_module)
    monkeypatch.setitem(sys.modules, "statsmodels.tsa", tsa_module)
    monkeypatch.setitem(sys.modules, "statsmodels.tsa.seasonal", seasonal_module)

    X = np.array([10.0, 12.0, 14.0, 16.0])
    transformer = ModifiedSTLResidualTransformer(period=2)

    residuals = transformer.transform_window(X)

    assert np.all(np.isclose(residuals, X - np.array([0.0, 1.0, 0.0, 1.0]) - np.median(X)))


def test_modified_stl_residual_validates_inputs():
    from numpy.testing import assert_raises
    from pysad.transform.preprocessing import ModifiedSTLResidualTransformer

    with assert_raises(ValueError):
        ModifiedSTLResidualTransformer(period=1)

    with assert_raises(ValueError):
        ModifiedSTLResidualTransformer(period=3, window_size=5)


def test_modified_stl_residual_fit_handles_base_transformer_tuple(monkeypatch):
    import numpy as np
    import sys
    import types
    from pysad.transform.preprocessing import ModifiedSTLResidualTransformer

    class DummySTL:
        def __init__(self, X, period, robust, **kwargs):
            self.X = X

        def fit(self):
            class Result:
                seasonal = np.zeros(4)

            return Result()

    statsmodels_module = types.ModuleType("statsmodels")
    tsa_module = types.ModuleType("statsmodels.tsa")
    seasonal_module = types.ModuleType("statsmodels.tsa.seasonal")
    seasonal_module.STL = DummySTL
    monkeypatch.setitem(sys.modules, "statsmodels", statsmodels_module)
    monkeypatch.setitem(sys.modules, "statsmodels.tsa", tsa_module)
    monkeypatch.setitem(sys.modules, "statsmodels.tsa.seasonal", seasonal_module)

    transformer = ModifiedSTLResidualTransformer(period=2, window_size=4)
    transformer.fit(np.array([[1.0], [2.0], [3.0], [4.0]]))

    assert transformer.window.get() == [1.0, 2.0, 3.0, 4.0]
