import pytest


def test_instance_standard_scaler():
    import numpy as np

    from pysad.transform.preprocessing import InstanceStandardScaler

    X = np.random.default_rng(0).random((100, 25))
    scaler = InstanceStandardScaler()

    scaled_X = scaler.fit_transform(X)
    assert np.all(np.isclose(scaled_X, (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)))

    scaler = scaler.fit(X)
    scaled_X = scaler.transform(X)
    assert np.all(np.isclose(scaled_X, (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)))


@pytest.mark.parametrize("value", [0.0, -3.0, 0.1, 1e308, 1e-300])
@pytest.mark.parametrize("num_features", [1, 3, 10])
def test_constant_finite_vectors_return_zeros(value, num_features):
    import numpy as np

    from pysad.transform.preprocessing import InstanceStandardScaler

    X = np.full(num_features, value)
    with np.errstate(all="raise"):
        scaled_X = InstanceStandardScaler().transform_partial(X)

    np.testing.assert_array_equal(scaled_X, np.zeros(num_features))
    np.testing.assert_array_equal(X, np.full(num_features, value))


@pytest.mark.parametrize("values", [[float("inf")], [float("nan")], [1.0, float("inf")]])
def test_nonfinite_vectors_keep_nan_outputs(values):
    import numpy as np

    from pysad.transform.preprocessing import InstanceStandardScaler

    with np.errstate(invalid="ignore"):
        scaled_X = InstanceStandardScaler().transform_partial(np.array(values))

    np.testing.assert_array_equal(scaled_X, np.full(len(values), np.nan))
