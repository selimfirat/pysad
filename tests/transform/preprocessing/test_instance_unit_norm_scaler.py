import pytest


def test_instance_unit_norm_scaler():
    import numpy as np

    from pysad.transform.preprocessing import InstanceUnitNormScaler

    X = np.random.rand(100, 25)
    scaler = InstanceUnitNormScaler()

    scaled_X = scaler.fit_transform(X)
    assert np.all(np.isclose(np.linalg.norm(scaled_X, axis=1), 1.0))

    scaler = scaler.fit(X)
    scaled_X = scaler.transform(X)
    assert np.all(np.isclose(np.linalg.norm(scaled_X, axis=1), 1.0))


@pytest.mark.parametrize("pow", [1, 2, 3])
def test_instance_unit_norm_scaler_pow(pow):
    import numpy as np

    from pysad.transform.preprocessing import InstanceUnitNormScaler

    X = np.random.rand(100, 25)
    scaler = InstanceUnitNormScaler(pow=pow)

    scaled_X = scaler.fit_transform(X)
    assert np.all(np.isclose(np.linalg.norm(scaled_X, ord=pow, axis=1), 1.0))

    scaler = scaler.fit(X)
    transformed_X = scaler.transform(X)
    assert np.all(np.isclose(np.linalg.norm(transformed_X, ord=pow, axis=1), 1.0))
    np.testing.assert_allclose(transformed_X, scaled_X)


def test_zero_vector_returns_zeros():
    import numpy as np

    from pysad.transform.preprocessing import InstanceUnitNormScaler

    with np.errstate(all="raise"):
        scaled_X = InstanceUnitNormScaler().fit_transform_partial(np.zeros(3))

    np.testing.assert_array_equal(scaled_X, np.zeros(3))


def test_batch_zero_rows_return_zeros():
    import numpy as np

    from pysad.transform.preprocessing import InstanceUnitNormScaler

    X = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
    with np.errstate(all="raise"):
        scaled_X = InstanceUnitNormScaler().fit_transform(X)

    np.testing.assert_allclose(scaled_X, [[0.0, 0.0, 0.0], [0.6, 0.8, 0.0]])
