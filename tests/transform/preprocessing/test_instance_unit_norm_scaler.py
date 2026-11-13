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


@pytest.mark.parametrize("norm_order", [1, 2, 3, float("inf")])
@pytest.mark.parametrize("num_features", [1, 3, 10])
def test_zero_vectors_return_zeros(norm_order, num_features):
    import numpy as np

    from pysad.transform.preprocessing import InstanceUnitNormScaler

    X = np.zeros(num_features)
    with np.errstate(all="raise"):
        scaled_X = InstanceUnitNormScaler(pow=norm_order).transform_partial(X)

    np.testing.assert_array_equal(scaled_X, np.zeros(num_features))
    np.testing.assert_array_equal(X, np.zeros(num_features))


@pytest.mark.parametrize("values", [[float("nan")], [float("inf")], [1.0, float("nan")]])
def test_nonfinite_vectors_keep_nonfinite_outputs(values):
    import numpy as np

    from pysad.transform.preprocessing import InstanceUnitNormScaler

    with np.errstate(invalid="ignore"):
        scaled_X = InstanceUnitNormScaler().transform_partial(np.array(values))

    assert not np.isfinite(scaled_X).all()


def test_batch_mixed_zero_and_normal_rows_under_errstate_raise():
    import numpy as np

    from pysad.transform.preprocessing import InstanceUnitNormScaler

    X = np.array(
        [
            [3.0, 4.0],
            [0.0, 0.0],
            [-1.0, 0.0],
        ]
    )
    with np.errstate(all="raise"):
        out = InstanceUnitNormScaler().fit_transform(X)

    np.testing.assert_allclose(out[0], [0.6, 0.8])
    np.testing.assert_array_equal(out[1], [0.0, 0.0])
    np.testing.assert_allclose(out[2], [-1.0, 0.0])


def test_all_zero_batch_returns_zeros():
    import numpy as np

    from pysad.transform.preprocessing import InstanceUnitNormScaler

    X = np.zeros((4, 3))
    with np.errstate(all="raise"):
        out = InstanceUnitNormScaler().fit_transform(X)

    np.testing.assert_array_equal(out, np.zeros((4, 3)))


@pytest.mark.parametrize("value", [1e-200, 1e-320, 1e200, 1e308])
def test_extreme_magnitude_vectors_scale_to_unit_norm(value):
    import numpy as np

    from pysad.transform.preprocessing import InstanceUnitNormScaler

    X = np.full(2, value)
    with np.errstate(all="raise"):
        scaled_X = InstanceUnitNormScaler().transform_partial(X)

    np.testing.assert_allclose(scaled_X, np.full(2, 1.0 / np.sqrt(2.0)))
    np.testing.assert_array_equal(X, np.full(2, value))


@pytest.mark.parametrize("value", [1e-200, 1e200])
def test_extreme_magnitude_batch_rows_scale_to_unit_norm(value):
    import numpy as np

    from pysad.transform.preprocessing import InstanceUnitNormScaler

    X = np.array([[value, value], [3.0, 4.0], [0.0, 0.0]])
    with np.errstate(all="raise"):
        out = InstanceUnitNormScaler().fit_transform(X)

    np.testing.assert_allclose(out[0], np.full(2, 1.0 / np.sqrt(2.0)))
    np.testing.assert_allclose(out[1], [0.6, 0.8])
    np.testing.assert_array_equal(out[2], [0.0, 0.0])


@pytest.mark.parametrize("pow", [1, 2, 3])
def test_extreme_magnitude_vectors_reach_unit_norm_for_every_pow(pow):
    import numpy as np

    from pysad.transform.preprocessing import InstanceUnitNormScaler

    X = np.array([1e200, 2e200, 3e200])
    with np.errstate(all="raise"):
        scaled_X = InstanceUnitNormScaler(pow=pow).transform_partial(X)

    assert np.isclose(np.linalg.norm(scaled_X, ord=pow), 1.0)
    np.testing.assert_allclose(
        scaled_X, np.array([1.0, 2.0, 3.0]) / np.linalg.norm([1.0, 2.0, 3.0], ord=pow)
    )
