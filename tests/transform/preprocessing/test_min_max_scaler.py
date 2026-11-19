import pytest


def test_fit_transform_matches_running_min_and_max():
    import numpy as np

    from pysad.statistics import MaxMeter, MinMeter
    from pysad.transform.preprocessing import MinMaxScaler

    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 4))

    scaler = MinMaxScaler()
    scaled_X = scaler.fit_transform(X)

    minimums = [MinMeter(keep_values=False) for _ in range(X.shape[1])]
    maximums = [MaxMeter(keep_values=False) for _ in range(X.shape[1])]
    expected = np.empty_like(X)
    for i, xi in enumerate(X):
        for minimum, maximum, x in zip(minimums, maximums, xi, strict=True):
            minimum.update(float(x))
            maximum.update(float(x))

        for j, (minimum, maximum, x) in enumerate(zip(minimums, maximums, xi, strict=True)):
            minimum_value = minimum.get()
            maximum_value = maximum.get()
            range_value = maximum_value - minimum_value
            expected[i, j] = 0.0 if range_value == 0.0 else (float(x) - minimum_value) / range_value

    np.testing.assert_allclose(scaled_X, expected)


def test_transform_partial_does_not_update_statistics():
    import numpy as np

    from pysad.transform.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    scaler.fit_partial(np.array([1.0, 2.0, 3.0]))

    first = scaler.transform_partial(np.array([1.0, 2.0, 3.0]))
    scaler.transform_partial(np.array([4.0, 5.0, 6.0]))
    second = scaler.transform_partial(np.array([1.0, 2.0, 3.0]))

    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("value", [0.0, -3.0, 100.0])
@pytest.mark.parametrize("num_features", [1, 3, 10])
def test_constant_feature_scales_to_zeros(value, num_features):
    import numpy as np

    from pysad.transform.preprocessing import MinMaxScaler

    X = np.full(num_features, value)
    scaler = MinMaxScaler()
    scaler.fit_partial(X)

    scaled_X = scaler.transform_partial(X)

    np.testing.assert_array_equal(scaled_X, np.zeros(num_features))
    np.testing.assert_array_equal(X, np.full(num_features, value))


def test_transform_partial_before_fit_returns_input_unchanged():
    import numpy as np

    from pysad.transform.preprocessing import MinMaxScaler

    X = np.array([1.0, -2.0, 3.0])

    np.testing.assert_array_equal(MinMaxScaler().transform_partial(X), X)


def test_outputs_stay_in_unit_interval():
    import numpy as np

    from pysad.transform.preprocessing import MinMaxScaler

    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 3))

    scaler = MinMaxScaler()
    for xi in X:
        scaled = scaler.fit_transform_partial(xi)
        assert np.all(scaled >= 0.0)
        assert np.all(scaled <= 1.0)


def test_single_feature_first_instance_scales_to_zero():
    import numpy as np

    from pysad.transform.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform_partial(np.array([7.0]))

    np.testing.assert_array_equal(scaled, np.array([0.0]))
