import pytest


def test_fit_transform_matches_running_mean_and_variance():
    import math

    import numpy as np

    from pysad.statistics import AverageMeter, VarianceMeter
    from pysad.transform.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 4))

    scaler = StandardScaler()
    scaled_X = scaler.fit_transform(X)

    means = [AverageMeter() for _ in range(X.shape[1])]
    variances = [VarianceMeter() for _ in range(X.shape[1])]
    expected = np.empty_like(X)
    for i, xi in enumerate(X):
        for mean, variance, x in zip(means, variances, xi, strict=True):
            mean.update(float(x))
            variance.update(float(x))

        for j, (mean, variance, x) in enumerate(zip(means, variances, xi, strict=True)):
            mean_value = mean.get()
            variance_value = variance.get()
            expected[i, j] = (
                0.0
                if variance_value == 0.0
                else (float(x) - mean_value) / math.sqrt(variance_value)
            )

    np.testing.assert_allclose(scaled_X, expected)


def test_transform_partial_does_not_update_statistics():
    import numpy as np

    from pysad.transform.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit_partial(np.array([1.0, 2.0, 3.0]))

    first = scaler.transform_partial(np.array([1.0, 2.0, 3.0]))
    scaler.transform_partial(np.array([4.0, 5.0, 6.0]))
    second = scaler.transform_partial(np.array([1.0, 2.0, 3.0]))

    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("value", [0.0, -3.0, 100.0])
@pytest.mark.parametrize("num_features", [1, 3, 10])
def test_constant_feature_scales_to_zeros(value, num_features):
    import numpy as np

    from pysad.transform.preprocessing import StandardScaler

    X = np.full(num_features, value)
    scaler = StandardScaler()
    scaler.fit_partial(X)

    with np.errstate(all="raise"):
        scaled_X = scaler.transform_partial(X)

    np.testing.assert_array_equal(scaled_X, np.zeros(num_features))
    np.testing.assert_array_equal(X, np.full(num_features, value))


def test_transform_partial_before_fit_returns_input_unchanged():
    import numpy as np

    from pysad.transform.preprocessing import StandardScaler

    X = np.array([1.0, -2.0, 3.0])

    np.testing.assert_array_equal(StandardScaler().transform_partial(X), X)


def test_single_feature_first_instance_scales_to_zero():
    import numpy as np

    from pysad.transform.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaled = scaler.fit_transform_partial(np.array([7.0]))

    np.testing.assert_array_equal(scaled, np.array([0.0]))
