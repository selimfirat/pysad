
def test_standard_absolute_deviation():
    from pysad.models.standard_absolute_deviation import StandardAbsoluteDeviation
    import numpy as np
    from numpy.testing import assert_raises

    np.random.seed(61)
    X = np.random.rand(150, 1)

    model = StandardAbsoluteDeviation(substracted_statistic="mean")
    model = model.fit(X)
    y_pred = model.score(X)
    assert y_pred.shape == (X.shape[0],)

    model = StandardAbsoluteDeviation(substracted_statistic="median")
    model = model.fit(X)
    y_pred = model.score(X)
    assert y_pred.shape == (X.shape[0],)

    with assert_raises(ValueError):
        StandardAbsoluteDeviation(substracted_statistic="asd")

    with assert_raises(ValueError):
        StandardAbsoluteDeviation(substracted_statistic=None)
