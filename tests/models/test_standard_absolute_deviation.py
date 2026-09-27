
def test_standard_absolute_deviation():
    from pysad.models import StandardAbsoluteDeviation
    import numpy as np
    from numpy.testing import assert_raises
    from pysad.utils import fix_seed

    fix_seed(61)
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


def test_absolute_deviation_rejects_multivariate_input():
    from pysad.models import MedianAbsoluteDeviation, StandardAbsoluteDeviation
    import numpy as np
    from numpy.testing import assert_raises

    for model in [MedianAbsoluteDeviation(), StandardAbsoluteDeviation()]:
        with assert_raises(ValueError):
            model.fit_partial(np.array([0.1, 0.2]))
