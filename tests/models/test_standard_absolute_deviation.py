
def test_standard_absolute_deviation():
    from pysad.models import StandardAbsoluteDeviation
    import numpy as np
    from numpy.testing import assert_raises
    from pysad.utils import fix_seed

    fix_seed(61)
    X = np.random.rand(150, 1)

    model = StandardAbsoluteDeviation(subtracted_statistic="mean")
    model = model.fit(X)
    y_pred = model.score(X)
    assert y_pred.shape == (X.shape[0],)

    model = StandardAbsoluteDeviation(subtracted_statistic="median")
    model = model.fit(X)
    y_pred = model.score(X)
    assert y_pred.shape == (X.shape[0],)

    with assert_raises(ValueError):
        StandardAbsoluteDeviation(subtracted_statistic="asd")

    with assert_raises(ValueError):
        StandardAbsoluteDeviation(subtracted_statistic=None)


def test_absolute_deviation_rejects_multivariate_input():
    from pysad.models import MedianAbsoluteDeviation, StandardAbsoluteDeviation
    import numpy as np
    from numpy.testing import assert_raises

    for model in [MedianAbsoluteDeviation(), StandardAbsoluteDeviation()]:
        with assert_raises(ValueError):
            model.fit_partial(np.array([0.1, 0.2]))


def test_substracted_statistic_deprecation():
    """Old spelling still works but emits FutureWarning."""
    import warnings
    from pysad.models import StandardAbsoluteDeviation
    import numpy as np
    from pysad.utils import fix_seed

    fix_seed(61)
    X = np.random.rand(150, 1)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model = StandardAbsoluteDeviation(substracted_statistic="median")
        assert len(w) == 1
        assert issubclass(w[0].category, FutureWarning)
        assert "substracted_statistic" in str(w[0].message)

    model = model.fit(X)
    y_pred = model.score(X)
    assert y_pred.shape == (X.shape[0],)


def test_both_spellings_raises():
    """Passing both old and new spelling at once is an error."""
    import pytest
    from pysad.models import StandardAbsoluteDeviation

    with pytest.raises(TypeError, match="Cannot specify both"):
        StandardAbsoluteDeviation(
            subtracted_statistic="mean",
            substracted_statistic="median",
        )
