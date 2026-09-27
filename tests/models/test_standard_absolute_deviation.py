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


def _expected_scores(values, statistic="mean", absolute=True):
    import numpy as np

    values = np.asarray(values, dtype=float)
    scores = []
    for t in range(1, len(values) + 1):
        window = values[:t]
        center = np.mean(window) if statistic == "mean" else np.median(window)
        std = np.std(window, ddof=0)
        score = (values[t - 1] - center) / (std + 1e-10)
        scores.append(abs(score) if absolute else score)
    return np.asarray(scores)


def test_standard_absolute_deviation_score_values_mean():
    from pysad.models import StandardAbsoluteDeviation
    import numpy as np
    from numpy.testing import assert_allclose

    values = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=float)
    X = values.reshape(-1, 1)

    model = StandardAbsoluteDeviation(subtracted_statistic="mean")
    scores = model.fit_score(X)

    expected = _expected_scores(values, statistic="mean", absolute=True)
    assert_allclose(scores, expected)
    assert scores[0] == expected[0]


def test_standard_absolute_deviation_score_values_median():
    from pysad.models import StandardAbsoluteDeviation
    import numpy as np
    from numpy.testing import assert_allclose

    values = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=float)
    X = values.reshape(-1, 1)

    model = StandardAbsoluteDeviation(subtracted_statistic="median")
    scores = model.fit_score(X)

    expected = _expected_scores(values, statistic="median", absolute=True)
    assert_allclose(scores, expected)


def test_standard_absolute_deviation_keeps_sign_when_absolute_false():
    from pysad.models import StandardAbsoluteDeviation
    import numpy as np
    from numpy.testing import assert_allclose

    values = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=float)
    X = values.reshape(-1, 1)

    model = StandardAbsoluteDeviation(subtracted_statistic="mean", absolute=False)
    scores = model.fit_score(X)

    expected = _expected_scores(values, statistic="mean", absolute=False)
    assert_allclose(scores, expected)
    assert np.any(scores < 0)


def test_standard_absolute_deviation_first_score_is_zero():
    from pysad.models import StandardAbsoluteDeviation
    import numpy as np

    X = np.array([[3.0]])
    score = StandardAbsoluteDeviation().fit_score(X)[0]
    # Single observation: deviation from its own mean is 0; variance is 0.
    assert score == 0.0
