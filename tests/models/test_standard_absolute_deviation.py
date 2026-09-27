
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


def test_standard_absolute_deviation_score_values():
    from pysad.models import StandardAbsoluteDeviation
    import numpy as np
    from numpy.testing import assert_allclose

    values = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=float)
    X = values.reshape(-1, 1)

    def expected_scores(statistic, absolute=True):
        scores = []
        for i, value in enumerate(values, start=1):
            seen = values[:i]
            center = statistic(seen)
            std = np.std(seen, ddof=0)
            score = (value - center) / (std + 1e-10)
            scores.append(abs(score) if absolute else score)
        return np.asarray(scores)

    mean_scores = StandardAbsoluteDeviation(
        subtracted_statistic="mean"
    ).fit_score(X)
    assert mean_scores[0] == 0.0
    assert_allclose(mean_scores, expected_scores(np.mean))

    median_scores = StandardAbsoluteDeviation(
        subtracted_statistic="median"
    ).fit_score(X)
    assert median_scores[0] == 0.0
    assert_allclose(median_scores, expected_scores(np.median))

    signed_scores = StandardAbsoluteDeviation(
        subtracted_statistic="mean",
        absolute=False,
    ).fit_score(X)
    assert signed_scores[0] == 0.0
    assert signed_scores[1] < 0.0
    assert_allclose(signed_scores, expected_scores(np.mean, absolute=False))


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
