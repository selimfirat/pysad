def test_standard_absolute_deviation():
    import numpy as np
    from numpy.testing import assert_raises

    from pysad.models import StandardAbsoluteDeviation
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
    import numpy as np
    from numpy.testing import assert_raises

    from pysad.models import MedianAbsoluteDeviation, StandardAbsoluteDeviation

    for model in [MedianAbsoluteDeviation(), StandardAbsoluteDeviation()]:
        with assert_raises(ValueError):
            model.fit_partial(np.array([0.1, 0.2]))


def _expected_scores(values, statistic="mean", absolute=True, window_size=None):
    import numpy as np

    values = np.asarray(values, dtype=float)
    scores = []
    for t in range(len(values)):
        # fit_score_partial scores each value against the values before it.
        window = values[:t]
        if len(window) == 0:
            scores.append(0.0)
            continue
        if statistic == "mean":
            center = np.mean(window)
        else:
            median_window = window if window_size is None else window[-window_size:]
            center = np.median(median_window)
        std = np.std(window, ddof=0)
        score = (values[t] - center) / (std + 1e-10)
        scores.append(abs(score) if absolute else score)
    return np.asarray(scores)


def test_standard_absolute_deviation_score_values_mean():
    import numpy as np
    from numpy.testing import assert_allclose

    from pysad.models import StandardAbsoluteDeviation

    values = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=float)
    X = values.reshape(-1, 1)

    model = StandardAbsoluteDeviation(subtracted_statistic="mean")
    scores = model.fit_score(X)

    expected = _expected_scores(values, statistic="mean", absolute=True)
    assert_allclose(scores, expected)
    assert scores[0] == expected[0]


def test_standard_absolute_deviation_score_values_median():
    import numpy as np
    from numpy.testing import assert_allclose

    from pysad.models import StandardAbsoluteDeviation

    values = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=float)
    X = values.reshape(-1, 1)

    model = StandardAbsoluteDeviation(subtracted_statistic="median")
    scores = model.fit_score(X)

    expected = _expected_scores(values, statistic="median", absolute=True)
    assert_allclose(scores, expected)


def test_standard_absolute_deviation_keeps_sign_when_absolute_false():
    import numpy as np
    from numpy.testing import assert_allclose

    from pysad.models import StandardAbsoluteDeviation

    values = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=float)
    X = values.reshape(-1, 1)

    model = StandardAbsoluteDeviation(subtracted_statistic="mean", absolute=False)
    scores = model.fit_score(X)

    expected = _expected_scores(values, statistic="mean", absolute=False)
    assert_allclose(scores, expected)
    assert np.any(scores < 0)


def test_standard_absolute_deviation_first_score_is_zero():
    import numpy as np

    from pysad.models import StandardAbsoluteDeviation

    X = np.array([[3.0]])
    score = StandardAbsoluteDeviation().fit_score(X)[0]
    # Single observation: deviation from its own mean is 0; variance is 0.
    assert score == 0.0


def test_standard_absolute_deviation_scores_zero_on_a_constant_stream():
    """#224: the variance went negative after three instances of 0.1, so the score was nan."""
    import numpy as np

    from pysad.models import StandardAbsoluteDeviation

    scores = StandardAbsoluteDeviation().fit_score(np.full((10, 1), 0.1))

    # Not exactly 0: the running mean of 0.1s is off by a rounding error, divided by the 1e-10 floor.
    np.testing.assert_allclose(scores, np.zeros(10), atol=1e-6)


def test_standard_absolute_deviation_median_uses_recent_window():
    import numpy as np
    from numpy.testing import assert_allclose

    from pysad.models import StandardAbsoluteDeviation

    values = np.array([3, 1, 4, 1, 5, 9, 2, 6, 2, 2, -1], dtype=float)
    window_size = 3
    model = StandardAbsoluteDeviation(subtracted_statistic="median", window_size=window_size)

    scores = model.fit_score(values.reshape(-1, 1))

    expected = _expected_scores(values, statistic="median", absolute=True, window_size=window_size)
    assert_allclose(scores, expected)


def test_standard_absolute_deviation_default_median_window_expires_old_values():
    import numpy as np
    from numpy.testing import assert_allclose

    from pysad.models import StandardAbsoluteDeviation

    values = np.concatenate([np.zeros(500), np.full(501, 10.0)])
    model = StandardAbsoluteDeviation(subtracted_statistic="median")

    scores = model.fit_score(values.reshape(-1, 1))

    expected = _expected_scores(values, statistic="median", window_size=500)
    assert_allclose(scores, expected)
    assert scores[1000] == 0.0


def test_standard_absolute_deviation_median_window_of_one():
    import numpy as np
    from numpy.testing import assert_allclose

    from pysad.models import StandardAbsoluteDeviation

    values = np.array([3, 1, 4, 1], dtype=float)
    scores = StandardAbsoluteDeviation(subtracted_statistic="median", window_size=1).fit_score(
        values.reshape(-1, 1)
    )

    assert_allclose(scores, _expected_scores(values, statistic="median", window_size=1))


def test_standard_absolute_deviation_rejects_invalid_window_size():
    import pytest

    from pysad.models import StandardAbsoluteDeviation

    for subtracted_statistic in ["mean", "median"]:
        for window_size in [0, -1]:
            with pytest.raises(ValueError, match="window_size"):
                StandardAbsoluteDeviation(subtracted_statistic, window_size=window_size)


def test_standard_absolute_deviation_mean_ignores_window_size():
    import numpy as np
    from numpy.testing import assert_allclose

    from pysad.models import StandardAbsoluteDeviation

    values = np.array([3, 1, 4, 1, 5, 9], dtype=float)
    scores = StandardAbsoluteDeviation(subtracted_statistic="mean", window_size=1).fit_score(
        values.reshape(-1, 1)
    )

    assert_allclose(scores, _expected_scores(values, statistic="mean"))


def test_standard_absolute_deviation_median_memory_stays_bounded():
    import pickle

    from pysad.models import StandardAbsoluteDeviation

    model = StandardAbsoluteDeviation(subtracted_statistic="median")
    for i in range(1_000):
        model.fit_partial([float(i % 7)])
    short_stream_size = len(pickle.dumps(model))

    for i in range(9_000):
        model.fit_partial([float((i + 1_000) % 7)])
    long_stream_size = len(pickle.dumps(model))

    assert long_stream_size < short_stream_size + 32
