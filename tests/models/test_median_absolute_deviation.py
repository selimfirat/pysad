import numpy as np
import pytest
from numpy.testing import assert_allclose

from pysad.models import MedianAbsoluteDeviation

VALUES = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=float)

# Hand-computed signed scores for VALUES with b=1.4826 and a window longer
# than the stream: fit_score_partial scores each value before it learns it, so
# the median and the MAD (Eq. 7 of the paper) are computed over the values
# before the current one, and the score is (value - median) / (b * MAD + 1e-10).
# For example, at the sixth value (9) the median of [3, 1, 4, 1, 5] is 3, the
# absolute deviations from it are [0, 2, 1, 2, 2] with median 2, so the score is
# 6 / (1.4826 * 2) = 2.023472. The first value has no earlier values and scores
# 0. The second is scored against [3], whose MAD is 0, so it scores -2 / 1e-10.
SIGNED_SCORES = np.array([0.0, -2e10, 1.348982, -1.348982, 2.023472, 2.023472, -0.505868, 1.011736])

# The same with window_size=4. At the sixth value (9) the window is
# [1, 4, 1, 5] with median 2.5, the absolute deviations are
# [1.5, 1.5, 1.5, 2.5] with median 1.5, so the score is 6.5 / 2.2239 = 2.922793.
WINDOW_4_SCORES = np.array(
    [0.0, -2e10, 1.348982, -1.348982, 2.023472, 2.922793, -0.843113, 0.843113]
)


def expected_scores(values, window_size=500, b=1.4826, absolute=True):
    scores = []
    for t in range(len(values)):
        window = values[max(0, t - window_size) : t]
        if len(window) == 0:
            scores.append(0.0)
            continue
        median = np.median(window)
        mad = np.median(np.abs(window - median))
        score = (values[t] - median) / (b * mad + 1e-10)
        scores.append(abs(score) if absolute else score)
    return np.asarray(scores)


def test_median_absolute_deviation_score_values():
    scores = MedianAbsoluteDeviation().fit_score(VALUES.reshape(-1, 1))

    assert_allclose(scores, np.abs(SIGNED_SCORES), rtol=1e-5)
    assert_allclose(scores, expected_scores(VALUES))
    assert scores[0] == 0.0


def test_median_absolute_deviation_signed_score_values():
    scores = MedianAbsoluteDeviation(absolute=False).fit_score(VALUES.reshape(-1, 1))

    assert_allclose(scores, SIGNED_SCORES, rtol=1e-5)
    assert_allclose(scores, expected_scores(VALUES, absolute=False))
    assert np.any(scores < 0)


def test_median_absolute_deviation_window_forgets_old_values():
    model = MedianAbsoluteDeviation(window_size=4, absolute=False)
    scores = model.fit_score(VALUES.reshape(-1, 1))

    assert_allclose(scores, WINDOW_4_SCORES, rtol=1e-5)
    assert_allclose(scores, expected_scores(VALUES, window_size=4, absolute=False))
    assert list(model.window) == [5, 9, 2, 6]


def test_median_absolute_deviation_follows_level_shift():
    rng = np.random.default_rng(0)
    x = np.r_[rng.normal(0, 1, 1000), rng.normal(10, 1, 1000)]
    model = MedianAbsoluteDeviation(window_size=200).fit(x.reshape(-1, 1))

    # The window holds only the new level, so 10 is normal and 6 and 13 are far from it.
    # Over the full history 6 would sit near the median and score about 0.
    assert model.score_partial(np.array([10.0])) < 1
    assert model.score_partial(np.array([13.0])) > 2.5
    assert model.score_partial(np.array([6.0])) > 3


def test_median_absolute_deviation_b_scales_scores():
    default_scores = MedianAbsoluteDeviation().fit_score(VALUES.reshape(-1, 1))
    unit_scores = MedianAbsoluteDeviation(b=1.0).fit_score(VALUES.reshape(-1, 1))

    # The first two values have no MAD above 0, so b does not change their scores.
    assert_allclose(unit_scores[2:], default_scores[2:] * 1.4826)


def test_median_absolute_deviation_invalid_window_size():
    with pytest.raises(ValueError, match="window_size"):
        MedianAbsoluteDeviation(window_size=0)


def test_median_absolute_deviation_scores_0_before_fit():
    assert MedianAbsoluteDeviation().score_partial(np.array([1.0])) == 0.0
