import numpy as np
from numpy.testing import assert_allclose

from pysad.models import MedianAbsoluteDeviation


VALUES = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=float)

# Hand-computed signed scores for VALUES with b=1.4826: at each step the
# running median and the running median of absolute deviations are taken
# over the values seen so far (including the current one), and the score is
# (value - median) / (b * mad). For example, at the sixth value (9) the
# median of [3, 1, 4, 1, 5, 9] is 3.5, the deviations so far are
# [0, 1, 1, 1, 2, 5.5] with median 1, so the score is 5.5 / 1.4826 = 3.7097.
SIGNED_SCORES = np.array(
    [0.0, -1.34898, 0.67449, -0.67449, 1.34898, 3.70970, -0.67449, 1.68623]
)


def expected_scores(values, b=1.4826, absolute=True):
    deviations, scores = [], []
    for t in range(1, len(values) + 1):
        median = np.median(values[:t])
        deviations.append(abs(values[t - 1] - median))
        mad = np.median(deviations)
        score = (values[t - 1] - median) / (b * mad + 1e-10)
        scores.append(abs(score) if absolute else score)
    return np.asarray(scores)


def test_median_absolute_deviation_score_values():
    scores = MedianAbsoluteDeviation().fit_score(VALUES.reshape(-1, 1))

    assert_allclose(scores, np.abs(SIGNED_SCORES), rtol=1e-5)
    assert_allclose(scores, expected_scores(VALUES))
    assert scores[0] == 0.0


def test_median_absolute_deviation_signed_score_values():
    scores = MedianAbsoluteDeviation(absolute=False).fit_score(
        VALUES.reshape(-1, 1)
    )

    assert_allclose(scores, SIGNED_SCORES, rtol=1e-5)
    assert_allclose(scores, expected_scores(VALUES, absolute=False))
    assert np.any(scores < 0)


def test_median_absolute_deviation_b_scales_scores():
    default_scores = MedianAbsoluteDeviation().fit_score(VALUES.reshape(-1, 1))
    unit_scores = MedianAbsoluteDeviation(b=1.0).fit_score(VALUES.reshape(-1, 1))
    nonzero = default_scores != 0

    assert_allclose(unit_scores[nonzero], default_scores[nonzero] * 1.4826)
