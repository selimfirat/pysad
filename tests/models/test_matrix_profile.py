import math

import numpy as np
import pytest

M = 20
HISTORY = 300


def _series():
    """A noisy sine of period 50 with a flat stretch at 700 and a ramp at 1000."""
    rng = np.random.default_rng(0)
    values = np.sin(2 * np.pi * np.arange(1500) / 50) + 0.1 * rng.normal(size=1500)
    values[700:720] = 0.3
    values[1000:1020] += np.linspace(0.0, 2.0, 20)
    return values


# STUMPY warns that m is long for the first 30 values, which this test needs.
@pytest.mark.filterwarnings("ignore:The window size")
def test_scores_match_stumpy_stumpi(monkeypatch):
    stumpy = pytest.importorskip("stumpy")

    from pysad.models import MatrixProfile

    values = _series()
    scores = MatrixProfile(window_length=M, history_size=HISTORY).fit_score(values.reshape(-1, 1))

    # The paper's exclusion zone of m / 2. STUMPY uses m / 4 by default.
    monkeypatch.setattr(stumpy.config, "STUMPY_EXCL_ZONE_DENOM", 2)
    first = M + math.ceil(M / 2)
    # The history grows to HISTORY values, and then it slides, as with egress.
    growing = stumpy.stumpi(values[:first], M, egress=False)
    sliding = stumpy.stumpi(values[: HISTORY + 1], M, egress=True)
    expected = []
    for i in range(first, len(values)):
        stream = growing if i <= HISTORY else sliding
        stream.update(values[i])
        expected.append(stream.left_P_[-1])

    np.testing.assert_array_equal(scores[:first], 0.0)
    np.testing.assert_allclose(scores[first:], expected, rtol=1e-9, atol=1e-9)


def test_inserted_discords_get_the_highest_scores():
    from pysad.models import MatrixProfile

    scores = MatrixProfile(window_length=M, history_size=HISTORY).fit_score(
        _series().reshape(-1, 1)
    )

    # Scores once the history is full. Each top score ends a subsequence that overlaps a discord.
    top = HISTORY + np.argsort(scores[HISTORY:])[-5:]
    assert all(700 <= i < 720 + M or 1000 <= i < 1020 + M for i in top)


def test_score_partial_does_not_change_the_model():
    from pysad.models import MatrixProfile

    values = _series()[:200]
    model = MatrixProfile(window_length=M, history_size=100).fit(values[:150].reshape(-1, 1))
    scores = model.score(values[150:].reshape(-1, 1))

    np.testing.assert_array_equal(model.score(values[150:].reshape(-1, 1)), scores)


def test_constant_subsequences():
    from pysad.models import MatrixProfile

    # A constant query is at distance 0 from a constant earlier subsequence, and at
    # sqrt(window_length) from any other subsequence.
    model = MatrixProfile(window_length=4, history_size=10).fit(np.ones((9, 1)))
    assert model.score_partial(np.array([1.0])) == 0.0

    model = MatrixProfile(window_length=4, history_size=10).fit(
        np.array([0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(-1, 1)
    )
    assert model.score_partial(np.array([0.0])) == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("params", "error"),
    [
        ({"window_length": 1}, ValueError),
        ({"window_length": 2.0}, TypeError),
        ({"window_length": 10, "history_size": 14}, ValueError),
    ],
)
def test_invalid_parameters(params, error):
    from pysad.models import MatrixProfile

    with pytest.raises(error):
        MatrixProfile(**params)


def test_rejects_multivariate_instances():
    from pysad.models import MatrixProfile

    with pytest.raises(ValueError, match="univariate"):
        MatrixProfile(window_length=4).fit_partial(np.array([1.0, 2.0]))
