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


# Scores of stumpy.stumpi 1.14.1 (STUMPY, BSD-3-Clause) for _short_series() with m = 4, made with
# stumpy.config.STUMPY_EXCL_ZONE_DENOM = 2 for the paper's exclusion zone of m / 2. One
# stumpi(values[:6], 4, egress=False) object is updated with values 6 to 12, and then one
# stumpi(values[:13], 4, egress=True) object is updated with values 13 to 29, so the history grows
# to 12 values and then slides. Each expected score is left_P_[-1] after the update.
STUMPI_SCORES = [
    3.862628621434302,
    2.5065301044117017,
    2.555994107889105,
    1.9880978905269473,
    1.2656422887911747,
    0.7956759366036217,
    0.034598868240575766,
    0.07872514234477174,
    0.01743094828767016,
    0.01126649178603427,
    0.018262351910034537,
    0.08463730036566065,
    0.032038476370741024,
    0.012383181599794268,
    0.6608079415719937,
    0.7615097814158512,
    1.8210664940128762,
    0.4263528320631701,
    0.3516080126001669,
    0.4370153376169141,
    0.08745135522233119,
    0.030865308300811023,
    0.012264151234654592,
    0.534941348201845,
]


def test_scores_match_stumpy_stumpi():
    from pysad.models import MatrixProfile

    values = np.sin(0.7 * np.arange(30))
    values[20:23] += 1.0
    scores = MatrixProfile(window_length=4, history_size=12).fit_score(values.reshape(-1, 1))

    first = 4 + math.ceil(4 / 2)
    np.testing.assert_array_equal(scores[:first], 0.0)
    np.testing.assert_allclose(scores[first:], STUMPI_SCORES, rtol=1e-9, atol=1e-9)


def test_inserted_discords_get_the_highest_scores():
    from pysad.models import MatrixProfile

    scores = MatrixProfile(window_length=M, history_size=HISTORY).fit_score(
        _series().reshape(-1, 1)
    )

    # Scores once the history is full. Each top score ends a subsequence that overlaps a discord.
    top = HISTORY + np.argsort(scores[HISTORY:])[-5:]
    assert all(700 <= i < 720 + M or 1000 <= i < 1020 + M for i in top)


def test_constant_offset_does_not_change_scores():
    """Adding a constant to the series must not change the z-normalized scores."""
    from pysad.models import DAMP, MatrixProfile

    rng = np.random.default_rng(0)
    values = np.sin(2 * np.pi * np.arange(600) / 50) + 0.1 * rng.normal(size=600)
    values[400:420] += 1.5  # the discord

    for model in (MatrixProfile, DAMP):
        base = model(window_length=M, history_size=HISTORY).fit_score(values.reshape(-1, 1))
        for offset in (1e6, 1e7):
            shifted = model(window_length=M, history_size=HISTORY).fit_score(
                (values + offset).reshape(-1, 1)
            )
            np.testing.assert_allclose(shifted, base, atol=1e-6)
            assert shifted.argmax() == base.argmax()


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
