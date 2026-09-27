import pytest


def test_relative_entropy_issue_example():
    from pysad.models import RelativeEntropy
    from sklearn.metrics import roc_auc_score
    import numpy as np

    rng = np.random.default_rng(0)
    x = rng.normal(0.5, 0.05, 3000)
    x[2000:2100] = rng.normal(0.9, 0.05, 100)  # anomalous segment
    x = np.clip(x, 0, 1)
    y = np.zeros(3000, dtype=int)
    y[2000:2100] = 1

    model = RelativeEntropy(min_val=0.0, max_val=1.0)
    scores = np.array([model.fit_score_partial(np.array([v])) for v in x])

    assert set(np.unique(scores)) <= {0.0, 1.0}
    assert roc_auc_score(y[52:], scores[52:]) > 0.5

    test_scores = model.score(np.array([[0.5], [0.9], [0.1]]))

    # A learned model must not return the same score for these three different points.
    assert len(set(test_scores.tolist())) > 1


def test_relative_entropy_score_partial_has_no_side_effects():
    from pysad.models import RelativeEntropy
    from pysad.utils import fix_seed
    import numpy as np
    import copy

    fix_seed(1)
    rng = np.random.default_rng(1)
    x = rng.normal(0.5, 0.05, 300)
    x = np.clip(x, 0, 1)

    model = RelativeEntropy(min_val=0.0, max_val=1.0)
    for v in x:
        model.fit_partial(np.array([v]))

    util_before = copy.deepcopy(model.util)
    P_before = copy.deepcopy(model.P)
    c_before = copy.deepcopy(model.c)
    m_before = model.m

    score1 = model.score_partial(np.array([0.7]))
    score2 = model.score_partial(np.array([0.7]))

    assert score1 == score2
    assert model.util == util_before
    assert len(model.P) == len(P_before)
    for hypothesis, hypothesis_before in zip(model.P, P_before):
        np.testing.assert_array_equal(hypothesis, hypothesis_before)
    assert model.c == c_before
    assert model.m == m_before


def test_relative_entropy_score_then_fit_matches_fit_score_partial():
    from pysad.models import RelativeEntropy
    import numpy as np

    rng = np.random.default_rng(2)
    x = rng.normal(0.5, 0.05, 300)
    x[150:160] = rng.normal(0.9, 0.05, 10)
    x = np.clip(x, 0, 1)

    model_a = RelativeEntropy(min_val=0.0, max_val=1.0)
    scores_a = []
    for v in x:
        xi = np.array([v])
        scores_a.append(model_a.score_partial(xi))
        model_a.fit_partial(xi)

    model_b = RelativeEntropy(min_val=0.0, max_val=1.0)
    scores_b = [model_b.fit_score_partial(np.array([v])) for v in x]

    assert scores_a == scores_b


def test_relative_entropy_constant_stream_scores_zero():
    from pysad.models import RelativeEntropy
    import numpy as np

    model = RelativeEntropy(min_val=0.5, max_val=0.5)

    scores = [model.fit_score_partial(np.array([0.5])) for _ in range(100)]

    assert all(score == 0.0 for score in scores)


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_relative_entropy_out_of_range_values_do_not_produce_nan_hypotheses():
    from pysad.models import RelativeEntropy
    import numpy as np

    window_size = 10
    model = RelativeEntropy(min_val=0.0, max_val=1.0, window_size=window_size)

    # In-range warm-up so at least one hypothesis is learned before the stream
    # goes out of range.
    rng = np.random.default_rng(3)
    for v in rng.normal(0.5, 0.05, 50):
        model.fit_score_partial(np.array([np.clip(v, 0, 1)]))

    # More than a window's worth of values above max_val.
    m_history = []
    for v in rng.normal(5.0, 0.1, 30):
        model.fit_score_partial(np.array([v]))
        m_history.append(model.m)

    assert not np.isnan(model.P).any()
    # Once the out-of-range window fills, every further out-of-range window quantizes
    # to the same top bin and must agree with an already-learned hypothesis, so `m`
    # stops changing well before the stream ends: it doesn't grow with every record.
    assert len(set(m_history[-window_size:])) == 1


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_relative_entropy_max_val_round_off_lands_in_top_bin():
    from pysad.models import RelativeEntropy
    import numpy as np

    # With these exact (min_val, max_val, num_bins), floating-point round-off makes
    # ceil((max_val - min_val) / stepSize) evaluate to num_bins + 1, one bin past the
    # histogram's (0, num_bins) range.
    window_size = 10
    model = RelativeEntropy(min_val=0.0, max_val=100.0, num_bins=29, window_size=window_size)

    m_history = []
    for _ in range(30):
        model.fit_score_partial(np.array([150.0]))
        m_history.append(model.m)

    assert not np.isnan(model.P).any()
    assert len(set(m_history[-window_size:])) == 1

    histogram = model._histogram([100.0] * window_size)
    assert np.isfinite(histogram).all()


def test_histogram_gives_each_bucket_its_own_bin():
    from pysad.models import RelativeEntropy
    import numpy as np

    model = RelativeEntropy(min_val=0, max_val=100, num_bins=5, window_size=50)

    # Ten values in each of the five buckets of width 20: (0, 20], (20, 40], (40, 60],
    # (60, 80], (80, 100].
    window = np.arange(1, 100, 2)  # 1, 3, ..., 99
    np.testing.assert_allclose(model._histogram(list(window)), [0.2] * 5)


def test_histogram_puts_min_val_in_first_bucket_and_max_val_in_last():
    from pysad.models import RelativeEntropy

    model = RelativeEntropy(min_val=0, max_val=100, num_bins=5, window_size=50)

    histogram = model._histogram([model.min_val] * 25 + [model.max_val] * 25)

    assert histogram[0] == 0.5
    assert histogram[-1] == 0.5
    assert histogram[1:-1].sum() == 0.0


def test_relative_entropy_flags_jump_between_top_two_buckets():
    from pysad.models import RelativeEntropy
    import numpy as np

    # A level shift from the 4th bucket (60, 80] to the 5th bucket (80, 100] must be
    # flagged just as often as a shift of the same size between two lower buckets.
    x_top_shift = np.r_[np.full(200, 70.0), np.full(200, 90.0)]
    scores_top_shift = RelativeEntropy(min_val=0, max_val=100, num_bins=5, window_size=50).fit_score(
        x_top_shift.reshape(-1, 1)
    )

    x_lower_shift = np.r_[np.full(200, 50.0), np.full(200, 70.0)]
    scores_lower_shift = RelativeEntropy(min_val=0, max_val=100, num_bins=5, window_size=50).fit_score(
        x_lower_shift.reshape(-1, 1)
    )

    assert scores_top_shift.sum() > 0
    assert scores_top_shift.sum() == scores_lower_shift.sum()
