import pytest


def _nab_reference_scores(x, min_val, max_val, num_bins=5, window_size=52):
    """Reference implementation of NAB's `handleRecord`
    (nab/detectors/relative_entropy/relative_entropy_detector.py), adapted to
    take a plain float per record and return a scalar score instead of a
    one-element list.
    """
    import math
    import numpy as np
    from scipy import stats

    N_bins = num_bins
    W = window_size
    T = stats.chi2.isf(0.01, N_bins - 1)
    c_th = 1
    stepSize = (max_val - min_val) / N_bins

    util = []
    P = []
    c = []
    m = 0

    def get_agreement_hypothesis(P_hat):
        index = -1
        minEntropy = float("inf")
        for i in range(m):
            entropy = 2 * W * stats.entropy(P_hat, P[i])
            if entropy < T and entropy < minEntropy:
                minEntropy = entropy
                index = i
        return index

    scores = []
    for value in x:
        anomalyScore = 0.0
        util.append(value)
        if stepSize != 0.0:
            if len(util) >= W:
                util_current = util[-W:]
                B_current = [math.ceil((v - min_val) / stepSize) for v in util_current]
                P_hat = np.histogram(B_current, bins=N_bins, range=(0, N_bins), density=True)[0]
                if m == 0:
                    P.append(P_hat)
                    c.append(1)
                    m = 1
                else:
                    index = get_agreement_hypothesis(P_hat)
                    if index != -1:
                        c[index] += 1
                        if c[index] <= c_th:
                            anomalyScore = 1.0
                    else:
                        anomalyScore = 1.0
                        P.append(P_hat)
                        c.append(1)
                        m += 1
        scores.append(anomalyScore)

    return scores


@pytest.mark.parametrize("window_size", [1, 2, 3, 52])
def test_relative_entropy_matches_nab_reference(window_size):
    from pysad.models import RelativeEntropy
    from pysad.utils import fix_seed
    import numpy as np

    fix_seed(0)
    rng = np.random.default_rng(0)
    x = rng.normal(0.5, 0.05, 500)
    x[200:220] = rng.normal(0.9, 0.05, 20)  # regime shift
    x = np.clip(x, 0, 1)

    reference_scores = _nab_reference_scores(x, min_val=0.0, max_val=1.0, window_size=window_size)

    model = RelativeEntropy(min_val=0.0, max_val=1.0, window_size=window_size)
    scores = [model.fit_score_partial(np.array([v])) for v in x]

    assert scores == reference_scores


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
