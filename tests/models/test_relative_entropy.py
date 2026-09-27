import pytest


def _reference_fit_scores(x, min_val, max_val, num_bins=5, window_size=52):
    """Loop-based reference for `RelativeEntropy.fit_score_partial`: NAB's `handleRecord`
    (nab/detectors/relative_entropy/relative_entropy_detector.py), adapted to take a plain float
    per record and return a scalar score, with the paper's quantizer (Fig. 1, steps 3-4b: bucket
    B = ceil((u - min_val) / stepSize) for u clipped to [min_val, max_val], one bin per bucket
    1..num_bins), and the relative entropy written out as in the paper instead of calling
    scipy.stats.entropy. As in NAB, a window ends at every value from the `window_size`-th on.

    Returns:
        tuple: `(scores, c, P)`, the score of every value, and the count and histogram of every
            learned hypothesis.
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

    def histogram(window):
        counts = [0] * N_bins
        for u in window:
            u = min(max(u, min_val), max_val)
            # min_val is in the first bucket; round-off may put max_val one bucket too high.
            B = min(max(math.ceil((u - min_val) / stepSize), 1), N_bins)
            counts[B - 1] += 1
        return np.array(counts) / len(window)

    def relative_entropy(p, q):  # D(p || q) = sum_k p_k log(p_k / q_k)
        total = 0.0
        for p_k, q_k in zip(p, q):
            if p_k > 0:
                if q_k == 0:
                    return float("inf")
                total += p_k * math.log(p_k / q_k)
        return total

    def get_agreement_hypothesis(P_hat):
        index = -1
        minEntropy = float("inf")
        for i in range(m):
            entropy = 2 * W * relative_entropy(P_hat, P[i])
            if entropy < T and entropy < minEntropy:
                minEntropy = entropy
                index = i
        return index

    scores = []
    for value in x:
        anomalyScore = 0.0
        util.append(value)
        if stepSize != 0.0 and len(util) >= W:
            P_hat = histogram(util[-W:])
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

    return scores, c, P


@pytest.mark.parametrize("window_size", [1, 2, 3, 52])
def test_relative_entropy_matches_reference(window_size):
    from pysad.models import RelativeEntropy
    import numpy as np

    # Spread over several buckets, so that window histograms vary and some test statistics
    # land near the threshold.
    rng = np.random.default_rng(0)
    x = rng.normal(0.5, 0.15, 500)
    x[200:220] = rng.normal(0.9, 0.05, 20)  # regime shift
    x = np.clip(x, 0, 1)

    reference_scores, reference_c, reference_P = _reference_fit_scores(x, min_val=0.0, max_val=1.0, window_size=window_size)

    model = RelativeEntropy(min_val=0.0, max_val=1.0, window_size=window_size, step=1)
    scores = [model.fit_score_partial(np.array([v])) for v in x]

    assert scores == reference_scores
    assert model.c == reference_c
    np.testing.assert_array_equal(model.P, np.array(reference_P))


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

    model = RelativeEntropy(min_val=0.0, max_val=1.0, step=1)
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

    model = RelativeEntropy(min_val=0.0, max_val=1.0, step=1)
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

    model_a = RelativeEntropy(min_val=0.0, max_val=1.0, step=1)
    scores_a = []
    for v in x:
        xi = np.array([v])
        scores_a.append(model_a.score_partial(xi))
        model_a.fit_partial(xi)

    model_b = RelativeEntropy(min_val=0.0, max_val=1.0, step=1)
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
    model = RelativeEntropy(min_val=0.0, max_val=1.0, window_size=window_size, step=1)

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
    # ceil((max_val - min_val) / stepSize) evaluate to num_bins + 1, one level past the levels
    # 1..num_bins that _histogram maps to bins 0..num_bins - 1; clipping the level must put
    # max_val in the top bin.
    window_size = 10
    model = RelativeEntropy(min_val=0.0, max_val=100.0, num_bins=29, window_size=window_size, step=1)
    assert np.ceil((model.max_val - model.min_val) / model.stepSize) == model.N_bins + 1

    m_history = []
    for _ in range(30):
        model.fit_score_partial(np.array([150.0]))
        m_history.append(model.m)

    assert not np.isnan(model.P).any()
    assert len(set(m_history[-window_size:])) == 1

    histogram = model._histogram([100.0] * window_size)
    np.testing.assert_array_equal(histogram, np.eye(model.N_bins)[-1])


def test_histogram_gives_each_bucket_its_own_bin():
    from pysad.models import RelativeEntropy
    import numpy as np

    model = RelativeEntropy(min_val=0, max_val=100, num_bins=5, window_size=50)

    # Ten values in each of the five buckets of width 20: (0, 20], (20, 40], (40, 60],
    # (60, 80], (80, 100].
    window = np.arange(1, 100, 2)  # 1, 3, ..., 99
    np.testing.assert_allclose(model._histogram(list(window)), [0.2] * 5)


def test_histogram_buckets_are_closed_on_the_right():
    from pysad.models import RelativeEntropy
    import numpy as np

    model = RelativeEntropy(min_val=0, max_val=100, num_bins=5, window_size=5)

    # The paper's level B = ceil((u - min_val) / stepSize) makes the buckets (0, 20], (20, 40],
    # (40, 60], (60, 80], (80, 100]: a value on an edge belongs to the bucket below it, and a
    # value just above an edge to the bucket above it.
    np.testing.assert_array_equal(model._histogram([20.0, 40.0, 60.0, 80.0, 100.0]), [0.2] * 5)
    np.testing.assert_array_equal(model._histogram([20.5, 40.5, 60.5, 80.5, 80.5]), [0.0, 0.2, 0.2, 0.2, 0.4])


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
    scores_top_shift = RelativeEntropy(min_val=0, max_val=100, num_bins=5, window_size=50, step=1).fit_score(
        x_top_shift.reshape(-1, 1)
    )

    x_lower_shift = np.r_[np.full(200, 50.0), np.full(200, 70.0)]
    scores_lower_shift = RelativeEntropy(min_val=0, max_val=100, num_bins=5, window_size=50, step=1).fit_score(
        x_lower_shift.reshape(-1, 1)
    )

    assert scores_top_shift.sum() > 0
    assert scores_top_shift.sum() == scores_lower_shift.sum()


def test_relative_entropy_default_step_tests_non_overlapping_windows():
    from pysad.models import RelativeEntropy
    import numpy as np

    rng = np.random.default_rng(0)
    x = np.clip(rng.normal(50, 10, 52 * 20), 0, 100)  # 20 non-overlapping windows of W=52

    model = RelativeEntropy(min_val=0, max_val=100, num_bins=5, window_size=52)
    model.fit_score(x.reshape(-1, 1))

    assert sum(model.c) == 20


def test_relative_entropy_step_one_reproduces_nab_sliding_windows():
    from pysad.models import RelativeEntropy
    import numpy as np

    rng = np.random.default_rng(0)
    n = 52 * 20
    x = np.clip(rng.normal(50, 10, n), 0, 100)

    model = RelativeEntropy(min_val=0, max_val=100, num_bins=5, window_size=52, step=1)
    model.fit_score(x.reshape(-1, 1))

    assert sum(model.c) == n - model.W + 1


def test_relative_entropy_values_that_do_not_close_a_window_score_zero():
    from pysad.models import RelativeEntropy
    import numpy as np

    window_size = 5
    model = RelativeEntropy(min_val=0.0, max_val=1.0, window_size=window_size)  # step defaults to window_size

    rng = np.random.default_rng(5)
    x = np.clip(rng.normal(0.5, 0.05, 23), 0, 1)

    scores = [model.fit_score_partial(np.array([v])) for v in x]

    # Only the value that closes a window (every window_size-th value here) can score
    # nonzero; every other value must score 0.0.
    for i, score in enumerate(scores, start=1):
        if i % window_size != 0:
            assert score == 0.0


@pytest.mark.parametrize("method", ["fit_partial", "score_partial", "fit_score_partial"])
@pytest.mark.parametrize("num_fitted", [3, 6])  # NaN would not fill / would close a window
@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_relative_entropy_rejects_nan_without_changing_the_model(method, num_fitted):
    from pysad.models import RelativeEntropy
    import numpy as np

    model = RelativeEntropy(min_val=0.0, max_val=1.0, window_size=5)
    for v in [0.1, 0.3, 0.5, 0.7, 0.9, 0.9, 0.9, 0.9, 0.9][:num_fitted]:
        model.fit_partial(np.array([v]))
    util_before, P_before, c_before, m_before = list(model.util), model.P.copy(), list(model.c), model.m

    with pytest.raises(ValueError, match="RelativeEntropy does not accept NaN values"):
        getattr(model, method)(np.array([np.nan]))

    assert model.util == util_before
    np.testing.assert_array_equal(model.P, P_before)
    assert model.c == c_before
    assert model.m == m_before

    # The rejected value leaves nothing behind that breaks later windows.
    for _ in range(10):
        model.fit_score_partial(np.array([0.9]))
    assert len(model.util) == num_fitted + 10


@pytest.mark.parametrize("step", [0, -1, 1.5, "1", True])
def test_relative_entropy_invalid_step_raises(step):
    from pysad.models import RelativeEntropy

    with pytest.raises(ValueError, match="step must be"):
        RelativeEntropy(min_val=0.0, max_val=1.0, step=step)


@pytest.mark.parametrize("window_size", [0, -1, 52.0, None, "52", True])
def test_relative_entropy_invalid_window_size_raises(window_size):
    from pysad.models import RelativeEntropy

    # The error names window_size, not the step resolved from it.
    with pytest.raises(ValueError, match="window_size must be an int >= 1"):
        RelativeEntropy(min_val=0.0, max_val=1.0, window_size=window_size)


def test_relative_entropy_accepts_numpy_integer_window_size_and_step():
    from pysad.models import RelativeEntropy
    import numpy as np

    for integer_type in (np.int32, np.int64, np.uint16):
        model = RelativeEntropy(min_val=0.0, max_val=1.0, window_size=integer_type(52))
        assert type(model.W) is int and model.W == 52
        assert type(model.step) is int and model.step == 52

        model = RelativeEntropy(min_val=0.0, max_val=1.0, window_size=integer_type(52), step=integer_type(1))
        assert type(model.step) is int and model.step == 1

    # e.g. window sizes from a grid built with np.arange
    rng = np.random.default_rng(0)
    x = np.clip(rng.normal(50, 10, 52 * 20), 0, 100)
    model = RelativeEntropy(min_val=0, max_val=100, num_bins=5, window_size=np.arange(52, 53)[0])
    model.fit_score(x.reshape(-1, 1))

    assert sum(model.c) == 20


@pytest.mark.parametrize("window_size", [1, 2, 3])
@pytest.mark.parametrize("step", [1, None])
def test_relative_entropy_small_windows_score_partial_matches_fit_score_partial(window_size, step):
    from pysad.models import RelativeEntropy
    import numpy as np

    rng = np.random.default_rng(4)
    x = np.clip(rng.normal(0.5, 0.05, 30), 0, 1)

    kwargs = dict(min_val=0.0, max_val=1.0, window_size=window_size, step=step)

    model_a = RelativeEntropy(**kwargs)
    scores_a = []
    for v in x:
        xi = np.array([v])
        scores_a.append(model_a.score_partial(xi))
        model_a.fit_partial(xi)

    model_b = RelativeEntropy(**kwargs)
    scores_b = [model_b.fit_score_partial(np.array([v])) for v in x]

    assert scores_a == scores_b
