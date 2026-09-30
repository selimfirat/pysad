import pytest


def _reference_fit_scores(x, min_val, max_val, num_bins=5, window_size=52, step=1, c_th=1):
    """Loop-based reference for `RelativeEntropy.fit_score_partial`: NAB's `handleRecord`
    (nab/detectors/relative_entropy/relative_entropy_detector.py), adapted to take a plain float
    per record and return a scalar score, with the paper's quantizer (Fig. 1, steps 3-4b: bucket
    B = ceil((u - min_val) / stepSize) for u clipped to [min_val, max_val], one bin per bucket
    1..num_bins), the relative entropy written out as in the paper instead of calling
    scipy.stats.entropy, and windows that end at the `window_size`-th value and every `step`
    values after it (`step=1` gives NAB's sliding windows, `step=window_size` the paper's
    non-overlapping ones), with the rarity threshold `c_th` that NAB fixes at 1 as an argument.

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
    stepSize = (max_val - min_val) / N_bins

    util = []
    P = []
    c = []
    m = 0
    next_window_end = W

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
        for p_k, q_k in zip(p, q, strict=True):
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
        if stepSize != 0.0 and len(util) == next_window_end:
            next_window_end += step
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
@pytest.mark.parametrize("step", [1, None, 7])  # 7 divides none of the window sizes
@pytest.mark.parametrize("c_th", [1, 3])
@pytest.mark.parametrize("driver", ["fit_score_partial", "score_partial_then_fit_partial"])
def test_relative_entropy_matches_reference(window_size, step, c_th, driver):
    import numpy as np

    from pysad.models import RelativeEntropy

    # Spread over several buckets, so that window histograms vary and some test statistics
    # land near the threshold.
    rng = np.random.default_rng(0)
    x = rng.normal(0.5, 0.15, 500)
    x[200:220] = rng.normal(0.9, 0.05, 20)  # regime shift
    x = np.clip(x, 0, 1)

    reference_scores, reference_c, reference_P = _reference_fit_scores(
        x,
        min_val=0.0,
        max_val=1.0,
        window_size=window_size,
        step=window_size if step is None else step,
        c_th=c_th,
    )

    # score_partial followed by fit_partial must score and learn exactly as fit_score_partial.
    model = RelativeEntropy(min_val=0.0, max_val=1.0, window_size=window_size, step=step, c_th=c_th)
    scores = []
    for v in x:
        xi = np.array([v])
        if driver == "fit_score_partial":
            scores.append(model.fit_score_partial(xi))
        else:
            scores.append(model.score_partial(xi))
            model.fit_partial(xi)

    assert scores == reference_scores
    assert model.c == reference_c
    np.testing.assert_array_equal(model.P, np.array(reference_P))


def test_relative_entropy_issue_example():
    import numpy as np
    from sklearn.metrics import roc_auc_score

    from pysad.models import RelativeEntropy

    rng = np.random.default_rng(0)
    x = rng.normal(0.5, 0.05, 3000)
    x[2000:2100] = rng.normal(0.9, 0.05, 100)  # anomalous segment
    x = np.clip(x, 0, 1)
    y = np.zeros(3000, dtype=int)
    y[2000:2100] = 1

    # With step=1 every value closes a window, so score() on held-out points after fitting
    # works at any fitted length.
    model = RelativeEntropy(min_val=0.0, max_val=1.0, step=1)
    scores = np.array([model.fit_score_partial(np.array([v])) for v in x])

    assert set(np.unique(scores)) <= {0.0, 1.0}
    assert roc_auc_score(y[52:], scores[52:]) > 0.5

    test_scores = model.score(np.array([[0.5], [0.9], [0.1]]))

    # A learned model must not return the same score for these three different points.
    assert len(set(test_scores.tolist())) > 1


def test_relative_entropy_issue_example_non_overlapping_windows():
    import numpy as np

    from pysad.models import RelativeEntropy

    rng = np.random.default_rng(0)
    x = rng.normal(0.5, 0.05, 3000)
    x[2000:2100] = rng.normal(0.9, 0.05, 100)  # anomalous segment
    x = np.clip(x, 0, 1)

    model = RelativeEntropy(
        min_val=0.0, max_val=1.0, step=52
    )  # the paper's windows: step = window_size
    scores = model.fit_score(x.reshape(-1, 1))

    # Only a value that closes a window (every 52nd value) can score nonzero, and the three
    # windows that overlap the anomalous segment (closed by the 2028th, 2080th and 2132nd
    # values) are flagged.
    flagged = np.flatnonzero(scores)
    assert ((flagged + 1) % 52 == 0).all()
    assert {2027, 2079, 2131} <= set(flagged.tolist())

    # score() on held-out points after fit() scores each point as the value that would follow
    # the fitted ones: it tells the points apart only when that value closes a window, i.e. when
    # the fitted length is one short of a multiple of 52, and returns all 0.0 otherwise.
    points = np.array([[0.5], [0.9], [0.1]])
    model = RelativeEntropy(min_val=0.0, max_val=1.0, step=52).fit(x[:2900].reshape(-1, 1))
    for num_fitted in range(2900, 3000):
        test_scores = model.score(points).tolist()
        if num_fitted % 52 == 51:  # 2911 and 2963
            assert len(set(test_scores)) > 1
        else:
            assert test_scores == [0.0, 0.0, 0.0]
        model.fit_partial(x[num_fitted : num_fitted + 1])


def test_relative_entropy_score_partial_has_no_side_effects():
    import copy

    import numpy as np

    from pysad.models import RelativeEntropy
    from pysad.utils import fix_seed

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
    for hypothesis, hypothesis_before in zip(model.P, P_before, strict=True):
        np.testing.assert_array_equal(hypothesis, hypothesis_before)
    assert model.c == c_before
    assert model.m == m_before


def test_relative_entropy_score_then_fit_matches_fit_score_partial():
    import numpy as np

    from pysad.models import RelativeEntropy

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
    import numpy as np

    from pysad.models import RelativeEntropy

    model = RelativeEntropy(min_val=0.5, max_val=0.5)

    scores = [model.fit_score_partial(np.array([0.5])) for _ in range(100)]

    assert all(score == 0.0 for score in scores)


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_relative_entropy_out_of_range_values_do_not_produce_nan_hypotheses():
    import numpy as np

    from pysad.models import RelativeEntropy

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
    import numpy as np

    from pysad.models import RelativeEntropy

    # With these exact (min_val, max_val, num_bins), floating-point round-off makes
    # ceil((max_val - min_val) / stepSize) evaluate to num_bins + 1, one level past the levels
    # 1..num_bins that _histogram maps to bins 0..num_bins - 1; clipping the level must put
    # max_val in the top bin.
    window_size = 10
    model = RelativeEntropy(
        min_val=0.0, max_val=100.0, num_bins=29, window_size=window_size, step=1
    )
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
    import numpy as np

    from pysad.models import RelativeEntropy

    model = RelativeEntropy(min_val=0, max_val=100, num_bins=5, window_size=50)

    # Ten values in each of the five buckets of width 20: (0, 20], (20, 40], (40, 60],
    # (60, 80], (80, 100].
    window = np.arange(1, 100, 2)  # 1, 3, ..., 99
    np.testing.assert_allclose(model._histogram(list(window)), [0.2] * 5)


def test_histogram_buckets_are_closed_on_the_right():
    import numpy as np

    from pysad.models import RelativeEntropy

    model = RelativeEntropy(min_val=0, max_val=100, num_bins=5, window_size=5)

    # The paper's level B = ceil((u - min_val) / stepSize) makes the buckets (0, 20], (20, 40],
    # (40, 60], (60, 80], (80, 100]: a value on an edge belongs to the bucket below it, and a
    # value just above an edge to the bucket above it.
    np.testing.assert_array_equal(model._histogram([20.0, 40.0, 60.0, 80.0, 100.0]), [0.2] * 5)
    np.testing.assert_array_equal(
        model._histogram([20.5, 40.5, 60.5, 80.5, 80.5]), [0.0, 0.2, 0.2, 0.2, 0.4]
    )


def test_histogram_puts_min_val_in_first_bucket_and_max_val_in_last():
    from pysad.models import RelativeEntropy

    model = RelativeEntropy(min_val=0, max_val=100, num_bins=5, window_size=50)

    histogram = model._histogram([model.min_val] * 25 + [model.max_val] * 25)

    assert histogram[0] == 0.5
    assert histogram[-1] == 0.5
    assert histogram[1:-1].sum() == 0.0


def test_relative_entropy_flags_jump_between_top_two_buckets():
    import numpy as np

    from pysad.models import RelativeEntropy

    # A level shift from the 4th bucket (60, 80] to the 5th bucket (80, 100] must be
    # flagged just as often as a shift of the same size between two lower buckets.
    x_top_shift = np.r_[np.full(200, 70.0), np.full(200, 90.0)]
    scores_top_shift = RelativeEntropy(
        min_val=0, max_val=100, num_bins=5, window_size=50, step=1
    ).fit_score(x_top_shift.reshape(-1, 1))

    x_lower_shift = np.r_[np.full(200, 50.0), np.full(200, 70.0)]
    scores_lower_shift = RelativeEntropy(
        min_val=0, max_val=100, num_bins=5, window_size=50, step=1
    ).fit_score(x_lower_shift.reshape(-1, 1))

    assert scores_top_shift.sum() > 0
    assert scores_top_shift.sum() == scores_lower_shift.sum()


@pytest.mark.parametrize("method", ["fit", "fit_score"])
def test_relative_entropy_step_window_size_tests_non_overlapping_windows(method):
    import numpy as np

    from pysad.models import RelativeEntropy

    rng = np.random.default_rng(0)
    x = np.clip(rng.normal(50, 10, 52 * 20), 0, 100)  # 20 non-overlapping windows of W=52

    model = RelativeEntropy(min_val=0, max_val=100, num_bins=5, window_size=52, step=52)
    getattr(model, method)(x.reshape(-1, 1))

    assert sum(model.c) == 20


@pytest.mark.parametrize("step_kwargs", [{}, {"step": 1}], ids=["default", "step=1"])
@pytest.mark.parametrize("method", ["fit", "fit_score"])
def test_relative_entropy_default_step_reproduces_nab_sliding_windows(method, step_kwargs):
    import numpy as np

    from pysad.models import RelativeEntropy

    rng = np.random.default_rng(0)
    n = 52 * 20
    x = np.clip(rng.normal(50, 10, n), 0, 100)

    model = RelativeEntropy(min_val=0, max_val=100, num_bins=5, window_size=52, **step_kwargs)
    getattr(model, method)(x.reshape(-1, 1))

    assert sum(model.c) == n - model.W + 1


@pytest.mark.parametrize(
    "step, flagged",
    [
        # Windows end at the 5th, 10th, 15th and 20th values. The first (all 0.1) is learned; the
        # second (0.1 x 2, 0.9 x 3) holds the shift and agrees with no hypothesis, so its closing
        # value, the 10th, scores 1.0; the third (all 0.9) agrees with the second.
        pytest.param(None, [9], id="step=None"),
        pytest.param(5, [9], id="step=window_size"),
        # Windows end at every value from the 5th on: the 8th value's window is the first to hold a
        # 0.9, the 12th value's the first to hold only 0.9s; every window in between agrees.
        pytest.param(1, [7, 11], id="step=1"),
        # Windows end at the 5th, 8th, 11th, 14th, ... values, counting from the first full window.
        pytest.param(3, [7, 13], id="step=3"),
    ],
)
@pytest.mark.parametrize("driver", ["fit_score_partial", "score_partial_then_fit_partial"])
def test_relative_entropy_only_the_value_closing_a_window_scores(step, flagged, driver):
    import numpy as np

    from pysad.models import RelativeEntropy

    # A level shift from bucket (0, 0.2] to bucket (0.8, 1] after the 7th value, inside the
    # second window of 5 values.
    x = np.r_[np.full(7, 0.1), np.full(16, 0.9)]

    model = RelativeEntropy(min_val=0.0, max_val=1.0, window_size=5, step=step)
    scores = []
    for v in x:
        xi = np.array([v])
        if driver == "fit_score_partial":
            scores.append(model.fit_score_partial(xi))
        else:
            scores.append(model.score_partial(xi))
            model.fit_partial(xi)

    expected = np.zeros(len(x))
    expected[flagged] = 1.0
    assert scores == expected.tolist()


@pytest.mark.parametrize(
    "c_th, flagged",
    [
        # Windows of 5 values end at the 5th, 10th, ..., 45th values: four windows of 0.1, then
        # five of 0.9. Only the first 0.9 window agrees with no hypothesis.
        pytest.param(1, [24], id="c_th=1"),
        # A window that agrees with a hypothesis is flagged until 3 windows have created or agreed
        # with it: the 2nd and 3rd windows (the first hypothesis is not exempt), the first 0.9
        # window, which creates the second hypothesis, and the two after it.
        pytest.param(3, [9, 14, 24, 29, 34], id="c_th=3"),
    ],
)
@pytest.mark.parametrize("driver", ["fit_score_partial", "score_partial_then_fit_partial"])
def test_relative_entropy_c_th_flags_a_state_until_it_recurs(c_th, flagged, driver):
    import numpy as np

    from pysad.models import RelativeEntropy

    x = np.r_[np.full(20, 0.1), np.full(25, 0.9)]

    # The paper's non-overlapping windows, so that c_th counts windows as in Fig. 1.
    model = RelativeEntropy(min_val=0.0, max_val=1.0, window_size=5, step=5, c_th=c_th)
    scores = []
    for v in x:
        xi = np.array([v])
        if driver == "fit_score_partial":
            scores.append(model.fit_score_partial(xi))
        else:
            scores.append(model.score_partial(xi))
            model.fit_partial(xi)

    expected = np.zeros(len(x))
    expected[flagged] = 1.0
    assert scores == expected.tolist()
    # c_th changes the scores only: the hypotheses and their counts are the same.
    assert model.c == [4, 5]


@pytest.mark.parametrize("c_th", [0, -1])
def test_relative_entropy_c_th_below_one_raises_value_error(c_th):
    from pysad.models import RelativeEntropy

    with pytest.raises(ValueError, match=f"c_th must be at least 1, got {c_th}"):
        RelativeEntropy(min_val=0.0, max_val=1.0, c_th=c_th)


@pytest.mark.parametrize("c_th", [1.0, 2.5, None, "1", True])
def test_relative_entropy_non_integer_c_th_raises_type_error(c_th):
    import re

    from pysad.models import RelativeEntropy

    with pytest.raises(TypeError, match=re.escape(f"c_th must be an int, got {c_th!r}")):
        RelativeEntropy(min_val=0.0, max_val=1.0, c_th=c_th)


def test_relative_entropy_accepts_numpy_integer_c_th():
    import numpy as np

    from pysad.models import RelativeEntropy

    model = RelativeEntropy(min_val=0.0, max_val=1.0, c_th=np.int64(3))

    assert type(model.c_th) is int and model.c_th == 3


@pytest.mark.parametrize("method", ["fit_partial", "score_partial", "fit_score_partial"])
@pytest.mark.parametrize(
    "num_fitted", [6, 9]
)  # NaN would not close / would close the second window
@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_relative_entropy_rejects_nan_without_changing_the_model(method, num_fitted):
    import numpy as np

    from pysad.models import RelativeEntropy

    model = RelativeEntropy(min_val=0.0, max_val=1.0, window_size=5)
    for v in [0.1, 0.3, 0.5, 0.7, 0.9, 0.9, 0.9, 0.9, 0.9][:num_fitted]:
        model.fit_partial(np.array([v]))
    util_before, P_before, c_before, m_before = (
        list(model.util),
        model.P.copy(),
        list(model.c),
        model.m,
    )

    with pytest.raises(ValueError, match="RelativeEntropy does not accept NaN values"):
        getattr(model, method)(np.array([np.nan]))

    assert list(model.util) == util_before
    assert model.num_fitted == num_fitted
    np.testing.assert_array_equal(model.P, P_before)
    assert model.c == c_before
    assert model.m == m_before

    # The rejected value leaves nothing behind that breaks later windows.
    for _ in range(10):
        model.fit_score_partial(np.array([0.9]))
    assert model.num_fitted == num_fitted + 10


@pytest.mark.parametrize("step", [1, None, 7])
@pytest.mark.parametrize("method", ["fit", "fit_score"])
def test_relative_entropy_keeps_only_the_current_window(method, step):
    import numpy as np

    from pysad.models import RelativeEntropy

    # Only the last window_size values are ever read, so the model must not keep the whole stream.
    rng = np.random.default_rng(0)
    x = rng.random(1000)

    model = RelativeEntropy(min_val=0.0, max_val=1.0, window_size=52, step=step)
    getattr(model, method)(x.reshape(-1, 1))

    assert model.num_fitted == 1000
    assert list(model.util) == x[-52:].tolist()


@pytest.mark.parametrize("step", [0, -1])
def test_relative_entropy_step_below_one_raises_value_error(step):
    from pysad.models import RelativeEntropy

    with pytest.raises(ValueError, match=f"step must be at least 1, got {step}"):
        RelativeEntropy(min_val=0.0, max_val=1.0, step=step)


@pytest.mark.parametrize("step", [1.5, 1.0, "1", True])
def test_relative_entropy_non_integer_step_raises_type_error(step):
    import re

    from pysad.models import RelativeEntropy

    # As in KNNCAD and RSHash: TypeError for a wrong type (bool is an int subclass, but not an
    # accepted integer), ValueError only for an int out of range.
    with pytest.raises(TypeError, match=re.escape(f"step must be None or an int, got {step!r}")):
        RelativeEntropy(min_val=0.0, max_val=1.0, step=step)


@pytest.mark.parametrize("window_size", [0, -1])
def test_relative_entropy_window_size_below_one_raises_value_error(window_size):
    from pysad.models import RelativeEntropy

    # The error names window_size, not the step resolved from it.
    with pytest.raises(ValueError, match=f"window_size must be at least 1, got {window_size}"):
        RelativeEntropy(min_val=0.0, max_val=1.0, window_size=window_size)


@pytest.mark.parametrize("window_size", [52.0, 1.5, None, "52", True])
def test_relative_entropy_non_integer_window_size_raises_type_error(window_size):
    import re

    from pysad.models import RelativeEntropy

    with pytest.raises(
        TypeError, match=re.escape(f"window_size must be an int, got {window_size!r}")
    ):
        RelativeEntropy(min_val=0.0, max_val=1.0, window_size=window_size)


@pytest.mark.parametrize("num_bins", [1, 0, -2])
def test_relative_entropy_num_bins_below_two_raises_value_error(num_bins):
    from pysad.models import RelativeEntropy

    # The threshold T has num_bins - 1 degrees of freedom: num_bins=1 made it nan, so every window
    # disagreed with every hypothesis and scored 1.0; 0 and -2 failed with unrelated errors.
    with pytest.raises(ValueError, match=f"num_bins must be at least 2, got {num_bins}"):
        RelativeEntropy(min_val=0.0, max_val=1.0, num_bins=num_bins)


@pytest.mark.parametrize("num_bins", [5.0, 2.5, None, "5", True])
def test_relative_entropy_non_integer_num_bins_raises_type_error(num_bins):
    import re

    from pysad.models import RelativeEntropy

    with pytest.raises(TypeError, match=re.escape(f"num_bins must be an int, got {num_bins!r}")):
        RelativeEntropy(min_val=0.0, max_val=1.0, num_bins=num_bins)


@pytest.mark.parametrize("name", ["min_val", "max_val"])
@pytest.mark.parametrize("value", [None, "0.5", True, [0.5]])
def test_relative_entropy_non_real_bounds_raise_type_error(name, value):
    import re

    from pysad.models import RelativeEntropy

    bounds = {"min_val": 0.0, "max_val": 1.0, name: value}
    with pytest.raises(TypeError, match=re.escape(f"{name} must be a real number, got {value!r}")):
        RelativeEntropy(**bounds)


@pytest.mark.parametrize("name", ["min_val", "max_val"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_relative_entropy_non_finite_bounds_raise_value_error(name, value):
    from pysad.models import RelativeEntropy

    bounds = {"min_val": 0.0, "max_val": 1.0, name: value}
    with pytest.raises(ValueError, match=f"{name} must be finite, got {value}"):
        RelativeEntropy(**bounds)


def test_relative_entropy_min_val_above_max_val_raises_value_error():
    import re

    from pysad.models import RelativeEntropy

    # A negative bucket width put every value in the top bucket, so no window was ever flagged.
    with pytest.raises(
        ValueError,
        match=re.escape("min_val must not exceed max_val, got min_val=1.0 and max_val=0.0"),
    ):
        RelativeEntropy(min_val=1.0, max_val=0.0)


def test_relative_entropy_accepts_numpy_bounds_and_num_bins():
    import numpy as np

    from pysad.models import RelativeEntropy

    model = RelativeEntropy(min_val=np.float32(0.0), max_val=np.int64(100), num_bins=np.uint8(5))

    assert type(model.min_val) is float and model.min_val == 0.0
    assert type(model.max_val) is float and model.max_val == 100.0
    assert type(model.N_bins) is int and model.N_bins == 5
    assert model.stepSize == 20.0


def test_relative_entropy_accepts_numpy_integer_window_size_and_step():
    import numpy as np

    from pysad.models import RelativeEntropy

    for integer_type in (np.int32, np.int64, np.uint16):
        model = RelativeEntropy(min_val=0.0, max_val=1.0, window_size=integer_type(52))
        assert type(model.W) is int and model.W == 52
        assert type(model.step) is int and model.step == 1

        model = RelativeEntropy(min_val=0.0, max_val=1.0, window_size=integer_type(52), step=None)
        assert type(model.step) is int and model.step == 52

        model = RelativeEntropy(
            min_val=0.0, max_val=1.0, window_size=integer_type(52), step=integer_type(1)
        )
        assert type(model.step) is int and model.step == 1

    # e.g. window sizes from a grid built with np.arange
    rng = np.random.default_rng(0)
    x = np.clip(rng.normal(50, 10, 52 * 20), 0, 100)
    model = RelativeEntropy(min_val=0, max_val=100, num_bins=5, window_size=np.arange(52, 53)[0])
    model.fit_score(x.reshape(-1, 1))

    assert sum(model.c) == 52 * 20 - 52 + 1


@pytest.mark.parametrize("window_size", [1, 2, 3])
@pytest.mark.parametrize("step", [1, None])
def test_relative_entropy_small_windows_score_partial_matches_fit_score_partial(window_size, step):
    import numpy as np

    from pysad.models import RelativeEntropy

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
