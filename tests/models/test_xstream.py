from collections import Counter, defaultdict

import numpy as np

from pysad.models import xStream


class _ReferenceChains:
    # A direct port of chains_add in the authors' code (cmuxstream-core, cpp/chain.cpp). The bin widths
    # and shifts are set once from the first window, as compute_deltamax in cpp/main.cpp does.

    def __init__(self, first_window, fs, rand_arr):
        self.fs = fs
        self.deltamax = (first_window.max(axis=0) - first_window.min(axis=0)) / 2.0
        self.deltamax[np.abs(self.deltamax) <= 0.0001] = 1.0
        self.shift = rand_arr * self.deltamax
        self.counts = defaultdict(Counter)

    def add(self, y, update):
        # Returns the score negated, so that higher is more anomalous.
        n_chains, depth = self.fs.shape
        total = 0.0
        for c in range(n_chains):
            prebin = np.zeros(len(y))
            used = np.zeros(len(y), dtype=bool)
            best = np.inf
            for d in range(depth):
                f = self.fs[c, d]
                if not used[f]:
                    prebin[f] = (y[f] + self.shift[c, f]) / self.deltamax[f]
                    used[f] = True
                else:
                    prebin[f] = 2.0 * prebin[f] - self.shift[c, f] / self.deltamax[f]
                key = tuple(np.floor(prebin).astype(int))
                if update:
                    self.counts[c, d][key] += 1
                best = min(best, np.log2(self.counts[c, d][key] + 1) + d + 1)
            total += best
        return -total / n_chains


def _model_and_projections(X, window_size):
    from pysad.models import xStream

    model = xStream(num_components=8, n_chains=6, depth=10, window_size=window_size, random_state=0)
    Y = np.vstack([model.streamhash.transform_partial(x) for x in X])
    return model, Y


def test_xstream_scores_with_the_bins_of_the_first_window():
    rng = np.random.RandomState(0)
    X = rng.randn(61, 6)
    Z = 2.0 * rng.randn(30, 6)
    model, Y = _model_and_projections(X, window_size=30)

    # After 61 instances, the reference is the second window.
    model.fit(X)
    reference = _ReferenceChains(Y[:30], model.hs_chains.fs, model.hs_chains.rand_arr)
    for y in Y[30:60]:
        reference.add(y, update=True)
    expected = [reference.add(model.streamhash.transform_partial(z), update=False) for z in Z]

    np.testing.assert_allclose(model.score(Z), expected, rtol=1e-12, atol=0)


def test_xstream_scores_each_instance_with_the_previous_window():
    # The windowed mode of cpp/main.cpp: each instance after the first window is scored with the counts
    # of the previous full window, and the counts are rebuilt once the current window is full.
    rng = np.random.RandomState(0)
    X = rng.randn(130, 6)
    X[::17] += 4.0
    model, Y = _model_and_projections(X, window_size=20)

    scores = model.fit_score(X)

    reference = _ReferenceChains(Y[:20], model.hs_chains.fs, model.hs_chains.rand_arr)
    for y in Y[:20]:
        reference.add(y, update=True)
    expected = []
    for start in range(20, 130, 20):
        window = Y[start : start + 20]
        expected += [reference.add(y, update=False) for y in window]
        reference.counts.clear()
        for y in window:
            reference.add(y, update=True)

    np.testing.assert_allclose(scores[20:], expected, rtol=1e-12, atol=0)


def test_xstream_fit_score_partial_matches_score_then_fit():
    from pysad.models import xStream

    X = np.random.RandomState(0).randn(65, 6)
    model = xStream(num_components=8, n_chains=6, depth=10, window_size=20, random_state=0)
    reference = xStream(num_components=8, n_chains=6, depth=10, window_size=20, random_state=0)

    for x in X:
        expected = reference.score_partial(x)
        reference.fit_partial(x)
        assert model.fit_score_partial(x) == expected

    assert model.hs_chains.counts == reference.hs_chains.counts
    assert model.hs_chains.counts_cur == reference.hs_chains.counts_cur


def test_xstream_explain_returns_per_feature_weights():
    rng = np.random.RandomState(0)
    X = rng.randn(120, 6)
    model = xStream(num_components=8, n_chains=6, depth=10, window_size=40, random_state=0)
    model.fit(X[:80])

    weights = model.score_explain_partial(X[81])

    assert weights.shape == (6,)
    assert np.all(weights >= 0.0)
    assert abs(weights.sum() - 1.0) < 1e-12


def test_xstream_explain_does_not_change_state():
    rng = np.random.RandomState(0)
    X = rng.randn(120, 6)
    model = xStream(num_components=8, n_chains=6, depth=10, window_size=40, random_state=0)
    model.fit(X[:80])

    before = model.score_partial(X[81])
    model.score_explain_partial(X[81])
    after = model.score_partial(X[81])

    assert before == after


def test_xstream_explain_is_reproducible():
    rng = np.random.RandomState(0)
    X = rng.randn(120, 6)

    model_a = xStream(num_components=8, n_chains=6, depth=10, window_size=40, random_state=0)
    model_a.fit(X[:80])
    weights_a = model_a.score_explain_partial(X[81])

    model_b = xStream(num_components=8, n_chains=6, depth=10, window_size=40, random_state=0)
    model_b.fit(X[:80])
    weights_b = model_b.score_explain_partial(X[81])

    np.testing.assert_array_equal(weights_a, weights_b)


def test_xstream_explain_rejects_nan():
    rng = np.random.RandomState(0)
    X = rng.randn(60, 6)
    model = xStream(num_components=8, n_chains=6, depth=10, window_size=30, random_state=0)
    model.fit(X[:30])

    anomalous = X[31].copy()
    anomalous[2] = np.nan
    try:
        model.score_explain_partial(anomalous)
    except ValueError:
        pass
    else:
        raise AssertionError("score_explain_partial did not reject NaN")
