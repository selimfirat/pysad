import gc
import weakref

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from pysad.models import Inqmad

PARAMS = {"input_shape": 3, "dim_x": 32, "gamma": 1.0, "initial_size": 20, "random_state": 0}


def _model(**kwargs):
    return Inqmad(**{**PARAMS, **kwargs})


def _states(model, X):
    psi = np.cos(np.atleast_2d(X) @ model.weights + model.offset)
    return psi / np.linalg.norm(psi, axis=1, keepdims=True)


def test_initial_sample_sets_rho_to_its_mean_state_and_tau_to_its_beta_quantile():
    """Paper Eq. 2 and Algorithm 1, step 6: once the initial sample is
    complete, rho is the mean of its psi psi^T and tau is the beta quantile
    of its densities psi^T rho psi.
    """
    X = np.random.default_rng(0).random((20, 3))
    model = _model(beta=0.2)
    model.fit(X[:19])
    assert model.rho is None

    model.fit_partial(X[19])

    psi = _states(model, X)
    rho = psi.T @ psi / 20
    np.testing.assert_allclose(model.rho, rho, atol=1e-12)
    assert model.rho.dtype == np.float64
    assert model.tau == pytest.approx(np.quantile(np.einsum("ni,ij,nj->n", psi, rho, psi), 0.2))


def test_normal_instance_updates_rho_with_forgetting():
    """Paper Eq. 4 and Algorithm 2: an instance whose density is at least
    tau updates rho to (1 - alpha) rho + alpha psi psi^T.
    """
    rng = np.random.default_rng(0)
    model = _model(alpha=0.3).fit(rng.random((20, 3)))
    x = np.array([0.5, 0.5, 0.5])
    psi = _states(model, x)[0]
    rho = model.rho.copy()
    assert psi @ rho @ psi >= model.tau

    model.fit_partial(x)

    np.testing.assert_allclose(model.rho, 0.7 * rho + 0.3 * np.outer(psi, psi), atol=1e-12)


def test_anomalous_instance_leaves_rho_unchanged():
    """Paper Algorithm 2, step 6: an instance whose density is below tau
    is classified as an anomaly and does not change rho.
    """
    rng = np.random.default_rng(0)
    model = _model().fit(rng.random((20, 3)))
    far_point = np.array([5.0, 5.0, 5.0])
    psi = _states(model, far_point)[0]
    rho = model.rho.copy()
    assert psi @ rho @ psi < model.tau

    model.fit_partial(far_point)

    np.testing.assert_array_equal(model.rho, rho)


def test_score_partial_is_negated_paper_density():
    """Paper Eq. 3: after the initial sample, the score is the negated
    density psi^T rho psi.
    """
    rng = np.random.default_rng(0)
    model = _model().fit(rng.random((30, 3)))
    q = rng.random(3)
    psi = _states(model, q)[0]

    assert model.score_partial(q) == pytest.approx(-psi @ model.rho @ psi, rel=1e-12)


def test_score_during_initial_sample_is_mean_squared_kernel():
    """While the initial sample fills up, an instance scores the negated
    mean of k(x, x_i)^2 over the sample so far, the density that
    psi^T rho psi approximates with rho the mean state of the sample.
    """
    rng = np.random.default_rng(0)
    X = rng.random((5, 3))
    q = rng.random(3)
    model = _model(gamma=0.7).fit(X)

    expected = -np.mean(np.exp(-0.7 * np.sum((X - q) ** 2, axis=1)) ** 2)

    assert model.score_partial(q) == pytest.approx(expected, rel=1e-12)


def test_first_instance_scores_zero():
    """Before anything has been fitted the density is 0, so the score is
    0.0, the most anomalous possible score, instead of an error.
    """
    x = np.array([0.5, 0.5, 0.5])

    score = _model().score_partial(x)
    assert type(score) is float and score == 0.0

    score = _model().fit_score_partial(x)
    assert type(score) is float and score == 0.0


def test_fit_score_matches_score_then_fit():
    """Paper Algorithm 2: fit_score_partial scores each instance against
    rho_t, and then fits it, through the initial sample and after it.
    """
    rng = np.random.default_rng(0)
    X = rng.random((50, 3))
    X[[10, 30]] += 5.0

    fit_scores = _model().fit_score(X)

    model = _model()
    expected = []
    for x in X:
        expected.append(model.score_partial(x))
        model.fit_partial(x)

    np.testing.assert_array_equal(fit_scores, expected)


def test_fit_score_ranks_interleaved_far_outliers_above_inliers():
    """Regression test for #120: scores are anomaly scores (higher is more
    anomalous), and each depends on the data seen so far.
    """
    rng = np.random.default_rng(0)
    X = rng.random((300, 3))
    y = np.zeros(300)
    outliers = rng.choice(300, size=30, replace=False)
    X[outliers] += 5.0
    y[outliers] = 1

    scores = _model().fit_score(X)

    assert roc_auc_score(y, scores) > 0.9


def test_score_follows_later_training():
    """Regression test for #120: scoring after more fitting reflects the
    updated model, here normal instances next to the scored one.
    """
    rng = np.random.default_rng(0)
    q = np.array([1.5, 1.5, 1.5])
    model = _model(alpha=0.5).fit(rng.random((20, 3)) * 3)
    before = model.score_partial(q)

    model.fit(np.tile(q + 0.1, (5, 1)))

    assert model.score_partial(q) < before


def test_docstring_keeps_math_backslashes():
    """Regression test for #120: LaTeX such as \\rho and \\tau must not
    turn into carriage returns or tabs in the rendered docstring.
    """
    assert "\r" not in Inqmad.__doc__
    assert "\t" not in Inqmad.__doc__


def test_scores_use_the_current_random_features():
    """Regression test for #171: the random Fourier features are read on
    every call, so replacing them takes effect.
    """
    rng = np.random.default_rng(0)
    X = rng.random((30, 3))
    q = np.array([0.5, 0.5, 0.5])

    model = _model()
    model.score_partial(q)
    other = _model(random_state=1)

    model.weights, model.offset = other.weights, other.offset
    model.fit(X)
    other.fit(X)

    assert model.score_partial(q) == other.score_partial(q)


def test_deleted_model_is_garbage_collected():
    """Regression test for #173: nothing at class or module level keeps a
    model or its random features alive after its last reference is gone.
    """
    rng = np.random.default_rng(0)
    model = _model().fit(rng.random((30, 3)))
    model.score_partial(rng.random(3))
    refs = [weakref.ref(model), weakref.ref(model.weights), weakref.ref(model.rho)]

    del model
    gc.collect()

    assert all(ref() is None for ref in refs)


@pytest.mark.parametrize("name", ["input_shape", "dim_x", "initial_size"])
@pytest.mark.parametrize("value", [0, -1])
def test_rejects_sizes_below_one(name, value):
    """Regression test for #172: a zero or negative size raises a
    ValueError that names the parameter.
    """
    with pytest.raises(ValueError, match=f"{name} must be at least 1, got {value}"):
        _model(**{name: value})


@pytest.mark.parametrize("name", ["input_shape", "dim_x", "initial_size"])
@pytest.mark.parametrize("value", [True, 3.0, "3", None])
def test_rejects_sizes_that_are_not_ints(name, value):
    """Regression test for #172: sizes must be ints, and a bool is
    rejected too.
    """
    with pytest.raises(TypeError, match=f"{name} must be an int"):
        _model(**{name: value})


@pytest.mark.parametrize("name", ["alpha", "beta"])
@pytest.mark.parametrize("value", [-0.1, 1.5, np.nan])
def test_rejects_rates_outside_the_unit_interval(name, value):
    with pytest.raises(ValueError, match=name):
        _model(**{name: value})


def test_accepts_numpy_int_sizes():
    """Regression test for #172: NumPy integers, such as a shape read off
    an array, are valid sizes.
    """
    model = _model(input_shape=np.int64(3), dim_x=np.int32(8), initial_size=np.int64(5))

    assert model.weights.shape == (3, 8)
    assert model.initial_size == 5


@pytest.mark.parametrize(
    "X",
    [
        np.zeros(2),
        np.zeros(4),
        np.zeros((1, 4)),
        np.float64(1.0),
        np.zeros((2, 3)),
        np.zeros((1, 1, 3)),
        [[]],
    ],
    ids=["too-few", "too-many", "row-too-many", "0-d", "two-rows", "3-d", "empty-row"],
)
@pytest.mark.parametrize("method", ["fit_partial", "score_partial", "fit_score_partial"])
@pytest.mark.parametrize("num_fitted", [5, 30])
def test_rejects_instances_of_the_wrong_shape(method, X, num_fitted):
    """Regression test for #172: an instance that is not one row of
    input_shape features raises a ValueError naming the expected shape and
    leaves the model unchanged, during the initial sample and after it.
    """
    rng = np.random.default_rng(0)
    model = _model().fit(rng.random((num_fitted, 3)))
    state = [None if a is None else a.copy() for a in (model.initial, model.rho)]

    with pytest.raises(ValueError, match=r"one instance of shape \(3,\)"):
        getattr(model, method)(X)

    for before, after in zip(state, (model.initial, model.rho), strict=True):
        np.testing.assert_array_equal(after, before)


@pytest.mark.parametrize("method", ["fit_partial", "score_partial", "fit_score_partial"])
def test_accepts_one_row_instances(method):
    """Regression test for #172: a (1, input_shape) row and a list are the
    same instance as the (input_shape,) array.
    """
    rng = np.random.default_rng(0)
    X = rng.random((30, 3))
    x = rng.random(3)

    results = []
    for instance in [x, x[None, :], list(x)]:
        model = _model().fit(X)
        result = getattr(model, method)(instance)
        results.append((result if method != "fit_partial" else None, model.rho.copy()))

    for score, rho in results[1:]:
        assert score == results[0][0]
        np.testing.assert_array_equal(rho, results[0][1])
