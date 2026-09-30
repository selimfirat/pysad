import gc
import weakref

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from pysad.models import Inqmad


def test_score_partial_follows_later_training():
    """Regression test for #120: scoring after a second fit_partial/fit
    must reflect the updated model, not the model as it stood at the
    first score_partial call (which used to get baked into a stale
    jit-compiled trace).
    """
    rng = np.random.default_rng(0)
    a = rng.random((50, 3))
    b = rng.random((50, 3)) + 5.0
    q = np.array([5.5, 5.5, 5.5])

    model = Inqmad(input_shape=3, dim_x=32, gamma=1.0, random_state=0)
    model.fit(a)
    score_after_a = model.score_partial(q)  # primes the jit cache, as in the issue's repro
    model.fit(b)
    score_after_b = model.score_partial(q)

    fresh = Inqmad(input_shape=3, dim_x=32, gamma=1.0, random_state=0)
    fresh.fit(a)
    fresh.fit(b)
    expected = fresh.score_partial(q)

    assert score_after_b != pytest.approx(score_after_a)
    assert score_after_b == pytest.approx(expected)


def test_score_orders_far_outliers_above_inliers():
    """Regression test for #120: scores must be anomaly scores (higher
    is more anomalous), not raw density (higher is more normal).
    """
    rng = np.random.default_rng(0)
    train = rng.random((200, 3))
    X = np.vstack([rng.random((20, 3)), rng.random((20, 3)) + 5.0])
    y = np.r_[np.zeros(20), np.ones(20)]

    model = Inqmad(input_shape=3, dim_x=32, gamma=1.0, random_state=0).fit(train)
    scores = model.score(X)

    assert roc_auc_score(y, scores) > 0.9


def test_fit_score_ranks_interleaved_far_outliers_above_inliers():
    """Regression test for #120: on the streaming path each score must
    depend on the data seen so far, not only on how many instances were
    fitted (a replacing update used to give exactly -1/t^2).
    """
    rng = np.random.default_rng(0)
    X = rng.random((300, 3))
    y = np.zeros(300)
    outliers = rng.choice(300, size=30, replace=False)
    X[outliers] += 5.0
    y[outliers] = 1

    scores = Inqmad(input_shape=3, dim_x=32, gamma=1.0, random_state=0).fit_score(X)

    assert not np.allclose(scores, -1.0 / np.arange(1, 301) ** 2)
    assert roc_auc_score(y, scores) > 0.9


def test_score_partial_depends_on_whole_fitted_history():
    """Regression test for #120: the model must remember every fitted
    instance, not only the last one, so two histories that end with the
    same row give different scores.
    """
    rng = np.random.default_rng(0)
    history = rng.random((199, 3))
    last = rng.random((1, 3))
    q = np.array([0.5, 0.5, 0.5])

    near = Inqmad(input_shape=3, dim_x=32, gamma=1.0, random_state=0).fit(
        np.vstack([history, last])
    )
    far = Inqmad(input_shape=3, dim_x=32, gamma=1.0, random_state=0).fit(
        np.vstack([history + 5.0, last])
    )

    assert near.score_partial(q) < far.score_partial(q)


def test_far_point_fitted_last_scores_above_inliers():
    """Regression test for #120: fitting one far point after many inliers
    must not make it the model's whole memory.
    """
    rng = np.random.default_rng(0)
    far_point = np.array([5.0, 5.0, 5.0])
    model = Inqmad(input_shape=3, dim_x=32, gamma=1.0, random_state=0).fit(
        np.vstack([rng.random((199, 3)), far_point])
    )

    inlier_scores = model.score(rng.random((20, 3)))

    assert model.score_partial(far_point) > inlier_scores.max()


def test_density_matrix_is_mean_of_fitted_states():
    """Regression test for #120: every fitted instance adds to the density
    matrix, across separate fit calls.
    """
    rng = np.random.default_rng(0)
    a = rng.random((7, 3))
    b = rng.random((5, 3)) + 5.0

    model = Inqmad(input_shape=3, dim_x=32, gamma=1.0, random_state=0)
    model.fit(a)
    model.fit(b)

    states = model._states(np.vstack([a, b]))
    rho = model.rho / model.num_fitted

    np.testing.assert_allclose(rho, np.einsum("ni,nj->ij", states, states) / 12, atol=1e-12)


def test_score_partial_is_negated_paper_density():
    """Regression test for #120: the score is the negated density
    estimate psi^T rho psi of the paper's Eq. 3, with rho normalised by
    the number of fitted instances.
    """
    rng = np.random.default_rng(0)
    model = Inqmad(input_shape=3, dim_x=32, gamma=1.0, random_state=0).fit(rng.random((50, 3)))
    q = rng.random(3)

    psi = np.cos(q @ model.weights + model.offset)
    psi /= np.linalg.norm(psi)
    rho = model.rho / model.num_fitted

    assert model.score_partial(q) == pytest.approx(-np.einsum("i,ij,j->", psi, rho, psi), rel=1e-12)


@pytest.mark.parametrize("num_fitted", [0, 5])
def test_score_partial_rejects_multiple_rows_with_pysad_message(num_fitted):
    """Regression test for #120: score_partial scores one instance, and
    several rows raise pysad's single-instance error rather than the one
    from converting the array to a scalar, before and after fitting.
    """
    rng = np.random.default_rng(0)
    model = Inqmad(input_shape=3, dim_x=32, gamma=1.0, random_state=0)
    if num_fitted:
        model.fit(rng.random((num_fitted, 3)))

    with pytest.raises(
        ValueError,
        match=r"one instance of shape \(3,\) or \(1, 3\), got an array of shape \(4, 3\)",
    ):
        model.score_partial(rng.random((4, 3)))


def test_first_instance_scores_zero():
    """Regression test for #120: before anything has been fitted rho is
    zero, so the density is 0 and the score is 0.0, the most anomalous
    possible score, instead of an error.
    """
    x = np.array([0.5, 0.5, 0.5])

    score = Inqmad(input_shape=3, dim_x=32, gamma=1.0, random_state=0).score_partial(x)
    assert type(score) is float and score == 0.0

    score = Inqmad(input_shape=3, dim_x=32, gamma=1.0, random_state=0).fit_score_partial(x)
    assert type(score) is float and score == 0.0


def test_fit_score_matches_score_then_fit():
    """Regression test for #120: fit_score_partial scores each instance
    against the density matrix of the instances before it and then fits
    it, as in the paper, which measures the density against rho_t before
    the update.
    """
    rng = np.random.default_rng(0)
    X = rng.random((50, 3))
    X[[10, 30]] += 5.0

    fit_scores = Inqmad(input_shape=3, dim_x=32, gamma=1.0, random_state=0).fit_score(X)

    model = Inqmad(input_shape=3, dim_x=32, gamma=1.0, random_state=0)
    expected = []
    for x in X:
        expected.append(model.score_partial(x))
        model.fit_partial(x)

    assert fit_scores[0] == 0.0
    np.testing.assert_array_equal(fit_scores, expected)


def test_fit_score_partial_leaves_out_the_instances_own_state():
    """Regression test for #120: fitting before scoring added the
    instance's own psi psi^T to rho, so with t fitted instances its
    density was ((t - 1) * d + 1) / t instead of d, its density against
    the earlier instances. Scoring first leaves the self-term out:
    fit_score_partial(x) equals the score of x before fit(x).
    """
    rng = np.random.default_rng(0)
    history = rng.random((9, 3))
    far_point = np.array([50.0, 50.0, 50.0])

    scored_first = Inqmad(input_shape=3, dim_x=32, gamma=1.0, random_state=0).fit(history)
    expected = scored_first.score_partial(far_point)

    model = Inqmad(input_shape=3, dim_x=32, gamma=1.0, random_state=0).fit(history)
    score = model.fit_score_partial(far_point)

    fitted_first = Inqmad(input_shape=3, dim_x=32, gamma=1.0, random_state=0).fit(
        np.vstack([history, far_point])
    )
    with_self_term = fitted_first.score_partial(far_point)

    assert score == expected
    # The score is the negated density, so the self-term 1/10 lowers it.
    assert with_self_term == pytest.approx((9 * expected - 1) / 10, rel=1e-5)
    assert score > with_self_term + 0.09


def test_docstring_keeps_math_backslashes():
    """Regression test for #120: LaTeX such as \\rho and \\tau must not
    turn into carriage returns or tabs in the rendered docstring.
    """
    assert "\r" not in Inqmad.__doc__
    assert "\t" not in Inqmad.__doc__


def test_scores_use_the_current_random_features():
    """Regression test for #171: the random Fourier features are read on
    every call, so replacing them after the model has scored takes effect
    instead of being ignored by a compiled trace.
    """
    rng = np.random.default_rng(0)
    X = rng.random((50, 3))
    q = np.array([0.5, 0.5, 0.5])

    model = Inqmad(input_shape=3, dim_x=32, gamma=1.0, random_state=0).fit(X)
    model.score_partial(q)
    other = Inqmad(input_shape=3, dim_x=32, gamma=1.0, random_state=1).fit(X)

    model.weights, model.offset = other.weights, other.offset
    model.rho, model.num_fitted = np.zeros_like(model.rho), 0
    model.fit(X)

    assert model.score_partial(q) == other.score_partial(q)


def test_deleted_model_is_garbage_collected():
    """Regression test for #173: nothing at class or module level keeps a
    model or its random features alive after its last reference is gone
    (the jit caches used to hold every model's feature map).
    """
    rng = np.random.default_rng(0)
    model = Inqmad(input_shape=3, dim_x=32, gamma=1.0, random_state=0).fit(rng.random((5, 3)))
    model.score_partial(rng.random(3))
    refs = [weakref.ref(model), weakref.ref(model.weights), weakref.ref(model.rho)]

    del model
    gc.collect()

    assert all(ref() is None for ref in refs)


def test_density_matrix_keeps_precision_on_long_streams():
    """Regression test for #175: rho accumulates in float64, so a million
    fits of the same instance still average to that instance's psi psi^T
    (float32 accumulation was off by about 5e-5 here). The states are
    added 10,000 at a time, since a million fit_partial calls are slow.
    """
    x = np.array([0.1, 0.2, 0.3])
    model = Inqmad(input_shape=3, dim_x=32, gamma=1.0, random_state=0)
    psi = model._states(x)[0]
    for _ in range(100):
        model._fit_states(np.tile(psi, (10_000, 1)))

    assert model.rho.dtype == np.float64
    np.testing.assert_allclose(
        model.rho / model.num_fitted, np.outer(psi, psi), rtol=1e-9, atol=1e-12
    )


@pytest.mark.parametrize("name", ["input_shape", "dim_x"])
@pytest.mark.parametrize("value", [0, -1])
def test_rejects_sizes_below_one(name, value):
    """Regression test for #172: a zero or negative input_shape or dim_x
    raises a ValueError that names the parameter, instead of an error from
    scikit-learn about n_components or an empty array.
    """
    kwargs = {"input_shape": 3, "dim_x": 32, "gamma": 1.0, name: value}

    with pytest.raises(ValueError, match=f"{name} must be at least 1, got {value}"):
        Inqmad(**kwargs)


@pytest.mark.parametrize("name", ["input_shape", "dim_x"])
@pytest.mark.parametrize("value", [True, 3.0, "3", None])
def test_rejects_sizes_that_are_not_ints(name, value):
    """Regression test for #172: input_shape and dim_x must be ints, and a
    bool, which np.zeros would take as a size of 0 or 1, is rejected too.
    """
    kwargs = {"input_shape": 3, "dim_x": 32, "gamma": 1.0, name: value}

    with pytest.raises(TypeError, match=f"{name} must be an int"):
        Inqmad(**kwargs)


def test_accepts_numpy_int_sizes():
    """Regression test for #172: NumPy integers, such as a shape read off
    an array, are valid sizes.
    """
    model = Inqmad(input_shape=np.int64(3), dim_x=np.int32(8), gamma=1.0, random_state=0)

    assert model.weights.shape == (3, 8)
    assert model.rho.shape == (8, 8)


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
def test_rejects_instances_of_the_wrong_shape(method, X):
    """Regression test for #172: an instance that is not one row of
    input_shape features raises a ValueError naming the expected shape,
    instead of a matmul error or a silently fitted batch, and leaves the
    model unchanged.
    """
    rng = np.random.default_rng(0)
    model = Inqmad(input_shape=3, dim_x=32, gamma=1.0, random_state=0).fit(rng.random((5, 3)))
    rho = model.rho.copy()

    with pytest.raises(ValueError, match=r"one instance of shape \(3,\)"):
        getattr(model, method)(X)

    np.testing.assert_array_equal(model.rho, rho)
    assert model.num_fitted == 5


@pytest.mark.parametrize("method", ["fit_partial", "score_partial", "fit_score_partial"])
def test_accepts_one_row_instances(method):
    """Regression test for #172: a (1, input_shape) row and a list are the
    same instance as the (input_shape,) array.
    """
    rng = np.random.default_rng(0)
    X = rng.random((5, 3))
    x = rng.random(3)

    results = []
    for instance in [x, x[None, :], list(x)]:
        model = Inqmad(input_shape=3, dim_x=32, gamma=1.0, random_state=0).fit(X)
        result = getattr(model, method)(instance)
        results.append((result if method != "fit_partial" else None, model.rho.copy()))

    for score, rho in results[1:]:
        assert score == results[0][0]
        np.testing.assert_array_equal(rho, results[0][1])
