import numpy as np
import pytest


class NABKNNCAD:
    """Reference port of ``handleRecord`` from NAB's nab/detectors/knncad/knncad_detector.py.

    ``result`` holds the p-value of the last record before alarm suppression (0.0 where NAB returns 0.0 early).
    """

    def __init__(self, probationary_period):
        import numpy as np

        self.buf = []
        self.training = []
        self.calibration = []
        self.scores = []
        self.record_count = 0
        self.pred = -1
        self.k = 27
        self.dim = 19
        self.sigma = np.diag(np.ones(self.dim))
        self.probationaryPeriod = probationary_period
        self.result = 0.0

    def metric(self, a, b):
        import numpy as np

        diff = a - np.array(b)
        return np.dot(np.dot(diff, self.sigma), diff.T)

    def ncm(self, item, item_in_array=False):
        import numpy as np

        arr = [self.metric(x, item) for x in self.training]
        return np.sum(np.partition(arr, self.k + item_in_array)[: self.k + item_in_array])

    def handle_record(self, value):
        import numpy as np

        self.result = 0.0
        self.buf.append(value)
        self.record_count += 1

        if len(self.buf) < self.dim:
            return 0.0

        new_item = self.buf[-self.dim :]
        if self.record_count < self.probationaryPeriod:
            self.training.append(new_item)
            return 0.0

        ost = self.record_count % self.probationaryPeriod
        if ost == 0 or ost == int(self.probationaryPeriod / 2):
            try:
                self.sigma = np.linalg.inv(np.dot(np.array(self.training).T, self.training))
            except np.linalg.LinAlgError:
                print("Singular Matrix at record", self.record_count)
        if len(self.scores) == 0:
            self.scores = [self.ncm(v, True) for v in self.training]

        new_score = self.ncm(new_item)
        result = 1.0 * len(np.where(np.array(self.scores) < new_score)[0]) / len(self.scores)
        self.result = result

        if self.record_count >= 2 * self.probationaryPeriod:
            self.training.pop(0)
            self.training.append(self.calibration.pop(0))

        self.scores.pop(0)
        self.calibration.append(new_item)
        self.scores.append(new_score)

        if self.pred > 0:
            self.pred -= 1
            return 0.5
        elif result >= 0.9965:
            self.pred = int(self.probationaryPeriod / 5)
        return result


def generate_stream(seed=61):
    import numpy as np

    rng = np.random.RandomState(seed)
    X = np.sin(np.arange(600) / 10.0) + rng.normal(0.0, 0.1, size=600)
    X[[250, 262, 400, 530]] += 8.0

    return X.reshape(-1, 1)


def test_knn_cad_fit_score_partial_matches_nab():
    import numpy as np

    from pysad.models import KNNCAD

    X = generate_stream()
    reference = NABKNNCAD(probationary_period=100)
    expected = np.array([reference.handle_record(x[0]) for x in X])

    model = KNNCAD(probationary_period=100)
    actual = np.array([model.fit_score_partial(x) for x in X])

    # The stream must raise alarms that suppress later scores to 0.5.
    assert np.sum(expected == 0.5) > 0
    np.testing.assert_array_equal(actual, expected)


def test_knn_cad_score_partial_is_nab_p_value():
    import numpy as np

    from pysad.models import KNNCAD

    X = generate_stream()
    reference = NABKNNCAD(probationary_period=100)
    model = KNNCAD(probationary_period=100)

    expected_p, actual_p, expected, actual = [], [], [], []
    for x in X:
        actual_p.append(model.score_partial(x))
        actual.append(model.fit_score_partial(x))
        expected.append(reference.handle_record(x[0]))
        expected_p.append(reference.result)

    np.testing.assert_array_equal(actual_p, expected_p)
    np.testing.assert_array_equal(actual, expected)


def test_knn_cad_score_partial_has_no_side_effects():
    import copy

    import numpy as np

    from pysad.models import KNNCAD

    X = generate_stream()
    model = KNNCAD(probationary_period=100)

    # Covers the uninitialized model, probation, the first scored record (scores still empty),
    # sigma refreshes (records 150, 200, ...) and the records right after an alarm.
    checked = {0, 17, 18, 98, 99, 100, 149, 199, 250, 251, 263, 349, 400, 401, 599}
    for i, x in enumerate(X):
        if i in checked:
            before = copy.deepcopy(vars(model))
            first = model.score_partial(x)
            second = model.score_partial(x)
            after = vars(model)

            assert first == second
            assert before.keys() == after.keys()
            for key in before:
                np.testing.assert_equal(after[key], before[key], err_msg=key)

        model.fit_score_partial(x)


def test_knn_cad_outlier_scores_higher():
    import numpy as np

    from pysad.models import KNNCAD

    rng = np.random.default_rng(0)
    model = KNNCAD(probationary_period=100).fit(rng.random((500, 1)))

    first_inlier, second_inlier, outlier = model.score(np.array([[0.5], [0.2], [9.0]]))

    assert outlier > first_inlier
    assert outlier > second_inlier


@pytest.mark.parametrize("period", [1, 19, 20, 47, 47.0])
def test_knn_cad_rejects_probationary_period_below_minimum(period):
    import re

    from pysad.models import KNNCAD

    # 1 and 19 leave the training set empty; 20-47 have too few training windows for the
    # calibration scores (np.partition kth out of bounds). All of them must raise at
    # construction, with a message that names the actual minimum (48) and the rejected value.
    with pytest.raises(
        ValueError,
        match=re.escape(f"at least 48 (window length 19 plus k=27 plus 2), got {period!r}"),
    ):
        KNNCAD(probationary_period=period)


@pytest.mark.parametrize(
    "period", [float("nan"), float("inf"), float("-inf"), 48.5, np.float64(100.5)]
)
def test_knn_cad_rejects_non_integral_probationary_period(period):
    import re

    from pysad.models import KNNCAD

    # NaN and inf compare False to any bound, so a plain `<` check lets them through and the
    # model then fails later with an unrelated error. A period with a fractional part is not a
    # whole number of records.
    with pytest.raises(ValueError, match=re.escape(f"got {period!r}")):
        KNNCAD(probationary_period=period)


@pytest.mark.parametrize("period", [None, "48", True, np.True_, 100 + 0j])
def test_knn_cad_rejects_non_real_probationary_period(period):
    from pysad.models import KNNCAD

    # None, strings and complex numbers are not periods, and neither are bools (bool is a
    # subclass of int, but True is not an accepted period of 1).
    with pytest.raises(TypeError, match="probationary_period must be"):
        KNNCAD(probationary_period=period)


@pytest.mark.parametrize("period", [100.0, np.float64(100.0), np.float32(100.0)])
def test_knn_cad_accepts_integral_float_probationary_period(period):
    from pysad.models import KNNCAD

    # NAB's probation period helper (nab/util.py, getProbationPeriod) returns a float such as
    # 750.0, which master accepted; it is converted to the matching int and scores the same.
    model = KNNCAD(probationary_period=period)

    assert type(model.probationaryPeriod) is int
    assert model.probationaryPeriod == 100

    X = generate_stream()
    np.testing.assert_array_equal(model.fit_score(X), KNNCAD(probationary_period=100).fit_score(X))


@pytest.mark.parametrize("period", [np.uint8(200), np.int8(100), np.int64(200)])
def test_knn_cad_stores_narrow_numpy_integer_probationary_period_as_python_int(period):
    from pysad.models import KNNCAD

    # Narrow NumPy dtypes (int8/uint8) overflow in `2 * probationaryPeriod` (knn_cad.py) once the
    # stream passes 2 * period records, unless the period is converted to a plain Python int at
    # construction.
    model = KNNCAD(probationary_period=period)

    assert type(model.probationaryPeriod) is int
    assert model.probationaryPeriod == int(period)

    X = np.random.default_rng(0).random((500, 1))
    with np.errstate(over="raise"):
        scores = model.fit_score(X)

    assert np.isfinite(scores).all()


def test_knn_cad_accepts_minimum_probationary_period():
    from pysad.models import KNNCAD

    X = np.random.default_rng(0).random((100, 1))

    model = KNNCAD(probationary_period=48)
    scores = model.fit_score(X)

    assert len(scores) == len(X)
    assert np.isfinite(scores).all()
    # Before the probationary period ends (records 1-47, i.e. indices 0-46) every score is 0.0.
    assert np.all(scores[:47] == 0.0)
    # From record 48 (index 47) on, the model is actually scoring: at least one score is nonzero.
    assert np.any(scores[47:] != 0.0)


def test_knn_cad_accepts_scalar_and_single_value_2d_instances():
    import numpy as np

    from pysad.models import KNNCAD

    X = generate_stream()
    model = KNNCAD(probationary_period=100)
    expected = np.array([model.fit_score_partial(x) for x in X])

    for instances in (X[:, 0], [x.reshape(1, 1) for x in X], list(X[:, 0].astype(float))):
        model = KNNCAD(probationary_period=100)
        actual = np.array([model.fit_score_partial(x) for x in instances])
        np.testing.assert_array_equal(actual, expected)

    # The batch path hands each row of a 1-D stream to the model as a scalar.
    np.testing.assert_array_equal(KNNCAD(probationary_period=100).fit_score(X[:, 0]), expected)


def test_knn_cad_rejects_multivariate_instances_without_changing_state():
    import numpy as np
    import pytest

    from pysad.models import KNNCAD

    X = generate_stream()
    model = KNNCAD(probationary_period=100)
    for x in X[:150]:
        model.fit_score_partial(x)
    buf, record_count = list(model.buf), model.record_count

    for bad in (np.array([0.1, 0.2, 0.3]), np.zeros((1, 2)), np.array([])):
        for method in (model.fit_partial, model.score_partial, model.fit_score_partial):
            with pytest.raises(ValueError, match="univariate"):
                method(bad)
    assert model.buf == buf
    assert model.record_count == record_count

    with pytest.raises(ValueError, match="univariate"):
        KNNCAD(probationary_period=100).fit_score(np.random.RandomState(0).rand(200, 3))
