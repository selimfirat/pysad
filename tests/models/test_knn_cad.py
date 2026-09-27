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
        return np.sum(np.partition(arr, self.k + item_in_array)[:self.k + item_in_array])

    def handle_record(self, value):
        import numpy as np

        self.result = 0.0
        self.buf.append(value)
        self.record_count += 1

        if len(self.buf) < self.dim:
            return 0.0

        new_item = self.buf[-self.dim:]
        if self.record_count < self.probationaryPeriod:
            self.training.append(new_item)
            return 0.0

        ost = self.record_count % self.probationaryPeriod
        if ost == 0 or ost == int(self.probationaryPeriod / 2):
            try:
                self.sigma = np.linalg.inv(np.dot(np.array(self.training).T, self.training))
            except np.linalg.LinAlgError:
                print('Singular Matrix at record', self.record_count)
        if len(self.scores) == 0:
            self.scores = [self.ncm(v, True) for v in self.training]

        new_score = self.ncm(new_item)
        result = 1. * len(np.where(np.array(self.scores) < new_score)[0]) / len(self.scores)
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
    from pysad.models import KNNCAD
    import numpy as np

    X = generate_stream()
    reference = NABKNNCAD(probationary_period=100)
    expected = np.array([reference.handle_record(x[0]) for x in X])

    model = KNNCAD(probationary_period=100)
    actual = np.array([model.fit_score_partial(x) for x in X])

    # The stream must raise alarms that suppress later scores to 0.5.
    assert np.sum(expected == 0.5) > 0
    np.testing.assert_array_equal(actual, expected)


def test_knn_cad_score_partial_is_nab_p_value():
    from pysad.models import KNNCAD
    import numpy as np

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
    from pysad.models import KNNCAD
    import copy
    import numpy as np

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
    from pysad.models import KNNCAD
    import numpy as np

    rng = np.random.default_rng(0)
    model = KNNCAD(probationary_period=100).fit(rng.random((500, 1)))

    first_inlier, second_inlier, outlier = model.score(np.array([[0.5], [0.2], [9.0]]))

    assert outlier > first_inlier
    assert outlier > second_inlier
