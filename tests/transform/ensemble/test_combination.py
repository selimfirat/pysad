import pytest

# pysad.transform.ensemble._combination copies the score combination functions of combo 0.1.3.
# The expected values below were computed with combo 0.1.3, so they pin the copy to its results.
SCORES = [
    [0.1, 0.5, 0.9, 0.2, 0.7, 0.4],
    [0.8, 0.2, 0.4, 0.6, 0.1, 0.3],
    [0.3, 0.3, 0.6, 0.9, 0.0, 0.5],
    [1.0, 0.0, 0.5, 0.2, 0.4, 0.8],
]


@pytest.mark.parametrize(
    "name, kwargs, expected",
    [
        (
            "average",
            {},
            [0.4666666666666666, 0.39999999999999997, 0.43333333333333335, 0.4833333333333334],
        ),
        ("maximization", {}, [0.9, 0.8, 0.9, 1.0]),
        ("median", {}, [0.45, 0.35, 0.4, 0.45]),
        (
            "aom",
            {"n_buckets": 3, "random_state": 0},
            [0.6999999999999998, 0.6, 0.6, 0.6666666666666666],
        ),
        (
            "aom",
            {"n_buckets": 2, "bootstrap_estimators": True, "random_state": 0},
            [0.9, 0.4, 0.6, 0.8],
        ),
        (
            "aom",
            {"n_buckets": 3, "method": "dynamic", "random_state": 0},
            [0.9, 0.4000000000000001, 0.6, 0.8000000000000002],
        ),
        ("moa", {"n_buckets": 3, "random_state": 0}, [0.65, 0.45, 0.6, 0.7]),
        (
            "moa",
            {"n_buckets": 2, "bootstrap_estimators": True, "random_state": 0},
            [0.6, 0.3, 0.46666666666666673, 0.43333333333333335],
        ),
        ("moa", {"n_buckets": 3, "method": "dynamic", "random_state": 0}, [0.65, 0.35, 0.55, 0.65]),
    ],
)
def test_combination_matches_combo(name, kwargs, expected):
    import numpy as np

    from pysad.transform.ensemble import _combination

    combined = getattr(_combination, name)(np.array(SCORES), **kwargs)

    np.testing.assert_allclose(combined, expected, rtol=1e-12)


def test_weighted_average_matches_combo():
    import numpy as np

    from pysad.transform.ensemble._combination import average

    weights = np.array([[1, 2, 3, 1, 2, 3]])

    np.testing.assert_allclose(
        average(np.array(SCORES), weights),
        [0.55, 0.34166666666666673, 0.425, 0.4916666666666667],
        rtol=1e-12,
    )
    with pytest.raises(ValueError, match="Bad input shape of estimator_weight"):
        average(np.array(SCORES), np.ones(6))


@pytest.mark.parametrize(
    "name, kwargs, error",
    [
        ("aom", {"n_buckets": 1}, ValueError),  # fewer than two buckets
        ("moa", {"n_buckets": 7}, ValueError),  # more buckets than estimators
        ("aom", {"n_buckets": 4}, ValueError),  # six estimators don't split into four equal buckets
        ("moa", {"n_buckets": 3, "method": "other"}, NotImplementedError),
    ],
)
def test_bucket_combination_rejects_invalid_buckets(name, kwargs, error):
    import numpy as np

    from pysad.transform.ensemble import _combination

    with pytest.raises(error):
        getattr(_combination, name)(np.array(SCORES), **kwargs)
