import pytest

# Fixed score matrix: 4 instances scored by 3 detectors.
SCORES = [
    [0.1, 0.5, 0.9],
    [0.8, 0.2, 0.4],
    [0.3, 0.3, 0.6],
    [1.0, 0.0, 0.5],
]

# Fixed score matrix with 6 detectors so that 3 static buckets of 2 detectors
# each make the bucket shuffling affect the result.
BUCKET_SCORES = [
    [0.1, 0.5, 0.9, 0.2, 0.7, 0.4],
    [0.8, 0.2, 0.4, 0.6, 0.1, 0.3],
    [0.3, 0.3, 0.6, 0.9, 0.0, 0.5],
    [1.0, 0.0, 0.5, 0.2, 0.4, 0.8],
]


def test_simple_ensemblers_output_values():
    import numpy as np

    from pysad.transform.ensemble import (
        AverageScoreEnsembler,
        MaximumScoreEnsembler,
        MedianScoreEnsembler,
    )

    scores = np.array(SCORES)

    np.testing.assert_allclose(
        MaximumScoreEnsembler().fit_transform(scores), np.max(scores, axis=1)
    )
    np.testing.assert_allclose(
        MedianScoreEnsembler().fit_transform(scores), np.median(scores, axis=1)
    )
    np.testing.assert_allclose(
        AverageScoreEnsembler().fit_transform(scores), np.mean(scores, axis=1)
    )


def test_weighted_average_ensembler_output_values():
    import numpy as np

    from pysad.transform.ensemble import AverageScoreEnsembler

    scores = np.array(SCORES)
    expected = np.array(
        [
            (1 * 0.1 + 2 * 0.5 + 3 * 0.9) / 6,
            (1 * 0.8 + 2 * 0.2 + 3 * 0.4) / 6,
            (1 * 0.3 + 2 * 0.3 + 3 * 0.6) / 6,
            (1 * 1.0 + 2 * 0.0 + 3 * 0.5) / 6,
        ]
    )

    for weights in (np.array([1.0, 2.0, 3.0]), np.array([[1.0, 2.0, 3.0]])):
        ensembled_scores = AverageScoreEnsembler(estimator_weights=weights).fit_transform(scores)
        np.testing.assert_allclose(ensembled_scores, expected)

    with pytest.raises(ValueError, match="Bad input shape of estimator_weights"):
        AverageScoreEnsembler(estimator_weights=np.ones(2)).fit_transform(scores)


@pytest.mark.parametrize(
    "ensembler_cls,kwargs",
    [
        ("AverageOfMaximumScoreEnsembler", {"n_buckets": 3, "random_state": 0}),
        (
            "AverageOfMaximumScoreEnsembler",
            {"n_buckets": 2, "bootstrap_estimators": True, "random_state": 0},
        ),
        (
            "AverageOfMaximumScoreEnsembler",
            {"n_buckets": 3, "method": "dynamic", "random_state": 0},
        ),
        ("MaximumOfAverageScoreEnsembler", {"n_buckets": 3, "random_state": 0}),
        (
            "MaximumOfAverageScoreEnsembler",
            {"n_buckets": 2, "bootstrap_estimators": True, "random_state": 0},
        ),
        (
            "MaximumOfAverageScoreEnsembler",
            {"n_buckets": 3, "method": "dynamic", "random_state": 0},
        ),
    ],
)
def test_bucket_ensemblers_stay_within_row_score_bounds(ensembler_cls, kwargs):
    import numpy as np

    from pysad.transform import ensemble as ensemble_module

    scores = np.array(BUCKET_SCORES)
    cls = getattr(ensemble_module, ensembler_cls)
    combined = cls(**kwargs).fit_transform(scores)
    assert combined.shape == (len(scores),)
    assert np.all(combined >= scores.min(axis=1) - 1e-12)
    assert np.all(combined <= scores.max(axis=1) + 1e-12)


@pytest.mark.parametrize(
    "ensembler_cls,kwargs,error",
    [
        ("AverageOfMaximumScoreEnsembler", {"n_buckets": 1}, ValueError),
        ("MaximumOfAverageScoreEnsembler", {"n_buckets": 7}, ValueError),
        ("AverageOfMaximumScoreEnsembler", {"n_buckets": 4}, ValueError),
        (
            "MaximumOfAverageScoreEnsembler",
            {"n_buckets": 3, "method": "other"},
            NotImplementedError,
        ),
    ],
)
def test_bucket_ensemblers_reject_invalid_buckets(ensembler_cls, kwargs, error):
    import numpy as np

    from pysad.transform import ensemble as ensemble_module

    cls = getattr(ensemble_module, ensembler_cls)
    with pytest.raises(error):
        cls(**kwargs).fit_transform(np.array(BUCKET_SCORES))


@pytest.mark.parametrize(
    "ensembler_cls,params_dict",
    [
        ("AverageScoreEnsembler", {}),
        ("MaximumScoreEnsembler", {}),
        ("MedianScoreEnsembler", {}),
        ("AverageOfMaximumScoreEnsembler", {"n_buckets": 3}),
        ("MaximumOfAverageScoreEnsembler", {"n_buckets": 3}),
    ],
)
def test_ensemblers_accept_list_scores(ensembler_cls, params_dict):
    """A plain list of scores must give the same result as the equivalent np.array (#208)."""
    import numpy as np

    from pysad.transform import ensemble as ensemble_module

    cls = getattr(ensemble_module, ensembler_cls)
    row = BUCKET_SCORES[0]

    np.random.seed(0)
    from_list = cls(**params_dict).fit_transform_partial(row)

    np.random.seed(0)
    from_array = cls(**params_dict).fit_transform_partial(np.array(row))

    np.testing.assert_allclose(from_list, from_array)


@pytest.mark.parametrize(
    "ensembler_cls",
    ["AverageOfMaximumScoreEnsembler", "MaximumOfAverageScoreEnsembler"],
)
@pytest.mark.parametrize("params", [{"bootstrap_estimators": True}, {"method": "dynamic"}])
def test_bucket_ensemblers_draw_distinct_buckets_with_an_int_seed(ensembler_cls, params):
    """An int random_state draws the buckets like a RandomState with that seed (#627)."""
    import numpy as np

    from pysad.transform import ensemble as ensemble_module

    cls = getattr(ensemble_module, ensembler_cls)
    scores = np.random.RandomState(1).rand(20)

    from_seed = cls(n_buckets=5, random_state=0, **params).transform_partial(scores)
    from_generator = cls(
        n_buckets=5, random_state=np.random.RandomState(0), **params
    ).transform_partial(scores)

    assert from_seed == from_generator
