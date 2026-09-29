import pytest


def test_ensemblers():
    import numpy as np
    from pysad.transform.ensemble import AverageScoreEnsembler, MaximumScoreEnsembler, MedianScoreEnsembler, \
    AverageOfMaximumScoreEnsembler, MaximumOfAverageScoreEnsembler

    scores = np.random.rand(100, 10)

    ensemblers = {
        AverageScoreEnsembler: {},
        MaximumScoreEnsembler: {},
        MedianScoreEnsembler: {},
        AverageOfMaximumScoreEnsembler: {},
        MaximumOfAverageScoreEnsembler: {}
    }

    for ensembler_cls, params_dict in ensemblers.items():
        ensembler = ensembler_cls(**params_dict)
        ensembled_scores = ensembler.fit_transform(scores)

        assert ensembled_scores.shape == (scores.shape[0], )

        ensembler = ensembler_cls(**params_dict).fit(scores)
        ensembled_scores = ensembler.transform(scores)

        assert ensembled_scores.shape == (scores.shape[0], )


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
    from pysad.transform.ensemble import AverageScoreEnsembler, MaximumScoreEnsembler, MedianScoreEnsembler

    scores = np.array(SCORES)

    np.testing.assert_allclose(MaximumScoreEnsembler().fit_transform(scores), np.max(scores, axis=1))
    np.testing.assert_allclose(MedianScoreEnsembler().fit_transform(scores), np.median(scores, axis=1))
    np.testing.assert_allclose(AverageScoreEnsembler().fit_transform(scores), np.mean(scores, axis=1))


def test_weighted_average_ensembler_output_values():
    import numpy as np
    from pysad.transform.ensemble import AverageScoreEnsembler

    scores = np.array(SCORES)
    weights = np.array([[1, 2, 3]])

    expected = np.array([
        (1 * 0.1 + 2 * 0.5 + 3 * 0.9) / 6,
        (1 * 0.8 + 2 * 0.2 + 3 * 0.4) / 6,
        (1 * 0.3 + 2 * 0.3 + 3 * 0.6) / 6,
        (1 * 1.0 + 2 * 0.0 + 3 * 0.5) / 6,
    ])

    ensembled_scores = AverageScoreEnsembler(estimator_weights=weights).fit_transform(scores)

    np.testing.assert_allclose(ensembled_scores, expected)


def test_bucket_ensemblers_output_values():
    import numpy as np
    from pyod.models.combination import aom, moa
    from pysad.transform.ensemble import AverageOfMaximumScoreEnsembler, MaximumOfAverageScoreEnsembler

    scores = np.array(BUCKET_SCORES)
    n_buckets = 3

    for ensembler_cls, combine in [(AverageOfMaximumScoreEnsembler, aom), (MaximumOfAverageScoreEnsembler, moa)]:
        np.random.seed(0)
        ensembled_scores = ensembler_cls(n_buckets=n_buckets).fit_transform(scores)

        # The ensembler combines one row at a time, so each row gets its own bucket shuffle.
        np.random.seed(0)
        expected = np.array([combine(row.reshape(1, -1), n_buckets=n_buckets)[0] for row in scores])

        np.testing.assert_allclose(ensembled_scores, expected)


def test_ensemblers_partial_matches_batch():
    import numpy as np
    from pysad.transform.ensemble import AverageScoreEnsembler, MaximumScoreEnsembler, MedianScoreEnsembler, \
        AverageOfMaximumScoreEnsembler, MaximumOfAverageScoreEnsembler

    scores = np.array(BUCKET_SCORES)

    ensemblers = {
        AverageScoreEnsembler: {},
        MaximumScoreEnsembler: {},
        MedianScoreEnsembler: {},
        AverageOfMaximumScoreEnsembler: {"n_buckets": 3},
        MaximumOfAverageScoreEnsembler: {"n_buckets": 3},
    }

    for ensembler_cls, params_dict in ensemblers.items():
        np.random.seed(0)
        batch_scores = ensembler_cls(**params_dict).fit_transform(scores)

        np.random.seed(0)
        ensembler = ensembler_cls(**params_dict)
        for row, batch_score in zip(scores, batch_scores, strict=True):
            partial_score = ensembler.fit_transform_partial(row)

            assert isinstance(partial_score, float)
            np.testing.assert_allclose(partial_score, batch_score)


@pytest.mark.parametrize("ensembler_cls,params_dict", [
    ("AverageScoreEnsembler", {}),
    ("MaximumScoreEnsembler", {}),
    ("MedianScoreEnsembler", {}),
    ("AverageOfMaximumScoreEnsembler", {"n_buckets": 3}),
    ("MaximumOfAverageScoreEnsembler", {"n_buckets": 3}),
])
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
