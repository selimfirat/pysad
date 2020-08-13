
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
