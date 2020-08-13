from pysad.transform.postprocessing import AveragePostprocessor, MaxPostprocessor, MedianPostprocessor, \
    ZScorePostprocessor
from pysad.transform.postprocessing import RunningZScorePostprocessor, \
    RunningMedianPostprocessor, RunningMaxPostprocessor, RunningAveragePostprocessor


def helper_get_scores():
    import numpy as np
    scores = np.random.rand(100)

    return scores


def test_postprocessors_shape():
    scores = helper_get_scores()

    postprocessors = {
        AveragePostprocessor: {},
        MaxPostprocessor: {},
        MedianPostprocessor: {},
        ZScorePostprocessor: {},
        RunningAveragePostprocessor: { "window_size": 30 },
        RunningMaxPostprocessor: { "window_size": 30 },
        RunningMedianPostprocessor: { "window_size": 30 },
        RunningZScorePostprocessor: { "window_size": 30 },
    }

    for postprocessor_cls, params_dict in postprocessors.items():
        postprocessor = postprocessor_cls(**params_dict)
        postprocessed_scores = postprocessor.fit_transform(scores)
        assert scores.shape == postprocessed_scores.shape
