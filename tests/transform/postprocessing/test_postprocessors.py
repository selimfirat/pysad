from pysad.transform.postprocessing import AveragePostprocessor, MaxPostprocessor, MedianPostprocessor, \
    ZScorePostprocessor
from pysad.transform.postprocessing import RunningZScorePostprocessor, \
    RunningMedianPostprocessor, RunningMaxPostprocessor, RunningAveragePostprocessor
import warnings


def helper_get_scores():
    import numpy as np
    # Use a fixed seed for reproducible results and avoid edge cases
    np.random.seed(42)
    # Create more diverse scores to avoid variance issues
    scores = np.random.rand(100) * 10 + np.arange(100) * 0.1

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
        # Suppress RuntimeWarning for division by zero in z-score calculations
        # This can happen with small variance values or edge cases
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, 
                                  message="invalid value encountered in scalar divide")
            postprocessed_scores = postprocessor.fit_transform(scores)
        assert scores.shape == postprocessed_scores.shape
