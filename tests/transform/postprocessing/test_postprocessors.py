import warnings

from pysad.transform.postprocessing import (
    AveragePostprocessor,
    MaxPostprocessor,
    MedianPostprocessor,
    RunningAveragePostprocessor,
    RunningMaxPostprocessor,
    RunningMedianPostprocessor,
    RunningZScorePostprocessor,
    ZScorePostprocessor,
)


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
        RunningAveragePostprocessor: {"window_size": 30},
        RunningMaxPostprocessor: {"window_size": 30},
        RunningMedianPostprocessor: {"window_size": 30},
        RunningZScorePostprocessor: {"window_size": 30},
    }

    for postprocessor_cls, params_dict in postprocessors.items():
        postprocessor = postprocessor_cls(**params_dict)
        # Suppress RuntimeWarning for division by zero in z-score calculations
        # This can happen with small variance values or edge cases
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message="invalid value encountered in scalar divide",
            )
            postprocessed_scores = postprocessor.fit_transform(scores)
        assert scores.shape == postprocessed_scores.shape


# Fixed scores. The last two values make every running postprocessor differ from its
# cumulative counterpart; without them RunningMaxPostprocessor(window_size=3) would
# match MaxPostprocessor exactly and a broken window would go unnoticed.
SCORES = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
WINDOW_SIZE = 3


def test_cumulative_postprocessors_output_values():
    import numpy as np

    scores = np.array(SCORES, dtype=float)

    for postprocessor_cls, reference in [
        (AveragePostprocessor, np.mean),
        (MaxPostprocessor, np.max),
        (MedianPostprocessor, np.median),
    ]:
        expected = [reference(scores[: i + 1]) for i in range(len(scores))]

        np.testing.assert_allclose(
            postprocessor_cls().fit_transform(scores), expected, err_msg=postprocessor_cls.__name__
        )


def test_running_postprocessors_output_values():
    import numpy as np

    scores = np.array(SCORES, dtype=float)

    for postprocessor_cls, reference in [
        (RunningAveragePostprocessor, np.mean),
        (RunningMaxPostprocessor, np.max),
        (RunningMedianPostprocessor, np.median),
    ]:
        expected = [
            reference(scores[max(0, i - WINDOW_SIZE + 1) : i + 1]) for i in range(len(scores))
        ]

        np.testing.assert_allclose(
            postprocessor_cls(window_size=WINDOW_SIZE).fit_transform(scores),
            expected,
            err_msg=postprocessor_cls.__name__,
        )


def test_zscore_postprocessors_output_values():
    import numpy as np

    scores = np.array(SCORES, dtype=float)

    for postprocessor, window_size in [
        (ZScorePostprocessor(), None),
        (RunningZScorePostprocessor(window_size=WINDOW_SIZE), WINDOW_SIZE),
    ]:
        postprocessed_scores = postprocessor.fit_transform(scores)

        # A single observation has zero deviation from its own mean.
        assert postprocessed_scores[0] == 0.0

        expected = []
        for i in range(1, len(scores)):
            start = 0 if window_size is None else max(0, i - window_size + 1)
            values = scores[start : i + 1]
            expected.append((scores[i] - np.mean(values)) / np.std(values, ddof=0))

        np.testing.assert_allclose(
            postprocessed_scores[1:], expected, err_msg=type(postprocessor).__name__
        )


def test_zscore_postprocessors_return_zero_for_constant_stream():
    import numpy as np

    scores = np.full(10, 3.5)

    for postprocessor in [
        ZScorePostprocessor(),
        RunningZScorePostprocessor(window_size=WINDOW_SIZE),
    ]:
        np.testing.assert_array_equal(
            postprocessor.fit_transform(scores),
            np.zeros_like(scores),
            err_msg=type(postprocessor).__name__,
        )


def test_postprocessors_partial_matches_batch():
    import numpy as np

    scores = np.array(SCORES, dtype=float)

    postprocessors = {
        AveragePostprocessor: {},
        MaxPostprocessor: {},
        MedianPostprocessor: {},
        ZScorePostprocessor: {},
        RunningAveragePostprocessor: {"window_size": WINDOW_SIZE},
        RunningMaxPostprocessor: {"window_size": WINDOW_SIZE},
        RunningMedianPostprocessor: {"window_size": WINDOW_SIZE},
        RunningZScorePostprocessor: {"window_size": WINDOW_SIZE},
    }

    for postprocessor_cls, params_dict in postprocessors.items():
        # The postprocessors are stateful, so each path gets its own fresh instance.
        batch_scores = postprocessor_cls(**params_dict).fit_transform(scores)

        postprocessor = postprocessor_cls(**params_dict)
        partial_scores = [postprocessor.fit_transform_partial(score) for score in scores]

        # All postprocessors should produce the same values through either API.
        np.testing.assert_allclose(partial_scores, batch_scores, err_msg=postprocessor_cls.__name__)
