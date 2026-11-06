def _stages():
    from pysad.models import LODA
    from pysad.transform.postprocessing import RunningZScorePostprocessor
    from pysad.transform.preprocessing import ShingleTransformer
    from pysad.transform.probability_calibration import ConformalProbabilityCalibrator

    return {
        "model": LODA(random_state=0),
        "preprocessors": [ShingleTransformer(shingle_size=3)],
        "postprocessors": [
            RunningZScorePostprocessor(window_size=50),
            ConformalProbabilityCalibrator(window_size=50),
        ],
    }


def test_stream_pipeline_matches_the_manual_loop():
    import numpy as np

    from pysad.models import StreamPipeline

    X = np.random.RandomState(0).rand(200, 2)

    stages = _stages()
    expected = []
    for x in X:
        for preprocessor in stages["preprocessors"]:
            x = preprocessor.fit_transform_partial(x)
        score = stages["model"].fit_score_partial(x)
        for postprocessor in stages["postprocessors"]:
            score = postprocessor.fit_transform_partial(score)
        expected.append(score)

    np.testing.assert_array_equal(StreamPipeline(**_stages()).fit_score(X), expected)


def test_stream_pipeline_score_partial_does_not_change_later_scores():
    import numpy as np

    from pysad.models import StreamPipeline

    X = np.random.RandomState(0).rand(200, 2)
    expected = StreamPipeline(**_stages()).fit_score(X)

    pipeline = StreamPipeline(**_stages())
    actual = []
    for x in X:
        actual.append(pipeline.fit_score_partial(x))
        pipeline.score_partial(x)

    np.testing.assert_array_equal(actual, expected)
