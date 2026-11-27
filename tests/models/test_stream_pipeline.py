import pytest


def _stages():
    from pysad.models import LODA
    from pysad.transform.postprocessing import ZScorePostprocessor
    from pysad.transform.preprocessing import ShingleTransformer
    from pysad.transform.probability_calibration import ConformalProbabilityCalibrator

    return {
        "model": LODA(random_state=0),
        "preprocessors": [ShingleTransformer(shingle_size=3)],
        "postprocessors": [
            ZScorePostprocessor(window_size=50),
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


def test_stream_pipeline_pipe_operator_composition_and_batch_parity():
    import numpy as np
    import pytest

    from pysad.models import LODA, StreamPipeline
    from pysad.transform.postprocessing import ThresholdFilter, ZScorePostprocessor
    from pysad.transform.preprocessing import InstanceStandardScaler, ShingleTransformer
    from pysad.transform.probability_calibration import ConformalProbabilityCalibrator

    X = np.random.RandomState(7).rand(150, 3)

    t1 = InstanceStandardScaler()
    t2 = ShingleTransformer(shingle_size=3)
    m = LODA(random_state=0)
    p1 = ZScorePostprocessor(window_size=30)
    p2 = ConformalProbabilityCalibrator(window_size=30)

    # Left-to-right, parenthesized chains, and incremental composition flatten identically.
    pipe_flat = t1 | t2 | m | p1 | p2
    assert isinstance(pipe_flat, StreamPipeline)
    assert pipe_flat.model is m
    assert pipe_flat.preprocessors == (t1, t2)
    assert pipe_flat.postprocessors == (p1, p2)

    pipe_grouped = (t1 | t2) | m | (p1 | p2)
    assert pipe_grouped.preprocessors == (t1, t2)
    assert pipe_grouped.postprocessors == (p1, p2)

    pipe_prepend = t1 | (t2 | (m | p1) | p2)
    assert pipe_prepend.preprocessors == (t1, t2)
    assert pipe_prepend.postprocessors == (p1, p2)

    # Batch fit_score matches partial fit_score_partial bit-for-bit.
    def make_pipe():
        return (
            InstanceStandardScaler()
            | ShingleTransformer(shingle_size=3)
            | LODA(random_state=0)
            | ZScorePostprocessor(window_size=30)
            | ConformalProbabilityCalibrator(window_size=30)
        )

    p_batch = make_pipe()
    p_stream = make_pipe()
    streamed = np.array([p_stream.fit_score_partial(x) for x in X])
    np.testing.assert_array_equal(p_batch.fit_score(X), streamed)

    # Batch fit + score matches partial fit_partial + score_partial bit-for-bit.
    p_fit_batch = make_pipe().fit(X[:100])
    p_fit_stream = make_pipe()
    for x in X[:100]:
        p_fit_stream.fit_partial(x)
    np.testing.assert_array_equal(
        p_fit_batch.score(X[100:]),
        np.array([p_fit_stream.score_partial(x) for x in X[100:]]),
    )

    # Standalone _TransformerChain and _PostprocessorChain parity with partial loops.
    t_chain_b = InstanceStandardScaler() | ShingleTransformer(shingle_size=3)
    t_chain_s = InstanceStandardScaler() | ShingleTransformer(shingle_size=3)
    np.testing.assert_array_equal(
        t_chain_b.fit_transform(X),
        np.array([t_chain_s.fit_transform_partial(x) for x in X]),
    )
    np.testing.assert_array_equal(
        t_chain_b.transform(X[:20]),
        np.array([t_chain_s.transform_partial(x) for x in X[:20]]),
    )

    raw_scores = np.random.RandomState(9).rand(80)
    p_chain_b = ZScorePostprocessor(window_size=15) | ThresholdFilter(threshold=1.0)
    p_chain_s = ZScorePostprocessor(window_size=15) | ThresholdFilter(threshold=1.0)
    np.testing.assert_array_equal(
        p_chain_b.fit_transform(raw_scores),
        np.array([p_chain_s.fit_transform_partial(float(s)) for s in raw_scores]),
    )
    np.testing.assert_array_equal(
        p_chain_b.transform(raw_scores[:20]),
        np.array([p_chain_s.transform_partial(float(s)) for s in raw_scores[:20]]),
    )

    # Length mismatch in fit_score raises before mutating stages.
    fresh = make_pipe()
    with pytest.raises(ValueError, match="same length"):
        fresh.fit_score(X[:5], np.zeros(3, dtype=np.int32))
    assert len(fresh.preprocessors[1].recent) == 0


def test_stream_pipeline_normalizes_stage_iterables_to_tuples():
    import numpy as np

    from pysad.models import LODA, StreamPipeline
    from pysad.transform.postprocessing import ZScorePostprocessor
    from pysad.transform.preprocessing import InstanceStandardScaler

    pipe = StreamPipeline(
        model=LODA(random_state=0),
        preprocessors=(s for s in [InstanceStandardScaler()]),
        postprocessors=(p for p in [ZScorePostprocessor(window_size=10)]),
    )
    assert isinstance(pipe.preprocessors, tuple)
    assert isinstance(pipe.postprocessors, tuple)
    X = np.random.RandomState(0).rand(10, 2)
    assert pipe.fit_score(X).shape == (10,)
    assert pipe.score(X).shape == (10,)


def _postprocessor_cases():
    import numpy as np

    from pysad.models import LODA, RelativeEntropy
    from pysad.transform.postprocessing import (
        AveragePostprocessor,
        MaxPostprocessor,
        MedianPostprocessor,
        NonparametricDynamicThresholdPostprocessor,
        QuantileFilter,
        ThresholdFilter,
        ZScorePostprocessor,
    )

    rng = np.random.RandomState(0)
    X = rng.rand(120, 3)
    return [
        (lambda: AveragePostprocessor(window_size=20), lambda: LODA(random_state=0), X, "score"),
        (lambda: MaxPostprocessor(window_size=20), lambda: LODA(random_state=0), X, "score"),
        (lambda: MedianPostprocessor(window_size=20), lambda: LODA(random_state=0), X, "score"),
        (lambda: ZScorePostprocessor(window_size=20), lambda: LODA(random_state=0), X, "score"),
        # NonparametricDynamicThresholdPostprocessor needs nonnegative scores;
        # RelativeEntropy emits 0.0/1.0 decisions, which it accepts.
        (
            lambda: NonparametricDynamicThresholdPostprocessor(window_size=20),
            lambda: RelativeEntropy(min_val=0.0, max_val=1.0),
            X[:, :1],
            "score",
        ),
        (lambda: QuantileFilter(q=0.9), lambda: LODA(random_state=0), X, "decision"),
        (lambda: ThresholdFilter(threshold=0.5), lambda: LODA(random_state=0), X, "decision"),
    ]


@pytest.mark.parametrize(
    "postprocessor_factory,model_factory,X,expected_kind", _postprocessor_cases()
)
def test_stream_pipeline_with_each_postprocessor(
    postprocessor_factory, model_factory, X, expected_kind
):
    import numpy as np

    from pysad.models import StreamPipeline

    batch_pipe = StreamPipeline(model=model_factory(), postprocessors=[postprocessor_factory()])
    stream_pipe = StreamPipeline(model=model_factory(), postprocessors=[postprocessor_factory()])
    assert batch_pipe.output_kind == expected_kind

    batch = batch_pipe.fit_score(X)
    assert batch.shape == (len(X),)
    assert np.isfinite(batch).all()

    stream = np.array([stream_pipe.fit_score_partial(x) for x in X])
    np.testing.assert_array_equal(batch, stream)


def _calibrator_cases():
    import numpy as np

    from pysad.transform.probability_calibration import (
        AnomalyLikelihoodProbabilityCalibrator,
        ConformalProbabilityCalibrator,
        ConformalTSFMCalibrator,
        GaussianTailProbabilityCalibrator,
        GroupConditionalConformalCalibrator,
        SPOTProbabilityCalibrator,
    )

    X = np.random.RandomState(0).rand(120, 3)
    return [
        (
            lambda: AnomalyLikelihoodProbabilityCalibrator(
                learning_period=5,
                num_estimation_samples=10,
                window_size=30,
                reestimation_period=5,
                short_window_size=3,
            ),
            X,
            "probability",
        ),
        (lambda: ConformalProbabilityCalibrator(window_size=30), X, "probability"),
        (
            lambda: ConformalTSFMCalibrator(alpha=0.1, gamma=0.05, window_size=30),
            X,
            "probability",
        ),
        (
            lambda: GaussianTailProbabilityCalibrator(window_size=30, short_window_size=3),
            X,
            "probability",
        ),
        (lambda: GroupConditionalConformalCalibrator(window_size=30), X, "probability"),
        (lambda: SPOTProbabilityCalibrator(num_init=20, level=0.8), X, "probability"),
    ]


@pytest.mark.parametrize("calibrator_factory,X,expected_kind", _calibrator_cases())
def test_stream_pipeline_with_each_calibrator(calibrator_factory, X, expected_kind):
    import numpy as np

    from pysad.models import LODA, StreamPipeline

    batch_pipe = StreamPipeline(model=LODA(random_state=0), postprocessors=[calibrator_factory()])
    stream_pipe = StreamPipeline(model=LODA(random_state=0), postprocessors=[calibrator_factory()])
    assert batch_pipe.output_kind == expected_kind

    batch = batch_pipe.fit_score(X)
    assert batch.shape == (len(X),)
    assert np.isfinite(batch).all()
    assert ((batch >= 0.0) & (batch <= 1.0)).all()

    stream = np.array([stream_pipe.fit_score_partial(x) for x in X])
    np.testing.assert_array_equal(batch, stream)


def _score_postprocessor_chain_cases():
    import numpy as np

    from pysad.models import LODA, RelativeEntropy
    from pysad.transform.postprocessing import (
        AveragePostprocessor,
        MaxPostprocessor,
        MedianPostprocessor,
        NonparametricDynamicThresholdPostprocessor,
        ZScorePostprocessor,
    )

    X = np.random.RandomState(0).rand(120, 3)
    # NonparametricDynamicThresholdPostprocessor needs nonnegative scores
    # (documented in its docstring), so it is paired with RelativeEntropy, which
    # emits 0.0/1.0 decisions on univariate data.
    return [
        (lambda: AveragePostprocessor(window_size=20), lambda: LODA(random_state=0), X),
        (lambda: MaxPostprocessor(window_size=20), lambda: LODA(random_state=0), X),
        (lambda: MedianPostprocessor(window_size=20), lambda: LODA(random_state=0), X),
        (lambda: ZScorePostprocessor(window_size=20), lambda: LODA(random_state=0), X),
        (
            lambda: NonparametricDynamicThresholdPostprocessor(window_size=20),
            lambda: RelativeEntropy(min_val=0.0, max_val=1.0),
            X[:, :1],
        ),
    ]


def _calibrator_factories():
    from pysad.transform.probability_calibration import (
        AnomalyLikelihoodProbabilityCalibrator,
        ConformalProbabilityCalibrator,
        ConformalTSFMCalibrator,
        GaussianTailProbabilityCalibrator,
        GroupConditionalConformalCalibrator,
        SPOTProbabilityCalibrator,
    )

    return [
        lambda: AnomalyLikelihoodProbabilityCalibrator(
            learning_period=5,
            num_estimation_samples=10,
            window_size=30,
            reestimation_period=5,
            short_window_size=3,
        ),
        lambda: ConformalProbabilityCalibrator(window_size=30),
        lambda: ConformalTSFMCalibrator(alpha=0.1, gamma=0.05, window_size=30),
        lambda: GaussianTailProbabilityCalibrator(window_size=30, short_window_size=3),
        lambda: GroupConditionalConformalCalibrator(window_size=30),
        lambda: SPOTProbabilityCalibrator(num_init=20, level=0.8),
    ]


def _decision_postprocessor_factories():
    from pysad.transform.postprocessing import QuantileFilter, ThresholdFilter

    return [
        lambda: QuantileFilter(q=0.9),
        lambda: ThresholdFilter(threshold=0.5),
    ]


@pytest.mark.parametrize(
    "postprocessor_factory,model_factory,X,calibrator_factory",
    [
        (post, model, x, cal)
        for post, model, x in _score_postprocessor_chain_cases()
        for cal in _calibrator_factories()
    ],
)
def test_stream_pipeline_score_postprocessor_then_calibrator(
    postprocessor_factory, model_factory, X, calibrator_factory
):
    import numpy as np

    from pysad.models import StreamPipeline

    batch_pipe = StreamPipeline(
        model=model_factory(),
        postprocessors=[postprocessor_factory(), calibrator_factory()],
    )
    stream_pipe = StreamPipeline(
        model=model_factory(),
        postprocessors=[postprocessor_factory(), calibrator_factory()],
    )
    assert batch_pipe.output_kind == "probability"

    batch = batch_pipe.fit_score(X)
    assert batch.shape == (len(X),)
    assert np.isfinite(batch).all()
    assert ((batch >= 0.0) & (batch <= 1.0)).all()

    stream = np.array([stream_pipe.fit_score_partial(x) for x in X])
    np.testing.assert_array_equal(batch, stream)


@pytest.mark.parametrize(
    "postprocessor_factory,model_factory,X,decision_factory",
    [
        (post, model, x, dec)
        for post, model, x in _score_postprocessor_chain_cases()
        for dec in _decision_postprocessor_factories()
    ],
)
def test_stream_pipeline_score_postprocessor_then_decision_postprocessor(
    postprocessor_factory, model_factory, X, decision_factory
):
    import numpy as np

    from pysad.models import StreamPipeline

    batch_pipe = StreamPipeline(
        model=model_factory(),
        postprocessors=[postprocessor_factory(), decision_factory()],
    )
    stream_pipe = StreamPipeline(
        model=model_factory(),
        postprocessors=[postprocessor_factory(), decision_factory()],
    )
    assert batch_pipe.output_kind == "decision"

    batch = batch_pipe.fit_score(X)
    assert batch.shape == (len(X),)
    assert np.isfinite(batch).all()
    assert ((batch == 0.0) | (batch == 1.0)).all()

    stream = np.array([stream_pipe.fit_score_partial(x) for x in X])
    np.testing.assert_array_equal(batch, stream)


@pytest.mark.parametrize(
    "decision_factory,calibrator_factory",
    [(dec, cal) for dec in _decision_postprocessor_factories() for cal in _calibrator_factories()],
)
def test_stream_pipeline_decision_postprocessor_then_calibrator_raises(
    decision_factory, calibrator_factory
):
    from pysad.models import LODA, StreamPipeline

    with pytest.raises(ValueError, match="probability calibrator"):
        StreamPipeline(
            model=LODA(random_state=0),
            postprocessors=[decision_factory(), calibrator_factory()],
        )
