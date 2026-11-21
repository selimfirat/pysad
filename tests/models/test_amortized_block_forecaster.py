import numpy as np
import pytest

from pysad.integrations import AmortizedBlockForecaster


def _constant_forecaster(context: np.ndarray, horizon: int) -> np.ndarray:
    """A forecaster that always predicts the mean of the context."""
    return np.full(horizon, context.mean())


def _linear_forecaster(context: np.ndarray, horizon: int) -> np.ndarray:
    """A forecaster that predicts a linear trend continuation."""
    if len(context) < 2:
        return np.full(horizon, context[-1])
    trend = context[-1] - context[-2]
    return np.array([context[-1] + trend * (i + 1) for i in range(horizon)])


def _perfect_forecaster(context: np.ndarray, horizon: int) -> np.ndarray:
    """A forecaster that perfectly predicts the next values (for testing with known sequences)."""
    # This is a placeholder - in real tests we'd use a known sequence
    return np.zeros(horizon)


def test_basic_scoring_with_constant_forecaster():
    """Scores match the absolute error against the context mean."""
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 5.0, 4.0, 3.0, 2.0])
    context_length = 3
    horizon = 2

    model = AmortizedBlockForecaster(_constant_forecaster, context_length, horizon)
    scores = [model.fit_score_partial(np.array([v])) for v in values]

    # First context_length values: context filling, score = 0
    assert scores[0] == 0.0
    assert scores[1] == 0.0
    assert scores[2] == 0.0

    # Value 4 (index 3): context is [1,2,3], mean=2, forecast=[2,2], score=|4-2|=2
    # Then context becomes [2,3,4]
    assert scores[3] == 2.0

    # Value 5 (index 4): buffer index 1, forecast=2, score=|5-2|=3
    # Then context becomes [3,4,5]
    assert scores[4] == 3.0

    # Value 10 (index 5): buffer exhausted, re-forecast with context [3,4,5], mean=4, score=|10-4|=6
    # Then context becomes [4,5,10]
    assert scores[5] == 6.0

    # Value 5 (index 6): buffer index 1, forecast=4, score=|5-4|=1
    # Then context becomes [5,10,5]
    assert scores[6] == 1.0

    # Value 4 (index 7): buffer exhausted, re-forecast with context [5,10,5], mean=20/3, score=|4-20/3|
    # Then context becomes [10,5,4]
    expected = abs(4.0 - 20.0 / 3.0)
    assert abs(scores[7] - expected) < 1e-10

    # Value 3 (index 8): buffer index 1, forecast=20/3, score=|3-20/3|
    # Then context becomes [5,4,3]
    expected = abs(3.0 - 20.0 / 3.0)
    assert abs(scores[8] - expected) < 1e-10

    # Value 2 (index 9): buffer exhausted, re-forecast with context [5,4,3], mean=4, score=|2-4|=2
    expected = abs(2.0 - 4.0)
    assert abs(scores[9] - expected) < 1e-10


def test_relative_scoring():
    """With relative=True, scores are absolute relative errors."""
    values = np.array([10.0, 10.0, 10.0, 12.0, 8.0, 10.0, 10.0])
    context_length = 3
    horizon = 2

    model = AmortizedBlockForecaster(_constant_forecaster, context_length, horizon, relative=True)
    scores = [model.fit_score_partial(np.array([v])) for v in values]

    # First 3 values: context filling, score = 0
    assert scores[0] == 0.0
    assert scores[1] == 0.0
    assert scores[2] == 0.0

    # Value 12: context [10,10,10], mean=10, forecast=10, relative error = |12-10|/10 = 0.2
    assert scores[3] == 0.2

    # Value 8: forecast=10, relative error = |8-10|/10 = 0.2
    assert scores[4] == 0.2


def test_relative_zero_forecast_raises():
    """Relative scoring with zero forecast raises ZeroDivisionError."""

    def zero_forecaster(context: np.ndarray, horizon: int) -> np.ndarray:
        return np.zeros(horizon)

    np.array([1.0, 2.0, 3.0, 4.0])
    model = AmortizedBlockForecaster(zero_forecaster, context_length=3, horizon=1, relative=True)
    model.fit_partial(np.array([1.0]))
    model.fit_partial(np.array([2.0]))
    model.fit_partial(np.array([3.0]))

    with pytest.raises(ZeroDivisionError):
        model.fit_score_partial(np.array([4.0]))


def test_probationary_period():
    """First probationary_period values get score 0.0."""
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    model = AmortizedBlockForecaster(
        _constant_forecaster, context_length=2, horizon=2, probationary_period=3
    )

    scores = [model.fit_score_partial(np.array([v])) for v in values]

    # First 3 values: probationary period, score = 0
    assert scores[0] == 0.0
    assert scores[1] == 0.0
    assert scores[2] == 0.0

    # Value 4 (index 3): num_values_=3 >= probationary_period, context is [2,3], mean=2.5, score=|4-2.5|=1.5
    assert scores[3] == 1.5

    # Value 5 (index 4): buffer index 1, forecast=2.5, score=|5-2.5|=2.5
    assert scores[4] == 2.5


def test_buffer_refill_works_correctly():
    """The forecast buffer is refilled after horizon steps."""
    values = np.arange(1.0, 11.0)  # 1, 2, ..., 10
    context_length = 3
    horizon = 3

    model = AmortizedBlockForecaster(_constant_forecaster, context_length, horizon)
    scores = [model.fit_score_partial(np.array([v])) for v in values]

    # Context filling: first 3 values
    assert scores[0] == 0.0
    assert scores[1] == 0.0
    assert scores[2] == 0.0

    # Block 1: values 4,5,6 with context [1,2,3], mean=2
    assert scores[3] == 2.0  # |4-2|
    assert scores[4] == 3.0  # |5-2|
    assert scores[5] == 4.0  # |6-2|

    # Block 2: values 7,8,9 with context [4,5,6], mean=5
    assert scores[6] == 2.0  # |7-5|
    assert scores[7] == 3.0  # |8-5|
    assert scores[8] == 4.0  # |9-5|

    # Block 3: value 10 with context [7,8,9], mean=8
    assert scores[9] == 2.0  # |10-8|


def test_score_partial_does_not_change_state():
    """Calling score_partial multiple times without fit_partial gives the same score."""
    np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    model = AmortizedBlockForecaster(_constant_forecaster, context_length=3, horizon=2)

    # Fill context
    model.fit_partial(np.array([1.0]))
    model.fit_partial(np.array([2.0]))
    model.fit_partial(np.array([3.0]))

    # Score the same value multiple times
    score1 = model.score_partial(np.array([4.0]))
    score2 = model.score_partial(np.array([4.0]))
    score3 = model.score_partial(np.array([4.0]))

    # All should be the same (using the same buffer entry)
    assert score1 == score2 == score3
    assert score1 == 2.0  # |4-2|


def test_fit_partial_advances_buffer():
    """Calling fit_partial advances the buffer index."""
    np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    model = AmortizedBlockForecaster(_constant_forecaster, context_length=3, horizon=2)

    model.fit_partial(np.array([1.0]))
    model.fit_partial(np.array([2.0]))
    model.fit_partial(np.array([3.0]))

    # Score and fit value 4
    score1 = model.score_partial(np.array([4.0]))
    model.fit_partial(np.array([4.0]))

    # Score and fit value 5
    score2 = model.score_partial(np.array([5.0]))
    model.fit_partial(np.array([5.0]))

    # Score value 6 (should trigger refill)
    score3 = model.score_partial(np.array([6.0]))

    assert score1 == 2.0  # |4-2|
    assert score2 == 3.0  # |5-2|
    assert score3 == 2.0  # |6-5|, context is now [3,4,5], mean=4... wait


def test_rejects_multivariate_instances():
    """The detector rejects multivariate input."""
    model = AmortizedBlockForecaster(_constant_forecaster, context_length=3, horizon=2)

    with pytest.raises(ValueError, match="univariate"):
        model.fit_partial(np.zeros(2))
    with pytest.raises(ValueError, match="univariate"):
        model.score_partial(np.zeros(2))


def test_rejects_invalid_forecaster():
    """Non-callable forecaster raises TypeError."""
    with pytest.raises(TypeError, match="callable"):
        AmortizedBlockForecaster("not callable", context_length=3, horizon=2)


def test_rejects_invalid_context_length():
    """context_length < 1 raises ValueError."""
    with pytest.raises(ValueError, match="context_length.*at least 1"):
        AmortizedBlockForecaster(_constant_forecaster, context_length=0, horizon=2)


def test_rejects_invalid_horizon():
    """horizon < 1 raises ValueError."""
    with pytest.raises(ValueError, match="horizon.*at least 1"):
        AmortizedBlockForecaster(_constant_forecaster, context_length=3, horizon=0)


def test_rejects_negative_probationary_period():
    """Negative probationary_period raises ValueError."""
    with pytest.raises(ValueError, match="probationary_period.*at least 0"):
        AmortizedBlockForecaster(
            _constant_forecaster, context_length=3, horizon=2, probationary_period=-1
        )


def test_forecaster_return_shape_validation():
    """Forecaster returning wrong shape raises ValueError on buffer refill."""

    def bad_forecaster(context: np.ndarray, horizon: int) -> np.ndarray:
        return np.zeros(horizon + 1)  # Wrong shape

    model = AmortizedBlockForecaster(bad_forecaster, context_length=3, horizon=2)
    model.fit_partial(np.array([1.0]))
    model.fit_partial(np.array([2.0]))

    with pytest.raises(ValueError, match="shape"):
        model.fit_partial(np.array([3.0]))


def test_known_sequence_with_linear_forecaster():
    """Test with a linear sequence and linear forecaster."""
    # Sequence: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 (linear trend +1)
    values = np.arange(1.0, 11.0)
    context_length = 3
    horizon = 2

    model = AmortizedBlockForecaster(_linear_forecaster, context_length, horizon)
    scores = [model.fit_score_partial(np.array([v])) for v in values]

    # Context filling
    assert scores[0] == 0.0
    assert scores[1] == 0.0
    assert scores[2] == 0.0

    # Value 4: context [1,2,3], trend=1, forecast=[4,5], score=|4-4|=0
    assert scores[3] == 0.0

    # Value 5: forecast=5, score=|5-5|=0
    assert scores[4] == 0.0

    # Value 6: context [3,4,5], trend=1, forecast=[6,7], score=|6-6|=0
    assert scores[5] == 0.0

    # Value 7: forecast=7, score=|7-7|=0
    assert scores[6] == 0.0

    # All scores should be 0 for a perfectly predicted linear sequence
    assert all(s == 0.0 for s in scores)


def test_anomaly_detection_on_spikes():
    """The detector assigns high scores to spikes in a predictable sequence."""
    # Linear sequence with isolated spikes
    values = np.arange(1.0, 21.0)
    values[10] = 100.0  # Large positive spike at index 10
    values[15] = -50.0  # Large negative spike at index 15

    model = AmortizedBlockForecaster(_linear_forecaster, context_length=5, horizon=3)
    scores = [model.fit_score_partial(np.array([v])) for v in values]

    # Context filling (first 5 values)
    for i in range(5):
        assert scores[i] == 0.0

    # Spike at index 10 (value 100.0) should have high score
    # Context before spike: [6,7,8,9,10], trend=1, forecast=[11,12,13], score=|100-11|=89
    assert scores[10] > 50.0

    # Spike at index 15 (value -50.0) should have high score
    # Context before spike is contaminated by first spike, but still gets high score
    assert scores[15] > 30.0

    # Normal values before the first spike should have low scores (near 0 for linear sequence)
    normal_scores_before_spike = [scores[i] for i in range(5, 10)]
    assert max(normal_scores_before_spike) < 1.0


def test_pickle_round_trip():
    """The model can be pickled and restored with same behavior."""
    import pickle

    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    model = AmortizedBlockForecaster(_constant_forecaster, context_length=3, horizon=2)
    model.fit(values[:5])

    restored = pickle.loads(pickle.dumps(model))

    # Compare scores on remaining values
    original_scores = model.fit_score(values[5:].reshape(-1, 1))
    restored_scores = restored.fit_score(values[5:].reshape(-1, 1))

    np.testing.assert_array_equal(original_scores, restored_scores)


def test_fit_score_batch_matches_incremental():
    """fit_score gives the same result as incremental fit_score_partial."""
    values = np.random.default_rng(42).normal(size=50)
    context_length = 5
    horizon = 3

    model1 = AmortizedBlockForecaster(_constant_forecaster, context_length, horizon)
    model2 = AmortizedBlockForecaster(_constant_forecaster, context_length, horizon)

    batch_scores = model1.fit_score(values.reshape(-1, 1))
    incremental_scores = [model2.fit_score_partial(np.array([v])) for v in values]

    np.testing.assert_allclose(batch_scores, incremental_scores, rtol=0, atol=1e-12)
