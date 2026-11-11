import warnings

from pysad.transform.postprocessing import (
    AveragePostprocessor,
    MaxPostprocessor,
    MedianPostprocessor,
    NonparametricDynamicThresholdPostprocessor,
    QuantileFilter,
    ThresholdFilter,
    ZScorePostprocessor,
)


def helper_get_scores():
    import numpy as np

    np.random.seed(42)
    scores = np.random.rand(100) * 10 + np.arange(100) * 0.1

    return scores


def test_postprocessors_shape():
    scores = helper_get_scores()

    cases = [
        (AveragePostprocessor, {}),
        (AveragePostprocessor, {"window_size": 30}),
        (MaxPostprocessor, {}),
        (MaxPostprocessor, {"window_size": 30}),
        (MedianPostprocessor, {}),
        (MedianPostprocessor, {"window_size": 30}),
        (ZScorePostprocessor, {}),
        (ZScorePostprocessor, {"window_size": 30}),
        (NonparametricDynamicThresholdPostprocessor, {"window_size": 30}),
        (QuantileFilter, {}),
        (ThresholdFilter, {"threshold": 5}),
    ]

    for postprocessor_cls, params_dict in cases:
        postprocessor = postprocessor_cls(**params_dict)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message="invalid value encountered in scalar divide",
            )
            postprocessed_scores = postprocessor.fit_transform(scores)
        assert scores.shape == postprocessed_scores.shape


# Fixed scores. The last two values make every windowed postprocessor differ from its
# cumulative counterpart; without them MaxPostprocessor(window_size=3) would
# match MaxPostprocessor() exactly and a broken window would go unnoticed.
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


def test_windowed_postprocessors_output_values():
    import numpy as np

    scores = np.array(SCORES, dtype=float)

    for postprocessor_cls, reference in [
        (AveragePostprocessor, np.mean),
        (MaxPostprocessor, np.max),
        (MedianPostprocessor, np.median),
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
        (ZScorePostprocessor(window_size=WINDOW_SIZE), WINDOW_SIZE),
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


def test_postprocessors_return_zero_before_the_first_fit():
    # StreamPipeline.score_partial transforms a score before any postprocessor or calibrator learns (#625).
    for postprocessor in [
        AveragePostprocessor(),
        AveragePostprocessor(window_size=WINDOW_SIZE),
        MaxPostprocessor(),
        MaxPostprocessor(window_size=WINDOW_SIZE),
        MedianPostprocessor(),
        MedianPostprocessor(window_size=WINDOW_SIZE),
        NonparametricDynamicThresholdPostprocessor(),
        QuantileFilter(),
        ZScorePostprocessor(),
        ZScorePostprocessor(window_size=WINDOW_SIZE),
    ]:
        assert postprocessor.transform_partial(0.7) == 0.0, type(postprocessor).__name__


def test_zscore_postprocessors_return_zero_for_constant_stream():
    import numpy as np

    # 0.1 is inexact in binary; its sum-of-squares variance went negative, giving nan (#224).
    for value in [3.5, 0.1]:
        scores = np.full(10, value)

        for postprocessor in [
            ZScorePostprocessor(),
            ZScorePostprocessor(window_size=WINDOW_SIZE),
        ]:
            np.testing.assert_array_equal(
                postprocessor.fit_transform(scores),
                np.zeros_like(scores),
                err_msg=f"{type(postprocessor).__name__} on {value}",
            )


def test_postprocessors_partial_matches_batch():
    import numpy as np

    scores = np.array(SCORES, dtype=float)

    cases = [
        (AveragePostprocessor, {}),
        (AveragePostprocessor, {"window_size": WINDOW_SIZE}),
        (MaxPostprocessor, {}),
        (MaxPostprocessor, {"window_size": WINDOW_SIZE}),
        (MedianPostprocessor, {}),
        (MedianPostprocessor, {"window_size": WINDOW_SIZE}),
        (ZScorePostprocessor, {}),
        (ZScorePostprocessor, {"window_size": WINDOW_SIZE}),
        (NonparametricDynamicThresholdPostprocessor, {"window_size": WINDOW_SIZE}),
        (QuantileFilter, {"q": 0.5}),
        (ThresholdFilter, {"threshold": 4}),
    ]

    for postprocessor_cls, params_dict in cases:
        batch_scores = postprocessor_cls(**params_dict).fit_transform(scores)

        postprocessor = postprocessor_cls(**params_dict)
        partial_scores = [postprocessor.fit_transform_partial(score) for score in scores]

        np.testing.assert_allclose(partial_scores, batch_scores, err_msg=postprocessor_cls.__name__)


def test_max_postprocessor_memory_does_not_grow_with_the_stream():
    import numpy as np

    # It used to keep every score of the stream (#388).
    scores = np.random.default_rng(0).random(10_000)
    postprocessor = MaxPostprocessor()
    out = postprocessor.fit_transform(scores)

    np.testing.assert_allclose(out, np.maximum.accumulate(scores))
    assert len(postprocessor.meter.lst) == 0


def test_dynamic_threshold_scores_a_spike_with_eq_4():
    import numpy as np

    scores = np.array([1.0, 2.0] * 20 + [10.0])
    # span=1 turns off the smoothing, so the window holds the raw scores.
    out = NonparametricDynamicThresholdPostprocessor(window_size=41, span=1).fit_transform(scores)

    # Every z with a threshold below the spike gives the same objective, and the largest wins.
    mean, std = scores.mean(), scores.std()
    z = max(z for z in np.arange(2.5, 10.5, 0.5) if mean + z * std < 10.0)
    np.testing.assert_array_equal(out[:-1], 0.0)
    np.testing.assert_allclose(out[-1], (10.0 - (mean + z * std)) / (mean + std))


def test_dynamic_threshold_flags_spikes_of_a_noisy_stream():
    import numpy as np

    scores = np.random.default_rng(0).exponential(size=3000)
    spikes = [1000, 1800, 2500]
    scores[spikes] += 30.0
    out = NonparametricDynamicThresholdPostprocessor(window_size=500, span=3).fit_transform(scores)

    assert (out[spikes] > 0).all()
    assert (out > 0).mean() < 0.01


def test_dynamic_threshold_pruning_removes_every_sequence_when_p_is_one():
    import numpy as np

    scores = np.random.default_rng(0).exponential(size=1000)
    scores[[300, 700]] += 30.0

    # A percent decrease cannot exceed 1, so pruning reclassifies every sequence as nominal.
    np.testing.assert_array_equal(
        NonparametricDynamicThresholdPostprocessor(window_size=200, p=1.0).fit_transform(scores),
        0.0,
    )


def test_dynamic_threshold_returns_zero_for_constant_stream():
    import numpy as np

    for value in [0.0, 0.1, 3.5]:
        np.testing.assert_array_equal(
            NonparametricDynamicThresholdPostprocessor(window_size=50).fit_transform(
                np.full(200, value)
            ),
            0.0,
        )


def test_dynamic_threshold_rejects_negative_scores():
    import pytest

    with pytest.raises(ValueError, match="nonnegative"):
        NonparametricDynamicThresholdPostprocessor().fit_partial(-1.0)


def test_quantile_filter_labels_scores_with_the_quantile_of_the_previous_scores():
    import numpy as np

    # With at most four previous scores, the quantile is exact.
    out = QuantileFilter(q=0.5).fit_transform(np.array([3.0, 1.0, 4.0, 2.0, 2.5]))

    # Previous medians: none, 3, 2, 3, 2.5.
    np.testing.assert_array_equal(out, [0.0, 0.0, 1.0, 0.0, 1.0])


def test_quantile_filter_flags_about_one_minus_q_of_the_scores():
    import numpy as np

    scores = np.random.default_rng(0).gamma(2.0, size=20_000)

    for q in [0.5, 0.9, 0.99]:
        out = QuantileFilter(q=q).fit_transform(scores)

        assert set(np.unique(out)) == {0.0, 1.0}
        assert abs(out.mean() - (1 - q)) < 0.01, q


def test_quantile_filter_transform_partial_does_not_learn():
    import numpy as np

    postprocessor = QuantileFilter(q=0.5).fit(np.array([1.0, 2.0, 3.0]))

    assert postprocessor.transform_partial(2.0) == 1.0
    assert postprocessor.transform_partial(1.5) == 0.0
    assert postprocessor.transform_partial(2.0) == 1.0


def test_threshold_filter_labels_scores_at_or_above_the_threshold():
    import numpy as np

    out = ThresholdFilter(threshold=4).fit_transform(np.array(SCORES, dtype=float))

    np.testing.assert_array_equal(out, [0, 0, 1, 0, 1, 1, 0, 1, 1, 0])


def test_postprocessors_batch_transform_matches_partial():
    import numpy as np
    import pytest

    with pytest.raises(ValueError, match=r"window_size must be a positive integer\."):
        NonparametricDynamicThresholdPostprocessor(window_size=0)

    train = np.array(SCORES, dtype=float)
    test = np.array([0.5, 2.5, 4.0, 7.5, 12.0])
    for cls, kwargs in [
        (AveragePostprocessor, {}),
        (AveragePostprocessor, {"window_size": WINDOW_SIZE}),
        (MaxPostprocessor, {}),
        (MaxPostprocessor, {"window_size": WINDOW_SIZE}),
        (MedianPostprocessor, {}),
        (MedianPostprocessor, {"window_size": WINDOW_SIZE}),
        (ZScorePostprocessor, {}),
        (ZScorePostprocessor, {"window_size": WINDOW_SIZE}),
        (NonparametricDynamicThresholdPostprocessor, {"window_size": 5, "span": 1}),
        (QuantileFilter, {"q": 0.5}),
        (ThresholdFilter, {"threshold": 4.0}),
    ]:
        unfitted = cls(**kwargs)
        np.testing.assert_allclose(
            unfitted.transform(test),
            [unfitted.transform_partial(float(s)) for s in test],
        )
        fitted = cls(**kwargs).fit(train)
        np.testing.assert_allclose(
            fitted.transform(test),
            [fitted.transform_partial(float(s)) for s in test],
        )
