import numpy as np
import pytest


def test_scalar_encoder_overlaps_fall_with_the_distance_of_the_buckets():
    from pysad.models.htm.encoders import RandomDistributedScalarEncoder

    encoder = RandomDistributedScalarEncoder(resolution=1.0, random_state=0)
    codes = {v: set(encoder.encode(float(v)).tolist()) for v in range(-40, 41)}

    assert all(len(code) == 21 for code in codes.values())
    for v in range(-40, 41):
        for distance in range(1, 41 - max(v, 0)):
            overlap = len(codes[v] & codes[v + distance])
            if distance < 21:
                assert overlap == 21 - distance
            else:
                assert overlap <= 2


def test_scalar_encoder_keeps_the_bits_of_a_value_and_does_not_learn_without_learn():
    from pysad.models.htm.encoders import RandomDistributedScalarEncoder

    encoder = RandomDistributedScalarEncoder(resolution=0.1, random_state=0)
    first = encoder.encode(5.0)
    preview = encoder.encode(9.0, learn=False)

    assert encoder.min_index_ == encoder.max_index_
    np.testing.assert_array_equal(encoder.encode(9.0), preview)
    np.testing.assert_array_equal(encoder.encode(5.0), first)
    # Values beyond the last bucket share its bits.
    np.testing.assert_array_equal(encoder.encode(1e9), encoder.encode(5.0 + 0.1 * 499))


def test_time_of_day_encoder_wraps_around_midnight():
    from pysad.models.htm.encoders import encode_time_of_day

    def bits(hours):
        return set(encode_time_of_day(hours * 3600.0).tolist())

    assert len(bits(0.0)) == 21
    assert bits(0.0) == bits(24.0) == bits(-24.0)
    assert len(bits(23.5) & bits(0.0)) > 15
    assert not bits(6.0) & bits(18.0)
    # The seconds do not count, as in NuPIC.
    assert bits(10.0) == set(encode_time_of_day(10 * 3600.0 + 59.0).tolist())


def _sequence(num_repeats):
    """A daily pattern of 24 hourly values, repeated."""
    hours = np.arange(24 * num_repeats)
    values = np.tile(np.r_[np.zeros(12), np.ones(12)], num_repeats)
    return np.column_stack([3600.0 * hours, values])


def test_raw_score_falls_on_a_learned_sequence_and_rises_on_a_new_one():
    from pysad.models import HTM

    model = HTM(min_val=0.0, max_val=1.0, likelihood=False, num_columns=512, random_state=0)
    scores = model.fit_score(_sequence(20))

    assert scores[:24].mean() > 0.9
    assert scores[-48:].mean() < 0.3
    # A value between the two levels, at an hour that the model knows.
    assert model.score_partial(np.array([3600.0 * 24 * 20, 0.5])) > 0.5


def test_likelihood_is_one_half_during_the_probationary_period():
    from pysad.models import HTM

    model = HTM(min_val=0.0, max_val=1.0, probationary_period=100, num_columns=256, random_state=0)
    scores = model.fit_score(_sequence(10))

    np.testing.assert_array_equal(scores[:100], 0.5)
    assert np.all((scores >= 0.0) & (scores <= 1.0))
    assert np.any(scores[100:] != 0.5)


def test_value_outside_the_range_so_far_is_a_spatial_anomaly():
    from pysad.models import HTM

    model = HTM(min_val=0.0, max_val=1.0, num_columns=256, random_state=0)
    model.fit(_sequence(2))

    assert model.score_partial(np.array([0.0, 1.06])) == 1.0
    assert model.score_partial(np.array([0.0, -0.06])) == 1.0
    # Within 5% of the range, the score is the likelihood, 0.5 during the probationary period.
    assert model.score_partial(np.array([0.0, 1.04])) == 0.5


def test_score_partial_does_not_change_the_model():
    from pysad.models import HTM

    X = _sequence(3)
    scored = HTM(min_val=0.0, max_val=1.0, num_columns=256, probationary_period=20, random_state=0)
    plain = HTM(min_val=0.0, max_val=1.0, num_columns=256, probationary_period=20, random_state=0)
    for x in X:
        # A new value, which needs new buckets of the encoder.
        scored.score_partial(x + [0.0, 0.5])
        scored.fit_partial(x)
        plain.fit_partial(x)

    np.testing.assert_array_equal(scored.score(X), plain.score(X))


def test_min_val_above_max_val_raises():
    from pysad.models import HTM

    with pytest.raises(ValueError, match="min_val"):
        HTM(min_val=1.0, max_val=0.0)
