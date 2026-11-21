"""Property-based tests for RingBuffer chronology and capacity.

The public result of ``RingBuffer`` (oldest-first items in :meth:`to_array`,
the stored count in :meth:`__len__`) must match a bounded
``collections.deque(maxlen=window_size)`` after every append, no matter how
many times the ring wraps around. The buffer never holds more than
``window_size`` items.
"""

from collections import deque

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pysad.utils import RingBuffer

MAX_EXAMPLES = 40

RING_BUFFER_SETTINGS = settings(
    max_examples=MAX_EXAMPLES,
    deadline=None,
    # The same examples on every run, so a failure in CI reproduces locally.
    derandomize=True,
)


@st.composite
def _scalar_stream(draw):
    """Draw a window size and a list of scalar values, long enough to wrap."""
    window_size = draw(st.integers(min_value=1, max_value=16))
    values = draw(
        st.lists(
            st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
            min_size=0,
            max_size=64,
        )
    )
    return window_size, values


@st.composite
def _vector_stream(draw):
    """Draw a window size and a list of same-shaped 1-D vectors, long enough to wrap."""
    num_features = draw(st.integers(min_value=1, max_value=4))
    window_size = draw(st.integers(min_value=1, max_value=16))
    values = draw(
        st.lists(
            st.lists(
                st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
                min_size=num_features,
                max_size=num_features,
            ),
            min_size=0,
            max_size=64,
        )
    )
    return window_size, [np.array(value, dtype=np.float64) for value in values]


@given(stream=_scalar_stream())
@RING_BUFFER_SETTINGS
def test_ring_buffer_scalars_match_bounded_deque(stream):
    """Chronology and capacity: scalars match a deque after every append."""
    window_size, values = stream
    buf = RingBuffer(window_size)
    ref = deque(maxlen=window_size)
    for value in values:
        slot = buf.append(value)
        assert 0 <= slot < window_size
        ref.append(value)
    assert len(buf) == len(ref)
    assert len(buf) <= window_size
    if values:
        np.testing.assert_array_equal(buf.to_array(), np.array(ref, dtype=np.float64))
    else:
        assert buf.to_array().size == 0


@given(stream=_vector_stream())
@RING_BUFFER_SETTINGS
def test_ring_buffer_vectors_match_bounded_deque(stream):
    """Chronology and capacity: vectors match a deque after every append."""
    window_size, values = stream
    buf = RingBuffer(window_size)
    ref = deque(maxlen=window_size)
    for value in values:
        buf.append(value)
        ref.append(value)
    assert len(buf) == len(ref)
    assert len(buf) <= window_size
    if values:
        expected = np.stack(list(ref))
        np.testing.assert_array_equal(buf.to_array(), expected)
    else:
        assert buf.to_array().size == 0


@given(stream=_scalar_stream())
@RING_BUFFER_SETTINGS
def test_active_slices_match_deque_without_oldest(stream):
    """active_slices holds every item except the oldest once the window is full."""
    window_size, values = stream
    buf = RingBuffer(window_size)
    ref = deque(maxlen=window_size)
    for value in values:
        buf.append(value)
        ref.append(value)
    first, second, active_count = buf.active_slices()
    if len(ref) < window_size:
        assert active_count == len(ref)
        assert second is None
        if len(ref):
            np.testing.assert_array_equal(first, np.array(ref, dtype=np.float64))
        else:
            assert first.size == 0
    else:
        assert active_count == window_size - 1
        parts = [part for part in (first, second) if len(part) > 0]
        active = np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)
        # first + second holds every item except the oldest; order within is not
        # chronological, so compare sorted multisets.
        np.testing.assert_array_equal(np.sort(active), np.sort(np.array(ref, dtype=np.float64)[1:]))


def test_wraparound_over_many_cycles_matches_deque():
    """More than six full cycles still yield the oldest-first deque contents."""
    window_size = 4
    values = [float(value) for value in range(25)]
    buf = RingBuffer(window_size)
    ref = deque(maxlen=window_size)
    for value in values:
        buf.append(value)
        ref.append(value)
    assert len(buf) == window_size
    np.testing.assert_array_equal(buf.to_array(), np.array(ref, dtype=np.float64))


@pytest.mark.parametrize("window_size", [1, 2, 3, 5])
def test_to_array_is_oldest_first_after_every_wraparound(window_size):
    """Each append past capacity keeps ``to_array`` in oldest-first order."""
    buf = RingBuffer(window_size)
    ref = deque(maxlen=window_size)
    for value in range(3 * window_size + 2):
        buf.append(float(value))
        ref.append(float(value))
        np.testing.assert_array_equal(buf.to_array(), np.array(ref, dtype=np.float64))


def test_to_array_vectors_oldest_first_after_wraparound():
    """Vector items stay in oldest-first order after the ring wraps around."""
    window_size = 3
    buf = RingBuffer(window_size)
    ref = deque(maxlen=window_size)
    for i in range(2 * window_size + 4):
        item = np.array([float(i), float(10 * i)])
        buf.append(item)
        ref.append(item)
        np.testing.assert_array_equal(buf.to_array(), np.stack(list(ref)))


@pytest.mark.parametrize("window_size", [1, 2, 3, 8])
def test_capacity_never_exceeds_window_size(window_size):
    """The count stays at or below window_size and reaches it once full."""
    buf = RingBuffer(window_size)
    for value in range(3 * window_size):
        buf.append(float(value))
        assert len(buf) <= window_size
    assert len(buf) == window_size


@pytest.mark.parametrize("window_size", [0, -1, -5])
def test_window_size_must_be_positive(window_size):
    """A capacity below 1 raises ValueError."""
    with pytest.raises(ValueError, match="window_size"):
        RingBuffer(window_size)
