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
from hypothesis.stateful import RuleBasedStateMachine, initialize, rule

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


# Hypothesis stateful property tests for RingBuffer chronology across multiple
# wraparounds. These use RuleBasedStateMachine to model sequences of append
# operations and verify chronology against a reference deque after every step.


STATEFUL_SETTINGS = settings(
    max_examples=20,
    deadline=None,
    database=None,
    derandomize=True,
)


class _ScalarRingBufferMachine(RuleBasedStateMachine):
    """State machine for testing scalar RingBuffer chronology across wraparounds."""

    window_size: int
    buf: RingBuffer
    ref: deque[float]
    step: int

    @initialize()
    def setup(self):
        # Use a small window size to force frequent wraparounds
        self.window_size = 4
        self.buf = RingBuffer(self.window_size)
        self.ref = deque(maxlen=self.window_size)
        self.step = 0

    @rule()
    def append_scalar(self):
        # Draw values from a finite set to make duplicates likely
        value = float(self.step * 7 % 13 - 5)  # Deterministic but varied sequence
        slot = self.buf.append(value)
        assert 0 <= slot < self.window_size
        self.ref.append(value)
        self.step += 1

    @rule()
    def check_chronology(self):
        # to_array must match deque contents in oldest-first order
        expected = np.array(self.ref, dtype=np.float64)
        actual = self.buf.to_array()
        np.testing.assert_array_equal(actual, expected)

    @rule()
    def check_length(self):
        # Length must match deque length and never exceed window_size
        assert len(self.buf) == len(self.ref)
        assert len(self.buf) <= self.window_size

    @rule()
    def check_active_slices(self):
        # active_slices must match deque without the oldest item when full
        first, second, active_count = self.buf.active_slices()
        if len(self.ref) < self.window_size:
            assert active_count == len(self.ref)
            assert second is None
            if len(self.ref):
                np.testing.assert_array_equal(first, np.array(self.ref, dtype=np.float64))
            else:
                assert first.size == 0
        else:
            assert active_count == self.window_size - 1
            parts = [part for part in (first, second) if len(part) > 0]
            active = np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)
            np.testing.assert_array_equal(
                np.sort(active), np.sort(np.array(self.ref, dtype=np.float64)[1:])
            )

    @rule()
    def check_oldest(self):
        # oldest() must match the leftmost item of the deque when full
        oldest_buf = self.buf.oldest()
        if len(self.ref) == 0:
            assert oldest_buf is None
        elif len(self.ref) < self.window_size:
            expected = float(self.ref[0])
            assert oldest_buf == expected
        else:
            expected = float(self.ref[0])
            assert oldest_buf == expected

    @rule()
    def check_window_view(self):
        # window_view must match to_array (chronological order)
        view = self.buf.window_view()
        np.testing.assert_array_equal(view, self.buf.to_array())


class _VectorRingBufferMachine(RuleBasedStateMachine):
    """State machine for testing vector RingBuffer chronology across wraparounds."""

    window_size: int
    num_features: int
    buf: RingBuffer
    ref: deque[np.ndarray]
    step: int
    has_appended: bool

    @initialize()
    def setup(self):
        self.window_size = 3
        self.num_features = 2
        self.buf = RingBuffer(self.window_size)
        self.ref = deque(maxlen=self.window_size)
        self.step = 0
        self.has_appended = False

    @rule()
    def append_vector(self):
        # Create a deterministic but varied vector
        base = self.step * 5
        item = np.array([float(base + 1), float(base * 2 + 3)], dtype=np.float64)
        self.buf.append(item)
        self.ref.append(item)
        self.step += 1
        self.has_appended = True

    @rule()
    def check_chronology(self):
        if not self.has_appended:
            # Before first append, buffer is empty with 1D shape
            assert self.buf.to_array().size == 0
            return
        expected = (
            np.stack(list(self.ref))
            if self.ref
            else np.empty((0, self.num_features), dtype=np.float64)
        )
        actual = self.buf.to_array()
        np.testing.assert_array_equal(actual, expected)

    @rule()
    def check_length(self):
        assert len(self.buf) == len(self.ref)
        assert len(self.buf) <= self.window_size

    @rule()
    def check_active_slices(self):
        first, second, active_count = self.buf.active_slices()
        if len(self.ref) < self.window_size:
            assert active_count == len(self.ref)
            assert second is None
            if len(self.ref):
                expected = np.stack(list(self.ref))
                np.testing.assert_array_equal(first, expected)
            else:
                assert first.size == 0
        else:
            assert active_count == self.window_size - 1
            parts = [part for part in (first, second) if len(part) > 0]
            if parts:
                active = np.concatenate(parts)
                expected = np.stack(list(self.ref))[1:]  # without oldest
                np.testing.assert_array_equal(np.sort(active, axis=0), np.sort(expected, axis=0))
            else:
                assert active_count == 0

    @rule()
    def check_oldest(self):
        oldest_buf = self.buf.oldest()
        if len(self.ref) == 0:
            assert oldest_buf is None
        else:
            expected = self.ref[0]
            np.testing.assert_array_equal(oldest_buf, expected)

    @rule()
    def check_window_view(self):
        if not self.has_appended:
            assert self.buf.window_view().size == 0
            return
        view = self.buf.window_view()
        np.testing.assert_array_equal(view, self.buf.to_array())


# Apply settings and get TestCase classes for pytest
_ScalarRingBufferMachine = STATEFUL_SETTINGS(_ScalarRingBufferMachine)
TestScalarRingBufferChronology = _ScalarRingBufferMachine.TestCase
TestScalarRingBufferChronology.__name__ = "TestScalarRingBufferChronology"
TestScalarRingBufferChronology.__qualname__ = "TestScalarRingBufferChronology"

_VectorRingBufferMachine = STATEFUL_SETTINGS(_VectorRingBufferMachine)
TestVectorRingBufferChronology = _VectorRingBufferMachine.TestCase
TestVectorRingBufferChronology.__name__ = "TestVectorRingBufferChronology"
TestVectorRingBufferChronology.__qualname__ = "TestVectorRingBufferChronology"
