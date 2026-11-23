"""
Tests for pysad.utils module functions and classes.
"""

import numpy as np
import pytest


class TestUtilsFunctions:
    """Tests for utility functions in pysad.utils."""

    @pytest.mark.parametrize("dtype", [np.int64, np.float32, np.float64])
    def test_scalar_bounds_are_python_floats(self, dtype):
        from pysad.utils import get_minmax_scalar

        low, high = get_minmax_scalar(np.array([[2, -3], [7, 1]], dtype=dtype))

        assert type(low) is float
        assert type(high) is float
        assert (low, high) == (-3.0, 7.0)

    def test_fix_seed(self):
        """Test fix_seed function for reproducible randomness."""
        from pysad.utils import fix_seed

        # Test with specific seed
        fix_seed(42)
        random_val1 = np.random.random()

        fix_seed(42)
        random_val2 = np.random.random()

        # Should be identical with same seed
        assert random_val1 == random_val2

        # Test with different seed
        fix_seed(123)
        random_val3 = np.random.random()

        # Should be different with different seed
        assert random_val1 != random_val3

    def test_get_minmax_array(self):
        """Test get_minmax_array function."""
        from pysad.utils import get_minmax_array

        # Test with 2D array
        X = np.array([[1.0, 5.0], [3.0, 2.0], [0.0, 8.0]])
        min_vals, max_vals = get_minmax_array(X)

        expected_min = np.array([0.0, 2.0])
        expected_max = np.array([3.0, 8.0])

        np.testing.assert_array_equal(min_vals, expected_min)
        np.testing.assert_array_equal(max_vals, expected_max)

    def test_get_minmax_array_single_feature(self):
        """Test get_minmax_array with single feature."""
        from pysad.utils import get_minmax_array

        X = np.array([[1.0], [3.0], [0.0], [5.0]])
        min_vals, max_vals = get_minmax_array(X)

        assert min_vals[0] == 0.0
        assert max_vals[0] == 5.0
        assert len(min_vals) == 1
        assert len(max_vals) == 1

    def test_get_minmax_array_identical_values(self):
        """Test get_minmax_array with identical values."""
        from pysad.utils import get_minmax_array

        X = np.array([[2.0, 3.0], [2.0, 3.0], [2.0, 3.0]])
        min_vals, max_vals = get_minmax_array(X)

        expected_min = np.array([2.0, 3.0])
        expected_max = np.array([2.0, 3.0])

        np.testing.assert_array_equal(min_vals, expected_min)
        np.testing.assert_array_equal(max_vals, expected_max)

    def test_get_minmax_scalar(self):
        """Test get_minmax_scalar function."""
        from pysad.utils import get_minmax_scalar

        # Test with 1D array
        x = np.array([1.0, 3.0, 0.0, 5.0, 2.0])
        min_val, max_val = get_minmax_scalar(x)

        assert min_val == 0.0
        assert max_val == 5.0

    def test_get_minmax_scalar_2d_array(self):
        """Test get_minmax_scalar with 2D array."""
        from pysad.utils import get_minmax_scalar

        x = np.array([[1.0, 3.0], [0.0, 5.0]])
        min_val, max_val = get_minmax_scalar(x)

        assert min_val == 0.0
        assert max_val == 5.0

    def test_get_minmax_scalar_single_value(self):
        """Test get_minmax_scalar with single value."""
        from pysad.utils import get_minmax_scalar

        x = np.array([42.0])
        min_val, max_val = get_minmax_scalar(x)

        assert min_val == 42.0
        assert max_val == 42.0

    def test_iterate_without_labels(self):
        """Test _iterate function without labels."""
        from pysad.utils import _iterate

        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        results = list(_iterate(X))

        assert len(results) == 3
        for i, (xi, yi) in enumerate(results):
            np.testing.assert_array_equal(xi, X[i])
            assert yi is None

    def test_iterate_with_labels(self):
        """Test _iterate function with labels."""
        from pysad.utils import _iterate

        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([0, 1])

        results = list(_iterate(X, y))

        assert len(results) == 2
        for i, (xi, yi) in enumerate(results):
            np.testing.assert_array_equal(xi, X[i])
            assert yi == y[i]

    def test_iterate_single_instance(self):
        """Test _iterate function with single instance."""
        from pysad.utils import _iterate

        X = np.array([[1.0, 2.0]])
        y = np.array([1])

        results = list(_iterate(X, y))

        assert len(results) == 1
        xi, yi = results[0]
        np.testing.assert_array_equal(xi, X[0])
        assert yi == y[0]

    def test_iterate_empty_array(self):
        """Test _iterate function with empty array."""
        from pysad.utils import _iterate

        X = np.array([]).reshape(0, 2)

        results = list(_iterate(X))

        assert len(results) == 0


class TestWindow:
    """Tests for Window class."""

    def test_window_update_exceeds_size(self):
        """Test Window update when exceeding size."""
        from pysad.utils import Window

        window = Window(window_size=3)

        # Add more items than window size
        for i in range(5):
            window.update(float(i))

        result = window.get()
        assert len(result) == 3
        assert result == [2.0, 3.0, 4.0]  # Should keep last 3

    def test_window_single_size(self):
        """Test Window with size 1."""
        from pysad.utils import Window

        window = Window(window_size=1)

        window.update(1.0)
        assert window.get() == [1.0]

        window.update(2.0)
        assert window.get() == [2.0]

    def test_window_get_returns_a_copy(self):
        """Changing the returned list does not change the window."""
        from pysad.utils import Window

        window = Window(window_size=3)
        window.update(1.0)
        window.update(2.0)

        result = window.get()
        result[0] = 9.0
        result.append(3.0)

        assert window.get() == [1.0, 2.0]
        assert list(window) == [1.0, 2.0]

    def test_window_get_returns_copy_before_wraparound(self):
        """get() returns a copy when buffer has not wrapped."""
        from pysad.utils import Window

        window = Window(window_size=5)
        for v in (1.0, 2.0, 3.0):
            window.update(v)

        result = window.get()
        result[0] = 9.0
        result.append(4.0)

        assert window.get() == [1.0, 2.0, 3.0]
        assert list(window) == [1.0, 2.0, 3.0]

    def test_window_get_returns_copy_after_wraparound(self):
        """get() returns a copy after buffer wraps around."""
        from pysad.utils import Window

        window = Window(window_size=3)
        for v in (1.0, 2.0, 3.0, 4.0, 5.0):
            window.update(v)

        # Buffer wrapped; internal storage is [4.0, 5.0, 3.0] with next_slot=2
        # window_view() returns concatenated array [3.0, 4.0, 5.0]
        result = window.get()
        result[0] = 9.0
        result.append(6.0)

        assert window.get() == [3.0, 4.0, 5.0]
        assert list(window) == [3.0, 4.0, 5.0]

    def test_window_get_returns_copy_at_exact_wrap_boundary(self):
        """get() returns a copy when next_slot == 0 (exact wrap boundary)."""
        from pysad.utils import Window

        window = Window(window_size=3)
        for v in (1.0, 2.0, 3.0):
            window.update(v)

        # Buffer is full, next_slot == 0; window_view() returns a view of points[:3]
        result = window.get()
        result[0] = 9.0
        result.append(4.0)

        assert window.get() == [1.0, 2.0, 3.0]
        assert list(window) == [1.0, 2.0, 3.0]

    def test_window_get_copy_independent_across_calls(self):
        """Each get() call returns an independent copy."""
        from pysad.utils import Window

        window = Window(window_size=3)
        for v in (1.0, 2.0, 3.0):
            window.update(v)

        result1 = window.get()
        result2 = window.get()

        result1[0] = 9.0
        result2[1] = 8.0

        assert window.get() == [1.0, 2.0, 3.0]
        assert result1 == [9.0, 2.0, 3.0]
        assert result2 == [1.0, 8.0, 3.0]

    def test_window_iter_returns_copy_across_wraparound(self):
        """Iterating the window yields a copy after wraparound."""
        from pysad.utils import Window

        window = Window(window_size=3)
        for v in (1.0, 2.0, 3.0, 4.0):
            window.update(v)

        items = list(window)
        items[0] = 9.0

        assert window.get() == [2.0, 3.0, 4.0]
        assert list(window) == [2.0, 3.0, 4.0]

    def test_ring_buffer_scalar_and_vector_wraparound(self):
        """Test RingBuffer chronological order and active slices for scalars and vectors."""
        from pysad.utils import RingBuffer

        scalar_buf = RingBuffer(window_size=3)
        assert len(scalar_buf) == 0
        np.testing.assert_array_equal(scalar_buf.to_array(), np.empty(0))

        for val in (10.0, 20.0, 30.0, 40.0, 50.0):
            scalar_buf.append(val)
        assert len(scalar_buf) == 3
        np.testing.assert_array_equal(scalar_buf.to_array(), [30.0, 40.0, 50.0])

        vec_buf = RingBuffer(window_size=3)
        for i in range(5):
            vec_buf.append(np.array([float(i), float(10 * i)]))
        np.testing.assert_array_equal(vec_buf.to_array(), [[2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
        first, second, active_count = vec_buf.active_slices()
        assert active_count == 2
        active = first if second is None else np.vstack([a for a in (first, second) if len(a) > 0])
        assert {tuple(row) for row in active} == {(3.0, 30.0), (4.0, 40.0)}

    @pytest.mark.parametrize("num_appended", [4, 5, 6, 7, 8, 9])
    def test_ring_buffer_active_slices_after_wraparound(self, num_appended):
        """After the buffer is full, active_slices drops the item that the next append expires."""
        from pysad.utils import RingBuffer

        buffer = RingBuffer(window_size=4)
        values = [float(v) for v in range(1, num_appended + 1)]
        for value in values:
            buffer.append(value)

        newer, older, active_count = buffer.active_slices()

        assert active_count == 3
        assert older is not None
        assert len(newer) + len(older) == active_count
        # The oldest stored item, values[-4], expires when the next item arrives.
        np.testing.assert_array_equal(np.concatenate((older, newer)), values[-3:])

    @pytest.mark.parametrize(
        ("first", "mismatched"),
        [(1.0, np.array([2.0])), (np.array([1.0, 2.0]), 3.0)],
    )
    def test_ring_buffer_rejects_a_different_item_shape(self, first, mismatched):
        """The first item fixes the shape accepted by later appends."""
        from pysad.utils import RingBuffer

        buffer = RingBuffer(window_size=2)
        buffer.append(first)

        with pytest.raises(ValueError, match="item must have shape"):
            buffer.append(mismatched)

    def test_ring_buffer_to_array_returns_a_copy(self):
        """Changing the returned array does not change buffered items."""
        from pysad.utils import RingBuffer

        buffer = RingBuffer(window_size=3)
        for value in (1.0, 2.0, 3.0, 4.0):
            buffer.append(value)

        result = buffer.to_array()
        result[0] = -1.0

        np.testing.assert_array_equal(buffer.to_array(), [2.0, 3.0, 4.0])


class TestArrayStreamer:
    """Tests for ArrayStreamer class."""

    def test_array_streamer_iter_without_labels(self):
        """Test ArrayStreamer iteration without labels."""
        from pysad.utils import ArrayStreamer

        streamer = ArrayStreamer(shuffle=False)
        X = np.array([[1, 2], [3, 4], [5, 6]])

        results = list(streamer.iter(X))
        assert len(results) == 3
        for i, xi in enumerate(results):
            np.testing.assert_array_equal(xi, X[i])

    def test_array_streamer_iter_with_labels(self):
        """Test ArrayStreamer iteration with labels."""
        from pysad.utils import ArrayStreamer

        streamer = ArrayStreamer(shuffle=False)
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])

        results = list(streamer.iter(X, y))
        assert len(results) == 2
        for i, (xi, yi) in enumerate(results):
            np.testing.assert_array_equal(xi, X[i])
            assert yi == y[i]

    def test_array_streamer_empty_array(self):
        """Test ArrayStreamer iteration with an empty array."""
        from pysad.utils import ArrayStreamer

        X = np.empty((0, 2))
        y = np.empty((0,), dtype=int)

        results = list(ArrayStreamer().iter(X))
        assert results == []

        labeled_results = list(ArrayStreamer().iter(X, y))
        assert labeled_results == []

    def test_array_streamer_first_row_memory(self):
        """Unshuffled streams yield the first row with bounded extra memory."""
        import tracemalloc

        from pysad.utils import ArrayStreamer

        X = np.broadcast_to(np.array([1, 2]), (100_000, 2))
        y = np.broadcast_to(3, (len(X),))
        for labels in (None, y):
            tracemalloc.start()
            try:
                first = next(ArrayStreamer(shuffle=False).iter(X, labels))
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()

            # Materializing 100,000 indices requires several megabytes.
            assert peak < 100_000
            if labels is None:
                np.testing.assert_array_equal(first, [1, 2])
            else:
                np.testing.assert_array_equal(first[0], [1, 2])
                assert first[1] == 3

    def test_array_streamer_seeded_row_order(self):
        """Seeded shuffling preserves the permutation and feature-label pairing."""
        from pysad.utils import ArrayStreamer

        X = np.arange(12).reshape(6, 2)
        y = np.arange(6)
        expected = [0, 1, 5, 2, 4, 3]
        streamer = ArrayStreamer(shuffle=True, random_state=42)

        np.testing.assert_array_equal(list(streamer.iter(X)), X[expected])
        rows = list(streamer.iter(X, y))
        np.testing.assert_array_equal([row for row, _ in rows], X[expected])
        np.testing.assert_array_equal([label for _, label in rows], y[expected])

    def test_array_streamer_length_mismatch(self):
        """Test ArrayStreamer rejects labels of a different length."""
        import pytest

        from pysad.utils import ArrayStreamer

        streamer = ArrayStreamer(shuffle=False)
        with pytest.raises(ValueError):
            list(streamer.iter(np.zeros((3, 2)), np.zeros(2)))

    def test_array_streamer_random_state(self):
        """Test that ArrayStreamer shuffles with its own seed, and with the global state by default."""
        from pysad.utils import ArrayStreamer, fix_seed

        X = np.arange(50)

        fix_seed(4)
        default = list(ArrayStreamer(shuffle=True).iter(X))

        assert default == list(ArrayStreamer(shuffle=True, random_state=4).iter(X))
        assert default != list(ArrayStreamer(shuffle=True, random_state=8).iter(X))


class TestPlotStream:
    """Tests for plot_stream."""

    def _use_agg_backend(self):
        import matplotlib

        matplotlib.use("Agg")

    def test_plot_stream_draws_score_and_threshold_lines(self):
        """With a threshold, the scores panel shows the score line and the threshold line."""
        self._use_agg_backend()
        import matplotlib.pyplot as plt
        import numpy as np

        from pysad.utils import plot_stream

        scores = np.array([0.1, 0.5, 0.9, 0.2, 0.8])

        _, axes = plot_stream(scores.reshape(-1, 1), scores)

        assert len(axes[1].get_lines()) == 1

        _, axes = plot_stream(scores.reshape(-1, 1), scores, threshold=0.5)

        assert len(axes[1].get_lines()) == 2
        threshold_line = axes[1].get_lines()[1]
        assert threshold_line.get_ydata()[0] == 0.5

        plt.close("all")

    def test_plot_stream_draws_confidence_band(self):
        """With lower and upper bounds, the scores panel shows the confidence band."""
        self._use_agg_backend()
        import matplotlib.pyplot as plt
        import numpy as np

        from pysad.utils import plot_stream

        scores = np.array([0.1, 0.5, 0.9, 0.2, 0.8])
        lower = scores - 0.1
        upper = scores + 0.1

        _, axes = plot_stream(scores.reshape(-1, 1), scores, lower=lower, upper=upper)

        assert len(axes[1].collections) >= 1

        plt.close("all")

    def test_plot_stream_draws_alert_spikes(self):
        """With alerts and labels, the alerts panel shows the spikes."""
        self._use_agg_backend()
        import matplotlib.pyplot as plt
        import numpy as np

        from pysad.utils import plot_stream

        scores = np.array([0.1, 0.5, 0.9, 0.2, 0.8])
        alerts = np.array([2])
        labels = np.array([0, 0, 0, 0, 1])

        _, axes = plot_stream(scores.reshape(-1, 1), scores, alerts=alerts, labels=labels)

        assert len(axes[2].collections) >= 1

        plt.close("all")

    def test_plot_stream_without_optional_args(self):
        """plot_stream works with only the features and the scores."""
        self._use_agg_backend()
        import matplotlib.pyplot as plt
        import numpy as np

        from pysad.utils import plot_stream

        rng = np.random.default_rng(0)
        X = rng.normal(size=(20, 3))
        scores = rng.random(20)

        fig, axes = plot_stream(X, scores)

        assert len(axes[0].get_lines()) == 3
        assert len(axes[1].get_lines()) == 1

        plt.close(fig)


class TestPlotStreamingRibbon:
    """Tests for plot_streaming_ribbon."""

    def _use_agg_backend(self):
        import matplotlib

        matplotlib.use("Agg")

    def test_plot_streaming_ribbon_draws_ribbon_and_alerts(self):
        """With a calibrator, the scores panel shows the conformal ribbon and alerts."""
        self._use_agg_backend()
        import matplotlib.pyplot as plt
        import numpy as np

        from pysad.transform.probability_calibration import ConformalTSFMCalibrator
        from pysad.utils import plot_streaming_ribbon

        scores = np.array([0.1, 0.5, 0.9, 0.2, 0.8])
        X = scores.reshape(-1, 1)
        calibrator = ConformalTSFMCalibrator(alpha=0.1, gamma=0.01, window_size=100)

        fig, axes = plot_streaming_ribbon(X, scores, calibrator)

        # Score line + ribbon (fill_between creates a collection)
        assert len(axes[1].get_lines()) >= 1
        assert len(axes[1].collections) >= 1
        # Legend should have "score" and "conformal ribbon"
        legend_texts = [t.get_text() for t in axes[1].get_legend().get_texts()]
        assert "score" in legend_texts
        assert "conformal ribbon" in legend_texts

        # Alerts panel should have alert spikes (score 0.9 > threshold)
        assert len(axes[2].collections) >= 1

        plt.close(fig)

    def test_plot_streaming_ribbon_with_labels(self):
        """With labels, the alerts panel shows both alerts and anomalies."""
        self._use_agg_backend()
        import matplotlib.pyplot as plt
        import numpy as np

        from pysad.transform.probability_calibration import ConformalTSFMCalibrator
        from pysad.utils import plot_streaming_ribbon

        scores = np.array([0.1, 0.5, 0.9, 0.2, 0.8])
        X = scores.reshape(-1, 1)
        labels = np.array([0, 0, 1, 0, 0])
        calibrator = ConformalTSFMCalibrator(alpha=0.1, gamma=0.01, window_size=100)

        fig, axes = plot_streaming_ribbon(X, scores, calibrator, labels=labels)

        # Alerts panel should have both alert and anomaly spikes
        assert len(axes[2].collections) >= 1
        ytick_labels = [t.get_text() for t in axes[2].get_yticklabels()]
        assert "alert" in ytick_labels
        assert "anomaly" in ytick_labels

        plt.close(fig)

    def test_plot_streaming_ribbon_without_lower_ribbon(self):
        """With ribbon_lower=None, only the threshold line is drawn."""
        self._use_agg_backend()
        import matplotlib.pyplot as plt
        import numpy as np

        from pysad.transform.probability_calibration import ConformalTSFMCalibrator
        from pysad.utils import plot_streaming_ribbon

        scores = np.array([0.1, 0.5, 0.9, 0.2, 0.8])
        X = scores.reshape(-1, 1)
        calibrator = ConformalTSFMCalibrator(alpha=0.1, gamma=0.01, window_size=100)

        fig, axes = plot_streaming_ribbon(X, scores, calibrator, ribbon_lower=None)

        # Should have score line + threshold line (no fill_between collection)
        assert len(axes[1].get_lines()) >= 2
        legend_texts = [t.get_text() for t in axes[1].get_legend().get_texts()]
        assert "score" in legend_texts
        assert "threshold" in legend_texts
        # No ribbon collection
        assert len(axes[1].collections) == 0

        plt.close(fig)

    def test_plot_streaming_ribbon_with_array_lower(self):
        """With array ribbon_lower, the ribbon uses per-step lower bounds."""
        self._use_agg_backend()
        import matplotlib.pyplot as plt
        import numpy as np

        from pysad.transform.probability_calibration import ConformalTSFMCalibrator
        from pysad.utils import plot_streaming_ribbon

        scores = np.array([0.1, 0.5, 0.9, 0.2, 0.8])
        X = scores.reshape(-1, 1)
        calibrator = ConformalTSFMCalibrator(alpha=0.1, gamma=0.01, window_size=100)
        ribbon_lower = np.array([0.0, 0.1, 0.2, 0.0, 0.1])

        fig, axes = plot_streaming_ribbon(X, scores, calibrator, ribbon_lower=ribbon_lower)

        assert len(axes[1].collections) >= 1
        legend_texts = [t.get_text() for t in axes[1].get_legend().get_texts()]
        assert "conformal ribbon" in legend_texts

        plt.close(fig)

    def test_plot_streaming_ribbon_does_not_modify_original_calibrator(self):
        """The original calibrator is not modified by the plotting function."""
        self._use_agg_backend()
        import matplotlib.pyplot as plt
        import numpy as np

        from pysad.transform.probability_calibration import ConformalTSFMCalibrator
        from pysad.utils import plot_streaming_ribbon

        scores = np.array([0.1, 0.5, 0.9, 0.2, 0.8])
        X = scores.reshape(-1, 1)
        calibrator = ConformalTSFMCalibrator(alpha=0.1, gamma=0.01, window_size=100)
        original_threshold = calibrator.threshold_

        plot_streaming_ribbon(X, scores, calibrator)
        plt.close("all")

        # Original calibrator should be unchanged
        assert calibrator.threshold_ == original_threshold
