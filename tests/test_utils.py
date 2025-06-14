"""
Tests for pysad.utils module functions and classes.
"""
import numpy as np


class TestUtilsFunctions:
    """Tests for utility functions in pysad.utils."""
    
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
    
    def test_window_initialization(self):
        """Test Window initialization."""
        from pysad.utils import Window
        
        window = Window(window_size=5)
        assert window.window_size == 5
        assert window.window == []
        
    def test_window_update_within_size(self):
        """Test Window update within size limit."""
        from pysad.utils import Window
        
        window = Window(window_size=3)
        
        window.update(1.0)
        window.update(2.0)
        
        assert window.get() == [1.0, 2.0]
        assert len(window.get()) == 2
        
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
        
    def test_window_get_copy(self):
        """Test that Window.get() returns the actual list."""
        from pysad.utils import Window
        
        window = Window(window_size=5)
        window.update(1.0)
        window.update(2.0)
        
        result1 = window.get()
        result2 = window.get()
        
        # Should be the same list
        assert result1 is result2
        
    def test_unlimited_window_initialization(self):
        """Test UnlimitedWindow initialization."""
        from pysad.utils.window import UnlimitedWindow
        
        window = UnlimitedWindow()
        assert window.window_size is None
        assert window.window == []
        
    def test_unlimited_window_growth(self):
        """Test UnlimitedWindow unlimited growth."""
        from pysad.utils.window import UnlimitedWindow
        
        window = UnlimitedWindow()
        
        # Add many items
        for i in range(100):
            window.update(float(i))
            
        result = window.get()
        assert len(result) == 100
        assert result[0] == 0.0
        assert result[-1] == 99.0
        
    def test_unlimited_window_no_size_limit(self):
        """Test that UnlimitedWindow doesn't remove old items."""
        from pysad.utils.window import UnlimitedWindow
        
        window = UnlimitedWindow()
        
        for i in range(10):
            window.update(float(i))
            
        # All items should be present
        result = window.get()
        expected = [float(i) for i in range(10)]
        assert result == expected


class TestArrayStreamer:
    """Tests for ArrayStreamer class."""
    
    def test_array_streamer_initialization(self):
        """Test ArrayStreamer initialization."""
        from pysad.utils import ArrayStreamer
        
        streamer = ArrayStreamer(shuffle=False)
        assert hasattr(streamer, 'shuffle')
        assert streamer.shuffle is False
        
        streamer = ArrayStreamer(shuffle=True)
        assert streamer.shuffle is True
        
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
            
    def test_array_streamer_shuffle(self):
        """Test ArrayStreamer with shuffle."""
        from pysad.utils import ArrayStreamer
        
        # Set seed for reproducible shuffle
        np.random.seed(42)
        streamer = ArrayStreamer(shuffle=True)
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        
        results = list(streamer.iter(X))
        assert len(results) == 4
        
        # Results should contain all original data but potentially in different order
        result_set = {tuple(xi) for xi in results}
        original_set = {tuple(X[i]) for i in range(len(X))}
        assert result_set == original_set


class TestPandasStreamer:
    """Tests for PandasStreamer class."""
    
    def test_pandas_streamer_initialization(self):
        """Test PandasStreamer initialization."""
        from pysad.utils import PandasStreamer
        
        streamer = PandasStreamer(shuffle=False)
        assert hasattr(streamer, 'shuffle')
        assert streamer.shuffle is False
        
    def test_pandas_streamer_with_dataframe(self):
        """Test PandasStreamer with pandas DataFrame."""
        import pandas as pd
        from pysad.utils import PandasStreamer
        
        # Create test DataFrame
        df = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [4.0, 5.0, 6.0]
        })
        
        streamer = PandasStreamer(shuffle=False)
        results = list(streamer.iter(df))
        
        assert len(results) == 3
        for i, row in enumerate(results):
            assert len(row) == 2
            assert row[0] == df.iloc[i, 0]
            assert row[1] == df.iloc[i, 1]
            
    def test_pandas_streamer_with_labels(self):
        """Test PandasStreamer with labels."""
        import pandas as pd
        from pysad.utils import PandasStreamer
        
        # Create test data
        df = pd.DataFrame({
            'feature1': [1.0, 2.0],
            'feature2': [3.0, 4.0]
        })
        labels = pd.Series([0, 1])
        
        streamer = PandasStreamer(shuffle=False)
        results = list(streamer.iter(df, labels))
        
        assert len(results) == 2
        for i, (row, label) in enumerate(results):
            assert len(row) == 2
            assert label == labels.iloc[i]
