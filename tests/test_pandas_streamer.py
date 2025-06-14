import pandas as pd
import numpy as np
import pytest
from pysad.utils.pandas_streamer import PandasStreamer


def test_pandas_streamer_iter_no_labels():
    df = pd.DataFrame(np.random.rand(10, 3))
    streamer = PandasStreamer(shuffle=False)
    result = list(streamer.iter(df))
    assert len(result) == 10
    assert all(isinstance(row, np.ndarray) for row in result)
    assert all(row.shape == (3,) for row in result)


def test_pandas_streamer_iter_with_labels():
    df = pd.DataFrame(np.random.rand(10, 3))
    labels = pd.Series(np.arange(10))
    streamer = PandasStreamer(shuffle=False)
    result = list(streamer.iter(df, labels))
    assert len(result) == 10
    for x, y in result:
        assert isinstance(x, np.ndarray)
        assert x.shape == (3,)
        assert isinstance(y, (int, np.integer, float, np.floating))


def test_pandas_streamer_iter_length_mismatch():
    df = pd.DataFrame(np.random.rand(10, 3))
    labels = pd.Series(np.arange(9))
    streamer = PandasStreamer(shuffle=False)
    with pytest.raises(AssertionError):
        list(streamer.iter(df, labels))


def test_pandas_streamer_shuffle():
    """Test that shuffle parameter works correctly."""
    df = pd.DataFrame(np.arange(20).reshape(10, 2))

    # Test with shuffle=True
    streamer_shuffle = PandasStreamer(shuffle=True)
    result_shuffle = list(streamer_shuffle.iter(df))

    # Test with shuffle=False
    streamer_no_shuffle = PandasStreamer(shuffle=False)
    result_no_shuffle = list(streamer_no_shuffle.iter(df))

    # Both should have same length
    assert len(result_shuffle) == len(result_no_shuffle) == 10

    # With shuffle=False, first row should be [0, 1]
    assert np.array_equal(result_no_shuffle[0], np.array([0, 1]))


def test_pandas_streamer_inheritance():
    """Test that PandasStreamer properly inherits from BaseStreamer."""
    streamer = PandasStreamer(shuffle=True)
    assert streamer.shuffle is True

    streamer = PandasStreamer(shuffle=False)
    assert streamer.shuffle is False


def test_pandas_streamer_empty_dataframe():
    """Test with empty dataframe."""
    df = pd.DataFrame()
    streamer = PandasStreamer(shuffle=False)
    result = list(streamer.iter(df))
    assert len(result) == 0
