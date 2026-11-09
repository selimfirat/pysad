import numpy as np
import polars as pl
import pytest

from pysad.utils import ArrayStreamer, PolarsStreamer


def test_polars_streamer_iter_no_labels():
    X = np.arange(30.0).reshape(10, 3)
    result = list(PolarsStreamer().iter(pl.DataFrame(X)))
    assert len(result) == 10
    assert all(isinstance(row, np.ndarray) for row in result)
    np.testing.assert_array_equal(np.stack(result), X)


def test_polars_streamer_iter_with_labels():
    X = np.arange(30.0).reshape(10, 3)
    y = np.arange(10)
    result = list(PolarsStreamer().iter(pl.DataFrame(X), pl.Series(y)))
    np.testing.assert_array_equal(np.stack([x for x, _ in result]), X)
    assert [label for _, label in result] == list(y)


def test_polars_streamer_iter_length_mismatch():
    df = pl.DataFrame(np.zeros((10, 3)))
    with pytest.raises(ValueError):
        list(PolarsStreamer().iter(df, pl.Series(np.arange(9))))


def test_polars_streamer_empty_dataframe():
    assert list(PolarsStreamer().iter(pl.DataFrame())) == []


def test_polars_streamer_random_state():
    df = pl.DataFrame({"a": np.arange(50)})

    shuffled = [x[0] for x in PolarsStreamer(shuffle=True, random_state=4).iter(df)]

    assert shuffled == list(ArrayStreamer(shuffle=True, random_state=4).iter(np.arange(50)))
