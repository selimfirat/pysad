import tracemalloc

import numpy as np
import pandas as pd
import pytest

from pysad.utils import ArrayStreamer, PandasStreamer


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


@pytest.mark.parametrize("shuffle", [False, True])
def test_pandas_streamer_iter_length_mismatch(shuffle):
    df = pd.DataFrame(np.random.rand(10, 3))
    labels = pd.Series(np.arange(9))
    streamer = PandasStreamer(shuffle=shuffle)
    with pytest.raises(ValueError):
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


def test_pandas_streamer_random_state():
    df = pd.DataFrame({"a": np.arange(50)})

    shuffled = [x[0] for x in PandasStreamer(shuffle=True, random_state=4).iter(df)]

    assert shuffled == list(ArrayStreamer(shuffle=True, random_state=4).iter(np.arange(50)))


@pytest.mark.parametrize(
    "data, expected",
    [
        ({"a": np.array([1, 2], dtype="int16")}, np.array([[1], [2]], dtype="int16")),
        ({"a": np.array([1, 2], dtype="float32")}, np.array([[1], [2]], dtype="float32")),
        ({"a": [True, False]}, np.array([[True], [False]])),
        ({"a": [1, 2], "b": [0.5, 1.5]}, np.array([[1, 0.5], [2, 1.5]])),
        ({"a": [1, 2], "b": ["x", "y"]}, np.array([[1, "x"], [2, "y"]], dtype=object)),
        ({"a": pd.array([1, None], dtype="Int64")}, np.array([[1], [np.nan]])),
        ({"a": pd.array([1, 2], dtype="Int64")}, np.array([[1], [2]], dtype="int64")),
        (
            {"a": pd.array([1, 2], dtype="Int64"), "b": pd.array([3, 4], dtype="Int64")},
            np.array([[1, 3], [2, 4]], dtype=object),
        ),
        (
            {"a": pd.array([True, None], dtype="boolean")},
            np.array([[True], [pd.NA]], dtype=object),
        ),
        (
            {"a": pd.date_range("2025-01-01", periods=2, tz="UTC")},
            np.array(
                [[pd.Timestamp("2025-01-01", tz="UTC")], [pd.Timestamp("2025-01-02", tz="UTC")]],
                dtype=object,
            ),
        ),
        ({"a": pd.Categorical([1, None])}, np.array([[1], [np.nan]])),
        ({"a": pd.Categorical(["x", "y"])}, np.array([["x"], ["y"]], dtype=object)),
        (
            {"a": pd.date_range("2025-01-01", periods=2)},
            np.array([["2025-01-01"], ["2025-01-02"]], dtype="datetime64[ns]"),
        ),
        ({}, np.empty((2, 0))),
    ],
)
@pytest.mark.parametrize("shuffle", [False, True])
def test_pandas_streamer_row_values_and_dtypes(data, expected, shuffle):
    df = pd.DataFrame(data, index=[7, 3])
    order = [0, 1] if not shuffle else [1, 0]

    rows = list(PandasStreamer(shuffle=shuffle, random_state=0).iter(df))

    assert len(rows) == 2
    for row, i in zip(rows, order, strict=True):
        assert isinstance(row, np.ndarray)
        assert row.dtype == expected.dtype
        pd.testing.assert_series_equal(pd.Series(row), pd.Series(expected[i]))


@pytest.mark.parametrize("label_kind", ["series", "frame", "multi_frame"])
@pytest.mark.parametrize("shuffle", [False, True])
def test_pandas_streamer_label_values_and_shapes(label_kind, shuffle):
    df = pd.DataFrame({"a": [10, 20]}, index=[7, 3])
    labels = pd.Series([1, 0], index=[3, 7], dtype="int16")
    expected = np.array([1, 0], dtype="int16")
    if label_kind != "series":
        labels = labels.to_frame()
        expected = expected[:, None]
    if label_kind == "multi_frame":
        labels["weight"] = [0.5, 1.5]
        expected = np.array([[1, 0.5], [0, 1.5]])
    order = [0, 1] if not shuffle else [1, 0]

    rows = list(PandasStreamer(shuffle=shuffle, random_state=0).iter(df, labels))

    assert len(rows) == 2
    for (row, label), i in zip(rows, order, strict=True):
        np.testing.assert_array_equal(row, [10 * (i + 1)])
        np.testing.assert_array_equal(label, expected[i])
        assert label.shape == expected[i].shape
        assert label.dtype == expected.dtype


@pytest.mark.parametrize("with_labels", [False, True])
def test_pandas_streamer_unshuffled_memory(with_labels):
    df = pd.DataFrame({"a": np.arange(100_000), "b": np.zeros(100_000)})
    labels = df if with_labels else None
    rows = PandasStreamer().iter(df, labels)

    tracemalloc.start()
    try:
        first = next(rows)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
        rows.close()

    np.testing.assert_array_equal(first[0] if with_labels else first, [0, 0])
    assert peak < 500_000


@pytest.mark.parametrize("as_frame", [False, True])
def test_pandas_streamer_nullable_labels(as_frame):
    df = pd.DataFrame({"a": [10, 20]})
    labels = pd.Series([1, None], dtype="Int64")
    if as_frame:
        labels = labels.to_frame()
    expected = np.array([1, np.nan])

    rows = list(PandasStreamer().iter(df, labels))

    for i, (_, label) in enumerate(rows):
        assert label.dtype == expected.dtype
        np.testing.assert_array_equal(label, expected[i : i + 1] if as_frame else expected[i])
