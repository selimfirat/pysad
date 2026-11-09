import tracemalloc

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


@pytest.mark.parametrize("shuffle", [False, True])
def test_polars_streamer_iter_length_mismatch(shuffle):
    df = pl.DataFrame(np.zeros((10, 3)))
    with pytest.raises(ValueError):
        list(PolarsStreamer(shuffle=shuffle).iter(df, pl.Series(np.arange(9))))


def test_polars_streamer_empty_dataframe():
    assert list(PolarsStreamer().iter(pl.DataFrame())) == []


def test_polars_streamer_random_state():
    df = pl.DataFrame({"a": np.arange(50)})

    shuffled = [x[0] for x in PolarsStreamer(shuffle=True, random_state=4).iter(df)]

    assert shuffled == list(ArrayStreamer(shuffle=True, random_state=4).iter(np.arange(50)))


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize(
    "data, expected",
    [
        ({"a": np.array([1, 2], dtype="int16")}, np.array([[1], [2]], dtype="int16")),
        ({"a": np.array([1, 2], dtype="float32")}, np.array([[1], [2]], dtype="float32")),
        ({"a": [True, False]}, np.array([[True], [False]])),
        ({"a": [1, 2], "b": [0.5, 1.5]}, np.array([[1, 0.5], [2, 1.5]])),
        ({"a": [1, 2], "b": ["x", "y"]}, np.array([[1, "x"], [2, "y"]], dtype=object)),
        ({"a": [1, None]}, np.array([[1], [np.nan]])),
        (
            {"a": pl.Series([1, None], dtype=pl.Int16)},
            np.array([[1], [np.nan]], dtype="float32"),
        ),
        ({"a": [True, None], "b": [1, 2]}, np.array([[True, 1], [None, 2]], dtype=object)),
        (
            {"a": np.array([1, 2], dtype="datetime64[ns]")},
            np.array([[1], [2]], dtype="datetime64[ns]"),
        ),
    ],
)
def test_polars_streamer_row_values_and_dtypes(data, expected, shuffle):
    rows = list(PolarsStreamer(shuffle=shuffle, random_state=0).iter(pl.DataFrame(data)))
    order = [1, 0] if shuffle else [0, 1]

    assert len(rows) == 2
    for row, i in zip(rows, order, strict=True):
        assert row.dtype == expected.dtype
        np.testing.assert_array_equal(row, expected[i])


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("label_kind", ["series", "frame", "multi_frame"])
def test_polars_streamer_label_values_and_shapes(label_kind, shuffle):
    features = pl.DataFrame({"a": [10, 20]})
    labels = pl.Series("label", [1, 0], dtype=pl.Int16)
    expected = np.array([1, 0], dtype="int16")
    if label_kind != "series":
        labels = labels.to_frame()
        expected = expected[:, None]
    if label_kind == "multi_frame":
        labels = labels.with_columns(pl.Series("weight", [0.5, 1.5]))
        expected = np.array([[1, 0.5], [0, 1.5]])
    order = [1, 0] if shuffle else [0, 1]

    rows = list(PolarsStreamer(shuffle=shuffle, random_state=0).iter(features, labels))

    assert len(rows) == 2
    for (row, label), i in zip(rows, order, strict=True):
        np.testing.assert_array_equal(row, [10 * (i + 1)])
        np.testing.assert_array_equal(label, expected[i])
        assert label.shape == expected[i].shape
        assert label.dtype == expected.dtype


@pytest.mark.parametrize("label_kind", ["none", "series", "frame"])
@pytest.mark.parametrize("mixed", [False, True])
def test_polars_streamer_unshuffled_memory(label_kind, mixed):
    second = ["x"] * 100_000 if mixed else np.zeros(100_000)
    features = pl.DataFrame({"a": np.arange(100_000), "b": second})
    labels = None
    if label_kind == "series":
        labels = features["a"]
    if label_kind == "frame":
        labels = features
    rows = PolarsStreamer().iter(features, labels)

    tracemalloc.start()
    try:
        first = next(rows)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
        rows.close()

    expected = np.array([0, "x"], dtype=object) if mixed else np.array([0, 0])
    np.testing.assert_array_equal(first if labels is None else first[0], expected)
    assert peak < 500_000


@pytest.mark.parametrize("as_frame", [False, True])
def test_polars_streamer_nulls_across_buffers(as_frame):
    values = [1] * 1024 + [None]
    features = pl.DataFrame({"a": pl.Series(values, dtype=pl.Int16)})
    labels = pl.Series("label", [True] * 1024 + [None])
    if as_frame:
        labels = labels.to_frame()

    rows = list(PolarsStreamer().iter(features, labels))

    assert len(rows) == 1025
    for i, (row, label) in enumerate(rows):
        assert row.dtype == np.dtype("float32")
        np.testing.assert_array_equal(row, [1 if i < 1024 else np.nan])
        expected = True if i < 1024 else None
        if as_frame:
            assert label.dtype == np.dtype(object)
            np.testing.assert_array_equal(label, [expected])
        else:
            assert label is expected
