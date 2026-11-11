import numpy as np
import pytest

from pysad.utils import CSVStreamer


def test_unlabeled_rows_preserve_file_order(tmp_path):
    path = tmp_path / "stream.csv"
    path.write_text("a,b\n3,4\n1,2\n", encoding="utf-8")

    rows = list(CSVStreamer().iter(path))

    assert all(isinstance(row, np.ndarray) and row.dtype == np.float64 for row in rows)
    np.testing.assert_array_equal(rows, [[3, 4], [1, 2]])


def test_selected_features_and_labels(tmp_path):
    path = tmp_path / "stream.csv"
    path.write_text('name,a,label,b\n"first, row",3,0,4\nsecond,1,1,2\n', encoding="utf-8")

    rows = list(CSVStreamer(feature_columns=[3, 1], label_column=2).iter(str(path)))

    np.testing.assert_array_equal([x for x, _ in rows], [[4, 3], [2, 1]])
    assert [label for _, label in rows] == [0.0, 1.0]
    assert all(isinstance(label, float) for _, label in rows)


def test_default_features_exclude_label(tmp_path):
    path = tmp_path / "stream.csv"
    path.write_text("a,label,b\n3,1,4\n", encoding="utf-8")

    [(row, label)] = CSVStreamer(label_column=1).iter(path)

    np.testing.assert_array_equal(row, [3, 4])
    assert label == 1.0


def test_headerless_custom_delimiter_and_encoding(tmp_path):
    path = tmp_path / "stream.csv"
    path.write_bytes('café;"3";4\n'.encode("latin-1"))
    streamer = CSVStreamer(
        feature_columns=[2, 1], has_header=False, delimiter=";", encoding="latin-1"
    )

    np.testing.assert_array_equal(list(streamer.iter(path)), [[4, 3]])


@pytest.mark.parametrize("contents", ["", "a,b\n"])
def test_empty_stream(tmp_path, contents):
    path = tmp_path / "stream.csv"
    path.write_text(contents, encoding="utf-8")

    assert list(CSVStreamer().iter(path)) == []


def test_iteration_is_lazy(tmp_path):
    path = tmp_path / "stream.csv"
    path.write_text("a,b\n1,2\ninvalid,3\n", encoding="utf-8")
    rows = CSVStreamer().iter(path)

    np.testing.assert_array_equal(next(rows), [1, 2])
    with pytest.raises(ValueError):
        next(rows)


def test_each_iteration_reopens_file(tmp_path):
    path = tmp_path / "stream.csv"
    path.write_text("a\n1\n", encoding="utf-8")
    streamer = CSVStreamer()
    np.testing.assert_array_equal(list(streamer.iter(path)), [[1]])

    path.write_text("a\n2\n", encoding="utf-8")

    np.testing.assert_array_equal(list(streamer.iter(path)), [[2]])


@pytest.mark.parametrize("columns", [{"feature_columns": [-1]}, {"label_column": -1}])
def test_negative_columns_are_rejected(columns):
    with pytest.raises(ValueError, match="non-negative"):
        CSVStreamer(**columns)
