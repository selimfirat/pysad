import csv
from collections.abc import Iterator, Sequence
from os import PathLike

import numpy as np


class CSVStreamer:
    """Read numeric CSV records one at a time in file order.

    Only selected columns are converted to numbers, so other columns may contain
    text. Features are float64 NumPy arrays of shape ``(num_features,)`` and labels
    are Python floats. Memory use depends on the size of a record, not the file.

    Args:
        feature_columns (Sequence[int] or None): Zero-based, non-negative column indices in the desired feature order. None selects every column except label_column (Default=None).
        label_column (int or None): Zero-based, non-negative index of the numeric label column. None yields features only (Default=None).
        has_header (bool): Whether to skip the first CSV record (Default=True).
        delimiter (str): Single-character field separator (Default=",").
        encoding (str): Encoding used to open the file (Default="utf-8").
    """

    def __init__(
        self,
        feature_columns: Sequence[int] | None = None,
        label_column: int | None = None,
        has_header: bool = True,
        delimiter: str = ",",
        encoding: str = "utf-8",
    ) -> None:
        if feature_columns is not None and any(column < 0 for column in feature_columns):
            raise ValueError("feature_columns must contain non-negative indices.")
        if label_column is not None and label_column < 0:
            raise ValueError("label_column must be a non-negative index.")
        self.feature_columns = tuple(feature_columns) if feature_columns is not None else None
        self.label_column = label_column
        self.has_header = has_header
        self.delimiter = delimiter
        self.encoding = encoding

    def iter(
        self, path: str | PathLike[str]
    ) -> Iterator[np.ndarray | tuple[np.ndarray, float]]:
        """Open a CSV file and yield its features and optional labels.

        Each call reads the file from the beginning. The file is opened on first
        iteration and closed on exhaustion, an error, or closing the iterator.
        Close the iterator explicitly if stopping early. Invalid numeric fields
        raise ValueError and missing selected columns raise IndexError when read.

        Args:
            path (str or os.PathLike[str]): Path to the CSV file.

        Yields:
            np.ndarray or tuple[np.ndarray, float]: Features, or a features-label pair.
        """
        with open(path, encoding=self.encoding, newline="") as file:
            reader = csv.reader(file, delimiter=self.delimiter)
            if self.has_header:
                next(reader, None)
            for row in reader:
                columns = self.feature_columns
                if columns is None:
                    columns = tuple(i for i in range(len(row)) if i != self.label_column)
                features = np.array([row[i] for i in columns], dtype=np.float64)
                if self.label_column is None:
                    yield features
                else:
                    yield features, float(row[self.label_column])
