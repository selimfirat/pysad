import os
import shutil
import tempfile
import urllib.request
from collections.abc import Iterator
from typing import Any

import numpy as np

from pysad.utils.array_streamer import ArrayStreamer


def _get_default_data_dir() -> str:
    """Returns the default directory for caching datasets.

    Defaults to the directory in the ``PYSAD_DATA_DIR`` environment variable if
    set, or ``~/.cache/pysad``.
    """
    return os.environ.get(
        "PYSAD_DATA_DIR", os.path.expanduser(os.path.join("~", ".cache", "pysad"))
    )


class Data:
    """A helper class to load and stream anomaly detection datasets.

    Datasets are loaded from the local filesystem or downloaded on demand from
    the Outlier Detection DataSets (ODDS) repository mirror
    (:cite:`rayana2016odds`, hosted at
    https://github.com/yzhao062/pyod/raw/master/notebooks/data/).

    Downloaded datasets are cached in a local directory. The cache directory
    defaults to ``~/.cache/pysad``, which can be overridden via the
    ``PYSAD_DATA_DIR`` environment variable or by passing an explicit
    ``data_base_path``.

    Args:
        data_base_path (str | None): Base directory containing data files or where
            downloaded datasets should be cached. If None, defaults to the directory
            in ``PYSAD_DATA_DIR`` if set, or ``~/.cache/pysad``.
    """

    def __init__(self, data_base_path: str | None = None) -> None:
        self.data_base_path = _get_default_data_dir() if data_base_path is None else data_base_path
        self._is_default_path = data_base_path is None

    def _get_data_files(self) -> list[str]:
        """Helper method to return the names of the known datasets.

        Returns:
            list[str]: List of dataset file names.
        """
        return [
            "arrhythmia.mat",
            "cardio.mat",
            "glass.mat",
            "ionosphere.mat",
            "letter.mat",
            "lympho.mat",
            "mnist.mat",
            "musk.mat",
            "optdigits.mat",
            "pendigits.mat",
            "pima.mat",
            "satellite.mat",
            "satimage-2.mat",
            "shuttle.mat",
            "vertebral.mat",
            "vowels.mat",
            "wbc.mat",
        ]

    def _download(self, file_name: str, target_path: str) -> None:
        """Downloads a dataset file from the remote mirror into target_path atomically.

        Args:
            file_name (str): Name of the remote file (e.g. 'cardio.mat').
            target_path (str): Destination file path.
        """
        url = f"https://github.com/yzhao062/pyod/raw/master/notebooks/data/{file_name}"
        target_dir = os.path.dirname(target_path)
        if target_dir:
            os.makedirs(target_dir, exist_ok=True)

        temp_file = tempfile.NamedTemporaryFile(
            dir=target_dir or None, prefix=f"tmp_{file_name}_", delete=False
        )
        temp_path = temp_file.name
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "pysad"})
            with urllib.request.urlopen(req) as response, temp_file:
                shutil.copyfileobj(response, temp_file)
            os.replace(temp_path, target_path)
        except Exception:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            raise

    def _load_via_txt(self, path: str) -> np.ndarray:
        """Loads the data file from .txt file.

        Args:
            path (str): The path of data.

        Returns:
            X (np.float64 array of shape (num_instances, num_features)): Feature vectors.
        """
        X = np.loadtxt(path, delimiter=",")

        return X

    def get_data(self, data_file: str, download: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Loads the data given the filename or path, downloading it if not present.

        Args:
            data_file (str): Path or filename of the data.
            download (bool): Whether to download the dataset if not present locally (Default=True).

        Returns:
            X (np.ndarray of shape (num_instances, num_features)): Feature vectors.
            y (np.ndarray of shape (num_instances,)): Labels.
        """
        if os.path.exists(data_file):
            data_path = data_file
        else:
            data_path = os.path.join(self.data_base_path, data_file)

        file_name = os.path.basename(data_file)

        if not os.path.exists(data_path):
            if self._is_default_path:
                local_fallback = os.path.join("data", data_file)
                if os.path.exists(local_fallback):
                    data_path = local_fallback
                else:
                    examples_fallback = os.path.join("examples", "data", data_file)
                    if os.path.exists(examples_fallback):
                        data_path = examples_fallback

        if not os.path.exists(data_path):
            if not download:
                raise FileNotFoundError(
                    f"Dataset file '{data_file}' was not found at '{data_path}' and download=False."
                )

            if file_name not in self._get_data_files():
                raise FileNotFoundError(
                    f"Dataset file '{data_file}' was not found at '{data_path}' and is not a known remote dataset. "
                    f"Available datasets: {self._get_data_files()}"
                )

            self._download(file_name, data_path)

        if data_path.endswith(".mat"):
            from scipy.io import loadmat

            f = loadmat(data_path)

            X = f["X"]
            y = f["y"].ravel()
        else:
            X = self._load_via_txt(data_path)

            y = X[:, -1].ravel()
            X = X[:, :-1]

        return X, y

    def get_iterator(
        self,
        data_file: str,
        shuffle: bool = True,
        seed: int | None = None,
        download: bool = True,
    ) -> Iterator[np.ndarray | tuple[np.ndarray, Any]]:
        """The iterator function

        Args:
            data_file (str): Path or filename of data.
            shuffle (bool): Whether to shuffle (Default=True).
            seed (int | None): Random seed (Default=None).
            download (bool): Whether to download the dataset if not present locally (Default=True).

        Returns:
            iterator (The iterator): pysad.utils.array_streamer.ArrayStreamer.iter method applied with (X, y), where X is the variable containing feature vectors and y is the variable containing labels.
        """
        if seed is not None:
            np.random.seed(seed)  # noqa: NPY002 - also seeds the models that draw from the global state.

        iterator = ArrayStreamer(shuffle=shuffle)

        X, y = self.get_data(data_file, download=download)

        return iterator.iter(X, y)
