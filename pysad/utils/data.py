import hashlib
import os
import shutil
import tempfile
import urllib.request
from collections.abc import Iterator
from typing import Any

import numpy as np

from pysad.utils.array_streamer import ArrayStreamer

# The ODDS files of PyOD's mirror at a fixed commit, so that the data cannot change without a
# change in PySAD.
_MIRROR = (
    "https://github.com/yzhao062/pyod/raw/bd110115c680d29561dd7e05d8f4b7af4f3512f7/notebooks/data/"
)
# The SHA-256 checksum of each file, which a download must match.
_CHECKSUMS = {
    "arrhythmia.mat": "e4d47e499591ae31e5d7f0cd983367092353eac240f48e14c9b36457edb3460d",
    "cardio.mat": "5e568b6a73711012c481fa6cc984ab9c68e9a4fa44d6aeb3b43efa381c3f695b",
    "glass.mat": "2c506e6e1e2ca1e82a905d10a90773151ff812c4bd969c71a12db5b2bb6e7921",
    "ionosphere.mat": "8ac5dfa6f47db4acdfe10615489b7f7dd015d79d5b381ce641e72baca3cfefb0",
    "letter.mat": "f4a229b0e2a6b0c9157ce27fa760bcbac0f948cd690b2f5d51c120e5450f2195",
    "lympho.mat": "031b907ecb50347720b08b38f8139d05635adeca9335a064d89d5569c27a5834",
    "mnist.mat": "9479e221f236684c09f3e11c7ad124725d34ecd301b9450425e8b113368413a6",
    "musk.mat": "50bbf414caceb66cfe6f6b1eb7c1ff7c6eb1e6d87173f7bb597ba55cd3dc5493",
    "optdigits.mat": "30c664cd6f3740a173277a27cd1028d471b06f0835a9902c9712749cec58c1c4",
    "pendigits.mat": "7bc32ac83b3d583443acea8b1342a54aaa59eaedb9996af405322f473f3c22bc",
    "pima.mat": "b6a5be8d548b9293d6958beceb1c77055591714d25d40d957eba87808456468f",
    "satellite.mat": "6feac3112b9c14e1c3e60afc437f2f3d29dc1000119c9f950c628078778d6aa0",
    "satimage-2.mat": "95c37a07cea2cb201f6900d14f521510e10b6b2f0074c06ac46d3157440ec67b",
    "shuttle.mat": "b6e396c92354d5dcf5036901e36dc58e2913f2d1a5e138c35a2bedf9c69db899",
    "vertebral.mat": "84b50b055f118f54b5c976650ccd284ab7c1adee822f8b3eb517c7e2c890d15e",
    "vowels.mat": "51d5397bad1ceeda22b7cd44fd58125c5d603c9c2edb53aa59933330e259ac7b",
    "wbc.mat": "afe6b26953cc196b9a480318c903e0edef22529c2f61bd02082438c6e70630a7",
}


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
    (:cite:`rayana2016odds`, hosted in the PyOD repository at a fixed commit). Each
    download must match a SHA-256 checksum. :meth:`datasets` lists the datasets, and
    :meth:`get_stream` gives the stream of a run of the PySAD benchmark.

    Downloaded datasets are cached in a local directory. The cache directory
    defaults to ``~/.cache/pysad``, which can be overridden via the
    ``PYSAD_DATA_DIR`` environment variable or by passing an explicit
    ``data_base_path``.

    Args:
        data_base_path (str or None): Base directory that contains the data files and caches the downloaded datasets. If None, the class uses the directory in ``PYSAD_DATA_DIR`` if set, or ``~/.cache/pysad`` (Default=None).
    """

    def __init__(self, data_base_path: str | None = None) -> None:
        self.data_base_path = _get_default_data_dir() if data_base_path is None else data_base_path
        self._is_default_path = data_base_path is None

    @staticmethod
    def datasets() -> list[str]:
        """Returns the file names of the datasets that this class can download, for example ``"cardio.mat"``.

        The benchmark of PySAD uses exactly these datasets.

        Returns:
            list[str]: The file names of the datasets.
        """
        return list(_CHECKSUMS)

    def _download(self, file_name: str, target_path: str) -> None:
        """Downloads a dataset file from the remote mirror into target_path atomically.

        Args:
            file_name (str): Name of the remote file (e.g. 'cardio.mat').
            target_path (str): Destination file path.
        """
        url = _MIRROR + file_name
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
            with open(temp_path, "rb") as f:
                checksum = hashlib.sha256(f.read()).hexdigest()
            if checksum != _CHECKSUMS[file_name]:
                raise ValueError(
                    f"The download of '{file_name}' has the SHA-256 checksum {checksum}, "
                    f"but PySAD expects {_CHECKSUMS[file_name]}."
                )
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
            np.float64 array of shape (num_instances, num_features): Feature vectors.
        """
        X = np.loadtxt(path, delimiter=",")

        return X

    def get_data(self, data_file: str, download: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Loads the data given the filename or path, downloading it if not present.

        Args:
            data_file (str): Path or filename of the data.
            download (bool): Whether to download the dataset if not present locally (Default=True).

        Returns:
            tuple[np.ndarray, np.ndarray]: The feature vectors of shape (num_instances, num_features) and the labels of shape (num_instances,).
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

            if file_name not in self.datasets():
                raise FileNotFoundError(
                    f"Dataset file '{data_file}' was not found at '{data_path}' and is not a known remote dataset. "
                    f"Available datasets: {self.datasets()}"
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
        random_state: int | np.random.RandomState | None = None,
        download: bool = True,
    ) -> Iterator[np.ndarray | tuple[np.ndarray, Any]]:
        """Returns an iterator over the instances and labels of a dataset, in shuffled order by default.

        Args:
            data_file (str): Path or filename of data.
            shuffle (bool): Whether to shuffle the instances (Default=True).
            random_state (int, np.random.RandomState or None): Seed or random number generator for the shuffle, as in :class:`ArrayStreamer`. None draws from NumPy's global random state (Default=None).
            download (bool): Whether to download the dataset if not present locally (Default=True).

        Returns:
            Iterator[tuple[np.ndarray, Any]]: Iterator over ``(X, y)`` pairs, where ``X`` is the feature vector of shape (num_features,) and ``y`` is its label.
        """
        X, y = self.get_data(data_file, download=download)

        return ArrayStreamer(shuffle=shuffle, random_state=random_state).iter(X, y)

    def describe(self, data_file: str, download: bool = True) -> dict[str, Any]:
        """Returns the number of points, the number of features and the anomaly rate of a dataset.

        Args:
            data_file (str): Path or filename of data.
            download (bool): Whether to download the dataset if not present locally (Default=True).

        Returns:
            dict: ``points`` (int), ``features`` (int) and ``anomaly_rate`` (float, the fraction of points with label 1).
        """
        X, y = self.get_data(data_file, download=download)

        return {"points": len(X), "features": X.shape[1], "anomaly_rate": float(np.mean(y))}

    def get_stream(
        self,
        data_file: str,
        random_state: int | np.random.RandomState | None = None,
        max_points: int | None = None,
        scale: bool = False,
        download: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns a shuffled dataset as arrays, in the order of :meth:`get_iterator`.

        With ``random_state=seed``, ``max_points=5000`` and ``scale=True``, this is the stream of the PySAD benchmark for that seed.

        Args:
            data_file (str): Path or filename of data.
            random_state (int, np.random.RandomState or None): Seed or random number generator for the shuffle. None draws from NumPy's global random state (Default=None).
            max_points (int or None): The number of first points of the shuffled stream to keep. None keeps all of them (Default=None).
            scale (bool): Whether to scale each feature to the range from 0 to 1, with the minimum and the maximum of the points that it keeps. A constant feature becomes 0 (Default=False).
            download (bool): Whether to download the dataset if not present locally (Default=True).

        Returns:
            tuple[np.ndarray, np.ndarray]: The feature vectors of shape (num_points, num_features) and the labels of shape (num_points,), in stream order.
        """
        stream = list(self.get_iterator(data_file, random_state=random_state, download=download))
        stream = stream[:max_points]
        X, y = np.array([x for x, _ in stream]), np.array([label for _, label in stream])
        if scale:
            mins, maxes = X.min(axis=0), X.max(axis=0)
            X = (X - mins) / np.where(maxes > mins, maxes - mins, 1.0)

        return X, y
