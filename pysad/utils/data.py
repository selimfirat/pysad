import csv
import gzip
import hashlib
import json
import os
import shutil
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Iterator
from typing import Any

import numpy as np
from scipy.io import arff

from pysad.utils._datasets import (
    _CHECKSUMS,
    _CREDITCARD,
    _CREDITCARD_CHECKSUM,
    _INSECTS,
    _INSECTS_FILES,
    _KDDCUP99,
    _KDDCUP99_ARCHIVE,
    _KDDCUP99_CHECKSUM,
    _MIRROR,
    _MSL_CHECKSUMS,
    _NAB,
    _NAB_CHECKSUMS,
    _NAB_WINDOWS,
    _NASA,
    _NASA_LABELS,
    _NASA_LABELS_CHECKSUM,
    _NASA_LABELS_URL,
    _SMAP_CHECKSUMS,
    _SMD,
    _SMD_CHECKSUMS,
    _SOURCES,
    _UCR,
    _UCR_ARCHIVE,
    _UCR_CHECKSUM,
    _UCR_FOLDER,
    _UCR_SERIES,
)
from pysad.utils.array_streamer import ArrayStreamer

# A download that fails with a server error or a network error is tried again, after 5 and then 10
# seconds. A connection that sends no data for 60 seconds counts as a network error. The download asks
# for gzip, because some servers send their files slowly, for example OpenML sends the 74 MB of KDD Cup
# 99 at about 100 KB/s, and as 2 MB with gzip.
_ATTEMPTS = 3
_RETRY_SECONDS = 5.0
_TIMEOUT_SECONDS = 60.0


def _source(name: str) -> tuple[str, str]:
    """The URL and the SHA-256 checksum of a remote file of a dataset."""
    if name == "nab/combined_windows.json":
        return _NAB + "labels/combined_windows.json", _NAB_WINDOWS
    if name.startswith("nab/"):
        return _NAB + "data/" + name[len("nab/") :], _NAB_CHECKSUMS[name[len("nab/") :]]
    if name == "creditcard/creditcard.arff":
        return _CREDITCARD, _CREDITCARD_CHECKSUM
    if name == _UCR_ARCHIVE:
        return _UCR, _UCR_CHECKSUM
    if name == _KDDCUP99_ARCHIVE:
        return _KDDCUP99, _KDDCUP99_CHECKSUM
    if name.startswith("smd/labels/"):
        return _SMD + "test_label/" + name[len("smd/labels/") :], _SMD_CHECKSUMS[
            name[len("smd/labels/") :]
        ][1]
    if name.startswith("smd/"):
        return _SMD + "test/" + name[len("smd/") :], _SMD_CHECKSUMS[name[len("smd/") :]][0]
    if name == _NASA_LABELS:
        return _NASA_LABELS_URL, _NASA_LABELS_CHECKSUM
    if name.startswith("smap/"):
        return _NASA + name[len("smap/") :], _SMAP_CHECKSUMS[name[len("smap/") :]]
    if name.startswith("msl/"):
        return _NASA + name[len("msl/") :], _MSL_CHECKSUMS[name[len("msl/") :]]
    if name.startswith("insects/"):
        drive_id, checksum = _INSECTS_FILES[name[len("insects/") :]]
        return _INSECTS + drive_id, checksum

    return _MIRROR + name, _CHECKSUMS[name]


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

    Datasets are loaded from the local filesystem or downloaded on demand. The multivariate
    datasets come from the Outlier Detection DataSets (ODDS) repository mirror
    (:cite:`rayana2016odds`, hosted in the PyOD repository at a fixed commit). The univariate
    series come from the Numenta Anomaly Benchmark (NAB) and the UCR Time Series Anomaly Archive.
    The multivariate streams in time order come from KDD Cup 99, the Server Machine Dataset (SMD),
    the NASA SMAP and MSL telemetry of Telemanom (:cite:`hundman2018detecting`), INSECTS and
    credit-card fraud (OpenML 1597). Each download must match a SHA-256 checksum.
    :meth:`datasets` lists the datasets, and
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
    def datasets(source: str = "odds") -> list[str]:
        """Returns the names of the datasets that this class can download.

        Args:
            source (str): The collection of the datasets (Default="odds"). These are the collections:

                - ``"odds"``: the 17 multivariate datasets of ODDS, for example ``"cardio.mat"``.
                - ``"nab"``: the 58 univariate series of the Numenta Anomaly Benchmark, for example ``"nab/realKnownCause/nyc_taxi.csv"``. The points in an anomaly window of NAB have the label 1.
                - ``"ucr"``: the 250 univariate series of the UCR Time Series Anomaly Archive, for example ``"ucr/025_UCR_Anomaly_DISTORTEDInternalBleeding14_2800_5607_5634.txt"``. The last two numbers of a name are the first and the last point of the anomaly, counted from 1. These points have the label 1. The number before them is the length of the training part, which has no anomalies. The first series download the archive of 184 MB.
                - ``"kddcup99"``: ``"kddcup99/http.csv"``, the 58,724 http connections of logged in users of the KDD Cup 99 10% file, in their original order. The 3 features are ``log(x + 0.1)`` of the duration, the bytes from the source and the bytes to the destination. The attacks have the label 1. The first use downloads the file of 74 MB, which the server compresses to 2 MB.
                - ``"smd"``: the test halves of the 28 machines of the Server Machine Dataset, for example ``"smd/machine-1-1.txt"``, with one point of 38 features each minute.
                - ``"smap"``: the test halves of the 54 telemetry channels of the SMAP satellite, for example ``"smap/P-1.npy"``, with 25 features. The first feature is the telemetry value, and the others are the one-hot encoded commands that the spacecraft received. The points in an anomaly range of Telemanom have the label 1. Channel P-2 has two rows of ranges, and both rows label it.
                - ``"msl"``: the test halves of the 27 telemetry channels of the MSL rover Curiosity, for example ``"msl/M-1.npy"``, with 55 features in the same layout as ``"smap"``.
                - ``"creditcard"``: ``"creditcard/creditcard.arff"``, the 284,807 transactions of OpenML 1597, version 1, in their original time order. The 30 features are Time (elapsed seconds), V1 through V28 and Amount. Class is returned separately as the label: 1 for fraud, 0 otherwise. The database is under ODbL 1.0 and its contents under DbCL 1.0; see :doc:`/guides/stream_data` for the source links and license details.
                - ``"insects"``: the 4 INSECTS streams of the ARCUS paper, for example ``"insects/abrupt.csv"``, with 33 features. The southern house mosquito is the anomaly class.

        Returns:
            list[str]: The names of the datasets, which :meth:`get_data` accepts.

        Raises:
            ValueError: If ``source`` is not one of the collections above.
        """
        if source == "odds":
            return list(_CHECKSUMS)
        if source == "nab":
            return ["nab/" + name for name in _NAB_CHECKSUMS]
        if source == "ucr":
            return ["ucr/" + name for name in _UCR_SERIES]
        if source == "kddcup99":
            return ["kddcup99/http.csv"]
        if source == "smd":
            return ["smd/" + name for name in _SMD_CHECKSUMS]
        if source == "smap":
            return ["smap/" + name for name in _SMAP_CHECKSUMS]
        if source == "msl":
            return ["msl/" + name for name in _MSL_CHECKSUMS]
        if source == "creditcard":
            return ["creditcard/creditcard.arff"]
        if source == "insects":
            return ["insects/" + name for name in _INSECTS_FILES]
        raise ValueError(f"source must be one of {_SOURCES}, got {source!r}.")

    def _download(self, file_name: str, target_path: str) -> None:
        """Downloads a dataset file from the remote mirror into target_path atomically.

        Args:
            file_name (str): Name of the remote file (e.g. 'cardio.mat').
            target_path (str): Destination file path.
        """
        url, expected = _source(file_name)
        target_dir = os.path.dirname(target_path)
        if target_dir:
            os.makedirs(target_dir, exist_ok=True)

        temp_file = tempfile.NamedTemporaryFile(
            dir=target_dir or None, prefix=f"tmp_{os.path.basename(file_name)}_", delete=False
        )
        temp_path = temp_file.name
        try:
            temp_file.close()
            req = urllib.request.Request(
                url, headers={"User-Agent": "pysad", "Accept-Encoding": "gzip"}
            )
            for attempt in range(_ATTEMPTS):
                try:
                    with (
                        urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS) as response,
                        open(temp_path, "wb") as f,
                    ):
                        gzipped = response.headers.get("Content-Encoding") == "gzip"
                        shutil.copyfileobj(
                            gzip.GzipFile(fileobj=response) if gzipped else response, f
                        )
                    break
                except OSError as error:  # URLError, or TimeoutError while reading the response
                    client_error = isinstance(error, urllib.error.HTTPError) and error.code < 500
                    if client_error or attempt == _ATTEMPTS - 1:
                        raise
                    time.sleep(_RETRY_SECONDS * 2**attempt)
            with open(temp_path, "rb") as f:
                checksum = hashlib.sha256(f.read()).hexdigest()
            if checksum != expected:
                raise ValueError(
                    f"The download of '{file_name}' has the SHA-256 checksum {checksum}, "
                    f"but PySAD expects {expected}."
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

    def get_data(
        self, data_file: str, download: bool = True, timestamps: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Loads the data given the filename or path, downloading it if not present.

        Args:
            data_file (str): Path or filename of the data.
            download (bool): Whether to download the dataset if not present locally (Default=True).
            timestamps (bool): Whether to give each point of a NAB series as ``[timestamp, value]``, with the timestamp in seconds since the Unix epoch, as :class:`~pysad.models.HTM` takes it. NAB gives local times without a time zone, and the timestamps read them as UTC (Default=False).

        Returns:
            tuple[np.ndarray, np.ndarray]: The feature vectors of shape (num_instances, num_features) and the labels of shape (num_instances,).

        Raises:
            ValueError: If ``timestamps`` is True and ``data_file`` is not a NAB series.
        """
        if timestamps and not data_file.startswith("nab/"):
            raise ValueError(f"Only the NAB series have timestamps, got {data_file!r}.")
        if os.path.exists(data_file):
            data_path = data_file
        else:
            data_path = os.path.join(self.data_base_path, data_file)

        # An ODDS dataset is known by its file name, and a stream of another collection by its full name.
        series = data_file.startswith(tuple(f"{source}/" for source in _SOURCES[1:]))
        file_name = data_file if series else os.path.basename(data_file)

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

            if not any(file_name in self.datasets(source) for source in _SOURCES):
                raise FileNotFoundError(
                    f"Dataset file '{data_file}' was not found at '{data_path}' and is not a known remote dataset. "
                    f"Available datasets: {self.datasets()}, and the streams of datasets(source) for each source in {_SOURCES[1:]}."
                )

            if file_name.startswith("ucr/"):
                self._extract_ucr(file_name, data_path)
            elif file_name.startswith("kddcup99/"):
                self._extract_kddcup99(data_path)
            else:
                self._download(file_name, data_path)

        if file_name == "creditcard/creditcard.arff":
            records, metadata = arff.loadarff(data_path)
            X = np.column_stack([records[n] for n in metadata.names() if n != "Class"])
            return X, records["Class"].astype(int)
        if file_name.startswith("nab/"):
            return self._load_nab(file_name, data_path, download, timestamps)
        if file_name.startswith("ucr/"):
            return self._load_ucr(file_name, data_path)
        if file_name.startswith("smd/"):
            return self._load_smd(file_name, data_path, download)
        if file_name.startswith(("smap/", "msl/")):
            return self._load_nasa(file_name, data_path, download)
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

    def _load_nab(
        self, name: str, data_path: str, download: bool, timestamps: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        """Loads a NAB series, with the timestamps if `timestamps` is True, and labels the points in its anomaly windows with 1."""
        windows_path = os.path.join(self.data_base_path, "nab", "combined_windows.json")
        if not os.path.exists(windows_path):
            if not download:
                raise FileNotFoundError(
                    f"The NAB labels were not found at '{windows_path}' and download=False."
                )
            self._download("nab/combined_windows.json", windows_path)
        with open(windows_path) as f:
            windows = json.load(f)[name[len("nab/") :]]
        with open(data_path, newline="") as f:
            rows = list(csv.reader(f))[1:]  # the header is "timestamp,value"

        times = np.array([row[0] for row in rows], dtype="datetime64[s]")
        y = np.zeros(len(rows), dtype=int)
        for start, end in windows:
            y[(times >= np.datetime64(start)) & (times <= np.datetime64(end))] = 1

        values = np.array([[float(row[1])] for row in rows])
        if timestamps:
            return np.column_stack([times.astype(np.float64), values]), y
        return values, y

    def _extract_ucr(self, name: str, data_path: str) -> None:
        """Copies a UCR series out of the archive, and downloads the archive if it is not present."""
        archive = os.path.join(self.data_base_path, _UCR_ARCHIVE)
        if not os.path.exists(archive):
            self._download(_UCR_ARCHIVE, archive)
        with zipfile.ZipFile(archive) as z, z.open(_UCR_FOLDER + name[len("ucr/") :]) as f:
            content = f.read()
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        with open(data_path, "wb") as f:
            f.write(content)

    def _load_ucr(self, name: str, data_path: str) -> tuple[np.ndarray, np.ndarray]:
        """Loads a UCR series and labels the points of its anomaly with 1."""
        with open(data_path) as f:
            X = np.array(f.read().split(), dtype=np.float64).reshape(
                -1, 1
            )  # some files use one line
        first, last = map(int, name.removesuffix(".txt").split("_")[-2:])
        y = np.zeros(len(X), dtype=int)
        y[first - 1 : last] = 1

        return X, y

    def _extract_kddcup99(self, data_path: str) -> None:
        """Writes the http stream of KDD Cup 99, and downloads the file of the cup if it is not present."""
        archive = os.path.join(self.data_base_path, _KDDCUP99_ARCHIVE)
        if not os.path.exists(archive):
            self._download(_KDDCUP99_ARCHIVE, archive)
        with open(archive) as f:
            lines = (line for line in f if line.strip() and not line.startswith(("@", "%")))
            # The columns 0, 2, 4, 5 and 11 are the duration, the service, the bytes from the
            # source, the bytes to the destination and logged_in. The last one is the label.
            rows = [r for r in csv.reader(lines) if r[2] == "http" and r[11] == "1"]
        X = np.log(np.array([[r[0], r[4], r[5]] for r in rows], dtype=np.float64) + 0.1)
        y = np.array([r[-1] != "normal" for r in rows], dtype=np.float64)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        np.savetxt(data_path, np.column_stack([X, y]), delimiter=",", fmt="%.17g")

    def _load_smd(self, name: str, data_path: str, download: bool) -> tuple[np.ndarray, np.ndarray]:
        """Loads the test half of an SMD machine and its labels."""
        labels_name = "smd/labels/" + name[len("smd/") :]
        labels_path = os.path.join(self.data_base_path, labels_name)
        if not os.path.exists(labels_path):
            if not download:
                raise FileNotFoundError(
                    f"The SMD labels were not found at '{labels_path}' and download=False."
                )
            self._download(labels_name, labels_path)

        return np.loadtxt(data_path, delimiter=","), np.loadtxt(labels_path, dtype=int)

    def _load_nasa(
        self, name: str, data_path: str, download: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        """Loads the test half of a SMAP or MSL channel and labels the points of its anomaly ranges with 1."""
        labels_path = os.path.join(self.data_base_path, _NASA_LABELS)
        if not os.path.exists(labels_path):
            if not download:
                raise FileNotFoundError(
                    f"The SMAP and MSL labels were not found at '{labels_path}' and download=False."
                )
            self._download(_NASA_LABELS, labels_path)
        spacecraft, file_name = name.split("/")
        X = np.load(data_path)
        y = np.zeros(len(X), dtype=int)
        with open(labels_path, newline="") as f:
            for row in csv.DictReader(f):
                # A channel can have more than one row. The ranges include their last point.
                if row["spacecraft"] == spacecraft.upper() and row["chan_id"] + ".npy" == file_name:
                    for start, end in json.loads(row["anomaly_sequences"]):
                        y[start : end + 1] = 1

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
