import contextlib
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
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.utils import check_random_state

from pysad.utils._datasets import (
    _CHECKSUMS,
    _CREDITCARD,
    _CREDITCARD_CHECKSUM,
    _INSECTS,
    _INSECTS_FILES,
    _INVERTED_LABELS,
    _KDDCUP99,
    _KDDCUP99_ARCHIVE,
    _KDDCUP99_CHECKSUM,
    _LICENSES,
    _MIRROR,
    _MSL_CHECKSUMS,
    _NAB,
    _NAB_CHECKSUMS,
    _NAB_REAL_WORLD,
    _NAB_WINDOWS,
    _NASA,
    _NASA_LABELS,
    _NASA_LABELS_CHECKSUM,
    _NASA_LABELS_URL,
    _SMAP_CHECKSUMS,
    _SMD,
    _SMD_CHECKSUMS,
    _SOURCES,
    _SWAT,
    _SWAT_CHECKSUMS,
    _TSB_DRIFT_ARCHIVE,
    _TSB_DRIFT_ARCHIVE_CHECKSUM,
    _TSB_DRIFT_ARCHIVE_KEY,
    _TSB_DRIFT_SERIES,
    _UCR,
    _UCR_ARCHIVE,
    _UCR_CHECKSUM,
    _UCR_FOLDER,
    _UCR_SERIES,
    _YAHOO,
    _YAHOO_CHECKSUMS,
)
from pysad.utils.array_streamer import ArrayStreamer

# A download that fails with a server error or a network error is tried again, after 5 and then 10
# seconds. A connection that sends no data for 60 seconds counts as a network error. The download asks
# for gzip, because some servers send their files slowly, for example OpenML sends the 74 MB of KDD Cup
# 99 at about 100 KB/s, and as 2 MB with gzip.
_ATTEMPTS = 3
_RETRY_SECONDS = 5.0
_TIMEOUT_SECONDS = 60.0

_SMAP_PREFIX = "smap/"
_CREDITCARD_FILE = "creditcard/creditcard.arff"
_TSB_DRIFT_PREFIX = "tsb-drift/"
_YAHOO_PREFIX = "yahoo/"
_SWAT_PREFIX = "swat/"

_DATASET_GROUPS: dict[str, tuple[str, Callable[[], Any]]] = {
    "odds": ("", lambda: _CHECKSUMS),
    "nab": ("nab/", lambda: _NAB_CHECKSUMS),
    "nab-real-world": ("nab/", lambda: _NAB_REAL_WORLD),
    "ucr": ("ucr/", lambda: _UCR_SERIES),
    "kddcup99": ("", lambda: ["kddcup99/http.csv"]),
    "smd": ("smd/", lambda: _SMD_CHECKSUMS),
    "smap": (_SMAP_PREFIX, lambda: _SMAP_CHECKSUMS),
    "msl": ("msl/", lambda: _MSL_CHECKSUMS),
    "creditcard": ("", lambda: [_CREDITCARD_FILE]),
    "insects": ("insects/", lambda: _INSECTS_FILES),
    "tsb-drift": (_TSB_DRIFT_PREFIX, lambda: _TSB_DRIFT_SERIES),
    "yahoo": (_YAHOO_PREFIX, lambda: _YAHOO_CHECKSUMS),
    "swat": (_SWAT_PREFIX, lambda: _SWAT_CHECKSUMS),
}


def _cleanup_file(path: str) -> None:
    """Remove a temporary file if it exists, ignoring OS errors."""
    if os.path.exists(path):
        with contextlib.suppress(OSError):
            os.remove(path)


def _fetch_url_to_file(url: str, dest_path: str) -> None:
    """Fetch `url` into `dest_path` with exponential backoff retries."""
    if not url.startswith(("https://", "http://")):
        raise ValueError(f"Unsupported URL scheme for download: {url!r}.")
    req = urllib.request.Request(url, headers={"User-Agent": "pysad", "Accept-Encoding": "gzip"})
    open_url = urllib.request.urlopen
    for attempt in range(_ATTEMPTS):
        try:
            with (
                open_url(req, timeout=_TIMEOUT_SECONDS) as response,
                open(dest_path, "wb") as f,
            ):
                gzipped = response.headers.get("Content-Encoding") == "gzip"
                shutil.copyfileobj(gzip.GzipFile(fileobj=response) if gzipped else response, f)
            return
        except OSError as error:  # URLError, or TimeoutError while reading the response
            client_error = isinstance(error, urllib.error.HTTPError) and error.code < 500
            if client_error or attempt == _ATTEMPTS - 1:
                raise
            time.sleep(_RETRY_SECONDS * 2**attempt)


def _load_npz_cache(
    cache_path: str, data_path: str
) -> tuple[NDArray[np.float64], NDArray[np.int64]] | None:
    """Return cached `(X, y)` arrays if fresh and readable, else `None`."""
    if not (
        os.path.exists(cache_path) and os.path.getmtime(cache_path) >= os.path.getmtime(data_path)
    ):
        return None
    try:
        with np.load(cache_path) as cached:
            return cached["X"], cached["y"]
    except (OSError, ValueError, KeyError):
        return None


def _save_npz_cache(
    cache_path: str, data_path: str, X: NDArray[np.float64], y: NDArray[np.int64]
) -> None:
    """Persist parsed arrays atomically next to `data_path`."""
    tmp_path = f"{data_path}.{os.getpid()}.tmp.npz"
    try:
        np.savez(tmp_path, X=X, y=y)
        os.replace(tmp_path, cache_path)
    except OSError:
        _cleanup_file(tmp_path)


def _parse_arff_table(
    data_path: str, label_column: str = "Class"
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Parse a numeric ARFF table into contiguous features and integer labels."""
    attributes: list[str] = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("%"):
                continue
            head = stripped.split(maxsplit=2)
            tag = head[0].lower()
            if tag == "@attribute" and len(head) >= 2:
                attributes.append(head[1].strip("'\""))
            elif tag == "@data":
                break
        raw_body = f.read()

    if "%" in raw_body:
        body = ",".join(
            line.strip()
            for line in raw_body.splitlines()
            if line.strip() and not line.lstrip().startswith("%")
        )
    else:
        body = raw_body.replace("\r", "").replace("\n", ",")
    body = body.replace("'", "").replace('"', "").strip(" \t,")

    num_cols = len(attributes)
    class_idx = attributes.index(label_column)
    arr = np.fromstring(body, dtype=np.float64, sep=",").reshape(-1, num_cols)
    if class_idx == num_cols - 1:
        X = np.ascontiguousarray(arr[:, :class_idx])
    else:
        feature_cols = [i for i in range(num_cols) if i != class_idx]
        X = np.ascontiguousarray(arr[:, feature_cols])
    return X, arr[:, class_idx].astype(int)


def _source(name: str) -> tuple[str, str]:
    """The URL and the SHA-256 checksum of a remote file of a dataset."""
    exact_resolvers: dict[str, Callable[[], tuple[str, str]]] = {
        "nab/combined_windows.json": lambda: (_NAB + "labels/combined_windows.json", _NAB_WINDOWS),
        _CREDITCARD_FILE: lambda: (_CREDITCARD, _CREDITCARD_CHECKSUM),
        "kddcup99/http.csv": lambda: (_KDDCUP99, _KDDCUP99_CHECKSUM),
        _KDDCUP99_ARCHIVE: lambda: (_KDDCUP99, _KDDCUP99_CHECKSUM),
        _NASA_LABELS: lambda: (_NASA_LABELS_URL, _NASA_LABELS_CHECKSUM),
    }
    if name in exact_resolvers:
        return exact_resolvers[name]()

    prefix_resolvers: tuple[tuple[str, Callable[[str], tuple[str, str]]], ...] = (
        ("smd/labels/", lambda rel: (_SMD + "test_label/" + rel, _SMD_CHECKSUMS[rel][1])),
        ("smd/", lambda rel: (_SMD + "test/" + rel, _SMD_CHECKSUMS[rel][0])),
        ("insects/", lambda rel: (_INSECTS + _INSECTS_FILES[rel][0], _INSECTS_FILES[rel][1])),
        ("ucr/", lambda _rel: (_UCR, _UCR_CHECKSUM)),
        (_TSB_DRIFT_PREFIX, lambda _rel: (_TSB_DRIFT_ARCHIVE, _TSB_DRIFT_ARCHIVE_CHECKSUM)),
        ("nab/", lambda rel: (_NAB + "data/" + rel, _NAB_CHECKSUMS[rel])),
        (_SMAP_PREFIX, lambda rel: (_NASA + rel, _SMAP_CHECKSUMS[rel])),
        ("msl/", lambda rel: (_NASA + rel, _MSL_CHECKSUMS[rel])),
        (_YAHOO_PREFIX, lambda rel: (_YAHOO + rel, _YAHOO_CHECKSUMS[rel])),
        (_SWAT_PREFIX, lambda rel: (_SWAT + rel, _SWAT_CHECKSUMS[rel])),
    )
    for prefix, resolve in prefix_resolvers:
        if name.startswith(prefix):
            return resolve(name[len(prefix) :])
    return _MIRROR + name, _CHECKSUMS[name]


def _license(name: str) -> str:
    """The license of the collection that a remote dataset file belongs to."""
    if name == _NASA_LABELS:
        return _LICENSES["telemanom"]
    for source in _SOURCES[1:]:
        if name.startswith(source + "/"):
            return _LICENSES[source]

    return _LICENSES["odds"]


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
    (:cite:`rayana2016odds`, hosted in the PyOD repository at a fixed commit). For
    ``vertebral.mat``, the labels are flipped from the ODDS file, because ODDS marks the 30
    healthy patients as the anomalies: the anomaly class of the returned labels is the 210
    unhealthy patients. The univariate
    series come from the Numenta Anomaly Benchmark (NAB), the UCR Time Series Anomaly Archive,
    and the 67 real production-traffic series of Yahoo S5 A1Benchmark.
    The multivariate streams in time order come from KDD Cup 99, the Server Machine Dataset (SMD),
    the NASA SMAP and MSL telemetry of Telemanom (:cite:`hundman2018detecting`), INSECTS and
    credit-card fraud (OpenML 1597), and SWaT (Secure Water Treatment) of iTrust, Singapore
    University of Technology and Design. The distribution-drift series come from TSB-drift, the
    top-75 subset of the TSB-AD-M benchmark introduced by Parrino et al. (KDD 2026,
    "In a Streaming World, Should You Stand Still? A Comprehensive Benchmark of Anomaly
    Detection in Streams", DOI 10.1145/3770855.3817495, CC BY 4.0) and extracted from the
    TSB-AD-M archive of Liu and Paparrizos (NeurIPS 2024). Each download must match a
    SHA-256 checksum.
    :meth:`datasets` lists the datasets, :meth:`metadata` gives the source, the license and the
    checksum of each dataset file, and
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

                - ``"odds"``: the 17 multivariate datasets of ODDS, for example ``"cardio.mat"``. For ``"vertebral.mat"``, the labels are flipped from the ODDS file, so the anomaly class is the unhealthy patients, not the healthy patients that ODDS marks as anomalies.
                - ``"nab"``: the 58 univariate series of the Numenta Anomaly Benchmark, for example ``"nab/realKnownCause/nyc_taxi.csv"``. The points in an anomaly window of NAB have the label 1.
                - ``"nab-real-world"``: the 47 real-world series of NAB, that is all NAB series except the 11 artificial ones under ``artificialNoAnomaly`` and ``artificialWithAnomaly``, for example ``"nab/realKnownCause/nyc_taxi.csv"``. :meth:`metadata` gives the source, the license, and the SHA-256 checksum of each of these series.
                - ``"ucr"``: the 250 univariate series of the UCR Time Series Anomaly Archive, for example ``"ucr/025_UCR_Anomaly_DISTORTEDInternalBleeding14_2800_5607_5634.txt"``. The last two numbers of a name are the first and the last point of the anomaly, counted from 1. These points have the label 1. The number before them is the length of the training part, which has no anomalies. The first series download the archive of 184 MB.
                - ``"kddcup99"``: ``"kddcup99/http.csv"``, the 58,724 http connections of logged in users of the KDD Cup 99 10% file, in their original order. The 3 features are ``log(x + 0.1)`` of the duration, the bytes from the source and the bytes to the destination. The attacks have the label 1. The first use downloads the file of 74 MB, which the server compresses to 2 MB.
                - ``"smd"``: the test halves of the 28 machines of the Server Machine Dataset, for example ``"smd/machine-1-1.txt"``, with one point of 38 features each minute.
                - ``"smap"``: the test halves of the 54 telemetry channels of the SMAP satellite, for example ``"smap/P-1.npy"``, with 25 features. The first feature is the telemetry value, and the others are the one-hot encoded commands that the spacecraft received. The points in an anomaly range of Telemanom have the label 1. Channel P-2 has two rows of ranges, and both rows label it.
                - ``"msl"``: the test halves of the 27 telemetry channels of the MSL rover Curiosity, for example ``"msl/M-1.npy"``, with 55 features in the same layout as ``"smap"``.
                - ``"creditcard"``: ``"creditcard/creditcard.arff"``, the 284,807 transactions of OpenML 1597, version 1, in their original time order. The 30 features are Time (elapsed seconds), V1 through V28 and Amount. Class is returned separately as the label: 1 for fraud, 0 otherwise. The database is under ODbL 1.0 and its contents under DbCL 1.0; see :doc:`/guides/stream_data` for the source links and license details.
                - ``"insects"``: the 4 INSECTS streams of the ARCUS paper, for example ``"insects/abrupt.csv"``, with 33 features. The southern house mosquito is the anomaly class.
                - ``"tsb-drift"``: the 75 series of the TSB-drift distribution-drift benchmark of Parrino et al. (KDD 2026), for example ``"tsb-drift/009_MSL_id_8_Sensor_tr_714_1st_1390.csv"``. The series are extracted from the TSB-AD-M archive, and the last column holds the binary anomaly label. The paper is under CC BY 4.0; the underlying TSB-AD series keep their original per-dataset licenses and the TSB-AD curation is provided under the Apache 2.0 license.
                - ``"yahoo"``: the 67 real production-traffic series of Yahoo S5 A1Benchmark, for example ``"yahoo/real_1.csv"``. The value column is the only feature. Label 1 marks an anomaly. Yahoo requires a signed Data Sharing Agreement for academic, non-commercial research.
                - ``"swat"``: SWaT (Secure Water Treatment) of iTrust, Singapore University of Technology and Design, mirrored by PaddlePaddle PaddleTS, for example ``"swat/swat_test.csv"``. Each point has 51 sensor and actuator readings and a binary Normal/Attack label; label 1 marks an attack. iTrust provides the data free for research upon request, and iTrust's terms of usage apply: credit iTrust and do not redistribute the data.

        Returns:
            list[str]: The names of the datasets, which :meth:`get_data` accepts.

        Raises:
            ValueError: If ``source`` is not one of the collections above.
        """
        if source not in _DATASET_GROUPS:
            raise ValueError(f"source must be one of {_SOURCES}, got {source!r}.")
        prefix, get_entries = _DATASET_GROUPS[source]
        entries = get_entries()
        return [prefix + name for name in entries] if prefix else list(entries)

    @staticmethod
    def metadata(data_file: str) -> dict[str, str]:
        """Returns the public source, license, and SHA-256 checksum of a remote dataset file.

        Args:
            data_file (str): The name of a dataset file, as :meth:`datasets` lists it, for example ``"smap/P-1.npy"``.

        Returns:
            dict[str, str]: ``"source"`` is the URL that :meth:`get_data` downloads, ``"license"`` is the license of the dataset, and ``"checksum"`` is the SHA-256 checksum that every download must match.

        Raises:
            KeyError: If ``data_file`` is not a known remote dataset.
        """
        url, checksum = _source(data_file)

        return {"source": url, "license": _license(data_file), "checksum": checksum}

    def _download(self, file_name: str, target_path: str) -> None:
        """Downloads a dataset file from the remote mirror into target_path atomically.

        Args:
            file_name (str): Name of the remote file (e.g. 'cardio.mat').
            target_path (str): Destination file path.
        """
        url, expected = _source(file_name)
        target_path = os.path.normpath(target_path)
        target_dir = os.path.dirname(target_path)
        if target_dir:
            os.makedirs(target_dir, exist_ok=True)

        fd, temp_path = tempfile.mkstemp(
            prefix=f"tmp_{os.path.basename(file_name)}_", dir=target_dir or None
        )
        os.close(fd)
        try:
            _fetch_url_to_file(url, temp_path)
            with open(temp_path, "rb") as f:
                checksum = hashlib.sha256(f.read()).hexdigest()
            if checksum != expected:
                raise ValueError(
                    f"The download of '{file_name}' has the SHA-256 checksum {checksum}, "
                    f"but PySAD expects {expected}."
                )
            os.replace(temp_path, target_path)
        except Exception:
            _cleanup_file(temp_path)
            raise

    def _load_via_txt(self, path: str) -> NDArray[np.float64]:
        """Loads the data file from .txt file.

        Args:
            path (str): The path of data.

        Returns:
            np.float64 array of shape (num_instances, num_features): Feature vectors.
        """
        X = np.loadtxt(path, delimiter=",")

        return X

    def _resolve_local_data_path(self, data_file: str) -> str:
        """Resolve an existing local path across explicit, default cache, and repo fallbacks."""
        if os.path.exists(data_file):
            return data_file
        data_path = os.path.join(self.data_base_path, data_file)
        if os.path.exists(data_path) or not self._is_default_path:
            return data_path
        for fallback in (
            os.path.join("data", data_file),
            os.path.join("examples", "data", data_file),
        ):
            if os.path.exists(fallback):
                return fallback
        return data_path

    def _ensure_remote_dataset(self, file_name: str, data_path: str, data_file: str) -> None:
        """Validate `file_name` and download or extract it when missing locally."""
        if not any(file_name in self.datasets(source) for source in _SOURCES):
            raise FileNotFoundError(
                f"Dataset file '{data_file}' was not found at '{data_path}' and is not a known remote dataset. "
                f"Available datasets: {self.datasets()}, and the streams of datasets(source) for each source in {_SOURCES[1:]}."
            )
        if file_name.startswith("ucr/"):
            self._extract_ucr(file_name, data_path)
        elif file_name.startswith("kddcup99/"):
            self._extract_kddcup99(data_path)
        elif file_name.startswith(_TSB_DRIFT_PREFIX):
            self._extract_tsb_drift(file_name, data_path)
        else:
            self._download(file_name, data_path)

    def _load_resolved_data(
        self, file_name: str, data_path: str, download: bool, timestamps: bool
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """Dispatch a resolved dataset file path to its format-specific loader."""
        if file_name == _CREDITCARD_FILE:
            return self._load_creditcard(data_path)

        prefix_loaders: tuple[
            tuple[
                str | tuple[str, ...],
                Callable[[], tuple[NDArray[np.float64], NDArray[np.int64]]],
            ],
            ...,
        ] = (
            ("nab/", lambda: self._load_nab(file_name, data_path, download, timestamps)),
            ("ucr/", lambda: self._load_ucr(file_name, data_path)),
            ("smd/", lambda: self._load_smd(file_name, data_path, download)),
            ((_SMAP_PREFIX, "msl/"), lambda: self._load_nasa(file_name, data_path, download)),
            (_TSB_DRIFT_PREFIX, lambda: self._load_tsb_drift(data_path)),
            (_YAHOO_PREFIX, lambda: self._load_yahoo(data_path)),
            (_SWAT_PREFIX, lambda: self._load_swat(data_path)),
        )
        for prefix, loader in prefix_loaders:
            if file_name.startswith(prefix):
                return loader()

        if data_path.endswith(".mat"):
            from scipy.io import loadmat

            f = loadmat(data_path)
            y = f["y"].ravel()
            if file_name in _INVERTED_LABELS:
                y = 1 - y
            return f["X"], y

        X = self._load_via_txt(data_path)
        y_txt: Any = X[:, -1].ravel()
        return X[:, :-1], y_txt

    def get_data(
        self, data_file: str, download: bool = True, timestamps: bool = False
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """Loads the data given the filename or path, downloading it if not present.

        Args:
            data_file (str): Path or filename of the data.
            download (bool): Whether to download the dataset if not present locally (Default=True).
            timestamps (bool): Whether to give each point of a NAB series as ``[timestamp, value]``, with the timestamp in seconds since the Unix epoch, as :class:`~pysad.models.HTM` takes it. NAB gives local times without a time zone, and the timestamps read them as UTC (Default=False).

        Returns:
            tuple[NDArray[np.float64], NDArray[np.int64]]: The feature vectors of shape (num_instances, num_features) and the labels of shape (num_instances,).

        Raises:
            ValueError: If ``timestamps`` is True and ``data_file`` is not a NAB series.
        """
        if timestamps and not data_file.startswith("nab/"):
            raise ValueError(f"Only the NAB series have timestamps, got {data_file!r}.")
        data_path = self._resolve_local_data_path(data_file)

        # An ODDS dataset is known by its file name, and a stream of another collection by its full name.
        series = data_file.startswith(tuple(f"{source}/" for source in _SOURCES[1:]))
        file_name = data_file if series else os.path.basename(data_file)

        if not os.path.exists(data_path):
            if not download:
                raise FileNotFoundError(
                    f"Dataset file '{data_file}' was not found at '{data_path}' and download=False."
                )
            self._ensure_remote_dataset(file_name, data_path, data_file)

        return self._load_resolved_data(file_name, data_path, download, timestamps)

    def _load_creditcard(self, data_path: str) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """Loads the numeric credit-card ARFF file and separates features from ``Class``."""
        cache_path = f"{data_path}.npz"
        cached = _load_npz_cache(cache_path, data_path)
        if cached is not None:
            return cached
        X, y = _parse_arff_table(data_path)
        _save_npz_cache(cache_path, data_path, X, y)
        return X, y

    def _load_nab(
        self, name: str, data_path: str, download: bool, timestamps: bool
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
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

    def _load_ucr(self, name: str, data_path: str) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
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

    def _load_smd(
        self, name: str, data_path: str, download: bool
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
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
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
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

    def _extract_tsb_drift(self, name: str, data_path: str) -> None:
        """Extracts a TSB-drift series out of the TSB-AD-M archive, downloading it if needed."""
        archive = os.path.join(self.data_base_path, _TSB_DRIFT_ARCHIVE_KEY)
        if not os.path.exists(archive):
            self._download(_TSB_DRIFT_ARCHIVE_KEY, archive)
        with (
            zipfile.ZipFile(archive) as z,
            z.open("TSB-AD-M/" + name[len(_TSB_DRIFT_PREFIX) :]) as f,
        ):
            content = f.read()
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        with open(data_path, "wb") as f:
            f.write(content)

    def _load_tsb_drift(self, data_path: str) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """Loads a TSB-drift CSV and separates the binary ``Label`` column from the features."""
        X = np.loadtxt(data_path, delimiter=",", skiprows=1)
        y = X[:, -1].ravel().astype(int)
        X = X[:, :-1]
        return X, y

    def _load_yahoo(self, data_path: str) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """Loads Yahoo S5 values and binary anomaly labels in file order.

        Each row contains ``timestamp,value,is_anomaly``. The timestamp counts hours and is
        not a feature. Label 1 marks an anomaly. A single row returns one feature row.
        """
        data = np.loadtxt(data_path, delimiter=",", skiprows=1, ndmin=2)
        return data[:, 1:2], data[:, 2].astype(np.int64)

    def _load_swat(self, data_path: str) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """Loads a SWaT CSV of 51 sensor and actuator readings and a binary Normal/Attack label.

        The file has a header row, which is skipped. Label 1 marks an attack and label 0
        marks normal operation.

        Args:
            data_path (str): Path of the downloaded SWaT CSV file.

        Returns:
            tuple[NDArray[np.float64], NDArray[np.int64]]: The feature vectors of shape
            (num_instances, 51) and the binary labels of shape (num_instances,).
        """
        data = np.loadtxt(data_path, delimiter=",", skiprows=1, ndmin=2)
        X = data[:, :-1]
        y = data[:, -1].ravel().astype(np.int64)
        return X, y

    def get_iterator(
        self,
        data_file: str,
        shuffle: bool = True,
        random_state: int | np.random.RandomState | None = None,
        download: bool = True,
    ) -> Iterator[tuple[NDArray[np.float64], Any]]:
        """Returns an iterator over the instances and labels of a dataset, in shuffled order by default.

        Args:
            data_file (str): Path or filename of data.
            shuffle (bool): Whether to shuffle the instances (Default=True).
            random_state (int, np.random.RandomState or None): Seed or random number generator for the shuffle, as in :class:`ArrayStreamer`. None draws from NumPy's global random state (Default=None).
            download (bool): Whether to download the dataset if not present locally (Default=True).

        Returns:
            Iterator[tuple[NDArray[np.float64], Any]]: Iterator over ``(X, y)`` pairs, where ``X`` is the feature vector of shape (num_features,) and ``y`` is its label.
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
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """Returns a shuffled dataset as arrays, in the order of :meth:`get_iterator`.

        With ``random_state=seed``, ``max_points=5000`` and ``scale=True``, this is the stream of the PySAD benchmark for that seed.

        Args:
            data_file (str): Path or filename of data.
            random_state (int, np.random.RandomState or None): Seed or random number generator for the shuffle. None draws from NumPy's global random state (Default=None).
            max_points (int or None): The number of first points of the shuffled stream to keep. None keeps all of them (Default=None).
            scale (bool): Whether to scale each feature to the range from 0 to 1, with the minimum and the maximum of the points that it keeps. A constant feature becomes 0 (Default=False).
            download (bool): Whether to download the dataset if not present locally (Default=True).

        Returns:
            tuple[NDArray[np.float64], NDArray[np.int64]]: The feature vectors of shape (num_points, num_features) and the labels of shape (num_points,), in stream order.
        """
        X, y = self.get_data(data_file, download=download)
        indices = list(range(len(X)))
        check_random_state(random_state).shuffle(indices)
        if max_points is not None:
            indices = indices[:max_points]
        X, y = X[indices], y[indices]
        if scale and len(X) > 0:
            mins, maxes = X.min(axis=0), X.max(axis=0)
            X = (X - mins) / np.where(maxes > mins, maxes - mins, 1.0)

        return X, y
