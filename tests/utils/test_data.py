import gzip
import hashlib
import io
import os
import tempfile
import urllib.error

import numpy as np
import pytest

import pysad.utils.data
from pysad.utils._datasets import _LICENSES, _NASA, _NASA_LABELS_URL, _SMAP_CHECKSUMS, _SMD
from pysad.utils.data import Data


class Response(io.BytesIO):
    """A response of urlopen with the given body and headers."""

    def __init__(self, body=b"", headers=None):
        super().__init__(body)
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def test_get_data_mat(monkeypatch, tmp_path):
    # Simulate scipy.io.loadmat
    class DummyF:
        def __getitem__(self, key):
            if key == "X":
                return np.ones((5, 2))
            if key == "y":
                return np.arange(5).reshape(-1, 1)

    def dummy_loadmat(path):
        return DummyF()

    dummy_file = tmp_path / "dummy.mat"
    dummy_file.write_bytes(b"dummy")

    monkeypatch.setattr("scipy.io.loadmat", dummy_loadmat)
    d = Data(data_base_path=str(tmp_path))
    X, y = d.get_data("dummy.mat", download=False)
    assert X.shape == (5, 2)
    assert np.all(y == np.arange(5))


def test_get_data_flips_the_vertebral_labels(monkeypatch, tmp_path):
    """The ODDS file of vertebral marks the healthy patients as the anomalies. Data flips the
    labels, so the anomaly class is the unhealthy patients."""
    shipped_y = np.zeros((240, 1), dtype=np.uint8)
    shipped_y[:30] = 1  # the 30 healthy patients that ODDS marks as anomalies

    class DummyF:
        def __getitem__(self, key):
            if key == "X":
                return np.ones((240, 6))
            if key == "y":
                return shipped_y

    monkeypatch.setattr("scipy.io.loadmat", lambda path: DummyF())
    (tmp_path / "vertebral.mat").write_bytes(b"dummy")
    d = Data(data_base_path=str(tmp_path))

    X, y = d.get_data("vertebral.mat", download=False)

    assert X.shape == (240, 6)
    assert y.dtype == shipped_y.dtype
    assert int(y.sum()) == 210
    np.testing.assert_array_equal(y, (shipped_y.ravel() == 0).astype(np.uint8))


def test_get_data_txt():
    # Create a temporary .txt file
    arr = np.hstack([np.random.rand(5, 2), np.arange(5).reshape(-1, 1)])
    with tempfile.NamedTemporaryFile(mode="w+t", suffix=".txt", delete=False) as f:
        np.savetxt(f, arr, delimiter=",")
        fname = os.path.basename(f.name)
        f.close()
        d = Data(data_base_path=os.path.dirname(f.name))
        X, y = d.get_data(fname, download=False)
        assert X.shape == (5, 2)
        assert np.allclose(y, np.arange(5))
    os.remove(f.name)


def test_get_iterator(monkeypatch):
    # Patch get_data to return fixed arrays
    d = Data()
    d.get_data = lambda data_file, download=True: (np.ones((3, 2)), np.arange(3))
    it = d.get_iterator("dummy.txt", shuffle=False)
    items = list(it)
    assert len(items) == 3
    for x, y in items:
        assert np.all(x == 1)
        assert y in [0, 1, 2]


def test_get_iterator_random_state_shuffles_without_the_global_state():
    """A random_state gives the same order each time, and the global random state stays as it was."""
    d = Data()
    d.get_data = lambda data_file, download=True: (np.arange(20.0).reshape(10, 2), np.arange(10))
    np.random.seed(0)
    before = np.random.get_state()[1].copy()

    first = [label for _, label in d.get_iterator("dummy.txt", random_state=42)]
    second = [label for _, label in d.get_iterator("dummy.txt", random_state=42)]

    assert first == second
    assert sorted(first) == list(range(10))
    np.testing.assert_array_equal(np.random.get_state()[1], before)


def test_get_stream_keeps_shuffles_and_scales():
    """get_stream is get_iterator as arrays, cut to max_points and scaled to [0, 1]."""
    d = Data()
    X = np.array([[0.0, 5.0], [2.0, 5.0], [4.0, 5.0], [6.0, 5.0], [8.0, 5.0]])
    d.get_data = lambda data_file, download=True: (X, np.arange(5))

    Xs, ys = d.get_stream("dummy.txt", random_state=1, max_points=3, scale=True)
    order = [label for _, label in d.get_iterator("dummy.txt", random_state=1)][:3]

    np.testing.assert_array_equal(ys, order)
    kept = X[order, 0]
    np.testing.assert_array_equal(Xs[:, 0], (kept - kept.min()) / (kept.max() - kept.min()))
    np.testing.assert_array_equal(Xs[:, 1], 0.0)  # a constant feature becomes 0


def test_data_init_default_path(monkeypatch):
    """Test Data class initialization with default cache path and PYSAD_DATA_DIR env var."""
    monkeypatch.delenv("PYSAD_DATA_DIR", raising=False)
    d = Data()
    assert d.data_base_path == os.path.expanduser(os.path.join("~", ".cache", "pysad"))

    monkeypatch.setenv("PYSAD_DATA_DIR", "/custom/cache/dir")
    d_env = Data()
    assert d_env.data_base_path == "/custom/cache/dir"


def test_data_init_custom_path():
    """Test Data class initialization with custom path."""
    d = Data(data_base_path="/custom/path")
    assert d.data_base_path == "/custom/path"


def test_datasets_content():
    """Test that datasets returns a list with the expected ODDS dataset names."""
    files = Data.datasets()

    assert isinstance(files, list)

    expected_files = [
        "arrhythmia.mat",
        "cardio.mat",
        "glass.mat",
        "ionosphere.mat",
        "shuttle.mat",
        "mnist.mat",
        "vowels.mat",
    ]

    for expected_file in expected_files:
        assert expected_file in files

    for file in files:
        assert file.endswith(".mat")


def test_get_data_download_and_cache(tmp_path, monkeypatch):
    """Test downloading a dataset on demand and subsequent cache hit."""
    monkeypatch.setenv("PYSAD_DATA_DIR", str(tmp_path))

    class DummyF:
        def __getitem__(self, key):
            if key == "X":
                return np.ones((10, 4))
            if key == "y":
                return np.zeros((10, 1))

    monkeypatch.setattr("scipy.io.loadmat", lambda path: DummyF())

    download_count = 0

    def dummy_urlopen(req, timeout):
        nonlocal download_count
        download_count += 1
        assert "cardio.mat" in req.full_url
        return Response(b"MATLAB dummy content")

    monkeypatch.setattr("urllib.request.urlopen", dummy_urlopen)
    monkeypatch.setitem(
        pysad.utils.data._CHECKSUMS,
        "cardio.mat",
        hashlib.sha256(b"MATLAB dummy content").hexdigest(),
    )

    data = Data()
    target_file = tmp_path / "cardio.mat"
    assert not target_file.exists()

    # First call: downloads
    X, y = data.get_data("cardio.mat")
    assert download_count == 1
    assert target_file.exists()
    assert target_file.read_bytes() == b"MATLAB dummy content"
    assert X.shape == (10, 4)

    # Second call: reads from cache without re-downloading
    X2, y2 = data.get_data("cardio.mat")
    assert download_count == 1
    assert X2.shape == (10, 4)


def test_get_data_download_false_raises(tmp_path, monkeypatch):
    """Test that download=False raises FileNotFoundError when the file is absent."""
    monkeypatch.setenv("PYSAD_DATA_DIR", str(tmp_path))
    data = Data()

    with pytest.raises(FileNotFoundError, match="download=False"):
        data.get_data("cardio.mat", download=False)


def test_get_data_unknown_dataset_raises(tmp_path, monkeypatch):
    """Test that requesting an unknown dataset raises FileNotFoundError."""
    monkeypatch.setenv("PYSAD_DATA_DIR", str(tmp_path))
    data = Data()

    with pytest.raises(FileNotFoundError, match="not a known remote dataset"):
        data.get_data("non_existent_dataset.mat")


def test_download_atomic_cleanup_on_error(tmp_path, monkeypatch):
    """Test that failed downloads clean up any temporary files."""
    monkeypatch.setenv("PYSAD_DATA_DIR", str(tmp_path))

    def failing_urlopen(req, timeout):
        raise urllib.error.URLError("Network connection failed")

    monkeypatch.setattr("urllib.request.urlopen", failing_urlopen)
    monkeypatch.setattr(pysad.utils.data, "_RETRY_SECONDS", 0.0)

    data = Data()
    with pytest.raises(urllib.error.URLError):
        data.get_data("cardio.mat")

    # Verify no target file or tmp files left
    files = list(tmp_path.iterdir())
    assert len(files) == 0


def test_download_tries_again_after_a_server_error(tmp_path, monkeypatch):
    calls = []

    def urlopen(req, timeout):
        calls.append(req.full_url)
        if len(calls) == 1:
            raise urllib.error.HTTPError(req.full_url, 504, "Gateway Time-out", {}, None)
        return Response(b"content")

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    monkeypatch.setattr(pysad.utils.data, "_RETRY_SECONDS", 0.0)
    monkeypatch.setitem(
        pysad.utils.data._CHECKSUMS, "cardio.mat", hashlib.sha256(b"content").hexdigest()
    )

    Data(data_base_path=str(tmp_path))._download("cardio.mat", str(tmp_path / "cardio.mat"))

    assert len(calls) == 2
    assert (tmp_path / "cardio.mat").read_bytes() == b"content"


def test_download_tries_again_after_a_read_timeout(tmp_path, monkeypatch):
    calls = []

    class StalledResponse(Response):
        def read(self, *args):
            raise TimeoutError("The read operation timed out")

    def urlopen(req, timeout):
        calls.append(timeout)
        return StalledResponse() if len(calls) == 1 else Response(b"content")

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    monkeypatch.setattr(pysad.utils.data, "_RETRY_SECONDS", 0.0)
    monkeypatch.setitem(
        pysad.utils.data._CHECKSUMS, "cardio.mat", hashlib.sha256(b"content").hexdigest()
    )

    Data(data_base_path=str(tmp_path))._download("cardio.mat", str(tmp_path / "cardio.mat"))

    assert len(calls) == 2
    assert all(timeout is not None for timeout in calls)
    assert (tmp_path / "cardio.mat").read_bytes() == b"content"


def test_download_decompresses_a_gzip_response(tmp_path, monkeypatch):
    headers = []

    def urlopen(req, timeout):
        headers.append(req.get_header("Accept-encoding"))
        return Response(gzip.compress(b"content"), headers={"Content-Encoding": "gzip"})

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    monkeypatch.setitem(
        pysad.utils.data._CHECKSUMS, "cardio.mat", hashlib.sha256(b"content").hexdigest()
    )

    Data(data_base_path=str(tmp_path))._download("cardio.mat", str(tmp_path / "cardio.mat"))

    assert headers == ["gzip"]
    assert (tmp_path / "cardio.mat").read_bytes() == b"content"


def test_download_does_not_try_again_after_a_client_error(tmp_path, monkeypatch):
    calls = []

    def urlopen(req, timeout):
        calls.append(req.full_url)
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    monkeypatch.setattr(pysad.utils.data, "_RETRY_SECONDS", 0.0)

    with pytest.raises(urllib.error.HTTPError):
        Data(data_base_path=str(tmp_path))._download("cardio.mat", str(tmp_path / "cardio.mat"))

    assert len(calls) == 1
    assert list(tmp_path.iterdir()) == []


def test_download_with_a_wrong_checksum_raises_and_leaves_no_file(tmp_path, monkeypatch):
    """A download that does not match the checksum of the dataset is not used or kept."""

    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout: Response(b"changed content"))
    data = Data(data_base_path=str(tmp_path))

    with pytest.raises(ValueError, match="SHA-256"):
        data.get_data("cardio.mat")
    assert list(tmp_path.iterdir()) == []


def test_describe_gives_the_facts_of_a_dataset():
    d = Data()
    d.get_data = lambda data_file, download=True: (
        np.zeros((8, 3)),
        np.array([0, 0, 1, 0, 0, 0, 1, 0]),
    )

    assert d.describe("dummy.txt") == {"points": 8, "features": 3, "anomaly_rate": 0.25}


def test_readme_counts_the_datasets_of_data():
    """The docs take the number of datasets from Data.datasets(), and the README must match it."""
    import pathlib
    import re

    readme = (pathlib.Path(__file__).parents[2] / "README.md").read_text()
    counts = {int(n) for n in re.findall(r"(\d+) labeled", readme)}
    counts |= {int(n) for n in re.findall(r"on (\d+) datasets", readme)}

    assert counts == {len(Data.datasets())}


def test_nab_datasets_are_listed_by_source():
    names = Data.datasets("nab")

    assert len(names) == 58
    assert "nab/realKnownCause/nyc_taxi.csv" in names
    assert all(name.startswith("nab/") and name.endswith(".csv") for name in names)
    assert len(Data.datasets()) == 17
    with pytest.raises(ValueError, match="source"):
        Data.datasets("yahoo")


def test_nab_real_world_datasets_are_listed_by_source():
    """The nab-real-world source lists the 47 real-world NAB series, without the 11 artificial ones."""
    names = Data.datasets("nab-real-world")

    assert len(names) == 47
    assert "nab/realKnownCause/nyc_taxi.csv" in names
    assert all(name.startswith("nab/") and name.endswith(".csv") for name in names)
    assert all(
        "artificialNoAnomaly" not in name and "artificialWithAnomaly" not in name for name in names
    )
    assert set(names) < set(Data.datasets("nab"))
    assert Data.metadata("nab/realKnownCause/nyc_taxi.csv")["license"] == _LICENSES["nab"]


def test_nab_series_are_labeled_by_their_anomaly_windows(tmp_path):
    """A point is an anomaly when its timestamp is in a window of combined_windows.json, ends included."""
    import json

    series = tmp_path / "nab" / "realKnownCause" / "nyc_taxi.csv"
    series.parent.mkdir(parents=True)
    series.write_text(
        "timestamp,value\n"
        "2014-07-01 00:00:00,1\n"
        "2014-07-01 00:30:00,2\n"
        "2014-07-01 01:00:00,30\n"
        "2014-07-01 01:30:00,4\n"
        "2014-07-01 02:00:00,5\n"
    )
    windows = {
        "realKnownCause/nyc_taxi.csv": [
            ["2014-07-01 00:30:00.000000", "2014-07-01 01:00:00.000000"]
        ]
    }
    (tmp_path / "nab" / "combined_windows.json").write_text(json.dumps(windows))

    X, y = Data(data_base_path=str(tmp_path)).get_data(
        "nab/realKnownCause/nyc_taxi.csv", download=False
    )

    np.testing.assert_array_equal(X[:, 0], [1, 2, 30, 4, 5])
    np.testing.assert_array_equal(y, [0, 1, 1, 0, 0])

    X, _ = Data(data_base_path=str(tmp_path)).get_data(
        "nab/realKnownCause/nyc_taxi.csv", download=False, timestamps=True
    )

    # 2014-07-01 00:00:00 UTC is 1404172800 seconds after the Unix epoch.
    np.testing.assert_array_equal(X[:, 0], 1404172800 + 1800 * np.arange(5))
    np.testing.assert_array_equal(X[:, 1], [1, 2, 30, 4, 5])


def test_only_nab_series_have_timestamps():
    with pytest.raises(ValueError, match="NAB"):
        Data().get_data("cardio.mat", timestamps=True)


def test_ucr_datasets_are_listed_by_source():
    names = Data.datasets("ucr")

    assert len(names) == 250
    assert names[0] == "ucr/001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.txt"
    assert all(name.startswith("ucr/") and name.endswith(".txt") for name in names)


def test_ucr_series_come_from_the_archive_and_are_labeled_by_their_names(tmp_path, monkeypatch):
    """The last two numbers of a name are the first and the last anomaly point, counted from 1."""
    import zipfile

    name = "001_UCR_Anomaly_test_2_3_4.txt"
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as z:
        # Some files have one value on each line, and others have all the values on one line.
        z.writestr(pysad.utils.data._UCR_FOLDER + name, "1\n2\n3\n4\n5\n6\n")
    content = archive.getvalue()
    downloads = []

    def urlopen(req, timeout):
        downloads.append(req.full_url)
        return Response(content)

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    monkeypatch.setattr(pysad.utils.data, "_UCR_CHECKSUM", hashlib.sha256(content).hexdigest())
    monkeypatch.setattr(pysad.utils.data, "_UCR_SERIES", [name])
    data = Data(data_base_path=str(tmp_path))

    X, y = data.get_data("ucr/" + name)
    data.get_data("ucr/" + name)

    np.testing.assert_array_equal(X[:, 0], [1, 2, 3, 4, 5, 6])
    np.testing.assert_array_equal(y, [0, 0, 1, 1, 0, 0])
    assert downloads == [pysad.utils.data._UCR]  # the second call reads the extracted file


def test_tsb_drift_series_come_from_the_archive_and_are_labeled_by_their_names(
    tmp_path, monkeypatch
):
    """TSB-drift series are extracted from the TSB-AD-M archive, Label column last."""
    import zipfile

    name = "009_TEST_id_1_Sensor_tr_3_1st_7.csv"
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("TSB-AD-M/" + name, "a,b,Label\n1.0,2.0,0\n3.0,4.0,1\n5.0,6.0,0\n")
    content = archive.getvalue()
    downloads = []

    def urlopen(req, timeout):
        downloads.append(req.full_url)
        return Response(content)

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    monkeypatch.setattr(
        pysad.utils.data, "_TSB_DRIFT_ARCHIVE_CHECKSUM", hashlib.sha256(content).hexdigest()
    )
    monkeypatch.setattr(pysad.utils.data, "_TSB_DRIFT_SERIES", (name,))
    data = Data(data_base_path=str(tmp_path))

    X, y = data.get_data("tsb-drift/" + name)
    data.get_data("tsb-drift/" + name)

    np.testing.assert_array_equal(X, [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    np.testing.assert_array_equal(y, [0, 1, 0])
    assert downloads == [pysad.utils.data._TSB_DRIFT_ARCHIVE]


def test_drift_streams_are_listed_by_source():
    assert Data.datasets("kddcup99") == ["kddcup99/http.csv"]
    assert len(Data.datasets("smd")) == 28
    assert "smd/machine-1-1.txt" in Data.datasets("smd")
    assert len(Data.datasets("smap")) == 54
    assert "smap/P-1.npy" in Data.datasets("smap")
    assert len(Data.datasets("msl")) == 27
    assert "msl/M-1.npy" in Data.datasets("msl")
    assert len(Data.datasets("insects")) == 4
    assert "insects/abrupt.csv" in Data.datasets("insects")
    assert len(Data.datasets("tsb-drift")) == 75
    assert "tsb-drift/009_MSL_id_8_Sensor_tr_714_1st_1390.csv" in Data.datasets("tsb-drift")


def test_metadata_gives_the_source_license_and_checksum_of_a_dataset_file():
    assert Data.metadata("smap/P-1.npy") == {
        "source": _NASA + "P-1.npy",
        "license": _LICENSES["smap"],
        "checksum": _SMAP_CHECKSUMS["P-1.npy"],
    }
    assert Data.metadata("msl/M-1.npy")["source"] == _NASA + "M-1.npy"
    assert Data.metadata("msl/M-1.npy")["license"] == _LICENSES["msl"]
    assert Data.metadata("telemanom/labeled_anomalies.csv") == {
        "source": _NASA_LABELS_URL,
        "license": _LICENSES["telemanom"],
        "checksum": pysad.utils.data._NASA_LABELS_CHECKSUM,
    }
    assert Data.metadata("nab/realKnownCause/nyc_taxi.csv")["license"] == _LICENSES["nab"]
    assert Data.metadata("nab/combined_windows.json")["license"] == _LICENSES["nab"]
    assert Data.metadata("cardio.mat") == {
        "source": pysad.utils.data._MIRROR + "cardio.mat",
        "license": _LICENSES["odds"],
        "checksum": pysad.utils.data._CHECKSUMS["cardio.mat"],
    }
    assert Data.metadata("kddcup99/http.csv")["license"] == _LICENSES["kddcup99"]
    ucr = "ucr/001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.txt"
    assert Data.metadata(ucr) == {
        "source": pysad.utils.data._UCR,
        "license": _LICENSES["ucr"],
        "checksum": pysad.utils.data._UCR_CHECKSUM,
    }
    assert Data.metadata("smd/machine-1-1.txt")["license"] == _LICENSES["smd"]
    assert Data.metadata("insects/abrupt.csv")["license"] == _LICENSES["insects"]
    assert Data.metadata("creditcard/creditcard.arff")["license"] == _LICENSES["creditcard"]
    series = "tsb-drift/009_MSL_id_8_Sensor_tr_714_1st_1390.csv"
    assert Data.metadata(series)["license"] == _LICENSES["tsb-drift"]


def test_metadata_of_an_unknown_file_raises_key_error():
    with pytest.raises(KeyError):
        Data.metadata("unknown/file.csv")


def _serve(monkeypatch, files):
    """Makes each download of a URL in ``files`` return its content, with a matching checksum."""

    monkeypatch.setattr(
        "urllib.request.urlopen", lambda req, timeout: Response(files[req.full_url])
    )
    real_source = pysad.utils.data._source

    def source(name):
        url, _ = real_source(name)
        return url, hashlib.sha256(files[url]).hexdigest()

    monkeypatch.setattr(pysad.utils.data, "_source", source)


def test_kddcup99_http_keeps_the_http_connections_of_logged_in_users(tmp_path, monkeypatch):
    rows = [
        "0,tcp,http,SF,181,5450," + "0," * 5 + "1," + "0," * 29 + "normal",
        "2,tcp,http,SF,10,20," + "0," * 5 + "0," + "0," * 29 + "normal",  # not logged in
        "0,tcp,smtp,SF,1,2," + "0," * 5 + "1," + "0," * 29 + "normal",  # not http
        "5,tcp,http,SF,0,0," + "0," * 5 + "1," + "0," * 29 + "back",
    ]
    arff = ("@relation KDDCup99\n@data\n" + "\n".join(rows) + "\n").encode()
    _serve(monkeypatch, {pysad.utils.data._KDDCUP99: arff})

    X, y = Data(data_base_path=str(tmp_path)).get_data("kddcup99/http.csv")

    np.testing.assert_allclose(X, np.log(np.array([[0, 181, 5450], [5, 0, 0]]) + 0.1))
    np.testing.assert_array_equal(y, [0, 1])


def test_smd_streams_load_the_test_half_and_its_labels(tmp_path, monkeypatch):
    url = pysad.utils.data._SMD
    _serve(
        monkeypatch,
        {
            url + "test/machine-1-1.txt": b"0.1,0.2\n0.3,0.4\n0.5,0.6\n",
            url + "test_label/machine-1-1.txt": b"0\n1\n0\n",
        },
    )

    X, y = Data(data_base_path=str(tmp_path)).get_data("smd/machine-1-1.txt")

    np.testing.assert_array_equal(X, [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    np.testing.assert_array_equal(y, [0, 1, 0])


def test_smap_and_msl_channels_label_every_point_of_their_anomaly_ranges(tmp_path, monkeypatch):
    def npy(array):
        f = io.BytesIO()
        np.save(f, array)
        return f.getvalue()

    smap, msl = np.arange(16.0).reshape(8, 2), np.arange(12.0).reshape(4, 3)
    labels = (
        "chan_id,spacecraft,anomaly_sequences,class,num_values\n"
        'P-2,SMAP,"[[1, 2]]",[point],8\n'
        'P-2,SMAP,"[[5, 5], [7, 7]]","[point, point]",8\n'
        'M-1,MSL,"[[0, 1]]",[point],4\n'
    )
    _serve(
        monkeypatch,
        {
            pysad.utils.data._NASA + "P-2.npy": npy(smap),
            pysad.utils.data._NASA + "M-1.npy": npy(msl),
            pysad.utils.data._NASA_LABELS_URL: labels.encode(),
        },
    )
    data = Data(data_base_path=str(tmp_path))

    X, y = data.get_data("smap/P-2.npy")
    np.testing.assert_array_equal(X, smap)
    np.testing.assert_array_equal(y, [0, 1, 1, 0, 0, 1, 0, 1])

    X, y = data.get_data("msl/M-1.npy")
    np.testing.assert_array_equal(X, msl)
    np.testing.assert_array_equal(y, [1, 1, 0, 0])


def test_insects_streams_have_the_label_in_the_last_column(tmp_path, monkeypatch):
    drive_id = pysad.utils.data._INSECTS_FILES["abrupt.csv"][0]
    _serve(monkeypatch, {pysad.utils.data._INSECTS + drive_id: b"0.5,0.25,0\n0.75,1.0,1\n"})

    X, y = Data(data_base_path=str(tmp_path)).get_data("insects/abrupt.csv")

    np.testing.assert_array_equal(X, [[0.5, 0.25], [0.75, 1.0]])
    np.testing.assert_array_equal(y, [0, 1])


def test_creditcard_keeps_transaction_order_and_separates_class_in_shared_cache(
    tmp_path, monkeypatch
):
    from pysad.utils import Data

    monkeypatch.setenv("PYSAD_DATA_DIR", str(tmp_path))
    names = ["Time", *(f"V{i}" for i in range(1, 29)), "Amount"]
    header = "@relation creditcard\n" + "".join(f"@attribute {n} numeric\n" for n in names)
    header += "@attribute Class {'0','1'}\n@data\n"
    rows = [
        "0," + "1," * 28 + "10.5,'0'",
        "0," + "2," * 28 + "20.5,'1'",
        "1," + "3," * 28 + "30.5,'0'",
    ]
    content = (header + "\n".join(rows) + "\n").encode()
    url = "https://openml.org/data/v1/download/1673544/creditcard.arff"
    _serve(monkeypatch, {url: content})

    assert Data.datasets("creditcard") == ["creditcard/creditcard.arff"]
    with pytest.raises(FileNotFoundError, match="download=False"):
        Data().get_data("creditcard/creditcard.arff", download=False)
    X, y = Data().get_data("creditcard/creditcard.arff")

    assert X.shape == (3, 30)
    assert X.dtype == np.float64
    np.testing.assert_array_equal(X[:, 0], [0, 0, 1])
    np.testing.assert_array_equal(X[:, 1:29], [[1] * 28, [2] * 28, [3] * 28])
    np.testing.assert_array_equal(X[:, -1], [10.5, 20.5, 30.5])
    np.testing.assert_array_equal(y, [0, 1, 0])
    assert (tmp_path / "creditcard/creditcard.arff").read_bytes() == content

    def offline(req, timeout):
        pytest.fail("A cached dataset must not need the network")

    monkeypatch.setattr("urllib.request.urlopen", offline)
    cached_X, cached_y = Data().get_data("creditcard/creditcard.arff", download=False)
    np.testing.assert_array_equal(cached_X, X)
    np.testing.assert_array_equal(cached_y, y)
    stream = list(Data().get_iterator("creditcard/creditcard.arff", shuffle=False))
    np.testing.assert_array_equal([x for x, _ in stream], X)
    np.testing.assert_array_equal([label for _, label in stream], y)

    arff_path = tmp_path / "creditcard" / "creditcard.arff"
    npz_path = tmp_path / "creditcard" / "creditcard.arff.npz"
    assert npz_path.exists()
    # Replacing the cached .arff with a newer mtime invalidates the sidecar cache.
    updated_content = (header + "0," + "4," * 28 + "40.5,'1'\n").encode()
    arff_path.write_bytes(updated_content)
    newer_mtime = npz_path.stat().st_mtime + 10.0
    os.utime(arff_path, (newer_mtime, newer_mtime))
    reloaded_X, reloaded_y = Data().get_data("creditcard/creditcard.arff", download=False)
    assert reloaded_X.shape == (1, 30)
    np.testing.assert_array_equal(reloaded_X[:, -1], [40.5])
    np.testing.assert_array_equal(reloaded_y, [1])


def test_data_pickle_round_trip(tmp_path):
    """Verify Data preserves cache path and cached data retrieval across pickle round-trip."""
    import pickle

    dataset_path = tmp_path / "sample.txt"
    arr = np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 1.0]])
    np.savetxt(dataset_path, arr, delimiter=",")

    data = Data(data_base_path=str(tmp_path))
    X_orig, y_orig = data.get_data("sample.txt", download=False)

    restored = pickle.loads(pickle.dumps(data))
    assert restored.data_base_path == data.data_base_path

    X_restored, y_restored = restored.get_data("sample.txt", download=False)
    np.testing.assert_array_equal(X_restored, X_orig)
    np.testing.assert_array_equal(y_restored, y_orig)


def _validate_dataset_structure(X, y, num_features_min: int = 1) -> None:
    """Validate that a loaded dataset has the expected structure.

    Args:
        X: Feature array of shape (num_instances, num_features).
        y: Label array of shape (num_instances,).
        num_features_min: Minimum expected number of features.
    """
    assert isinstance(X, np.ndarray), "X must be a numpy array"
    assert X.ndim == 2, f"X must be 2D, got ndim={X.ndim}"
    assert X.shape[0] > 0, "X must have at least one instance"
    assert X.shape[1] >= num_features_min, (
        f"X must have at least {num_features_min} feature(s), got {X.shape[1]}"
    )
    assert isinstance(y, np.ndarray), "y must be a numpy array"
    assert y.ndim == 1, f"y must be 1D, got ndim={y.ndim}"
    assert y.shape[0] == X.shape[0], (
        f"y must have the same length as X: {y.shape[0]} vs {X.shape[0]}"
    )
    assert set(np.unique(y)).issubset({0, 1}), (
        f"y must contain only binary labels (0/1), got {set(np.unique(y))}"
    )


# A simple Response mock used across test helpers.
# (The Response class is defined at line 15 for reuse across tests.)


def _load_odds_mat(monkeypatch, data_file: str, data_base_path: str) -> tuple:
    """Load an ODDS .mat dataset with a mocked scipy loadmat."""

    class DummyF:
        def __getitem__(self, key):
            if key == "X":
                return np.ones((10, 3))
            if key == "y":
                return np.array([0, 1] * 5).reshape(-1, 1)

    monkeypatch.setattr("scipy.io.loadmat", lambda p: DummyF())
    import hashlib

    from pysad.utils._datasets import _CHECKSUMS

    monkeypatch.setitem(_CHECKSUMS, data_file, hashlib.sha256(b"dummy").hexdigest())
    d = Data(data_base_path=data_base_path)
    X, y = d.get_data(data_file, download=False)
    _validate_dataset_structure(X, y, num_features_min=3)
    return X, y


def _load_nab_series(monkeypatch, data_file: str, data_base_path: str) -> tuple:
    """Load a NAB series with mocked combined_windows.json."""
    import json
    import os

    d = Data(data_base_path=data_base_path)

    # Use the full NAB path as expected by get_data
    full_data_file = "nab/" + data_file

    # Create the NAB data file
    nab_dir = os.path.join(data_base_path, "nab")
    os.makedirs(nab_dir, exist_ok=True)
    series_dir = os.path.join(nab_dir, "realKnownCause")
    os.makedirs(series_dir, exist_ok=True)
    series_path = os.path.join(series_dir, "nyc_taxi.csv")
    with open(series_path, "w") as f:
        f.write(
            "timestamp,value\n2014-07-01 00:00:00,1\n2014-07-01 00:30:00,2\n2014-07-01 01:00:00,30\n2014-07-01 01:30:00,4\n2014-07-01 02:00:00,5\n"
        )

    # Set up combined_windows.json with key matching name[len("nab/") :]
    windows_path = os.path.join(nab_dir, "combined_windows.json")
    windows = {
        "realKnownCause/nyc_taxi.csv": [
            ["2014-07-01 00:30:00.000000", "2014-07-01 01:00:00.000000"]
        ]
    }
    with open(windows_path, "w") as f:
        json.dump(windows, f)

    X, y = d.get_data(full_data_file, download=False)
    _validate_dataset_structure(X, y)
    return X, y


def _load_ucr_series(monkeypatch, data_file: str, data_base_path: str) -> tuple:
    """Load a UCR series with a mocked archive download."""
    import hashlib
    import io
    import os
    import zipfile

    from pysad.utils.data import Data as Data_d

    d = Data_d(data_base_path=data_base_path)

    name = data_file
    # Create the UCR data file in the expected location
    ucr_dir = os.path.join(data_base_path, "ucr")
    os.makedirs(ucr_dir, exist_ok=True)
    data_path = os.path.join(ucr_dir, name)
    with open(data_path, "w") as f:
        f.write("1\n2\n3\n4\n5\n")

    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(f"ucr/{name}", "1\n2\n3\n4\n5\n")
    content = archive.getvalue()

    # Monkeypatch following the existing test pattern
    def urlopen(req, timeout):
        return type("R", (), {"read": lambda self, *a: content, "headers": {}}())()

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    monkeypatch.setattr(pysad.utils.data, "_UCR_CHECKSUM", hashlib.sha256(content).hexdigest())
    monkeypatch.setattr(pysad.utils.data, "_UCR_SERIES", [name])
    X, y = d.get_data("ucr/" + name, download=False)
    _validate_dataset_structure(X, y, num_features_min=1)
    return X, y


def _load_smd_stream(monkeypatch, data_file: str, data_base_path: str) -> tuple:
    """Load an SMD machine stream with mocked download."""
    import os

    from pysad.utils.data import Data as Data_d

    d = Data_d(data_base_path=data_base_path)

    # Create the SMD data and labels files in the expected locations
    smd_dir = os.path.join(data_base_path, "smd")
    os.makedirs(smd_dir, exist_ok=True)

    # Data file
    data_path = os.path.join(smd_dir, data_file)
    with open(data_path, "w") as f:
        f.write("0.1,0.2\n0.3,0.4\n0.5,0.6\n")

    # Labels file (matching the data file name)
    "smd/labels/" + data_file
    labels_path = os.path.join(smd_dir, "labels", data_file)
    os.makedirs(os.path.dirname(labels_path), exist_ok=True)
    with open(labels_path, "w") as f:
        f.write("0\n1\n0\n")

    # Directly set the SMD module attribute to avoid monkeypatch.setitem issues
    import pysad.utils.data as dmod

    dmod._SMD = "smd/"

    X, y = d.get_data("smd/" + data_file, download=False)
    _validate_dataset_structure(X, y, num_features_min=2)
    return X, y


def _load_smap_channel(monkeypatch, data_file: str, data_base_path: str) -> tuple:
    """Load a SMAP channel with mocked download."""
    import io
    import os

    import numpy as np

    from pysad.utils.data import Data as Data_d

    d = Data_d(data_base_path=data_base_path)

    # Create the SMAP data file
    smap_dir = os.path.join(data_base_path, "smap")
    os.makedirs(smap_dir, exist_ok=True)
    data_path = os.path.join(smap_dir, data_file)

    # Create a dummy .npy file with 25 features (1 telemetry + 24 one-hot commands)
    arr = np.ones((10, 25), dtype=np.float64)
    f = io.BytesIO()
    np.save(f, arr)
    with open(data_path, "wb") as out:
        out.write(f.getvalue())

    # Create the labels file
    labels_path = os.path.join(data_base_path, "telemanom", "labeled_anomalies.csv")
    os.makedirs(os.path.dirname(labels_path), exist_ok=True)
    with open(labels_path, "w") as f:
        f.write(
            "chan_id,spacecraft,anomaly_sequences,class,num_values\n"
            f'{data_file.removesuffix(".npy")},SMAP,"[[2, 4]]",[point],10\n'
        )

    # Monkeypatch the NASA URL base to local path
    import pysad.utils.data as dmod

    dmod._NASA = "smap/"

    X, y = d.get_data("smap/" + data_file, download=False)
    _validate_dataset_structure(X, y, num_features_min=25)
    return X, y


def _load_msl_channel(monkeypatch, data_file: str, data_base_path: str) -> tuple:
    """Load an MSL channel with mocked download."""
    import io
    import os

    import numpy as np

    from pysad.utils.data import Data as Data_d

    d = Data_d(data_base_path=data_base_path)

    # Create the MSL data file
    msl_dir = os.path.join(data_base_path, "msl")
    os.makedirs(msl_dir, exist_ok=True)
    data_path = os.path.join(msl_dir, data_file)

    # Create a dummy .npy file with 55 features (1 telemetry + 54 one-hot commands)
    arr = np.ones((10, 55), dtype=np.float64)
    f = io.BytesIO()
    np.save(f, arr)
    with open(data_path, "wb") as out:
        out.write(f.getvalue())

    # Create the labels file
    labels_path = os.path.join(data_base_path, "telemanom", "labeled_anomalies.csv")
    os.makedirs(os.path.dirname(labels_path), exist_ok=True)
    with open(labels_path, "w") as f:
        f.write(
            "chan_id,spacecraft,anomaly_sequences,class,num_values\n"
            f'{data_file.removesuffix(".npy")},MSL,"[[2, 4]]",[point],10\n'
        )

    # Monkeypatch the NASA URL base to local path
    import pysad.utils.data as dmod

    dmod._NASA = "msl/"

    X, y = d.get_data("msl/" + data_file, download=False)
    _validate_dataset_structure(X, y, num_features_min=55)
    return X, y


def _load_creditcard_arff(monkeypatch, data_base_path: str) -> tuple:
    """Load the credit-card ARFF dataset with mocked download."""
    from pysad.utils import Data

    d = Data(data_base_path=data_base_path)

    names = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    header = "@relation creditcard\n" + "".join(f"@attribute {n} numeric\n" for n in names)
    header += "@attribute Class {'0','1'}\n@data\n"
    rows = [
        "0," + "1," * 28 + "10.5,'0'",
        "0," + "2," * 28 + "20.5,'1'",
        "1," + "3," * 28 + "30.5,'0'",
    ]
    content = (header + "\n".join(rows) + "\n").encode()

    # Create the ARFF file in the expected location
    import os

    arff_dir = os.path.join(data_base_path, "creditcard")
    os.makedirs(arff_dir, exist_ok=True)
    arff_path = os.path.join(arff_dir, "creditcard.arff")
    with open(arff_path, "w") as f:
        f.write(content.decode())

    # Follow existing test pattern: set checksum via monkeypatch so it is reverted after the test
    # The _source function uses "creditcard/creditcard.arff" as key for checksum lookup
    # When download=False and file exists, checksum is checked but we can set it to match
    import hashlib

    import pysad.utils.data as dmod

    # Read back the file content to compute checksum
    with open(arff_path, "rb") as f:
        file_content = f.read()
    monkeypatch.setitem(
        dmod._CHECKSUMS, "creditcard/creditcard.arff", hashlib.sha256(file_content).hexdigest()
    )

    # Monkeypatch urlopen to avoid actual network calls
    def urlopen(req, timeout):
        return Response(content)

    monkeypatch.setattr("urllib.request.urlopen", urlopen)

    X, y = d.get_data("creditcard/creditcard.arff", download=False)
    _validate_dataset_structure(X, y, num_features_min=30)
    return X, y


_SAMPLE_DATASETS_PER_SOURCE = {
    "odds": ["arrhythmia.mat", "cardio.mat", "ionosphere.mat"],
    "nab": ["realKnownCause/nyc_taxi.csv"],
    "ucr": ["001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.txt"],
    "kddcup99": ["http.csv"],
    "smd": ["machine-1-1.txt"],
    "smap": ["P-1.npy"],
    "msl": ["M-1.npy"],
    "insects": ["abrupt.csv"],
    "creditcard": ["creditcard.arff"],
}


def test_datasets_list_by_source(monkeypatch):
    """Test that datasets() returns the expected number of datasets per source."""
    from pysad.utils.data import Data

    d = Data()
    expected_counts = {
        "odds": 17,
        "nab": 58,
        "nab-real-world": 47,
        "ucr": 250,
        "kddcup99": 1,
        "smd": 28,
        "smap": 54,
        "msl": 27,
        "insects": 4,
        "creditcard": 1,
        "tsb-drift": 75,
    }
    for source, expected in expected_counts.items():
        names = d.datasets(source)
        assert isinstance(names, list)
        assert len(names) == expected, (
            f"Expected {expected} datasets for source '{source}', got {len(names)}"
        )


@pytest.mark.parametrize("data_file", ["arrhythmia.mat", "cardio.mat", "ionosphere.mat"])
def test_dataset_structure_odds(monkeypatch, tmp_path, data_file):
    """Test that ODDS .mat datasets load correctly and have expected structure."""
    import hashlib

    # Create a dummy .mat file so get_data can find it
    from pysad.utils._datasets import _CHECKSUMS

    dummy_mat = tmp_path / data_file
    dummy_mat.write_bytes(b"MATLAB dummy content")

    monkeypatch.setitem(_CHECKSUMS, data_file, hashlib.sha256(b"MATLAB dummy content").hexdigest())

    class DummyF:
        def __getitem__(self, key):
            if key == "X":
                return np.ones((10, 3))
            if key == "y":
                return np.array([0, 1] * 5).reshape(-1, 1)

    monkeypatch.setattr("scipy.io.loadmat", lambda p: DummyF())

    d = Data(data_base_path=str(tmp_path))
    X, y = d.get_data(data_file, download=False)
    _validate_dataset_structure(X, y, num_features_min=3)


def test_dataset_structure_nab(monkeypatch, tmp_path):
    """Test that NAB series load correctly and have expected structure."""
    _load_nab_series(monkeypatch, "realKnownCause/nyc_taxi.csv", str(tmp_path))


def test_dataset_structure_ucr(monkeypatch, tmp_path):
    """Test that UCR series load correctly and have expected structure."""
    _load_ucr_series(
        monkeypatch, "001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.txt", str(tmp_path)
    )


def test_dataset_structure_smd(monkeypatch, tmp_path):
    """Test that SMD machine streams load correctly and have expected structure."""
    _load_smd_stream(monkeypatch, "machine-1-1.txt", str(tmp_path))


def test_dataset_structure_smap(monkeypatch, tmp_path):
    """Test that SMAP channels load correctly and have expected structure."""
    _load_smap_channel(monkeypatch, "P-1.npy", str(tmp_path))


def test_dataset_structure_msl(monkeypatch, tmp_path):
    """Test that MSL channels load correctly and have expected structure."""
    _load_msl_channel(monkeypatch, "M-1.npy", str(tmp_path))


def test_dataset_structure_creditcard(monkeypatch, tmp_path):
    """Test that credit-card ARFF dataset loads correctly and has expected structure."""
    _load_creditcard_arff(monkeypatch, str(tmp_path))


def test_smd_streaming_iterator_preserves_labels_and_shuffles(tmp_path, monkeypatch):
    """Test that get_iterator for SMD streams preserves labels and supports shuffling with random_state."""
    # Restore original module variables that may have been modified by earlier tests
    monkeypatch.setattr(pysad.utils.data, "_SMD", _SMD)
    monkeypatch.setattr(pysad.utils.data, "_NASA", _NASA)
    _serve(
        monkeypatch,
        {
            _SMD + "test/machine-1-1.txt": b"0.1,0.2\n0.3,0.4\n0.5,0.6\n0.7,0.8\n",
            _SMD + "test_label/machine-1-1.txt": b"0\n1\n0\n1\n",
        },
    )
    data = Data(data_base_path=str(tmp_path))

    # Test shuffle=False preserves order
    stream = list(data.get_iterator("smd/machine-1-1.txt", shuffle=False))
    X_stream = np.array([x for x, _ in stream])
    y_stream = np.array([y for _, y in stream])
    np.testing.assert_array_equal(X_stream, [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
    np.testing.assert_array_equal(y_stream, [0, 1, 0, 1])

    # Test shuffle=True with random_state gives deterministic order
    stream1 = list(data.get_iterator("smd/machine-1-1.txt", shuffle=True, random_state=42))
    stream2 = list(data.get_iterator("smd/machine-1-1.txt", shuffle=True, random_state=42))
    y1 = [y for _, y in stream1]
    y2 = [y for _, y in stream2]
    assert y1 == y2, "Same random_state must give same shuffle order"
    assert sorted(y1) == [0, 0, 1, 1], "Shuffle must preserve all labels"

    # Test different random_state gives different order (with high probability)
    stream3 = list(data.get_iterator("smd/machine-1-1.txt", shuffle=True, random_state=123))
    y3 = [y for _, y in stream3]
    # Very unlikely to be same order with different seeds for 4 items
    assert y1 != y3 or y1 == [0, 0, 1, 1], (
        "Different random_state should typically give different order"
    )


def test_smap_streaming_iterator_preserves_labels_and_shuffles(tmp_path, monkeypatch):
    """Test that get_iterator for SMAP streams preserves labels and supports shuffling with random_state."""
    # Restore original module variables that may have been modified by earlier tests
    monkeypatch.setattr(pysad.utils.data, "_NASA", _NASA)

    def npy(array):
        f = io.BytesIO()
        np.save(f, array)
        return f.getvalue()

    smap_data = np.arange(20.0).reshape(10, 2)
    labels = (
        'chan_id,spacecraft,anomaly_sequences,class,num_values\nP-1,SMAP,"[[2, 4]]",[point],10\n'
    )
    _serve(
        monkeypatch,
        {
            _NASA + "P-1.npy": npy(smap_data),
            _NASA_LABELS_URL: labels.encode(),
        },
    )
    data = Data(data_base_path=str(tmp_path))

    # Expected labels: points 2,3,4 are anomalies (0-indexed: indices 2,3,4)
    expected_y = np.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 0])

    # Test shuffle=False preserves order
    stream = list(data.get_iterator("smap/P-1.npy", shuffle=False))
    X_stream = np.array([x for x, _ in stream])
    y_stream = np.array([y for _, y in stream])
    np.testing.assert_array_equal(X_stream, smap_data)
    np.testing.assert_array_equal(y_stream, expected_y)

    # Test shuffle=True with random_state gives deterministic order
    stream1 = list(data.get_iterator("smap/P-1.npy", shuffle=True, random_state=42))
    stream2 = list(data.get_iterator("smap/P-1.npy", shuffle=True, random_state=42))
    y1 = [y for _, y in stream1]
    y2 = [y for _, y in stream2]
    assert y1 == y2, "Same random_state must give same shuffle order"
    assert sorted(y1) == sorted(expected_y), "Shuffle must preserve all labels"


def test_msl_streaming_iterator_preserves_labels_and_shuffles(tmp_path, monkeypatch):
    """Test that get_iterator for MSL streams preserves labels and supports shuffling with random_state."""
    # Restore original module variables that may have been modified by earlier tests
    monkeypatch.setattr(pysad.utils.data, "_NASA", _NASA)

    def npy(array):
        f = io.BytesIO()
        np.save(f, array)
        return f.getvalue()

    msl_data = np.arange(15.0).reshape(5, 3)
    labels = 'chan_id,spacecraft,anomaly_sequences,class,num_values\nM-1,MSL,"[[0, 1]]",[point],5\n'
    _serve(
        monkeypatch,
        {
            _NASA + "M-1.npy": npy(msl_data),
            _NASA_LABELS_URL: labels.encode(),
        },
    )
    data = Data(data_base_path=str(tmp_path))

    # Expected labels: points 0,1 are anomalies
    expected_y = np.array([1, 1, 0, 0, 0])

    # Test shuffle=False preserves order
    stream = list(data.get_iterator("msl/M-1.npy", shuffle=False))
    X_stream = np.array([x for x, _ in stream])
    y_stream = np.array([y for _, y in stream])
    np.testing.assert_array_equal(X_stream, msl_data)
    np.testing.assert_array_equal(y_stream, expected_y)

    # Test shuffle=True with random_state gives deterministic order
    stream1 = list(data.get_iterator("msl/M-1.npy", shuffle=True, random_state=42))
    stream2 = list(data.get_iterator("msl/M-1.npy", shuffle=True, random_state=42))
    y1 = [y for _, y in stream1]
    y2 = [y for _, y in stream2]
    assert y1 == y2, "Same random_state must give same shuffle order"
    assert sorted(y1) == sorted(expected_y), "Shuffle must preserve all labels"


def test_smd_streaming_get_stream_shuffles_scales_and_limits(tmp_path, monkeypatch):
    """Test that get_stream for SMD shuffles, scales to [0,1], and limits to max_points."""
    # Restore original module variables that may have been modified by earlier tests
    monkeypatch.setattr(pysad.utils.data, "_SMD", _SMD)
    monkeypatch.setattr(pysad.utils.data, "_NASA", _NASA)
    _serve(
        monkeypatch,
        {
            _SMD + "test/machine-1-1.txt": b"0.0,10.0\n2.0,20.0\n4.0,30.0\n6.0,40.0\n8.0,50.0\n",
            _SMD + "test_label/machine-1-1.txt": b"0\n1\n0\n1\n0\n",
        },
    )
    data = Data(data_base_path=str(tmp_path))

    # Test max_points limits the stream
    Xs, ys = data.get_stream("smd/machine-1-1.txt", random_state=1, max_points=3, scale=False)
    assert Xs.shape == (3, 2)
    assert ys.shape == (3,)
    assert set(ys).issubset({0, 1})

    # Test scale=True scales features to [0, 1]
    Xs_scaled, ys_scaled = data.get_stream(
        "smd/machine-1-1.txt", random_state=1, max_points=3, scale=True
    )
    assert Xs_scaled.shape == (3, 2)
    # First feature: original values [0, 2, 4, 6, 8] -> shuffled subset scaled to [0, 1]
    # Second feature: original values [10, 20, 30, 40, 50] -> shuffled subset scaled to [0, 1]
    assert np.all(Xs_scaled >= 0.0) and np.all(Xs_scaled <= 1.0), (
        "Scaled features must be in [0, 1]"
    )

    # Test random_state gives deterministic shuffle
    Xs1, ys1 = data.get_stream("smd/machine-1-1.txt", random_state=42, max_points=3, scale=False)
    Xs2, ys2 = data.get_stream("smd/machine-1-1.txt", random_state=42, max_points=3, scale=False)
    np.testing.assert_array_equal(Xs1, Xs2)
    np.testing.assert_array_equal(ys1, ys2)


def test_smap_streaming_get_stream_shuffles_scales_and_limits(tmp_path, monkeypatch):
    """Test that get_stream for SMAP shuffles, scales to [0,1], and limits to max_points."""
    # Restore original module variables that may have been modified by earlier tests
    monkeypatch.setattr(pysad.utils.data, "_NASA", _NASA)

    def npy(array):
        f = io.BytesIO()
        np.save(f, array)
        return f.getvalue()

    smap_data = np.array(
        [
            [0.0, 10.0],
            [2.0, 20.0],
            [4.0, 30.0],
            [6.0, 40.0],
            [8.0, 50.0],
            [10.0, 60.0],
            [12.0, 70.0],
            [14.0, 80.0],
            [16.0, 90.0],
            [18.0, 100.0],
        ]
    )
    labels = (
        'chan_id,spacecraft,anomaly_sequences,class,num_values\nP-1,SMAP,"[[2, 4]]",[point],10\n'
    )
    _serve(
        monkeypatch,
        {
            _NASA + "P-1.npy": npy(smap_data),
            _NASA_LABELS_URL: labels.encode(),
        },
    )
    data = Data(data_base_path=str(tmp_path))

    # Test max_points limits the stream
    Xs, ys = data.get_stream("smap/P-1.npy", random_state=1, max_points=4, scale=False)
    assert Xs.shape == (4, 2)
    assert ys.shape == (4,)
    assert set(ys).issubset({0, 1})

    # Test scale=True scales features to [0, 1]
    Xs_scaled, ys_scaled = data.get_stream("smap/P-1.npy", random_state=1, max_points=4, scale=True)
    assert Xs_scaled.shape == (4, 2)
    assert np.all(Xs_scaled >= 0.0) and np.all(Xs_scaled <= 1.0), (
        "Scaled features must be in [0, 1]"
    )

    # Test random_state gives deterministic shuffle
    Xs1, ys1 = data.get_stream("smap/P-1.npy", random_state=42, max_points=4, scale=False)
    Xs2, ys2 = data.get_stream("smap/P-1.npy", random_state=42, max_points=4, scale=False)
    np.testing.assert_array_equal(Xs1, Xs2)
    np.testing.assert_array_equal(ys1, ys2)


def test_msl_streaming_get_stream_shuffles_scales_and_limits(tmp_path, monkeypatch):
    """Test that get_stream for MSL shuffles, scales to [0,1], and limits to max_points."""
    # Restore original module variables that may have been modified by earlier tests
    monkeypatch.setattr(pysad.utils.data, "_NASA", _NASA)

    def npy(array):
        f = io.BytesIO()
        np.save(f, array)
        return f.getvalue()

    msl_data = np.array(
        [
            [0.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [4.0, 30.0, 300.0],
            [6.0, 40.0, 400.0],
            [8.0, 50.0, 500.0],
        ]
    )
    labels = 'chan_id,spacecraft,anomaly_sequences,class,num_values\nM-1,MSL,"[[0, 1]]",[point],5\n'
    _serve(
        monkeypatch,
        {
            _NASA + "M-1.npy": npy(msl_data),
            _NASA_LABELS_URL: labels.encode(),
        },
    )
    data = Data(data_base_path=str(tmp_path))

    # Test max_points limits the stream
    Xs, ys = data.get_stream("msl/M-1.npy", random_state=1, max_points=3, scale=False)
    assert Xs.shape == (3, 3)
    assert ys.shape == (3,)
    assert set(ys).issubset({0, 1})

    # Test scale=True scales features to [0, 1]
    Xs_scaled, ys_scaled = data.get_stream("msl/M-1.npy", random_state=1, max_points=3, scale=True)
    assert Xs_scaled.shape == (3, 3)
    assert np.all(Xs_scaled >= 0.0) and np.all(Xs_scaled <= 1.0), (
        "Scaled features must be in [0, 1]"
    )

    # Test random_state gives deterministic shuffle
    Xs1, ys1 = data.get_stream("msl/M-1.npy", random_state=42, max_points=3, scale=False)
    Xs2, ys2 = data.get_stream("msl/M-1.npy", random_state=42, max_points=3, scale=False)
    np.testing.assert_array_equal(Xs1, Xs2)
    np.testing.assert_array_equal(ys1, ys2)


def test_smd_smap_msl_streaming_anomaly_rate_preserved(tmp_path, monkeypatch):
    """Test that streaming preserves the anomaly rate of SMD, SMAP, and MSL datasets."""
    # Restore original module variables that may have been modified by earlier tests
    monkeypatch.setattr(pysad.utils.data, "_SMD", _SMD)
    monkeypatch.setattr(pysad.utils.data, "_NASA", _NASA)

    def npy(array):
        f = io.BytesIO()
        np.save(f, array)
        return f.getvalue()

    _serve(
        monkeypatch,
        {
            _SMD + "test/machine-1-1.txt": b"0.1,0.2\n0.3,0.4\n0.5,0.6\n0.7,0.8\n",
            _SMD + "test_label/machine-1-1.txt": b"0\n1\n0\n1\n",
            _NASA + "P-1.npy": npy(np.arange(20.0).reshape(10, 2)),
            _NASA + "M-1.npy": npy(np.arange(15.0).reshape(5, 3)),
            _NASA_LABELS_URL: (
                b"chan_id,spacecraft,anomaly_sequences,class,num_values\n"
                b'P-1,SMAP,"[[2, 4]]",[point],10\n'
                b'M-1,MSL,"[[0, 1]]",[point],5\n'
            ),
        },
    )
    data = Data(data_base_path=str(tmp_path))

    # SMD: 2 anomalies out of 4 = 0.5
    X_smd, y_smd = data.get_data("smd/machine-1-1.txt")
    smd_anomaly_rate = float(np.mean(y_smd))
    stream_smd = list(data.get_iterator("smd/machine-1-1.txt", shuffle=False))
    y_stream_smd = np.array([y for _, y in stream_smd])
    assert float(np.mean(y_stream_smd)) == smd_anomaly_rate

    # SMAP: 3 anomalies out of 10 = 0.3
    X_smap, y_smap = data.get_data("smap/P-1.npy")
    smap_anomaly_rate = float(np.mean(y_smap))
    stream_smap = list(data.get_iterator("smap/P-1.npy", shuffle=False))
    y_stream_smap = np.array([y for _, y in stream_smap])
    assert float(np.mean(y_stream_smap)) == smap_anomaly_rate

    # MSL: 2 anomalies out of 5 = 0.4
    X_msl, y_msl = data.get_data("msl/M-1.npy")
    msl_anomaly_rate = float(np.mean(y_msl))
    stream_msl = list(data.get_iterator("msl/M-1.npy", shuffle=False))
    y_stream_msl = np.array([y for _, y in stream_msl])
    assert float(np.mean(y_stream_msl)) == msl_anomaly_rate

    # Also test with get_stream (shuffled but same labels)
    _, ys_smd = data.get_stream("smd/machine-1-1.txt", random_state=1, scale=False)
    _, ys_smap = data.get_stream("smap/P-1.npy", random_state=1, scale=False)
    _, ys_msl = data.get_stream("msl/M-1.npy", random_state=1, scale=False)
    assert float(np.mean(ys_smd)) == smd_anomaly_rate
    assert float(np.mean(ys_smap)) == smap_anomaly_rate
    assert float(np.mean(ys_msl)) == msl_anomaly_rate
