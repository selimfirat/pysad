import hashlib
import io
import os
import tempfile
import urllib.error

import numpy as np
import pytest

import pysad.utils.data
from pysad.utils.data import Data


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


def test_datasets():
    files = Data.datasets()
    assert isinstance(files, list)
    assert len(files) == 17
    assert "arrhythmia.mat" in files
    assert "cardio.mat" in files
    assert all(f.endswith(".mat") for f in files)


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


def test_get_iterator_without_seed(monkeypatch):
    """Test get_iterator without seed parameter."""
    d = Data()
    d.get_data = lambda data_file, download=True: (np.ones((3, 2)), np.arange(3))

    # Should work without seed
    it = d.get_iterator("dummy.txt", shuffle=False)
    items = list(it)
    assert len(items) == 3


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


def test_load_via_txt_method():
    """Test the _load_via_txt method directly."""
    arr = np.random.rand(4, 3)
    with tempfile.NamedTemporaryFile(mode="w+t", suffix=".txt", delete=False) as f:
        np.savetxt(f, arr, delimiter=",")
        f.close()

        d = Data()
        result = d._load_via_txt(f.name)
        assert result.shape == (4, 3)
        assert np.allclose(result, arr)

    os.remove(f.name)


def test_datasets_content():
    """Test that datasets returns the expected ODDS dataset names."""
    files = Data.datasets()

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

    class DummyResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def dummy_urlopen(req):
        nonlocal download_count
        download_count += 1
        assert "cardio.mat" in req.full_url
        return DummyResponse(b"MATLAB dummy content")

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

    def failing_urlopen(req):
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

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def urlopen(req):
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


def test_download_does_not_try_again_after_a_client_error(tmp_path, monkeypatch):
    calls = []

    def urlopen(req):
        calls.append(req.full_url)
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    monkeypatch.setattr(pysad.utils.data, "_RETRY_SECONDS", 0.0)

    with pytest.raises(urllib.error.HTTPError):
        Data(data_base_path=str(tmp_path))._download("cardio.mat", str(tmp_path / "cardio.mat"))

    assert len(calls) == 1
    assert list(tmp_path.iterdir()) == []


def test_explicit_data_base_path_wins(tmp_path):
    """Test that an explicit data_base_path is preserved and used."""
    custom_dir = tmp_path / "my_custom_dir"
    custom_dir.mkdir()
    d = Data(data_base_path=str(custom_dir))
    assert d.data_base_path == str(custom_dir)


def test_download_with_a_wrong_checksum_raises_and_leaves_no_file(tmp_path, monkeypatch):
    """A download that does not match the checksum of the dataset is not used or kept."""

    class DummyResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr("urllib.request.urlopen", lambda req: DummyResponse(b"changed content"))
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

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def urlopen(req):
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


def test_drift_streams_are_listed_by_source():
    assert Data.datasets("kddcup99") == ["kddcup99/http.csv"]
    assert len(Data.datasets("smd")) == 28
    assert "smd/machine-1-1.txt" in Data.datasets("smd")
    assert len(Data.datasets("insects")) == 4
    assert "insects/abrupt.csv" in Data.datasets("insects")


def _serve(monkeypatch, files):
    """Makes each download of a URL in ``files`` return its content, with a matching checksum."""

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr("urllib.request.urlopen", lambda req: Response(files[req.full_url]))
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


def test_insects_streams_have_the_label_in_the_last_column(tmp_path, monkeypatch):
    drive_id = pysad.utils.data._INSECTS_FILES["abrupt.csv"][0]
    _serve(monkeypatch, {pysad.utils.data._INSECTS + drive_id: b"0.5,0.25,0\n0.75,1.0,1\n"})

    X, y = Data(data_base_path=str(tmp_path)).get_data("insects/abrupt.csv")

    np.testing.assert_array_equal(X, [[0.5, 0.25], [0.75, 1.0]])
    np.testing.assert_array_equal(y, [0, 1])
