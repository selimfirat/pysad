import io
import os
import tempfile
import urllib.error

import numpy as np
import pytest

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


def test_get_data_files():
    d = Data()
    files = d._get_data_files()
    assert isinstance(files, list)
    assert len(files) == 17
    assert "arrhythmia.mat" in files
    assert "cardio.mat" in files
    assert all(f.endswith(".mat") for f in files)


def test_get_iterator_with_seed(monkeypatch):
    """Test get_iterator with seed parameter to ensure np.random.seed is called."""
    d = Data()
    d.get_data = lambda data_file, download=True: (np.ones((3, 2)), np.arange(3))

    # Mock np.random.seed to track if it's called
    seed_called = []
    original_seed = np.random.seed

    def mock_seed(seed):
        seed_called.append(seed)
        original_seed(seed)

    monkeypatch.setattr("numpy.random.seed", mock_seed)

    # Test with seed
    it = d.get_iterator("dummy.txt", shuffle=False, seed=42)
    list(it)  # Consume iterator
    assert 42 in seed_called


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


def test_get_data_files_content():
    """Test that _get_data_files returns expected ODDS dataset names."""
    d = Data()
    files = d._get_data_files()

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

    data = Data()
    with pytest.raises(urllib.error.URLError):
        data.get_data("cardio.mat")

    # Verify no target file or tmp files left
    files = list(tmp_path.iterdir())
    assert len(files) == 0


def test_explicit_data_base_path_wins(tmp_path):
    """Test that an explicit data_base_path is preserved and used."""
    custom_dir = tmp_path / "my_custom_dir"
    custom_dir.mkdir()
    d = Data(data_base_path=str(custom_dir))
    assert d.data_base_path == str(custom_dir)
