import numpy as np
import os
import tempfile
from pysad.utils.data import Data


def test_get_data_mat(monkeypatch):
    # Simulate scipy.io.loadmat
    class DummyF:
        def __getitem__(self, key):
            if key == 'X':
                return np.ones((5, 2))
            if key == 'y':
                return np.arange(5).reshape(-1, 1)
    def dummy_loadmat(path):
        return DummyF()
    monkeypatch.setattr("scipy.io.loadmat", dummy_loadmat)
    d = Data(data_base_path=".")
    X, y = d.get_data("dummy.mat")
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
        X, y = d.get_data(fname)
        assert X.shape == (5, 2)
        assert np.allclose(y, np.arange(5))
    os.remove(f.name)


def test_get_iterator(monkeypatch):
    # Patch get_data to return fixed arrays
    d = Data()
    d.get_data = lambda data_file: (np.ones((3, 2)), np.arange(3))
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
    assert "arrhythmia.mat" in files


def test_get_iterator_with_seed(monkeypatch):
    """Test get_iterator with seed parameter to ensure np.random.seed is called."""
    d = Data()
    d.get_data = lambda data_file: (np.ones((3, 2)), np.arange(3))
    
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
    d.get_data = lambda data_file: (np.ones((3, 2)), np.arange(3))
    
    # Should work without seed
    it = d.get_iterator("dummy.txt", shuffle=False)
    items = list(it)
    assert len(items) == 3


def test_data_init_default_path():
    """Test Data class initialization with default path."""
    d = Data()
    assert d.data_base_path == "data"


def test_data_init_custom_path():
    """Test Data class initialization with custom path."""
    d = Data(data_base_path="/custom/path")
    assert d.data_base_path == "/custom/path"


def test_load_via_txt_method():
    """Test the _load_via_txt method directly."""
    # Create a temporary .txt file
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
    """Test that _get_data_files returns expected file names."""
    d = Data()
    files = d._get_data_files()
    
    # Check some specific files are present
    expected_files = [
        'arrhythmia.mat', 'cardio.mat', 'glass.mat', 
        'gisette_sampled.txt', 'pima-indians_sampled.txt'
    ]
    
    for expected_file in expected_files:
        assert expected_file in files
    
    # Check that all files have proper extensions
    for file in files:
        assert file.endswith('.mat') or file.endswith('.txt')
