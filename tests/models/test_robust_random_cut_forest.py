import importlib.abc
import sys


class _BlockPkgResources(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name == "pkg_resources":
            raise ModuleNotFoundError("No module named 'pkg_resources'", name=name)
        return None


def test_rrcf_imports_without_pkg_resources(monkeypatch):
    from pysad.models import RobustRandomCutForest

    for name in [name for name in sys.modules if name == "rrcf" or name.startswith("rrcf.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.delitem(sys.modules, "pkg_resources", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_BlockPkgResources()] + sys.meta_path)

    RobustRandomCutForest()

    assert "pkg_resources" not in sys.modules
    assert sys.modules["rrcf"].__version__ == "0.4.4"


def test_shingle_size_affects_point_dimension():
    """shingle_size must change the dimensionality of points inserted into the trees."""
    import numpy as np
    from pysad.models import RobustRandomCutForest

    X = np.random.RandomState(0).rand(20)[:, None]
    for shingle_size in [1, 4]:
        model = RobustRandomCutForest(num_trees=1, shingle_size=shingle_size, tree_size=32, random_state=0)
        model.fit(X)
        # After fitting, the leaves should have the expected dimension
        tree = model.forest[0]
        leaf_x = next(iter(tree.leaves.values())).x
        assert tree.ndim == shingle_size
        
