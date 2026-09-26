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
