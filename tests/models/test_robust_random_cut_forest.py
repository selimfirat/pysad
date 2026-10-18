import importlib.abc
import sys

import numpy as np


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


def _fitted_forest(**params):
    from pysad.models import RobustRandomCutForest

    model = RobustRandomCutForest(random_state=0, **params)
    for x in np.random.RandomState(0).rand(40, 2):
        model.fit_partial(x)

    return model


def test_rrcf_trees_keep_independent_uniform_samples():
    model = _fitted_forest(num_trees=400, tree_size=10)

    samples = [frozenset(tree.leaves) for tree in model.forest]
    assert all(len(sample) == 10 for sample in samples)
    assert all(
        sample == {index for _, index in heap}
        for sample, heap in zip(samples, model.samples, strict=True)
    )
    assert len(set(samples)) > 1

    # A uniform sample of 10 out of 40 keeps each instance in a quarter of the trees.
    kept = np.array([sum(index in sample for sample in samples) for index in range(40)]) / 400
    np.testing.assert_allclose(kept, 0.25, atol=0.11)


def test_rrcf_large_time_decay_keeps_the_latest_instances():
    model = _fitted_forest(num_trees=5, tree_size=10, time_decay=50.0)

    for tree in model.forest:
        assert sorted(tree.leaves) == list(range(30, 40))
