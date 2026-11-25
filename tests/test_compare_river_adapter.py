"""The optional River adapter benchmark harness.

River is a test dependency, so these tests run whenever river is installed. They verify the
harness imports cleanly, reports whether river is available, and scores through the
``PySADToRiverAdapter`` adapter in prequential order.
"""

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("river")

_BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(_BENCHMARKS))  # lets the harness import the local benchmark.py module
spec = spec_from_file_location("compare_river_adapter", _BENCHMARKS / "compare_river_adapter.py")
harness = module_from_spec(spec)
sys.modules["compare_river_adapter"] = harness
spec.loader.exec_module(harness)


def test_installed_reports_river_available():
    assert harness.installed() is True


def test_libraries_include_adapter_and_river():
    libs = harness.libraries()
    assert "PySAD (adapter)" in libs
    assert "River" in libs
    assert "HalfSpaceTrees" in libs["River"]
    assert "LODA" in libs["PySAD (adapter)"]


def test_run_scores_through_the_adapter(monkeypatch):
    rng = np.random.RandomState(0)
    X = rng.rand(120, 3)
    y = np.zeros(120, dtype=int)
    y[::10] = 1

    def fake_load(dataset, seed, points=None):
        return X, y

    monkeypatch.setattr(harness.benchmark, "load", fake_load)

    row = harness._run("PySAD (adapter)", "LODA", "odds/annthyroid.mat")
    assert row["library"] == "PySAD (adapter)"
    assert row["detector"] == "LODA"
    assert row["dataset"] == "odds/annthyroid.mat"
    assert isinstance(row["ms_per_point"], float)
    assert row["ms_per_point"] >= 0.0
    assert not np.isnan(row["auroc"])
    assert 0.0 <= row["auroc"] <= 1.0
    assert not np.isnan(row["aupr"])


def test_run_scores_a_native_river_detector(monkeypatch):
    rng = np.random.RandomState(0)
    X = rng.rand(120, 3)
    y = np.zeros(120, dtype=int)
    y[::10] = 1

    def fake_load(dataset, seed, points=None):
        return X, y

    monkeypatch.setattr(harness.benchmark, "load", fake_load)

    row = harness._run("River", "HalfSpaceTrees", "odds/annthyroid.mat")
    assert row["library"] == "River"
    assert row["detector"] == "HalfSpaceTrees"
    assert not np.isnan(row["auroc"])
    assert 0.0 <= row["auroc"] <= 1.0


def test_single_class_stream_gives_nan_metrics(monkeypatch):
    rng = np.random.RandomState(0)
    X = rng.rand(80, 2)
    y = np.zeros(80, dtype=int)

    def fake_load(dataset, seed, points=None):
        return X, y

    monkeypatch.setattr(harness.benchmark, "load", fake_load)

    row = harness._run("PySAD (adapter)", "LODA", "odds/annthyroid.mat")
    assert np.isnan(row["auroc"])
    assert np.isnan(row["aupr"])


def test_main_exits_early_without_river(monkeypatch, capsys):
    monkeypatch.setattr(harness, "installed", lambda: False)
    monkeypatch.setattr(sys, "argv", ["compare_river_adapter.py"])
    assert harness.main() == 0
    captured = capsys.readouterr()
    assert "river is not installed" in captured.out
