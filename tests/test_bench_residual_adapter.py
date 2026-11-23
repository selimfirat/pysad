"""The deterministic benchmark of the residual adapter with a duck-typed forecaster."""

import csv
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_BENCH_PATH = _ROOT / "benchmarks" / "bench_residual_adapter.py"
_SPEC = importlib.util.spec_from_file_location("bench_residual_adapter", _BENCH_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
_BENCH_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BENCH_MODULE)
run_benchmark = _BENCH_MODULE.run_benchmark


def test_run_benchmark_returns_deterministic_result():
    """run_benchmark returns the expected keys, a finite timing, and a deterministic flag."""
    result = run_benchmark(points=200, window_size=20, seed=3, repeats=2, warmup=20)
    assert set(result) == {
        "points",
        "window_size",
        "seed",
        "repeats",
        "warmup",
        "ms_per_point",
        "deterministic",
        "min_score",
        "max_score",
    }
    assert result["points"] == 200
    assert result["window_size"] == 20
    assert result["seed"] == 3
    assert result["repeats"] == 2
    assert result["warmup"] == 20
    assert result["deterministic"] is True
    assert result["ms_per_point"] >= 0
    assert np.isfinite(result["min_score"])
    assert np.isfinite(result["max_score"])


def test_run_benchmark_is_deterministic_across_seeds():
    """Two runs with the same settings give the same timing structure and determinism flag."""
    first = run_benchmark(points=300, window_size=30, seed=11, repeats=1, warmup=30)
    second = run_benchmark(points=300, window_size=30, seed=11, repeats=1, warmup=30)
    assert first["deterministic"] is True
    assert second["deterministic"] is True
    assert first["min_score"] == second["min_score"]
    assert first["max_score"] == second["max_score"]


def test_residual_adapter_benchmark_runs_and_is_deterministic(tmp_path):
    """The benchmark script exits zero, reports a deterministic run, and writes a CSV row."""
    output = tmp_path / "results.csv"
    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/bench_residual_adapter.py",
            "--points",
            "500",
            "--window-size",
            "20",
            "--seed",
            "3",
            "--repeats",
            "2",
            "--output",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Deterministic: True" in result.stdout
    with output.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    row = rows[0]
    assert row["deterministic"] == "True"
    assert float(row["ms_per_point"]) >= 0
    assert int(row["points"]) == 500
    assert int(row["window_size"]) == 20
    assert int(row["seed"]) == 3
