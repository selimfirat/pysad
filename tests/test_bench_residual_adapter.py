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


def test_comparison_reports_reproducible_quality_and_timing():
    """Compare calibrated residuals and detectors on the same seeded stream."""
    settings = dict(points=500, window_size=20, seed=3, repeats=1, warmup=30)
    first = _BENCH_MODULE.run_comparison(**settings)
    second = _BENCH_MODULE.run_comparison(**settings)
    assert [row["method"] for row in first] == [
        "ConformalResidual",
        "LODA",
        "StreamingAutoregressiveDetector",
    ]
    for row, repeated in zip(first, second, strict=True):
        assert row["deterministic"] is True
        assert 0 <= row["auroc"] <= 1
        assert row["ms_per_point"] >= 0
        assert row["auroc"] == repeated["auroc"]
        assert row["min_score"] == repeated["min_score"]
        assert row["max_score"] == repeated["max_score"]
    assert 0 <= first[0]["min_score"] <= first[0]["max_score"] <= 1


def test_comparison_cli_loads_duck_typed_forecaster(tmp_path):
    """Load a local factory without a forecasting dependency and write comparison rows."""
    factory = tmp_path / "forecaster.py"
    factory.write_text(
        "class Forecaster:\n"
        "    def forecast(self, horizon=1):\n"
        "        return [10.0]\n"
        "    def learn_one(self, y):\n"
        "        pass\n"
        "def create():\n"
        "    return Forecaster()\n"
    )
    output = tmp_path / "comparison.csv"
    result = subprocess.run(
        [
            sys.executable,
            str(_BENCH_PATH),
            "--compare",
            "--forecaster",
            f"{factory}:create",
            "--points",
            "500",
            "--repeats",
            "1",
            "--output",
            str(output),
        ],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "ConformalResidual" in result.stdout
    assert "AUROC" in result.stdout
    with output.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert all(row["deterministic"] == "True" for row in rows)
    assert all(np.isfinite(float(row["auroc"])) for row in rows)


def test_comparison_matches_public_prequential_pipeline():
    """AUROC uses scores before updates and excludes the warm-up labels."""
    from sklearn.metrics import roc_auc_score

    from pysad.integrations import ForecastingAnomalyDetector, TSFMResidualAdapter
    from pysad.transform.probability_calibration import ConformalTSFMCalibrator

    class ConstantForecaster:
        def forecast(self, horizon=1):
            return [10.0]

        def learn_one(self, y):
            pass

    points, seed, warmup = 500, 3, 30
    rng = np.random.default_rng(seed)
    time_index = np.arange(points)
    labels = rng.random(points) < 0.02
    stream = 10 + 2 * np.sin(2 * np.pi * time_index / 50)
    stream[labels] += 5
    stream += rng.normal(0, 0.5, points)
    pipeline = (
        ForecastingAnomalyDetector(forecaster=ConstantForecaster())
        | TSFMResidualAdapter(window_size=20, min_std=1e-8)
        | ConformalTSFMCalibrator(alpha=0.1, gamma=0.01, window_size=20)
    )
    expected = np.array([pipeline.fit_score_partial(np.array([x])) for x in stream])[warmup:]
    rows = _BENCH_MODULE.run_comparison(
        forecaster_factory=ConstantForecaster,
        points=points,
        window_size=20,
        seed=seed,
        repeats=1,
        warmup=warmup,
    )
    assert rows[0]["auroc"] == roc_auc_score(labels[warmup:], expected)
    assert rows[0]["min_score"] == expected.min()
    assert rows[0]["max_score"] == expected.max()


def test_comparison_without_anomalies_reports_nan_auroc():
    """A single label class cannot define AUROC."""
    rows = _BENCH_MODULE.run_comparison(points=2, seed=3, repeats=1, warmup=0)
    assert all(np.isnan(row["auroc"]) for row in rows)
