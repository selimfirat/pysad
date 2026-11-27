"""The optional embedding-stream benchmark preset of benchmarks/bench_embedding_stream.py."""

import importlib.util
from pathlib import Path


def load_benchmark():
    """The embedding-stream benchmark script as a module, loaded by file path."""
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "bench_embedding_stream.py"
    spec = importlib.util.spec_from_file_location("pysad_bench_embedding_stream", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_embedding_stream_benchmark_runs_without_encoder():
    """The built-in preset runs with no encoder file and no new dependency."""
    benchmark = load_benchmark()
    result = benchmark.run_benchmark(
        points=200,
        seed=7,
        num_features=8,
        anomaly_fraction=0.05,
        repeats=1,
        warmup=32,
        num_anchors=5,
    )
    expected = {
        "points",
        "seed",
        "num_features",
        "anomaly_fraction",
        "repeats",
        "warmup",
        "num_anchors",
        "temperature",
        "ms_per_point",
        "deterministic",
        "auroc",
        "min_score",
        "max_score",
    }
    assert set(result) == expected
    assert result["deterministic"] is True
    assert isinstance(result["ms_per_point"], float)
    assert result["ms_per_point"] >= 0.0
    assert isinstance(result["auroc"], float)
    assert 0.0 <= result["auroc"] <= 1.0


def test_embedding_stream_benchmark_gives_the_same_scores_on_rerun():
    """Two runs on the same stream give exactly the same AUROC and timings."""
    benchmark = load_benchmark()
    kwargs = dict(
        points=150,
        seed=3,
        num_features=8,
        anomaly_fraction=0.05,
        repeats=1,
        warmup=24,
        num_anchors=4,
    )
    result_a = benchmark.run_benchmark(**kwargs)
    result_b = benchmark.run_benchmark(**kwargs)
    assert result_a["deterministic"] is True
    assert result_b["deterministic"] is True
    assert result_a["auroc"] == result_b["auroc"]
    assert result_a["min_score"] == result_b["min_score"]
    assert result_a["max_score"] == result_b["max_score"]
