"""The dimensionality sweep of benchmarks/benchmark.py."""

import importlib.util
import sys
from pathlib import Path


def load_benchmark():
    """The benchmark script as a module, loaded by file path."""
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "benchmark.py"
    spec = importlib.util.spec_from_file_location("pysad_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dimension_rows_cover_the_sweep():
    benchmark = load_benchmark()
    rows = benchmark.dimension_rows(["LODA", "xStream"], [2, 5], 50, 0)
    assert [(row["detector"], row["dimension"]) for row in rows] == [
        ("LODA", 2),
        ("LODA", 5),
        ("xStream", 2),
        ("xStream", 5),
    ]
    assert all(row["points"] == 50 for row in rows)
    assert all(row["ms_per_point"] >= 0 for row in rows)


def test_every_sweep_detector_runs_at_one_dimension():
    benchmark = load_benchmark()
    rows = benchmark.dimension_rows(benchmark.DIMENSION_DETECTORS, [2], 30, 0)
    assert [row["detector"] for row in rows] == benchmark.DIMENSION_DETECTORS
    assert all(row["dimension"] == 2 for row in rows)


def test_dimension_outputs(tmp_path, monkeypatch):
    benchmark = load_benchmark()
    monkeypatch.setattr(benchmark, "OUT", tmp_path)
    rows = [
        {"detector": "LODA", "dimension": 2, "points": 50, "ms_per_point": 0.01},
        {"detector": "LODA", "dimension": 10, "points": 50, "ms_per_point": 0.02},
        {"detector": "xStream", "dimension": 2, "points": 50, "ms_per_point": 0.03},
    ]
    benchmark.write_csv(
        "dimensions.csv",
        benchmark.DIMENSION_FIELDS,
        [[row[field] for field in benchmark.DIMENSION_FIELDS] for row in rows],
    )
    benchmark.write_dimensions_svg(rows, ["LODA", "xStream"])
    csv_text = (tmp_path / "dimensions.csv").read_text()
    assert csv_text.splitlines()[0] == "detector,dimension,points,ms_per_point"
    assert "LODA,10,50,0.02" in csv_text
    svg = (tmp_path / "dimensions.svg").read_text()
    assert svg.startswith("<svg") and svg.strip().endswith("</svg>")
    assert "LODA" in svg and "xStream" in svg


def test_dimensions_command_writes_both_outputs(tmp_path, monkeypatch):
    benchmark = load_benchmark()
    monkeypatch.setattr(benchmark, "OUT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["benchmark.py", "--dimensions", "--dimension-points", "20"])
    benchmark.main()
    csv_lines = (tmp_path / "dimensions.csv").read_text().splitlines()
    assert len(csv_lines) == 1 + len(benchmark.DIMENSION_DETECTORS) * len(benchmark.DIMENSIONS)
    assert (tmp_path / "dimensions.svg").exists()
