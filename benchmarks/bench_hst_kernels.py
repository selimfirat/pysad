"""Benchmark the HalfSpaceTrees traversal hot path.

The benchmark compares the current NumPy traversal with the optional PyO3
kernel. It checks both paths before timing them. The benchmark measures path
traversal only. It does not measure mass updates or anomaly scoring.

Build the Rust extension before running this script:

    cargo build --manifest-path rust/Cargo.toml --release
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import sys
import time
import tracemalloc
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from statistics import median
from types import ModuleType

import numpy as np

from pysad.models import HalfSpaceTrees

_MODULE_NAME = "pysad_kernels"
_ARTIFACT_NAMES = (
    "libpysad_kernels.dylib",
    "libpysad_kernels.so",
    "pysad_kernels.dll",
)


def _find_release_artifact() -> Path:
    """Return the release artifact from the local Cargo build."""
    release_dir = Path(__file__).resolve().parents[1] / "rust" / "target" / "release"
    for name in _ARTIFACT_NAMES:
        artifact = release_dir / name
        if artifact.exists():
            return artifact
    return release_dir / _ARTIFACT_NAMES[0]


def load_rust_kernels(extension: Path | None = None) -> ModuleType:
    """Load the PyO3 module from an installed package or a Cargo artifact.

    Args:
        extension: Optional path to a compiled extension.

    Returns:
        The imported `pysad_kernels` module.

    Raises:
        FileNotFoundError: If the requested extension does not exist.
        ImportError: If Python cannot load the extension.
    """
    if extension is None:
        try:
            return import_module(_MODULE_NAME)
        except ImportError:
            extension = _find_release_artifact()

    if not extension.exists():
        raise FileNotFoundError(f"Rust extension does not exist: {extension}")

    loader = importlib.machinery.ExtensionFileLoader(_MODULE_NAME, str(extension))
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, extension, loader=loader)
    if spec is None:
        raise ImportError(f"Python cannot load the Rust extension: {extension}")

    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def _create_model(
    num_features: int,
    num_trees: int,
    max_depth: int,
    seed: int,
) -> HalfSpaceTrees:
    """Create a reproducible HalfSpaceTrees model for the benchmark."""
    return HalfSpaceTrees(
        feature_mins=np.zeros(num_features, dtype=np.float64),
        feature_maxes=np.ones(num_features, dtype=np.float64),
        num_trees=num_trees,
        max_depth=max_depth,
        random_state=seed,
    )


def _time_calls(call: Callable[[], None], repeats: int, num_points: int) -> float:
    """Return the median time per point in milliseconds."""
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        durations.append(time.perf_counter() - start)
    return median(durations) * 1000.0 / num_points


def _measure_memory(call: Callable[[], None], repeats: int) -> tuple[float, float]:
    """Return the median current and peak memory in MiB across repeats."""
    current_samples = []
    peak_samples = []
    for _ in range(repeats):
        tracemalloc.start()
        call()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        current_samples.append(current / (1024 * 1024))
        peak_samples.append(peak / (1024 * 1024))
    return median(current_samples), median(peak_samples)


def run_benchmark(
    num_features: int = 10,
    num_trees: int = 25,
    max_depth: int = 15,
    num_points: int = 1000,
    repeats: int = 10,
    seed: int = 42,
    extension: Path | None = None,
) -> dict[str, float]:
    """Run the traversal benchmark and return the timing and memory results.

    Args:
        num_features: The number of features in each point.
        num_trees: The number of trees in the model.
        max_depth: The maximum depth of each tree.
        num_points: The number of points in each timing run.
        repeats: The number of timing runs.
        seed: The random seed for the model and points.
        extension: Optional path to a compiled extension.

    Returns:
        A mapping with the NumPy time, Rust time, speedup, and memory usage
        in milliseconds and MiB.

    Raises:
        RuntimeError: If the Rust and NumPy paths differ.
        ValueError: If the feature count exceeds the Rust index type.
    """
    if num_features > np.iinfo(np.uint16).max + 1:
        raise ValueError("The Rust kernel supports at most 65536 features.")

    rust_module = load_rust_kernels(extension)
    rust_path = rust_module.half_space_trees_path
    model = _create_model(num_features, num_trees, max_depth, seed)
    rng = np.random.default_rng(seed)
    points = rng.random((num_points, num_features), dtype=np.float64)
    split_atts = model.split_atts.astype(np.uint16, copy=False)
    split_values = model.split_values

    numpy_path = model._path_numpy(points[0]).copy()
    rust_result = np.asarray(rust_path(points[0], split_atts, split_values, max_depth))
    if not np.array_equal(numpy_path, rust_result):
        raise RuntimeError("The Rust and NumPy traversal paths differ.")

    def run_numpy() -> None:
        for point in points:
            model._path_numpy(point)

    def run_rust() -> None:
        for point in points:
            rust_path(point, split_atts, split_values, max_depth)

    for point in points[:10]:
        model._path_numpy(point)
        rust_path(point, split_atts, split_values, max_depth)

    numpy_ms = _time_calls(run_numpy, repeats, num_points)
    rust_ms = _time_calls(run_rust, repeats, num_points)

    numpy_current_mib, numpy_peak_mib = _measure_memory(run_numpy, repeats)
    rust_current_mib, rust_peak_mib = _measure_memory(run_rust, repeats)

    return {
        "numpy_ms_per_point": numpy_ms,
        "rust_ms_per_point": rust_ms,
        "speedup": numpy_ms / rust_ms,
        "numpy_current_mib": numpy_current_mib,
        "numpy_peak_mib": numpy_peak_mib,
        "rust_current_mib": rust_current_mib,
        "rust_peak_mib": rust_peak_mib,
        "memory_speedup_current": numpy_current_mib / rust_current_mib
        if rust_current_mib > 0
        else float("inf"),
        "memory_speedup_peak": numpy_peak_mib / rust_peak_mib
        if rust_peak_mib > 0
        else float("inf"),
    }


def _parse_args() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=int, default=10, help="Number of features")
    parser.add_argument("--trees", type=int, default=25, help="Number of trees")
    parser.add_argument("--depth", type=int, default=15, help="Maximum tree depth")
    parser.add_argument("--points", type=int, default=1000, help="Points per timing run")
    parser.add_argument("--repeats", type=int, default=10, help="Number of timing runs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--extension",
        type=Path,
        default=None,
        help="Path to a compiled pysad_kernels extension",
    )
    parser.add_argument(
        "--numpy-only",
        action="store_true",
        help="Run only the NumPy path (skip Rust extension check)",
    )
    return parser.parse_args()


def main() -> int:
    """Run the benchmark command."""
    args = _parse_args()

    if args.numpy_only:
        return _run_numpy_only(args)

    try:
        results = run_benchmark(
            num_features=args.features,
            num_trees=args.trees,
            max_depth=args.depth,
            num_points=args.points,
            repeats=args.repeats,
            seed=args.seed,
            extension=args.extension,
        )
    except (FileNotFoundError, ImportError) as error:
        print(f"Rust extension is not available: {error}", file=sys.stderr)
        print(
            "If the Rust extension is not available, run "
            "`cargo build --manifest-path rust/Cargo.toml --release` "
            "or use --numpy-only to test NumPy path alone.",
            file=sys.stderr,
        )
        return 2

    print(f"Features: {args.features}")
    print(f"Trees: {args.trees}")
    print(f"Depth: {args.depth}")
    print(f"Points per run: {args.points}")
    print(f"Timing runs: {args.repeats}")
    print(f"NumPy traversal: {results['numpy_ms_per_point']:.6f} ms/point")
    print(f"Rust traversal: {results['rust_ms_per_point']:.6f} ms/point")
    print(f"Speedup: {results['speedup']:.2f}x")
    print(f"NumPy memory (current): {results['numpy_current_mib']:.3f} MiB")
    print(f"NumPy memory (peak): {results['numpy_peak_mib']:.3f} MiB")
    print(f"Rust memory (current): {results['rust_current_mib']:.3f} MiB")
    print(f"Rust memory (peak): {results['rust_peak_mib']:.3f} MiB")
    print(f"Memory speedup (current): {results['memory_speedup_current']:.2f}x")
    print(f"Memory speedup (peak): {results['memory_speedup_peak']:.2f}x")
    print("Paths match: yes")
    return 0


def _run_numpy_only(args: argparse.Namespace) -> int:
    """Run benchmark with NumPy path only."""
    model = _create_model(args.features, args.trees, args.depth, args.seed)
    rng = np.random.default_rng(args.seed)
    points = rng.random((args.points, args.features), dtype=np.float64)

    def run_numpy() -> None:
        for point in points:
            model._path_numpy(point)

    for point in points[:10]:
        model._path_numpy(point)

    numpy_ms = _time_calls(run_numpy, args.repeats, args.points)
    numpy_current_mib, numpy_peak_mib = _measure_memory(run_numpy, args.repeats)

    print(f"Features: {args.features}")
    print(f"Trees: {args.trees}")
    print(f"Depth: {args.depth}")
    print(f"Points per run: {args.points}")
    print(f"Timing runs: {args.repeats}")
    print(f"NumPy traversal: {numpy_ms:.6f} ms/point")
    print(f"NumPy memory (current): {numpy_current_mib:.3f} MiB")
    print(f"NumPy memory (peak): {numpy_peak_mib:.3f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
