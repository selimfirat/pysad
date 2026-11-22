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


def run_benchmark(
    num_features: int = 10,
    num_trees: int = 25,
    max_depth: int = 15,
    num_points: int = 1000,
    repeats: int = 10,
    seed: int = 42,
    extension: Path | None = None,
) -> dict[str, float]:
    """Run the traversal benchmark and return the timing results.

    Args:
        num_features: The number of features in each point.
        num_trees: The number of trees in the model.
        max_depth: The maximum depth of each tree.
        num_points: The number of points in each timing run.
        repeats: The number of timing runs.
        seed: The random seed for the model and points.
        extension: Optional path to a compiled extension.

    Returns:
        A mapping with the NumPy time, Rust time, and speedup in milliseconds
        and times.

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

    numpy_path = model._path(points[0]).copy()
    rust_result = np.asarray(rust_path(points[0], split_atts, split_values, max_depth))
    if not np.array_equal(numpy_path, rust_result):
        raise RuntimeError("The Rust and NumPy traversal paths differ.")

    def run_numpy() -> None:
        for point in points:
            model._path(point)

    def run_rust() -> None:
        for point in points:
            rust_path(point, split_atts, split_values, max_depth)

    for point in points[:10]:
        model._path(point)
        rust_path(point, split_atts, split_values, max_depth)

    numpy_ms = _time_calls(run_numpy, repeats, num_points)
    rust_ms = _time_calls(run_rust, repeats, num_points)
    return {
        "numpy_ms_per_point": numpy_ms,
        "rust_ms_per_point": rust_ms,
        "speedup": numpy_ms / rust_ms,
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
    return parser.parse_args()


def main() -> int:
    """Run the benchmark command."""
    args = _parse_args()
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
            "`cargo build --manifest-path rust/Cargo.toml --release`.",
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
    print("Paths match: yes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
