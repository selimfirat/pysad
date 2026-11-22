"""Benchmark the RRCF traversal and CoDisp hot paths.

The benchmark compares the current NumPy traversal with the optional PyO3
kernels. It checks both paths before timing them. The benchmark measures
leaf routing and CoDisp computation only. It does not measure tree updates
or random cut generation.

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

from pysad.models import RobustRandomCutForest

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
    tree_size: int,
    seed: int,
) -> RobustRandomCutForest:
    """Create a reproducible RobustRandomCutForest model for the benchmark."""
    return RobustRandomCutForest(
        num_trees=num_trees,
        tree_size=tree_size,
        shingle_size=1,
        time_decay=0.0,
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
    num_trees: int = 50,
    tree_size: int = 256,
    num_points: int = 1000,
    repeats: int = 10,
    seed: int = 42,
    extension: Path | None = None,
) -> dict[str, float]:
    """Run the RRCF traversal and CoDisp benchmark and return the timing results.

    Args:
        num_features: The number of features in each point.
        num_trees: The number of trees in the model.
        tree_size: The maximum number of instances per tree.
        num_points: The number of points in each timing run.
        repeats: The number of timing runs.
        seed: The random seed for the model and points.
        extension: Optional path to a compiled extension.

    Returns:
        A mapping with the NumPy time, Rust time, and speedup in milliseconds
        and times.

    Raises:
        RuntimeError: If the Rust and NumPy results differ.
    """
    rust_module = load_rust_kernels(extension)
    rust_route = rust_module.rrcf_route_to_leaf
    rust_codisp = rust_module.rrcf_codisp_sum

    model = _create_model(num_features, num_trees, tree_size, seed)
    rng = np.random.default_rng(seed)

    # Fit some data first to build trees
    fit_points = rng.random((tree_size * 2, num_features), dtype=np.float64)
    for x in fit_points:
        model.fit_partial(x)

    # Generate test points
    test_points = rng.random((num_points, num_features), dtype=np.float64)

    # Get arena arrays for Rust kernels
    arena = model._arena
    roots = arena.roots.astype(np.int32, copy=False)
    dim = arena.dim.astype(np.int32, copy=False)
    cut = arena.cut.astype(np.float64, copy=False)
    left = arena.left.astype(np.int32, copy=False)
    right = arena.right.astype(np.int32, copy=False)
    parent = arena.parent.astype(np.int32, copy=False)
    sibling = arena.sibling.astype(np.int32, copy=False)
    count = arena.count.astype(np.int32, copy=False)

    def run_numpy_route_and_codisp(point: np.ndarray) -> float:
        """NumPy implementation of route_to_leaf + codisp_sum."""
        # Route to leaf (similar to score_partial logic)
        non_empty_roots = roots[roots >= 0]
        if len(non_empty_roots) == 0:
            return 0.0

        # Find leaf for each tree
        cand_leaves = non_empty_roots.copy()
        for k in range(len(cand_leaves)):
            s = int(cand_leaves[k])
            while (d := int(dim[s])) >= 0:
                s = int(left[s]) if point[d] <= cut[s] else int(right[s])
            cand_leaves[k] = s

        # Check for duplicates
        is_dup = (arena.bbox_low[cand_leaves] == point).all(axis=1)

        final_slots = np.empty(len(non_empty_roots), dtype=np.int32)
        init_counts = np.empty(len(non_empty_roots), dtype=np.int32)
        n_done = 0

        if is_dup.any():
            dup_slots = cand_leaves[is_dup]
            nd = len(dup_slots)
            final_slots[:nd] = dup_slots
            init_counts[:nd] = 0
            n_done = nd
            non_dup_leaves = cand_leaves[~is_dup]
        else:
            non_dup_leaves = cand_leaves

        if len(non_dup_leaves) > 0:
            final_slots[n_done : n_done + len(non_dup_leaves)] = non_dup_leaves
            init_counts[n_done : n_done + len(non_dup_leaves)] = count[non_dup_leaves]
            n_done += len(non_dup_leaves)

        # Compute CoDisp sum
        total = 0.0
        for i in range(n_done):
            leaf = final_slots[i]
            if leaf < 0:
                continue
            init_count = float(init_counts[i])
            best = init_count
            curr = leaf
            while (p := int(parent[curr])) >= 0:
                sib = int(sibling[curr])
                if sib >= 0:
                    ratio = float(count[sib]) / (float(count[curr]) + 1.0)
                    if ratio > best:
                        best = ratio
                curr = p
            total += best

        return total / num_trees

    def run_rust_route_and_codisp(point: np.ndarray) -> float:
        """Rust implementation of route_to_leaf + codisp_sum."""
        leaves = np.asarray(rust_route(point, roots, dim, cut, left, right))
        # Filter to non-empty trees
        non_empty_mask = roots >= 0
        leaves = leaves[non_empty_mask]
        if len(leaves) == 0:
            return 0.0

        # Check for duplicates
        is_dup = (arena.bbox_low[leaves] == point).all(axis=1)

        final_slots = np.empty(len(leaves), dtype=np.int32)
        init_counts = np.empty(len(leaves), dtype=np.int32)
        n_done = 0

        if is_dup.any():
            dup_slots = leaves[is_dup]
            nd = len(dup_slots)
            final_slots[:nd] = dup_slots
            init_counts[:nd] = 0
            n_done = nd
            leaves = leaves[~is_dup]

        # For non-duplicates, we need to route further using existing cuts
        # The Rust route_to_leaf already routes to leaf using existing cuts
        # So leaves already contains the final leaf nodes
        if len(leaves) > 0:
            final_slots[n_done : n_done + len(leaves)] = leaves
            init_counts[n_done : n_done + len(leaves)] = count[leaves]
            n_done += len(leaves)

        total = rust_codisp(
            final_slots[:n_done],
            init_counts[:n_done],
            parent,
            sibling,
            count,
        )
        return total / num_trees

    # Verify results match
    for point in test_points[:10]:
        numpy_score = run_numpy_route_and_codisp(point)
        rust_score = run_rust_route_and_codisp(point)
        if not np.isclose(numpy_score, rust_score, rtol=1e-10, atol=1e-12):
            raise RuntimeError(
                f"Rust and NumPy scores differ: numpy={numpy_score}, rust={rust_score}"
            )

    def run_numpy() -> None:
        for point in test_points:
            run_numpy_route_and_codisp(point)

    def run_rust() -> None:
        for point in test_points:
            run_rust_route_and_codisp(point)

    # Warmup
    for point in test_points[:10]:
        run_numpy_route_and_codisp(point)
        run_rust_route_and_codisp(point)

    numpy_ms = _time_calls(run_numpy, repeats, num_points)
    rust_ms = _time_calls(run_rust, repeats, num_points)

    return {
        "numpy_ms_per_point": numpy_ms,
        "rust_ms_per_point": rust_ms,
        "speedup": numpy_ms / rust_ms if rust_ms > 0 else float("inf"),
    }


def _parse_args() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=int, default=10, help="Number of features")
    parser.add_argument("--trees", type=int, default=50, help="Number of trees")
    parser.add_argument(
        "--tree-size", type=int, default=256, help="Tree size (max instances per tree)"
    )
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
            tree_size=args.tree_size,
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
    except RuntimeError as error:
        print(f"Benchmark verification failed: {error}", file=sys.stderr)
        return 1

    print(f"Features: {args.features}")
    print(f"Trees: {args.trees}")
    print(f"Tree size: {args.tree_size}")
    print(f"Points per run: {args.points}")
    print(f"Timing runs: {args.repeats}")
    print(f"NumPy traversal + CoDisp: {results['numpy_ms_per_point']:.6f} ms/point")
    print(f"Rust traversal + CoDisp: {results['rust_ms_per_point']:.6f} ms/point")
    print(f"Speedup: {results['speedup']:.2f}x")
    print("Results match: yes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
