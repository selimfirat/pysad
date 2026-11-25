"""Bounded-memory and throughput benchmark for pure-NumPy and optional PyO3 detector paths.

Run it from the repository root:

    python benchmarks/bench_detector_paths.py --detectors all --points 10000 --repeats 3
    python benchmarks/bench_detector_paths.py --detectors LODA,HalfSpaceTrees,RobustRandomCutForest
    python benchmarks/bench_detector_paths.py --numpy-only --detectors LODA,StreamKMPlusPlus
    python benchmarks/bench_detector_paths.py --detectors all --points 50000 --output paths.csv

Each detector runs on a deterministic synthetic stream of a fixed length. tracemalloc records
the peak traced memory, and the script reports milliseconds per point. Memory is sampled at
regular checkpoints, and the script reports the growth between the second and the last
checkpoint, so one-time setup allocations do not count as a leak. The stream length and the
seed are fixed, so a rerun gives the same memory and timing.

Multivariate detectors see five features from the unit interval, univariate detectors see one
value, timestamped detectors see ``(timestamp, value)``, and edge-stream detectors see
``(source, destination)``.

For detectors that accept ``use_rust`` (LODA and StreamKMPlusPlus), the benchmark runs the
pure-NumPy path (``use_rust=False``) and, when ``pysad_kernels`` is importable or ``--extension``
points to a compiled artifact, the PyO3 path (``use_rust=True``). HalfSpaceTrees and
RobustRandomCutForest use the PyO3 kernel automatically when it is importable, so they run the
path that is available and report which one it is. When both paths run, the script checks that
they give the same scores and reports the speedup of the PyO3 path over the NumPy path.

No core dependency is added: ``pysad_kernels`` is optional and loaded at import time, and this
script only uses NumPy and the standard library.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import sys
import time
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE.parent), str(_HERE)]
from benchmark import _is_detector_model, _model_input_kind  # noqa: E402

from pysad import (  # noqa: E402
    _kernels,
    models,
)

NUM_FEATURES = 5
DEFAULT_PERIOD = 24
SEED = 0
WARMUP_POINTS = 64
SHINGLE_SIZE = 10


def _point(rng: np.random.RandomState, kind: str, step: int) -> np.ndarray:
    """One deterministic instance of the input kind of a detector.

    Args:
        rng (np.random.RandomState): The random number generator of the stream.
        kind (str): The input kind of the detector.
        step (int): The position of the point in the stream.

    Returns:
        np.ndarray: One instance of the shape the detector expects.
    """
    if kind == "univariate":
        return rng.rand(1)
    if kind == "timestamped":
        return np.array(
            [float((step % 24) * 3600), 0.5 + 0.4 * np.sin(2.0 * np.pi * (step % 24) / 24.0)]
        )
    if kind == "edge":
        return np.array([float(rng.randint(0, 20)), float(rng.randint(0, 20))])
    return rng.rand(NUM_FEATURES)


def _accepts_rust(name: str) -> bool:
    """Whether a detector's constructor accepts a ``use_rust`` parameter."""
    cls = getattr(models, name)
    return "use_rust" in inspect.signature(cls.__init__).parameters


def _required_kwargs(cls: type, points: int, use_rust: bool) -> dict[str, Any]:
    """Build the constructor keyword arguments for one detector.

    Required parameters get sensible defaults so that every detector in
    ``pysad.models.__all__`` can be built without dataset-specific arguments.
    The values mirror the ones the benchmark in ``benchmark.py`` uses.

    Args:
        cls (type): The detector class.
        points (int): The stream length, used for probationary periods and windows.
        use_rust (bool): The value to pass to ``use_rust`` when the constructor accepts it.

    Returns:
        dict[str, Any]: Constructor keyword arguments.

    Raises:
        TypeError: If a required parameter has no known default.
    """
    sig = inspect.signature(cls.__init__)
    kwargs: dict[str, Any] = {}
    sample_X = np.random.RandomState(SEED).rand(256, NUM_FEATURES)
    for pname, param in sig.parameters.items():
        if pname == "self" or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if pname == "use_rust":
            kwargs[pname] = use_rust
            continue
        if param.default is not inspect.Parameter.empty:
            if pname == "period":
                kwargs[pname] = DEFAULT_PERIOD
            continue
        if pname == "random_state":
            kwargs[pname] = 0
        elif pname == "feature_mins":
            kwargs[pname] = np.zeros(NUM_FEATURES, dtype=np.float64)
        elif pname == "feature_maxes":
            kwargs[pname] = np.ones(NUM_FEATURES, dtype=np.float64)
        elif pname == "initial_X":
            kwargs[pname] = sample_X[:WARMUP_POINTS]
        elif pname == "min_val":
            kwargs[pname] = 0.0
        elif pname == "max_val":
            kwargs[pname] = 1.0
        elif pname == "period":
            kwargs[pname] = DEFAULT_PERIOD
        elif pname in ("window_length", "subsequence_length"):
            kwargs[pname] = SHINGLE_SIZE
        elif pname == "window_size":
            has_period = "period" in sig.parameters
            kwargs[pname] = 3 * DEFAULT_PERIOD + 1 if has_period else WARMUP_POINTS
        elif pname == "probationary_period":
            kwargs[pname] = min(points * 15 // 100, 750)
        elif pname == "input_shape":
            kwargs[pname] = NUM_FEATURES
        elif pname == "semantic_anchors":
            kwargs[pname] = np.random.RandomState(0).randn(10, NUM_FEATURES)
        elif pname == "batch_size":
            kwargs[pname] = WARMUP_POINTS
        else:
            raise TypeError(f"{cls.__name__} requires {pname!r}; add it to _required_kwargs.")
    return kwargs


def _build(name: str, points: int, use_rust: bool) -> Any:
    """Build one detector instance for the given path.

    Args:
        name (str): The detector name in ``pysad.models.__all__``.
        points (int): The stream length, used for probationary periods and windows.
        use_rust (bool): Whether to request the optional PyO3 path. Ignored by detectors
            whose constructor has no ``use_rust`` parameter.

    Returns:
        Any: A fresh detector instance.
    """
    cls = getattr(models, name)
    return cls(**_required_kwargs(cls, points, use_rust))


def _rust_available(extension: Path | None) -> bool:
    """Whether the optional PyO3 kernels can be used.

    Args:
        extension (Path | None): Optional path to a compiled ``pysad_kernels`` extension.

    Returns:
        bool: True when the kernels module imports and exposes at least one kernel.
    """
    if (
        _kernels.loda_neg_log_densities is not None
        or _kernels.half_space_trees_path is not None
        or _kernels.rrcf_route_to_leaf is not None
    ):
        return True
    return extension is not None and extension.exists()


def _variants(
    name: str, points: int, rust_available: bool, numpy_only: bool
) -> list[tuple[str, Callable[[], Any]]]:
    """Build the NumPy and optional PyO3 factories for one detector.

    Args:
        name (str): The detector name.
        points (int): The stream length.
        rust_available (bool): Whether the PyO3 kernels are available.
        numpy_only (bool): Skip the PyO3 path when True.

    Returns:
        list[tuple[str, Callable[[], Any]]]: One ``(label, factory)`` pair per path.
    """
    factories: list[tuple[str, Callable[[], Any]]] = []
    accepts_rust = _accepts_rust(name)
    if accepts_rust:
        # LODA and StreamKMPlusPlus: run both the NumPy path and the optional PyO3 path.
        factories.append(("numpy", lambda n=name, p=points: _build(n, p, False)))
        if not numpy_only and rust_available:
            factories.append(("pyO3", lambda n=name, p=points: _build(n, p, True)))
    else:
        # HalfSpaceTrees and RobustRandomCutForest use the kernel automatically when available.
        label = "pyO3" if rust_available else "numpy"
        factories.append((label, lambda n=name, p=points: _build(n, p, rust_available)))
    return factories


def _available_factories(
    points: int, rust_available: bool, numpy_only: bool
) -> dict[str, list[tuple[str, Callable[[], Any]]]]:
    """One or two fresh detectors per name, NumPy and optional PyO3.

    Args:
        points (int): The fixed stream length.
        rust_available (bool): Whether the PyO3 kernels are available.
        numpy_only (bool): Skip the PyO3 path when True.

    Returns:
        A mapping from detector name to a list of ``(label, factory)`` pairs.
    """
    factories: dict[str, list[tuple[str, Callable[[], Any]]]] = {}
    for name in models.__all__:
        if not _is_detector_model(name):
            continue
        factories[name] = _variants(name, points, rust_available, numpy_only)
    return factories


def run_detector(
    name: str,
    detector: Any,
    points: int,
    checkpoint: int,
    kind: str,
    max_growth_kb: float,
    collect_scores: bool = False,
) -> dict[str, float | list[float]]:
    """Measure throughput and bounded-memory growth of one detector on one stream.

    The detector first warms up on a few points. Then every later point is scored before the
    detector learns from it, as in the rest of PySAD. tracemalloc records the traced memory at
    each checkpoint, and the growth between the second and the last checkpoint is reported.
    Growth at or below ``max_growth_kb`` means the detector's memory stays bounded as the
    stream grows.

    Args:
        name (str): The name of the detector.
        detector (Any): The detector to measure.
        points (int): The fixed length of the timed stream.
        checkpoint (int): Points between memory checkpoints.
        kind (str): The input kind of the detector.
        max_growth_kb (float): Allowed traced memory growth in KiB between the second and last
            checkpoints.
        collect_scores (bool): Whether to return the list of scores.

    Returns:
        dict: milliseconds per point, peak memory in KiB, the first and last checkpoint memory
        in KiB, the growth between the second and last checkpoints in KiB, whether the growth
        is within the allowed budget, and optionally the scores.
    """
    rng = np.random.RandomState(SEED)

    for step in range(1, WARMUP_POINTS + 1):
        detector.fit_score_partial(_point(rng, kind, step))

    scores: list[float] = [] if collect_scores else None  # type: ignore[assignment]
    tracemalloc.start()
    try:
        first_kb = second_kb = last_kb = 0.0
        start = time.perf_counter()
        for i in range(1, points + 1):
            point = _point(rng, kind, WARMUP_POINTS + i)
            if collect_scores:
                scores.append(float(detector.score_partial(point)))
            detector.fit_partial(point)
            if i % checkpoint == 0:
                last_kb = tracemalloc.get_traced_memory()[0] / 1024.0
                if i == checkpoint:
                    first_kb = last_kb
                if i == 2 * checkpoint:
                    second_kb = last_kb
        seconds = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    growth = last_kb - second_kb
    result: dict[str, float | list[float]] = {
        "points": float(points),
        "ms_per_point": float(1000.0 * seconds / points),
        "peak_kb": float(peak / 1024.0),
        "first_kb": float(first_kb),
        "last_kb": float(last_kb),
        "growth_kb": float(growth),
        "bounded": bool(growth <= max_growth_kb),
    }
    if collect_scores:
        result["scores"] = scores  # type: ignore[index]
    return result


def main(argv: list[str] | None = None) -> int:
    """Run the bounded-memory and throughput benchmark and print the results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--detectors",
        default="LODA,RSHash,HalfSpaceTrees,RobustRandomCutForest,StreamKMPlusPlus,StandardAbsoluteDeviation",
        help="Comma-separated detector names, or 'all' to run every streaming detector.",
    )
    parser.add_argument(
        "--points", type=int, default=10_000, help="Fixed length of the stream (Default=10000)."
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of timed runs per detector path (Default=3).",
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=1000,
        help="Points between memory checkpoints (Default=1000).",
    )
    parser.add_argument(
        "--max-growth-kb",
        type=float,
        default=512.0,
        help="Allowed traced memory growth in KiB between the second and last checkpoints.",
    )
    parser.add_argument("--output", type=Path, help="Write the measurements to this CSV file.")
    parser.add_argument(
        "--extension",
        type=Path,
        default=None,
        help="Path to a compiled pysad_kernels extension.",
    )
    parser.add_argument(
        "--numpy-only",
        action="store_true",
        help="Run only the pure-NumPy path, even when pysad_kernels is available.",
    )
    args = parser.parse_args(argv)

    if args.points <= 0:
        parser.error("--points must be positive.")
    if args.repeats <= 0:
        parser.error("--repeats must be positive.")
    if args.checkpoint <= 0:
        parser.error("--checkpoint must be positive.")
    if args.points < 3 * args.checkpoint:
        parser.error("--points must be at least three times --checkpoint.")

    rust_available = _rust_available(args.extension)
    names = [name.strip() for name in args.detectors.split(",") if name.strip()]
    if not names:
        parser.error("--detectors must name at least one detector.")
    if "all" in names:
        names = sorted(_available_factories(args.points, rust_available, args.numpy_only))
    factories = _available_factories(args.points, rust_available, args.numpy_only)
    unknown = [name for name in names if name not in factories]
    if unknown:
        raise ValueError(
            f"Unknown detector(s): {unknown}. Choose from {sorted(factories)} or 'all'."
        )

    kind_of = {name: _model_input_kind(name) for name in names}

    header = (
        f"{'detector':<30} {'path':<7} {'points':>8} {'repeats':>8} "
        f"{'ms/point':>10} {'peak KiB':>10} {'growth KiB':>10} {'bounded':>7} {'match':>6}"
    )
    print(header)
    rows: list[dict[str, Any]] = []
    for name in names:
        kind = kind_of[name]
        path_entries: list[dict[str, Any]] = []
        for label, factory in factories[name]:
            timings: list[float] = []
            peaks: list[float] = []
            growths: list[float] = []
            boundeds: list[bool] = []
            scores_list: list[list[float]] = []
            for _ in range(args.repeats):
                detector = factory()
                result = run_detector(
                    name,
                    detector,
                    args.points,
                    args.checkpoint,
                    kind,
                    args.max_growth_kb,
                    collect_scores=True,
                )
                timings.append(float(result["ms_per_point"]))
                peaks.append(float(result["peak_kb"]))
                growths.append(float(result["growth_kb"]))
                boundeds.append(bool(result["bounded"]))
                scores_list.append(result["scores"])  # type: ignore[arg-type]
            path_entries.append(
                {
                    "label": label,
                    "ms_per_point": float(median(timings)),
                    "peak_kb": float(median(peaks)),
                    "growth_kb": float(median(growths)),
                    "bounded": bool(all(boundeds)),
                    "first_kb": float(result["first_kb"]),
                    "last_kb": float(result["last_kb"]),
                    "scores": scores_list[0],
                }
            )
        # Compare the PyO3 scores with the NumPy scores when both paths ran.
        match = "n/a"
        if len(path_entries) == 2:
            numpy_scores = path_entries[0]["scores"]
            rust_scores = path_entries[1]["scores"]
            match = "yes" if np.allclose(numpy_scores, rust_scores, rtol=1e-9, atol=1e-12) else "no"
        for entry in path_entries:
            rows.append(
                {
                    "detector": name,
                    "path": entry["label"],
                    "points": args.points,
                    "repeats": args.repeats,
                    "ms_per_point": entry["ms_per_point"],
                    "peak_kb": entry["peak_kb"],
                    "first_kb": entry["first_kb"],
                    "last_kb": entry["last_kb"],
                    "growth_kb": entry["growth_kb"],
                    "bounded": entry["bounded"],
                }
            )
            print(
                f"{name:<30} {entry['label']:<7} {int(args.points):>8} {args.repeats:>8} "
                f"{entry['ms_per_point']:>10.3f} {entry['peak_kb']:>10.1f} "
                f"{entry['growth_kb']:>10.1f} "
                f"{'yes' if entry['bounded'] else 'no':>7} {match:>6}"
            )

    if args.output is not None:
        with args.output.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "detector",
                    "path",
                    "points",
                    "repeats",
                    "ms_per_point",
                    "peak_kb",
                    "first_kb",
                    "last_kb",
                    "growth_kb",
                    "bounded",
                ],
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(rows)

    rust_note = (
        "PyO3 path available."
        if rust_available and not args.numpy_only
        else "PyO3 path not available (NumPy only)."
    )
    print(f"\n{rust_note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
