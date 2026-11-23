"""Reproducible edge-memory and throughput benchmark for pure-Python streaming detectors.

Run it from the repository root:

    python benchmarks/bench_stream.py --detectors all --points 10000 --repeats 3
    python benchmarks/bench_stream.py --detectors LODA,RSHash,MIDAS --points 50000 --output stream.csv

Each detector runs on a deterministic synthetic stream of a fixed length. tracemalloc records
the peak traced memory, and the script reports milliseconds per point. The stream length and the
seed are fixed, so a rerun gives the same memory and timing. Multivariate detectors see five
features from the unit interval, univariate detectors see one value, timestamped detectors see
``(timestamp, value)``, and edge-stream detectors see ``(source, destination)``.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE.parent), str(_HERE)]
from benchmark import DETECTORS, UNIVARIATE, _is_detector_model, _model_input_kind  # noqa: E402

from pysad import models  # noqa: E402

NUM_FEATURES = 5
DEFAULT_PERIOD = 24
SEED = 0
WARMUP_POINTS = 64


def _available_factories(points: int) -> dict[str, Callable[[], Any]]:
    """One fresh detector per name, with the arguments the benchmark runner uses.

    Args:
        points (int): The fixed stream length, passed to univariate and timestamped factories.

    Returns:
        A mapping from detector name to a no-argument factory that builds the detector.
    """
    sample_X = np.random.RandomState(SEED).rand(256, NUM_FEATURES)
    factories: dict[str, Callable[[], Any]] = {}
    for name in models.__all__:
        if not _is_detector_model(name):
            continue
        kind = _model_input_kind(name)
        if kind == "multivariate" and name in DETECTORS:
            factories[name] = lambda f=DETECTORS[name]: f(sample_X, 0)  # type: ignore[misc]
        elif kind in ("univariate", "timestamped") and name in UNIVARIATE:
            factories[name] = lambda f=UNIVARIATE[name]: f(points, DEFAULT_PERIOD)  # type: ignore[misc]
        elif kind == "edge":
            cls = getattr(models, name)
            factories[name] = lambda c=cls: c(random_state=0)  # type: ignore[misc]
    return factories


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


def run_detector(name: str, detector: Any, points: int, repeats: int) -> dict[str, float | int]:
    """Measure the peak memory and the milliseconds per point of one detector.

    The detector first warms up on a few points. Then every later point is scored before the
    detector learns from it, as in the rest of PySAD. tracemalloc records the peak traced memory
    of the timed run.

    Args:
        name (str): The name of the detector.
        detector (Any): The detector to measure.
        points (int): The fixed length of the stream.
        repeats (int): The number of timed runs.

    Returns:
        A mapping with the stream length, the number of timed runs, the median milliseconds per
        point, and the peak traced memory in KiB and MiB.
    """
    kind = _model_input_kind(name)
    rng = np.random.RandomState(SEED)

    for step in range(1, WARMUP_POINTS + 1):
        detector.fit_score_partial(_point(rng, kind, step))

    timings = []
    peak_kb = 0.0
    for _ in range(repeats):
        tracemalloc.start()
        try:
            start = time.perf_counter()
            for i in range(1, points + 1):
                detector.fit_score_partial(_point(rng, kind, WARMUP_POINTS + i))
            seconds = time.perf_counter() - start
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        timings.append(1000.0 * seconds / points)
        peak_kb = max(peak_kb, peak / 1024.0)

    timings.sort()
    return {
        "points": float(points),
        "repeats": float(repeats),
        "ms_per_point": float(timings[len(timings) // 2]),
        "peak_kb": peak_kb,
        "peak_mb": peak_kb / 1024.0,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the edge-memory and throughput benchmark and print the results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--detectors",
        default="LODA,RSHash,StandardAbsoluteDeviation,MIDAS",
        help="Comma-separated detector names, or 'all' to run every streaming detector.",
    )
    parser.add_argument(
        "--points", type=int, default=10_000, help="Fixed length of the stream (Default=10000)."
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of timed runs per detector (Default=3).",
    )
    parser.add_argument("--output", type=Path, help="Write the measurements to this CSV file.")
    args = parser.parse_args(argv)

    if args.points <= 0:
        parser.error("--points must be positive.")
    if args.repeats <= 0:
        parser.error("--repeats must be positive.")

    names = [name.strip() for name in args.detectors.split(",") if name.strip()]
    if not names:
        parser.error("--detectors must name at least one detector.")
    if "all" in names:
        names = sorted(_available_factories(args.points))
    factories = _available_factories(args.points)
    detectors = {}
    for name in names:
        if name not in factories:
            raise ValueError(f"Unknown detector: {name}. Choose from {sorted(factories)} or 'all'.")
        detectors[name] = factories[name]

    header = (
        f"{'detector':<28} {'points':>9} {'repeats':>8} "
        f"{'ms/point':>10} {'peak KiB':>10} {'peak MiB':>10}"
    )
    print(header)
    rows = []
    for name, factory in detectors.items():
        detector = factory()
        result = run_detector(name, detector, args.points, args.repeats)
        rows.append({"detector": name, **result})
        print(
            f"{name:<28} {int(result['points']):>9} {int(result['repeats']):>8} "
            f"{result['ms_per_point']:>10.3f} {result['peak_kb']:>10.1f} {result['peak_mb']:>10.3f}"
        )

    if args.output is not None:
        with args.output.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["detector", "points", "repeats", "ms_per_point", "peak_kb", "peak_mb"],
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
