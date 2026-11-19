"""Check that detector memory stays bounded over a long stream.

Run it from the repository root:

    python benchmarks/memory_stability.py
    python benchmarks/memory_stability.py --detectors all --points 6000 --checkpoint 2000
    python benchmarks/memory_stability.py --detectors LODA,RSHash --points 100000 --checkpoint 10000

Each detector runs on a synthetic stream that is generated one point at a time, so
the script itself never stores the stream. Multivariate detectors see five features
drawn from the unit interval, univariate detectors see one feature, timestamped
detectors see ``(timestamp, value)``, and edge-stream detectors see ``(src, dst)``.
``tracemalloc`` records the traced memory at regular checkpoints, and the run fails
when the traced memory grows by more than ``--max-growth-kb`` between the second
checkpoint and the last one. Growth is measured between late checkpoints, not from
the start, so one-time setup allocations do not count as a leak.
"""

from __future__ import annotations

import argparse
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
WARMUP_POINTS = 1000
DEFAULT_PERIOD = 24


def _available_factories(points: int) -> dict[str, Callable[[], Any]]:
    sample_X = np.random.RandomState(0).rand(256, NUM_FEATURES)
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


def _detectors(names: list[str], points: int) -> dict[str, Any]:
    available = _available_factories(points)
    if names == ["all"]:
        names = sorted(available)
    detectors = {}
    for name in names:
        if name not in available:
            raise ValueError(f"Unknown detector: {name}. Choose from {sorted(available)} or 'all'.")
        detectors[name] = available[name]()
    return detectors


def _point(rng: np.random.RandomState, kind: str, step: int) -> np.ndarray:
    if kind == "univariate":
        return rng.rand(1)
    if kind == "timestamped":
        return np.array(
            [float((step % 24) * 3600), 0.5 + 0.4 * np.sin(2.0 * np.pi * (step % 24) / 24.0)]
        )
    if kind == "edge":
        return np.array([float(rng.randint(0, 20)), float(rng.randint(0, 20))])
    return rng.rand(NUM_FEATURES)


def run_detector(name: str, detector: Any, points: int, checkpoint: int) -> dict[str, float]:
    kind = _model_input_kind(name)
    rng = np.random.RandomState(0)

    for step in range(1, WARMUP_POINTS + 1):
        detector.fit_score_partial(_point(rng, kind, step))

    tracemalloc.start()
    try:
        first_kb = second_kb = last_kb = 0.0
        start = time.perf_counter()
        for i in range(1, points + 1):
            detector.fit_score_partial(_point(rng, kind, WARMUP_POINTS + i))

            if i % checkpoint == 0:
                last_kb = tracemalloc.get_traced_memory()[0] / 1024.0
                if i == checkpoint:
                    first_kb = last_kb
                if i == 2 * checkpoint:
                    second_kb = last_kb

        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    growth = last_kb - second_kb

    return {
        "points": float(points),
        "seconds": elapsed,
        "peak_kb": peak / 1024.0,
        "first_kb": first_kb,
        "last_kb": last_kb,
        "growth_kb": growth,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Long-run memory stability benchmark.")
    parser.add_argument(
        "--detectors",
        default="LODA,MedianAbsoluteDeviation,StandardAbsoluteDeviation",
        help="Comma-separated detector names, or 'all' to run every streaming detector.",
    )
    parser.add_argument("--points", type=int, default=1_000_000, help="Points per detector.")
    parser.add_argument(
        "--checkpoint", type=int, default=100_000, help="Points between memory checkpoints."
    )
    parser.add_argument(
        "--max-growth-kb",
        type=float,
        default=512.0,
        help="Allowed traced memory growth in KiB between the second and last checkpoints.",
    )
    args = parser.parse_args(argv)

    if args.checkpoint <= 0:
        parser.error("--checkpoint must be positive.")

    if args.points < 3 * args.checkpoint:
        parser.error("--points must be at least three times --checkpoint.")

    names = [name.strip() for name in args.detectors.split(",") if name.strip()]
    if not names:
        parser.error("--detectors must name at least one detector.")
    detectors = _detectors(names, args.points)
    failed = False

    header = f"{'detector':<28} {'points':>9} {'seconds':>9} "
    header += f"{'peak KiB':>10} {'growth KiB':>11}  result"
    print(header)
    for name, detector in detectors.items():
        result = run_detector(name, detector, args.points, args.checkpoint)
        passed = result["growth_kb"] <= args.max_growth_kb
        failed = failed or not passed
        print(
            f"{name:<28} {int(result['points']):>9} {result['seconds']:>9.1f} "
            f"{result['peak_kb']:>10.1f} {result['growth_kb']:>11.1f}  "
            f"{'PASS' if passed else 'FAIL'}"
        )

    if failed:
        print(f"Memory grew by more than {args.max_growth_kb} KiB for at least one detector.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
