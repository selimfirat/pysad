"""Check that detector memory stays bounded over a long stream.

Run it from the repository root:

    python benchmarks/memory_stability.py
    python benchmarks/memory_stability.py --detectors LODA,RSHash --points 100000

Each detector runs on a synthetic stream that is generated one point at a time, so
the script itself never stores the stream. The multivariate detectors see five
features drawn from the unit interval, and the univariate detectors see the first
feature only. ``tracemalloc`` records the traced memory at regular checkpoints, and
the run fails when the traced memory grows by more than ``--max-growth-kb`` between
the second checkpoint and the last one. Growth is measured between late checkpoints,
not from the start, so one-time setup allocations do not count as a leak. Scores are
discarded as they are produced, so a growing score list cannot mask a leak or fake
one. ``tracemalloc`` traces Python allocations; the fixed-size NumPy buffers of these
detectors are not traced, so a flat result means no Python-side state grows with the
stream, which is where an unbounded list or dict leak would show.

This script is not part of the pytest suite: a million point run would slow every
test run, and the benchmark workflow does not run it either. Run it by hand when a
change touches detector state, and report its numbers with the change.
"""

import argparse
import sys
import time
import tracemalloc

import numpy as np

from pysad import models

NUM_FEATURES = 5
WARMUP_POINTS = 1000


def _detectors(names: list[str]) -> dict[str, object]:
    available = {
        "LODA": lambda: models.LODA(random_state=0),
        "RSHash": lambda: models.RSHash(
            np.zeros(NUM_FEATURES), np.ones(NUM_FEATURES), random_state=0
        ),
        "MedianAbsoluteDeviation": lambda: models.MedianAbsoluteDeviation(),
        "StandardAbsoluteDeviation": lambda: models.StandardAbsoluteDeviation(),
    }
    detectors = {}
    for name in names:
        if name not in available:
            raise ValueError(f"Unknown detector: {name}. Choose from {sorted(available)}.")

        detectors[name] = available[name]()

    return detectors


def _points(rng: np.random.RandomState, univariate: bool):
    if univariate:
        return rng.rand(1)

    return rng.rand(NUM_FEATURES)


def run_detector(name: str, detector: object, points: int, checkpoint: int) -> dict[str, float]:
    univariate = name in {"MedianAbsoluteDeviation", "StandardAbsoluteDeviation"}
    rng = np.random.RandomState(0)

    for _ in range(WARMUP_POINTS):
        detector.fit_score_partial(_points(rng, univariate))

    tracemalloc.start()
    try:
        traced: list[float] = []
        start = time.perf_counter()
        for i in range(1, points + 1):
            detector.fit_score_partial(_points(rng, univariate))

            if i % checkpoint == 0:
                traced.append(tracemalloc.get_traced_memory()[0] / 1024.0)

        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    growth = traced[-1] - traced[1] if len(traced) > 1 else 0.0

    return {
        "points": float(points),
        "seconds": elapsed,
        "peak_kb": peak / 1024.0,
        "first_kb": traced[0],
        "last_kb": traced[-1],
        "growth_kb": growth,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Long-run memory stability benchmark.")
    parser.add_argument(
        "--detectors",
        default="LODA,MedianAbsoluteDeviation,StandardAbsoluteDeviation",
        help="Comma separated detector names. RSHash is also available.",
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

    if args.points < 3 * args.checkpoint:
        parser.error("--points must be at least three times --checkpoint.")

    names = [name.strip() for name in args.detectors.split(",") if name.strip()]
    detectors = _detectors(names)
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
