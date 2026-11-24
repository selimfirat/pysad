"""Reproducible benchmark of the public detectors on streams with drift.

Run it from the repository root:

    python benchmarks/bench_tsb_drift.py --points 10000 --seed 0 --repeats 3 --output drift.csv

The script loads each stream with drift from benchmarks.benchmark (KDD http, INSECTS A/I/IG/IR,
SMD 1-1/2-1/3-1), keeps at most ``--points`` points in time order, scales each feature to the range
from 0 to 1, and scores every public multivariate detector with ``fit_score_partial``. It reports
streaming AUROC and AUPR after the warm-up, milliseconds per point (throughput), and traced memory
from ``tracemalloc``. A second run on the same stream must give exactly the same scores, so the run
is reproducible.

The metrics exclude the warm-up points. Memory is the peak traced allocation of a timed run.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE.parent), str(_HERE)]

from benchmark import DETECTORS, DRIFT, load, warmup  # noqa: E402

from pysad.utils import Data, fix_seed  # noqa: E402

DEFAULT_SEED = 0
DEFAULT_REPEATS = 3


def run_detector(name, factory, X, y, seed, repeats):
    """Benchmark one detector on one drift stream.

    The detector is rebuilt for every timed run, so the constructor setup does not count. Each run
    scores every point with ``fit_score_partial`` before the detector learns from it, as the rest of
    the benchmark does. The first warm-up points do not count in AUROC and AUPR.

    Args:
        name (str): The name of the detector.
        factory (callable): Builds the detector from ``(X, seed)``.
        X (np.ndarray): The stream of shape ``(num_points, num_features)``.
        y (np.ndarray): The binary labels of shape ``(num_points,)``.
        seed (int): The random seed of the stream and the detector.
        repeats (int): The number of timed runs. The median time and memory are reported.

    Returns:
        dict: The detector, AUROC, AUPR, milliseconds per point, peak memory in MiB, and whether a
        fresh rebuild gives exactly the same scores.
    """
    cutoff = warmup(len(X))
    y_eval = y[cutoff:]

    def timed_run():
        fix_seed(seed)
        model = factory(X, seed)
        with threadpool_limits(1):
            tracemalloc.start()
            start = time.perf_counter()
            scores = np.array([float(model.fit_score_partial(x)) for x in X])
            seconds = time.perf_counter() - start
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        return scores, seconds, peak

    runs = [timed_run() for _ in range(repeats)]
    seconds = float(np.median([run[1] for run in runs]))
    peak = float(np.median([run[2] for run in runs]))
    scores = runs[-1][0]

    # Determinism: a fresh rebuild on the same stream must give exactly the same scores.
    fresh_scores, _, _ = timed_run()
    deterministic = bool(np.array_equal(scores, fresh_scores))

    both = 0 < y_eval.sum() < len(y_eval)
    auroc = float(roc_auc_score(y_eval, scores[cutoff:])) if both else float("nan")
    aupr = float(average_precision_score(y_eval, scores[cutoff:])) if both else float("nan")

    return {
        "detector": name,
        "auroc": round(auroc, 4),
        "aupr": round(aupr, 4),
        "ms_per_point": round(1000.0 * seconds / len(X), 3),
        "peak_mib": round(peak / (1024.0 * 1024.0), 3),
        "deterministic": deterministic,
    }


def run_benchmark(streams, detectors, points=None, seed=DEFAULT_SEED, repeats=DEFAULT_REPEATS):
    """Run the TSB-drift benchmark and return one row per stream and detector.

    Args:
        streams (list[str]): The drift streams to benchmark.
        detectors (list[str]): The public multivariate detectors to benchmark.
        points (int): The number of points to keep of each stream, or None for all thinned points.
        seed (int): The random seed of the streams and the detectors (Default=0).
        repeats (int): The number of timed runs per detector (Default=3).

    Returns:
        list[dict]: One row per stream and detector, with stream, detector, points, features,
        warm-up, seed, AUROC, AUPR, milliseconds per point, peak memory in MiB, and determinism.
    """
    rows = []
    for stream in streams:
        X, y = load(stream, seed, points)
        for name in detectors:
            factory = DETECTORS[name]
            result = run_detector(name, factory, X, y, seed, repeats)
            rows.append(
                {
                    "stream": stream,
                    "detector": name,
                    "points": len(X),
                    "features": int(X.shape[1]),
                    "warmup": warmup(len(X)),
                    "seed": seed,
                    **result,
                }
            )
    return rows


FIELDS = [
    "stream",
    "detector",
    "points",
    "features",
    "warmup",
    "seed",
    "auroc",
    "aupr",
    "ms_per_point",
    "peak_mib",
    "deterministic",
]


def _print_table(rows):
    """Print the results as a table."""
    widths = {field: len(field) for field in FIELDS}
    for row in rows:
        for field in FIELDS:
            widths[field] = max(widths[field], len(str(row.get(field, ""))))

    def line(row):
        return "  ".join(str(row.get(field, "")).ljust(widths[field]) for field in FIELDS)

    print(line({field: field for field in FIELDS}))
    print(line({field: "-" * widths[field] for field in FIELDS}))
    for row in rows:
        print(line(row))


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--detectors",
        default=",".join(sorted(DETECTORS)),
        help="Comma-separated detector names (default: all public multivariate detectors).",
    )
    parser.add_argument(
        "--streams",
        default=",".join(sorted(DRIFT)),
        help="Comma-separated drift streams (default: all streams with drift).",
    )
    parser.add_argument(
        "--points", type=int, default=None, help="Points to keep of each stream (Default=all)."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed (Default=0).")
    parser.add_argument(
        "--repeats", type=int, default=DEFAULT_REPEATS, help="Timed runs per detector (Default=3)."
    )
    parser.add_argument("--output", type=Path, help="Write the results to this CSV file.")
    args = parser.parse_args()

    detectors = [name.strip() for name in args.detectors.split(",") if name.strip()]
    streams = [name.strip() for name in args.streams.split(",") if name.strip()]

    unknown = set(detectors) - set(DETECTORS)
    if unknown:
        parser.error(f"unknown detectors: {', '.join(sorted(unknown))}")
    unknown = set(streams) - set(DRIFT)
    if unknown:
        parser.error(f"unknown streams: {', '.join(sorted(unknown))}")

    args.detectors = detectors
    args.streams = streams
    return args


def main() -> int:
    """Run the benchmark command."""
    args = _parse_args()

    for stream in args.streams:
        # Download each stream once before the timed runs.
        Data().get_data(stream)

    rows = run_benchmark(
        args.streams, args.detectors, points=args.points, seed=args.seed, repeats=args.repeats
    )
    _print_table(rows)

    if args.output is not None:
        with args.output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    all_deterministic = all(row["deterministic"] for row in rows)
    return 0 if all_deterministic else 1


if __name__ == "__main__":
    raise SystemExit(main())
