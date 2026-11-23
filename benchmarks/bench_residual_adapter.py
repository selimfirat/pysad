"""Deterministic benchmark of the zero-dependency TSFM residual adapter with a duck-typed forecaster.

Run it from the repository root:

    python benchmarks/bench_residual_adapter.py
    python benchmarks/bench_residual_adapter.py --points 5000 --window-size 100 --seed 7 --repeats 3

The benchmark chains :class:`~pysad.integrations.ForecastingAnomalyDetector` with a duck-typed
forecaster (a plain object with ``forecast`` and ``learn_one`` methods, so no River or other
forecasting library is needed) and :class:`~pysad.integrations.TSFMResidualAdapter`. It scores a
synthetic univariate stream one point at a time with a fixed seed, reports the milliseconds per
point of the full pipeline, and checks that a second run on the same stream gives exactly the same
scores, so the benchmark is deterministic. It needs no dependency beyond NumPy.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE.parent)]

from pysad.core.base_model import BaseModel  # noqa: E402
from pysad.integrations import ForecastingAnomalyDetector, TSFMResidualAdapter  # noqa: E402


class _MovingAverageForecaster:
    """A deterministic duck-typed forecaster: the mean of the last ``window`` values.

    It implements :class:`~pysad.integrations.ForecasterProtocol` (``forecast`` and ``learn_one``)
    without River or any other forecasting library.
    """

    def __init__(self, window: int = 10) -> None:
        self.window = window
        self.history: list[float] = []

    def forecast(self, horizon: int = 1) -> list[float]:
        """Return the mean of the last ``window`` values as the one-step forecast."""
        if not self.history:
            return [0.0]
        return [sum(self.history[-self.window :]) / min(len(self.history), self.window)]

    def learn_one(self, y: float) -> None:
        """Add the observed value ``y`` to the history."""
        self.history.append(float(y))


def _make_stream(points: int, seed: int) -> np.ndarray:
    """Return a deterministic synthetic univariate stream with a seasonal pattern and spikes."""
    rng = np.random.default_rng(seed)
    time_index = np.arange(points)
    level = 10.0 + 2.0 * np.sin(2.0 * np.pi * time_index / 50.0)
    spikes = rng.random(points) < 0.02
    level[spikes] += 5.0
    noise: np.ndarray = rng.normal(0.0, 0.5, size=points)
    result: np.ndarray = level + noise
    return result


def _build_pipeline(window_size: int) -> BaseModel:
    """Build the forecasting detector chained with the residual adapter."""
    forecaster = _MovingAverageForecaster(window=10)
    detector = ForecastingAnomalyDetector(forecaster=forecaster, probationary_period=0)
    adapter = TSFMResidualAdapter(window_size=window_size, min_std=1e-8)
    return detector | adapter


def _score_stream(pipeline: BaseModel, stream: np.ndarray, warmup: int) -> np.ndarray:
    """Score every point after warm-up with fit_score_partial and return the scores."""
    for x in stream[:warmup]:
        pipeline.fit_score_partial(np.array([x]))
    scores = np.empty(len(stream) - warmup, dtype=np.float64)
    for i, x in enumerate(stream[warmup:]):
        scores[i] = pipeline.fit_score_partial(np.array([x]))
    return scores


def run_benchmark(
    points: int = 2000,
    window_size: int = 100,
    seed: int = 7,
    repeats: int = 3,
    warmup: int = 64,
) -> dict[str, float | int | bool]:
    """Run the deterministic benchmark and return the results.

    Args:
        points: The number of points in the stream (Default=2000).
        window_size: The rolling window of the residual adapter (Default=100).
        seed: The random seed of the stream (Default=7).
        repeats: The number of timed runs (Default=3).
        warmup: The number of first points whose scores do not count in the timing (Default=64).

    Returns:
        A mapping with the stream settings, the median milliseconds per point, the determinism
        flag, and the score range.
    """
    stream = _make_stream(points, seed)
    warmup = min(warmup, points // 2)
    num_points = len(stream) - warmup

    def timed_run() -> np.ndarray:
        pipeline = _build_pipeline(window_size)
        return _score_stream(pipeline, stream, warmup)

    durations = []
    scores = np.empty(0, dtype=np.float64)
    for _ in range(repeats):
        start = time.perf_counter()
        scores = timed_run()
        durations.append(time.perf_counter() - start)
    ms_per_point = 1000.0 * float(np.median(durations)) / num_points

    # Determinism: a fresh run on the same stream must give exactly the same scores.
    second = timed_run()
    deterministic = bool(np.array_equal(scores, second))

    return {
        "points": points,
        "window_size": window_size,
        "seed": seed,
        "repeats": repeats,
        "warmup": warmup,
        "ms_per_point": ms_per_point,
        "deterministic": deterministic,
        "min_score": float(scores.min()),
        "max_score": float(scores.max()),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=2000, help="Points in the stream")
    parser.add_argument("--window-size", type=int, default=100, help="Adapter rolling window")
    parser.add_argument("--seed", type=int, default=7, help="Random seed of the stream")
    parser.add_argument("--repeats", type=int, default=3, help="Number of timed runs")
    parser.add_argument("--warmup", type=int, default=64, help="Points whose scores do not count")
    parser.add_argument("--output", type=Path, help="Write the results to this CSV file.")
    return parser.parse_args()


def main() -> int:
    """Run the benchmark command."""
    args = _parse_args()
    results = run_benchmark(
        points=args.points,
        window_size=args.window_size,
        seed=args.seed,
        repeats=args.repeats,
        warmup=args.warmup,
    )
    print(f"Points: {results['points']}")
    print(f"Window size: {results['window_size']}")
    print(f"Seed: {results['seed']}")
    print(f"Timed runs: {results['repeats']}")
    print(f"Warm-up points: {results['warmup']}")
    print(f"Median ms per point: {results['ms_per_point']:.4f}")
    print(f"Deterministic: {results['deterministic']}")
    print(f"Score range: [{results['min_score']:.4f}, {results['max_score']:.4f}]")
    if args.output is not None:
        with args.output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results.keys()))
            writer.writeheader()
            writer.writerow(results)
    return 0 if results["deterministic"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
