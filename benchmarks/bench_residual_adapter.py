"""Benchmark duck-typed residual pipelines, with an optional detector comparison.

Run from the repository root:

    python benchmarks/bench_residual_adapter.py
    python benchmarks/bench_residual_adapter.py --compare --points 5000 --seed 7
    python benchmarks/bench_residual_adapter.py --compare --forecaster my_forecaster.py:create

Comparison mode chains ForecastingAnomalyDetector, TSFMResidualAdapter, and
ConformalTSFMCalibrator. It compares this pipeline with LODA and
StreamingAutoregressiveDetector on one synthetic seasonal stream with labeled spikes.
The default moving average is a stand-in, not a TSFM quality measurement.
Use a local factory to compare an existing TSFM without adding dependencies.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE.parent)]

from pysad.core.base_model import BaseModel  # noqa: E402
from pysad.integrations import (  # noqa: E402
    ForecasterProtocol,
    ForecastingAnomalyDetector,
    TSFMResidualAdapter,
)
from pysad.models import LODA, StreamingAutoregressiveDetector  # noqa: E402
from pysad.transform.probability_calibration import ConformalTSFMCalibrator  # noqa: E402


class _MovingAverageForecaster:
    """Forecast the mean of the last values without an external library.

    Args:
        window (int): Number of recent values to retain (Default=10).
    """

    def __init__(self, window: int = 10) -> None:
        self.history: deque[float] = deque(maxlen=window)

    def forecast(self, horizon: int = 1) -> list[float]:
        """Return the mean of the last ``window`` values as the one-step forecast."""
        if not self.history:
            return [0.0]
        return [sum(self.history) / len(self.history)]

    def learn_one(self, y: float) -> None:
        """Add the observed value ``y`` to the history."""
        self.history.append(float(y))


def _make_stream(points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a seeded seasonal stream and binary spike labels."""
    rng = np.random.default_rng(seed)
    time_index = np.arange(points)
    level = 10.0 + 2.0 * np.sin(2.0 * np.pi * time_index / 50.0)
    spikes = rng.random(points) < 0.02
    level[spikes] += 5.0
    noise: np.ndarray = rng.normal(0.0, 0.5, size=points)
    result: np.ndarray = level + noise
    return result, spikes.astype(np.int64)


def _build_pipeline(window_size: int) -> BaseModel:
    """Build the forecasting detector chained with the residual adapter."""
    forecaster = _MovingAverageForecaster(window=10)
    detector = ForecastingAnomalyDetector(forecaster=forecaster, probationary_period=0)
    adapter = TSFMResidualAdapter(window_size=window_size, min_std=1e-8)
    return detector | adapter


def _measure(
    factory: Callable[[], BaseModel],
    stream: np.ndarray,
    warmup: int,
    repeats: int,
) -> tuple[np.ndarray, float, bool]:
    """Time scoring and learning after warm-up, then compare a fresh run."""
    durations = []
    previous = None
    deterministic = True
    scores = np.empty(len(stream) - warmup, dtype=np.float64)
    for run in range(repeats + 1):
        pipeline = factory()
        for x in stream[:warmup]:
            pipeline.fit_score_partial(x)
        start = time.perf_counter()
        for i, x in enumerate(stream[warmup:]):
            scores[i] = pipeline.fit_score_partial(x)
        elapsed = time.perf_counter() - start
        if run < repeats:
            durations.append(elapsed)
        if previous is not None:
            deterministic = deterministic and bool(np.array_equal(scores, previous))
        previous = scores.copy()
    ms_per_point = 1000.0 * float(np.median(durations)) / len(scores)
    return scores, ms_per_point, deterministic


def run_benchmark(
    points: int = 2000,
    window_size: int = 100,
    seed: int = 7,
    repeats: int = 3,
    warmup: int = 64,
) -> dict[str, float | int | bool]:
    """Run the deterministic benchmark and return the results.

    Args:
        points (int): The number of points in the stream (Default=2000).
        window_size (int): The rolling window of the residual adapter (Default=100).
        seed (int): The random seed of the stream (Default=7).
        repeats (int): The number of timed runs (Default=3).
        warmup (int): Initial points excluded from timing (Default=64).

    Returns:
        A mapping with the stream settings, the median milliseconds per point, the determinism
        flag, and the score range.
    """
    stream, _ = _make_stream(points, seed)
    warmup = min(warmup, points // 2)
    scores, ms_per_point, deterministic = _measure(
        factory=lambda: _build_pipeline(window_size),
        stream=stream[:, None],
        warmup=warmup,
        repeats=repeats,
    )

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


def run_comparison(
    forecaster_factory: Callable[[], ForecasterProtocol] = _MovingAverageForecaster,
    points: int = 2000,
    window_size: int = 100,
    seed: int = 7,
    repeats: int = 3,
    warmup: int = 64,
) -> list[dict[str, str | float | int | bool]]:
    """Compare calibrated residuals and PySAD detectors on the same stream.

    Each run starts with fresh models. Each method scores before learning.
    Warm-up scores do not enter AUROC or timing. Labels never enter training.
    Timing excludes model creation and includes forecasting, scoring, and learning.
    If the scored labels contain only one class, AUROC is NaN.

    Args:
        forecaster_factory (Callable): Create a fresh forecaster with forecast(horizon)
            and learn_one(y) methods (Default=_MovingAverageForecaster).
        points (int): Number of stream values (Default=2000).
        window_size (int): Residual and calibration window size (Default=100).
        seed (int): Seed for the stream and LODA (Default=7).
        repeats (int): Number of timed runs per method (Default=3).
        warmup (int): Initial values excluded from metrics (Default=64).

    Returns:
        list[dict]: Settings, AUROC, timing, repeatability, and score range for each method.
    """
    stream, labels = _make_stream(points, seed)
    warmup = min(warmup, points // 2)

    def residual_pipeline() -> BaseModel:
        return (
            ForecastingAnomalyDetector(forecaster=forecaster_factory())
            | TSFMResidualAdapter(window_size=window_size, min_std=1e-8)
            | ConformalTSFMCalibrator(alpha=0.1, gamma=0.01, window_size=window_size)
        )

    factories: dict[str, Callable[[], BaseModel]] = {
        "ConformalResidual": residual_pipeline,
        "LODA": lambda: LODA(random_state=seed),
        "StreamingAutoregressiveDetector": StreamingAutoregressiveDetector,
    }
    results = []
    scored_labels = labels[warmup:]
    for name, factory in factories.items():
        scores, ms_per_point, deterministic = _measure(
            factory=factory,
            stream=stream[:, None],
            warmup=warmup,
            repeats=repeats,
        )
        auroc = (
            float(roc_auc_score(scored_labels, scores))
            if np.unique(scored_labels).size == 2
            else float("nan")
        )
        results.append(
            {
                "method": name,
                "points": points,
                "window_size": window_size,
                "seed": seed,
                "repeats": repeats,
                "warmup": warmup,
                "auroc": auroc,
                "ms_per_point": ms_per_point,
                "deterministic": deterministic,
                "min_score": float(scores.min()),
                "max_score": float(scores.max()),
            }
        )
    return results


def _load_forecaster(spec: str) -> Callable[[], ForecasterProtocol]:
    """Load a factory from a local path:function specification."""
    file_path, name = spec.rsplit(":", 1)
    module_spec = importlib.util.spec_from_file_location("benchmark_forecaster", file_path)
    if module_spec is None or module_spec.loader is None:
        raise ValueError(f"Cannot load forecaster factory from {file_path}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return getattr(module, name)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=2000, help="Points in the stream")
    parser.add_argument("--window-size", type=int, default=100, help="Adapter rolling window")
    parser.add_argument("--seed", type=int, default=7, help="Random seed of the stream")
    parser.add_argument("--repeats", type=int, default=3, help="Number of timed runs")
    parser.add_argument("--warmup", type=int, default=64, help="Points whose scores do not count")
    parser.add_argument(
        "--compare", action="store_true", help="Compare calibrated residuals and detectors"
    )
    parser.add_argument("--forecaster", help="Local path.py:factory for comparison mode")
    parser.add_argument("--output", type=Path, help="Write the results to this CSV file.")
    args = parser.parse_args()
    if args.forecaster and not args.compare:
        parser.error("--forecaster requires --compare")
    return args


def main() -> int:
    """Run the benchmark command."""
    args = _parse_args()
    settings = dict(
        points=args.points,
        window_size=args.window_size,
        seed=args.seed,
        repeats=args.repeats,
        warmup=args.warmup,
    )
    if args.compare:
        factory = _load_forecaster(args.forecaster) if args.forecaster else _MovingAverageForecaster
        rows = run_comparison(forecaster_factory=factory, **settings)
        for row in rows:
            print(
                f"{row['method']}: AUROC={row['auroc']:.4f}, "
                f"ms per point={row['ms_per_point']:.4f}, "
                f"deterministic={row['deterministic']}"
            )
    else:
        rows = [run_benchmark(**settings)]
        _print_result(rows[0])
    if args.output is not None:
        with args.output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return 0 if all(row["deterministic"] for row in rows) else 1


def _print_result(results: dict[str, float | int | bool]) -> None:
    """Print the residual adapter timing result."""
    print(f"Points: {results['points']}")
    print(f"Window size: {results['window_size']}")
    print(f"Seed: {results['seed']}")
    print(f"Timed runs: {results['repeats']}")
    print(f"Warm-up points: {results['warmup']}")
    print(f"Median ms per point: {results['ms_per_point']:.4f}")
    print(f"Deterministic: {results['deterministic']}")
    print(f"Score range: [{results['min_score']:.4f}, {results['max_score']:.4f}]")


if __name__ == "__main__":
    raise SystemExit(main())
