"""Benchmark a custom detector with the PySAD benchmark runner.

This script shows how to benchmark a detector that is not part of PySAD. The detector lives in
its own file, and the runner loads it with ``--custom path/to/file.py:ClassName``.

Run the benchmark runner on this detector, with a LaTeX summary table:

    python benchmarks/benchmark.py --custom examples/example_custom_model_benchmark.py:EWMAZScore --latex

By default the runner adds the custom model to the parts that match its input kind. A
multivariate model runs on the ODDS datasets and the streams with drift. A univariate model runs
on the NAB and UCR series. Custom models are not stored in the result cache, so each run scores
them again. See :doc:`/benchmarks/run`.
"""

import sys
from pathlib import Path

import numpy as np

from pysad.core import BaseModel

ROOT = Path(__file__).resolve().parents[1]

# Make the repository benchmarks directory importable so the example also works when run directly
# as ``python examples/example_custom_model_benchmark.py``. The installed ``benchmarks`` package
# would otherwise shadow the one in this repository.
BENCHMARKS = ROOT / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))


class EWMAZScore(BaseModel):
    """Scores each point by its distance from an exponentially weighted mean.

    Args:
        alpha (float): The weight of the newest point, between 0 and 1 (Default=0.05).
    """

    _input_kind = "multivariate"

    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.mean_ = None
        self.var_ = None

    def fit_partial(self, X, y=None):
        if self.mean_ is None:
            self.mean_ = X.astype(float)
            self.var_ = np.ones_like(self.mean_)
            return self

        diff = X - self.mean_
        self.mean_ += self.alpha * diff
        self.var_ = (1 - self.alpha) * (self.var_ + self.alpha * diff**2)
        return self

    def score_partial(self, X):
        if self.mean_ is None:
            return 0.0

        z = (X - self.mean_) / np.sqrt(self.var_ + 1e-12)
        return float(np.abs(z).max())


def main():
    rng = np.random.default_rng(0)
    num_features = 3
    num_points = 600
    normal = rng.normal(size=(num_points, num_features))
    labels = np.zeros(num_points, dtype=int)
    # A few clear anomalies at the end of the stream.
    normal[-10:] += 5.0
    labels[-10:] = 1

    # Load this file as a custom model, exactly as --custom does.
    model_path = f"{Path(__file__)}:EWMAZScore"
    from benchmark import _load_custom_model, _make_custom_factory, score_stream, warmup

    cls = _load_custom_model(model_path)
    factory = _make_custom_factory(cls, "multivariate")
    model = factory(normal, 0)

    scores, ms_per_point = score_stream(model, normal)

    cut = warmup(len(normal))
    from pysad.evaluation import AUROCMetric

    metric = AUROCMetric()
    for label, score in zip(labels[cut:], scores[cut:], strict=False):
        metric.update(int(label), float(score))

    print(f"AUROC: {metric.get():.3f}")
    print(f"ms per point: {ms_per_point}")


if __name__ == "__main__":
    main()
