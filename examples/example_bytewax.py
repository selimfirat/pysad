"""Score a stream with PySAD under Bytewax, keeping one detector per source key.

This recipe runs a PySAD detector inside a Bytewax dataflow:

- Bytewax routes every string key to its own state, so keep one detector per source key.
- Score each point before the detector learns from it (prequential order) with
  fit_score_partial.
- Bytewax is not a PySAD dependency. Install it in your application environment to run
  the dataflow. If it is not installed, this script falls back to a synthetic stream so
  it runs anywhere for demonstration.

Usage:
    python examples/example_bytewax.py
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pysad.evaluation import AUROCMetric
from pysad.models import LODA
from pysad.utils import fix_seed, generate_stream

NUM_KEYS = 3
NUM_POINTS = 600
REPORT_EVERY = 100


def build_flow() -> Any | None:
    """Build the Bytewax dataflow if bytewax is installed, else return None."""
    try:
        import bytewax.operators as op
        from bytewax.connectors.stdio import StdOutSink
        from bytewax.dataflow import Dataflow
        from bytewax.testing import TestingSource
    except ImportError:
        return None

    def score_point(model: LODA | None, point: tuple[str, list[float]]) -> tuple[LODA, float]:
        """Score one keyed point, learn from it, and return the detector and its score."""
        key, values = point
        if model is None:
            model = LODA(random_state=0)
        score = model.fit_score_partial(np.asarray(values, dtype=float))
        return model, score

    flow = Dataflow("pysad_bytewax")
    source = op.input(
        "source",
        flow,
        TestingSource(
            [
                ("server-a", [0.1, 0.2, 0.3]),
                ("server-b", [0.4, 0.5, 0.6]),
                ("server-a", [1.0, 1.1, 1.2]),
            ]
        ),
    )
    scores = op.stateful_map("score", source, score_point)
    op.output("scores", scores, StdOutSink())
    return flow


def run_synthetic() -> None:
    """Run the keyed-state pattern on a synthetic stream without bytewax."""
    fix_seed(0)
    X, y = generate_stream(num_instances=NUM_POINTS, num_features=3, random_state=42)

    models: dict[str, LODA] = {}
    metric = AUROCMetric()

    for index in range(NUM_POINTS):
        key = f"server-{index % NUM_KEYS}"
        if key not in models:
            models[key] = LODA(random_state=0)
        model = models[key]
        score = model.fit_score_partial(X[index])
        metric.update(int(y[index]), score)

        if (index + 1) % REPORT_EVERY == 0:
            try:
                auroc = f"{metric.get():.3f}"
            except ValueError:
                auroc = "N/A (single class so far)"
            print(f"  [{index + 1}] score={score:.4f} label={int(y[index])} AUROC={auroc}")

    try:
        final_auroc = f"{metric.get():.3f}"
    except ValueError:
        final_auroc = "N/A (single class)"
    print(
        f"Finished synthetic stream. Points: {NUM_POINTS}, keys: {NUM_KEYS}, "
        f"Final AUROC: {final_auroc}"
    )


def main() -> None:
    """Run the dataflow recipe, or the synthetic fallback when bytewax is missing."""
    flow = build_flow()
    if flow is None:
        print("bytewax is not installed; running a synthetic stream instead.")
        print("Install bytewax in your application environment to run the dataflow recipe.")
        run_synthetic()
    else:
        print("Bytewax dataflow built. Run it with: python -m bytewax.run example_bytewax:flow")


if __name__ == "__main__":
    main()
