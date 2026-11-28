"""ClusterFuzzLite / Atheris fuzz target for core streaming detectors, pipelines, and windows."""

from __future__ import annotations

import struct
import sys

import numpy as np

from pysad.models import LODA, HalfSpaceTrees, RSHash, xStream
from pysad.transform.postprocessing import ThresholdFilter
from pysad.transform.preprocessing import StandardScaler
from pysad.transform.probability_calibration import ConformalProbabilityCalibrator
from pysad.utils.window import RingBuffer, Window


def _decode_stream(data: bytes) -> tuple[int, list[list[float]]]:
    """Converts arbitrary fuzz bytes into a detector selector and finite feature vectors."""
    if len(data) < 2:
        return 0, []
    selector = data[0]
    num_features = 1 + (data[1] % 4)
    raw = data[2:]
    step_bytes = 4 * num_features
    max_steps = min(32, len(raw) // step_bytes)
    points: list[list[float]] = []
    for idx in range(max_steps):
        chunk = raw[idx * step_bytes : (idx + 1) * step_bytes]
        vals = [
            float(np.clip(v, -1e4, 1e4)) if np.isfinite(v) else 0.0
            for v in struct.unpack(f"<{num_features}f", chunk)
        ]
        points.append(vals)
    return selector, points


def fuzz_one_input(data: bytes) -> None:
    """Runs one fuzzed stream across a detector, pipeline, RingBuffer, and Window."""
    selector, points = _decode_stream(data)
    if not points:
        return
    dim = len(points[0])
    mins = np.full(dim, -1e4, dtype=np.float64)
    maxes = np.full(dim, 1e4, dtype=np.float64)
    detectors = [
        LODA(num_bins=6, num_random_cuts=8, random_state=0),
        HalfSpaceTrees(
            feature_mins=mins,
            feature_maxes=maxes,
            num_trees=4,
            max_depth=3,
            window_size=8,
            random_state=0,
        ),
        RSHash(
            feature_mins=mins,
            feature_maxes=maxes,
            num_components=4,
            num_hash_fns=2,
            decay=0.5,
            random_state=0,
        ),
        xStream(num_components=6, n_chains=4, depth=3, window_size=8, random_state=0),
    ]
    model = detectors[selector % len(detectors)]
    pipeline = (
        StandardScaler()
        | LODA(num_bins=5, num_random_cuts=6, random_state=0)
        | ConformalProbabilityCalibrator(window_size=8)
        | ThresholdFilter(threshold=0.95)
    )
    ring = RingBuffer(window_size=8)
    win = Window(window_size=8)

    for row in points:
        arr = np.asarray(row, dtype=np.float64)
        try:
            score = model.fit_score_partial(arr)
            assert isinstance(score, float)
            decision = pipeline.fit_predict_partial(arr)
            assert decision in (0, 1)
            ring.append(arr)
            win.update(score)
        except ValueError:
            pass


def main() -> None:
    """Entrypoint when invoked under Atheris / ClusterFuzzLite."""
    import atheris

    atheris.Setup(sys.argv, fuzz_one_input)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
