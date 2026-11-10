"""Regenerate the JSON data behind the README performance badges.

The badges show throughput and detector count. Their values come from
the benchmark data in docs/benchmark/ and from the detector catalog, so they
stay in sync with the measured performance of the library.

Run it from the repository root:

    python scripts/generate_performance_badges.py
"""

import csv
import json
from pathlib import Path

from pysad import models

# These names are in pysad.models.__all__ but are not anomaly detectors.
# NullModel, RandomModel and PerfectModel are test doubles, and StreamPipeline
# composes models instead of detecting anomalies.
_NOT_DETECTORS = {"NullModel", "RandomModel", "PerfectModel", "StreamPipeline"}

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "benchmark" / "results.csv"
OUT_DIR = ROOT / "docs" / "_static"


def _detector_count() -> int:
    """Return the number of anomaly detectors exported by pysad.models."""
    return len([name for name in models.__all__ if name not in _NOT_DETECTORS])


def _median_points_per_second() -> float:
    """Return the median points per second across the benchmark runs.

    The benchmark records milliseconds per point in results.csv. Points per
    second is 1000 divided by that value. The median is robust to very fast or
    very slow outlier runs.
    """
    rates = []
    with open(RESULTS, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                ms_per_point = float(row["ms_per_point"])
            except (TypeError, ValueError):
                continue
            if ms_per_point > 0:
                rates.append(1000.0 / ms_per_point)

    rates.sort()
    if not rates:
        return 0.0
    mid = len(rates) // 2
    if len(rates) % 2 == 0:
        return (rates[mid - 1] + rates[mid]) / 2.0
    return rates[mid]


def _format_rate(rate: float) -> str:
    """Format a points-per-second rate for a badge message."""
    if rate >= 1000:
        return f"{rate / 1000.0:.1f}k points/s"
    return f"{rate:.0f} points/s"


def _badge(label: str, message: str, color: str) -> dict:
    """Build a shields.io endpoint badge JSON object."""
    return {
        "schemaVersion": 1,
        "label": label,
        "message": message,
        "color": color,
    }


def main() -> None:
    """Write the two badge JSON files to docs/_static/."""
    count = _detector_count()
    rate = _format_rate(_median_points_per_second())

    badges = {
        "throughput.json": _badge("throughput", rate, "green"),
        "detectors.json": _badge("detectors", str(count), "blue"),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, data in badges.items():
        path = OUT_DIR / filename
        path.write_text(json.dumps(data, indent=2) + "\n")
        print(f"wrote {path}: {data['label']} = {data['message']}")


if __name__ == "__main__":
    main()
