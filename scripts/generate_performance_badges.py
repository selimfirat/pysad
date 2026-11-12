"""Regenerate the JSON data behind the README performance badges.

The badges show median throughput and separate univariate and multivariate detector counts. Their values come from
the benchmark data in docs/benchmark/ and from the detector catalog, so they
stay in sync with the measured performance of the library.

Run it from the repository root:

    python scripts/generate_performance_badges.py
"""

import csv
import json
import sys
from pathlib import Path

from pysad import models

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "benchmark" / "results.csv"
OUT_DIR = ROOT / "docs" / "_static"


def _detector_counts() -> dict[str, int]:
    """Count detectors by input type, including timestamped series as univariate."""
    counts = {"univariate": 0, "multivariate": 0}
    for name in models.__all__:
        cls = getattr(models, name)
        if not cls._is_detector:
            continue
        kind = cls._input_kind
        if kind == "multivariate" and hasattr(sys.modules[cls.__module__], "check_univariate"):
            kind = "univariate"
        group = "univariate" if kind in ("univariate", "timestamped") else "multivariate"
        counts[group] += 1
    return counts


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
    """Write the performance and detector-count badges to docs/_static/."""
    counts = _detector_counts()
    rate = _format_rate(_median_points_per_second())

    badges = {
        "throughput.json": _badge("median throughput", rate, "green"),
        "univariate_detectors.json": _badge(
            "univariate detectors", str(counts["univariate"]), "blue"
        ),
        "multivariate_detectors.json": _badge(
            "multivariate detectors", str(counts["multivariate"]), "blue"
        ),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, data in badges.items():
        path = OUT_DIR / filename
        path.write_text(json.dumps(data, indent=2) + "\n")
        print(f"wrote {path}: {data['label']} = {data['message']}")


if __name__ == "__main__":
    main()
