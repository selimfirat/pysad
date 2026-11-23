"""Regenerate the JSON data behind the performance badges.

The badges show median throughput, memory growth, and separate detector counts.
Benchmark data and the detector catalog provide the values.

Run it from the repository root:

    python scripts/generate_performance_badges.py
"""

import csv
import json
from pathlib import Path
from statistics import median

from pysad import models

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "benchmark" / "results.csv"
MEMORY_RESULTS = ROOT / "docs" / "benchmark" / "memory.csv"
OUT_DIR = ROOT / "docs" / "_static"


def _detector_counts() -> dict[str, int]:
    """Count detectors by input type, including timestamped series as univariate."""
    counts = {"univariate": 0, "multivariate": 0}
    for name in models.__all__:
        cls = getattr(models, name)
        if not cls._is_detector:
            continue
        kind = cls._effective_input_kind()
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

    return median(rates) if rates else 0.0


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

    with MEMORY_RESULTS.open(newline="") as handle:
        growth = max(float(row["growth_kb"]) for row in csv.DictReader(handle))

    badges = {
        "memory.json": _badge("max memory growth", f"{growth:.1f} KiB", "00757a"),
        "throughput.json": _badge("median throughput", rate, "00757a"),
        "univariate_detectors.json": _badge(
            "univariate detectors", str(counts["univariate"]), "00565a"
        ),
        "multivariate_detectors.json": _badge(
            "multivariate detectors", str(counts["multivariate"]), "00565a"
        ),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, data in badges.items():
        path = OUT_DIR / filename
        path.write_text(json.dumps(data, indent=2) + "\n")
        print(f"wrote {path}: {data['label']} = {data['message']}")


if __name__ == "__main__":
    main()
