from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def export_report(
    results: dict[str, float | None],
    path: str | Path,
    format: str = "json",
    *,
    model: str | None = None,
    postprocessor: str | None = None,
    dataset: str | None = None,
) -> None:
    """Write evaluation results to a report file.

    Takes the metric name to value mapping returned by
    :func:`~pysad.evaluation.evaluate_stream` or
    :func:`~pysad.evaluation.evaluate_postprocessor` and serializes it to JSON,
    CSV, or Markdown. ``None`` values are written as ``null`` in JSON and as an
    empty string in CSV and Markdown. Parent directories of ``path`` are created
    if they do not exist.

    Args:
        results (dict[str, float or None]): Metric names mapped to their final values.
        path (str or Path): Output file path. The extension is ignored; ``format``
            decides the serialization.
        format (str): One of ``"json"``, ``"csv"``, or ``"markdown"``
            (Default="json").
        model (str or None): Optional model name recorded in the report (Default=None).
        postprocessor (str or None): Optional postprocessor or calibrator name
            (Default=None).
        dataset (str or None): Optional dataset name (Default=None).

    Raises:
        ValueError: If ``format`` is not one of ``"json"``, ``"csv"``, or
            ``"markdown"``.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, str | None] = {
        "model": model,
        "postprocessor": postprocessor,
        "dataset": dataset,
    }

    if format == "json":
        document: dict[str, Any] = {
            "metadata": {key: value for key, value in metadata.items() if value is not None},
            "results": {name: value for name, value in results.items()},
        }
        output_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    elif format == "csv":
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["model", "postprocessor", "dataset", "metric", "value"])
            for name, value in results.items():
                writer.writerow(
                    [
                        metadata["model"] or "",
                        metadata["postprocessor"] or "",
                        metadata["dataset"] or "",
                        name,
                        "" if value is None else value,
                    ]
                )
    elif format == "markdown":
        lines: list[str] = ["# Evaluation Report", ""]
        if any(entry is not None for entry in metadata.values()):
            lines.extend(["| Metadata | Value |", "|---|---|"])
            for key, entry in metadata.items():
                if entry is not None:
                    lines.append(f"| {key} | {entry} |")
            lines.append("")
        lines.extend(["| Metric | Value |", "|---|---|"])
        for name, value in results.items():
            lines.append(f"| {name} | {'' if value is None else value} |")
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        raise ValueError(
            f"Unsupported report format: {format!r}. Use 'json', 'csv', or 'markdown'."
        )
