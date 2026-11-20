"""Tests for the cross-library benchmark comparison SVG generator."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from scripts import generate_comparison_chart as chart


def _boxes_overlap(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def test_generated_comparison_svgs_are_valid_and_in_sync(tmp_path, monkeypatch):
    """Running main() produces valid light, dark, and adaptive SVGs matching committed files."""
    monkeypatch.setattr(chart, "OUT_DIR", tmp_path)
    chart.main()

    expected_names = {"comparison.svg", "comparison-light.svg", "comparison-dark.svg"}
    assert {p.name for p in tmp_path.glob("*.svg")} == expected_names

    for name in expected_names:
        generated = (tmp_path / name).read_text(encoding="utf-8")
        committed = (chart.ROOT / "docs" / "images" / name).read_text(encoding="utf-8")
        assert generated == committed, (
            f"docs/images/{name} is out of date; run `python scripts/generate_comparison_chart.py`"
        )

        root = ET.fromstring(generated)
        assert root.tag.endswith("svg")
        for label in ("LoOP", "LODA", "River LODA", "CapyMOA RRCF", "PyOD IForest"):
            assert label in generated


def test_highlight_labels_stay_within_canvas_and_do_not_overlap():
    """Highlighted detector labels do not collide with one another or leave the plot canvas."""
    points = chart.load_comparison_points()
    boxes: list[tuple[str, tuple[float, float, float, float]]] = []
    for point in points:
        spec = chart.HIGHLIGHT_LABELS.get((point.library, point.raw_name))
        if spec is None:
            continue
        box = chart.label_bounding_box(point, spec)
        assert box[0] >= 0.0 and box[2] <= chart.WIDTH, (
            f"{spec.text} exceeds horizontal bounds: {box}"
        )
        assert box[1] >= chart.MARGIN_TOP - 8.0 and box[3] <= chart.HEIGHT - 24.0, (
            f"{spec.text} exceeds plot vertical bounds: {box}"
        )
        boxes.append((spec.text, box))

    for idx, (name_a, box_a) in enumerate(boxes):
        for name_b, box_b in boxes[idx + 1 :]:
            assert not _boxes_overlap(box_a, box_b), (
                f"Label overlap between {name_a} {box_a} and {name_b} {box_b}"
            )
        for point in points:
            if chart.HIGHLIGHT_LABELS.get((point.library, point.raw_name)) and (
                chart.HIGHLIGHT_LABELS[(point.library, point.raw_name)].text == name_a
            ):
                continue
            px = chart._x_px(point.throughput)
            py = chart._y_px(point.auroc)
            marker_box = (px - 4.0, py - 4.0, px + 4.0, py + 4.0)
            assert not _boxes_overlap(box_a, marker_box), (
                f"Label {name_a} {box_a} covers marker {point.library}:{point.raw_name} at ({px:.1f}, {py:.1f})"
            )
