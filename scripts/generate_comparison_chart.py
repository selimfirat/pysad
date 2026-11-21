"""Regenerate the cross-library benchmark scatter plots in docs/images/.

Reads the same-hardware comparison tables in docs/benchmark/ and builds three
brand-matched SVG variants (adaptive light/dark for READMEs, plus explicit light
and dark variants for Sphinx documentation).

Run from the repository root:

    python scripts/generate_comparison_chart.py
"""

from __future__ import annotations

import csv
import html
import math
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "docs" / "benchmark"
OUT_DIR = ROOT / "docs" / "images"

WIDTH = 960
HEIGHT = 460
MARGIN_LEFT = 74.0
MARGIN_RIGHT = 32.0
MARGIN_TOP = 72.0
MARGIN_BOTTOM = 54.0
PLOT_W = WIDTH - MARGIN_LEFT - MARGIN_RIGHT
PLOT_H = HEIGHT - MARGIN_TOP - MARGIN_BOTTOM

LOG_X_MIN = 0.0  # 1 point/s
LOG_X_MAX = 5.25  # ~178,000 points/s
Y_MIN = 0.44
Y_MAX = 0.88

_CLASS_RE = re.compile(r":class:`~pysad\.models\.([A-Za-z_]\w*)`")


@dataclass(frozen=True)
class BenchmarkPoint:
    """One benchmarked detector plotted in throughput-vs-AUROC space."""

    library: str
    raw_name: str
    label: str
    auroc: float
    ms_per_point: float
    throughput: float


@dataclass(frozen=True)
class LabelSpec:
    """Label text and relative placement for a highlighted detector."""

    text: str
    dx: float
    dy: float
    anchor: str = "start"


# Carefully tuned offsets around each highlighted marker to keep all labels
# legible without overlapping neighboring points or labels.
HIGHLIGHT_LABELS: dict[tuple[str, str], LabelSpec] = {
    ("pysad", "LocalOutlierProbability"): LabelSpec("LoOP", -10, -8, "end"),
    ("pysad", "RSHash"): LabelSpec("RSHash", 6, -28, "start"),
    ("pysad", "IForestASD"): LabelSpec("IForestASD", -10, -10, "end"),
    ("pysad", "ExactStorm"): LabelSpec("ExactStorm", 10, -14, "start"),
    ("pysad", "HalfSpaceTrees"): LabelSpec("HalfSpaceTrees", 6, -32, "start"),
    ("pysad", "LODA"): LabelSpec("LODA", 12, -8, "start"),
    ("pysad", "RobustRandomCutForest"): LabelSpec("RRCF", -10, -9, "end"),
    ("pysad", "xStream"): LabelSpec("xStream", -10, 16, "end"),
    ("pysad", "TEDA"): LabelSpec("TEDA", -10, -6, "end"),
    ("river", "river.anomaly.LODA"): LabelSpec("River LODA", -16, -4, "end"),
    ("river", "river.anomaly.HalfSpaceTrees"): LabelSpec("River HST", 10, 14, "start"),
    ("river", "river.anomaly.LocalOutlierFactor"): LabelSpec("River LOF", -10, 4, "end"),
    ("river", "river.anomaly.OneClassSVM"): LabelSpec("River OCSVM", -10, 4, "end"),
    ("capymoa", "capymoa.anomaly.RobustRandomCutForest"): LabelSpec(
        "CapyMOA RRCF", 10, -8, "start"
    ),
    ("capymoa", "capymoa.anomaly.StreamingIsolationForest"): LabelSpec(
        "CapyMOA SIF", -10, 14, "end"
    ),
    ("capymoa", "capymoa.anomaly.StreamRHF"): LabelSpec("CapyMOA StreamRHF", 10, 4, "start"),
    ("capymoa", "capymoa.anomaly.AdaptiveIsolationForest"): LabelSpec(
        "CapyMOA AIF", 10, 10, "start"
    ),
    ("capymoa", "capymoa.anomaly.Loda"): LabelSpec("CapyMOA LODA", -10, 26, "end"),
    ("capymoa", "capymoa.anomaly.RSHash"): LabelSpec("CapyMOA RSHash", 10, 4, "start"),
    ("capymoa", "capymoa.anomaly.HalfSpaceTrees"): LabelSpec("CapyMOA HST", -10, 4, "end"),
    ("pyod", "pyod.models.iforest.IForest"): LabelSpec("PyOD IForest", 10, -9, "start"),
    ("pyod", "pyod.models.knn.KNN"): LabelSpec("PyOD KNN", -14, -18, "end"),
    ("pyod", "pyod.models.hbos.HBOS"): LabelSpec("PyOD HBOS", 10, 28, "start"),
    ("pyod", "pyod.models.lof.LOF"): LabelSpec("PyOD LOF", -10, 14, "end"),
}


@dataclass(frozen=True)
class ThemePalette:
    """Color definitions for one SVG theme variant."""

    text: str
    muted: str
    grid: str
    chance: str
    pysad: str
    river: str
    capymoa: str
    pyod: str


LIGHT_THEME = ThemePalette(
    text="#0d1b22",
    muted="#5b6f78",
    grid="#dce3e2",
    chance="#8b99a0",
    pysad="#00757a",
    river="#5c6bc0",
    capymoa="#c26d1f",
    pyod="#64748b",
)

DARK_THEME = ThemePalette(
    text="#edf5f6",
    muted="#99b0b6",
    grid="#1e343d",
    chance="#64748b",
    pysad="#2dd4bf",
    river="#818cf8",
    capymoa="#f59e0b",
    pyod="#94a3b8",
)


def _clean_detector_name(raw: str) -> tuple[str, str]:
    """Return `(library_key, short_name)` from a comparison CSV Detector cell."""
    match = _CLASS_RE.search(raw)
    if match:
        return "pysad", match.group(1)
    stripped = raw.strip("` ")
    if stripped.startswith("river."):
        return "river", stripped
    if stripped.startswith("capymoa."):
        return "capymoa", stripped
    if stripped.startswith("pyod."):
        return "pyod", stripped
    raise ValueError(f"Unexpected detector entry in comparison CSV: {raw!r}")


def load_comparison_points(bench_dir: Path = BENCH_DIR) -> list[BenchmarkPoint]:
    """Load unique benchmarked detectors from the same-hardware comparison CSVs."""
    seen: set[tuple[str, str]] = set()
    points: list[BenchmarkPoint] = []
    for filename in ("comparison_river.csv", "comparison_capymoa.csv", "comparison_pyod.csv"):
        csv_path = bench_dir / filename
        with open(csv_path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                lib, name = _clean_detector_name(row["Detector"])
                key = (lib, name)
                if key in seen:
                    continue
                ms = float(row["Time per point (ms)"])
                auroc = float(row["Mean AUROC"])
                if ms <= 0.0:
                    continue
                seen.add(key)
                spec = HIGHLIGHT_LABELS.get(key)
                label = spec.text if spec is not None else name
                points.append(
                    BenchmarkPoint(
                        library=lib,
                        raw_name=name,
                        label=label,
                        auroc=auroc,
                        ms_per_point=ms,
                        throughput=1000.0 / ms,
                    )
                )
    return points


def _x_px(throughput: float) -> float:
    return MARGIN_LEFT + (math.log10(throughput) - LOG_X_MIN) / (LOG_X_MAX - LOG_X_MIN) * PLOT_W


def _y_px(auroc: float) -> float:
    return MARGIN_TOP + (Y_MAX - auroc) / (Y_MAX - Y_MIN) * PLOT_H


def _pysad_pareto_frontier(points: list[BenchmarkPoint]) -> list[BenchmarkPoint]:
    """Return PySAD points on the speed-vs-AUROC Pareto frontier, sorted by throughput."""
    pysad_pts = sorted((p for p in points if p.library == "pysad"), key=lambda p: -p.throughput)
    frontier: list[BenchmarkPoint] = []
    best_auroc = -1.0
    for point in pysad_pts:
        if point.auroc > best_auroc:
            frontier.append(point)
            best_auroc = point.auroc
    frontier.reverse()
    return frontier


def label_bounding_box(point: BenchmarkPoint, spec: LabelSpec) -> tuple[float, float, float, float]:
    """Return approximate `(x1, y1, x2, y2)` SVG coordinates for overlap checking."""
    px = _x_px(point.throughput)
    py = _y_px(point.auroc)
    tx = px + spec.dx
    ty = py + spec.dy
    width = max(24.0, len(spec.text) * 6.5)
    height = 12.0
    if spec.anchor == "end":
        x1, x2 = tx - width, tx
    elif spec.anchor == "middle":
        x1, x2 = tx - width / 2.0, tx + width / 2.0
    else:
        x1, x2 = tx, tx + width
    y1, y2 = ty - height + 2.0, ty + 2.0
    return (x1, y1, x2, y2)


def _theme_css(mode: str) -> str:
    sans = 'Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",Helvetica,Arial,sans-serif'
    mono = '"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace'

    def rules(p: ThemePalette) -> str:
        return (
            f".title{{fill:{p.text};font-size:14px;font-weight:600}}"
            f".cue,.legend-text{{fill:{p.muted};font-size:12px}}"
            f".axis-label{{fill:{p.muted};font-size:12px;font-weight:500}}"
            f".tick{{fill:{p.muted};font-family:{mono};font-size:11px}}"
            f".grid{{stroke:{p.grid};stroke-width:1}}"
            f".chance{{stroke:{p.chance};stroke-width:1;stroke-dasharray:4 4}}"
            f".chance-text{{fill:{p.muted};font-size:11px}}"
            f".frontier{{stroke:{p.pysad};stroke-width:1.4;stroke-dasharray:5 4;opacity:0.45;fill:none}}"
            f".leader{{stroke:{p.muted};stroke-width:1;opacity:0.45}}"
            f".m-pysad{{fill:{p.pysad}}}"
            f".m-river{{fill:{p.river}}}"
            f".m-capymoa{{fill:{p.capymoa}}}"
            f".m-pyod{{fill:none;stroke:{p.pyod};stroke-width:2}}"
            f".lbl-pysad{{fill:{p.pysad};font-size:11.5px;font-weight:600}}"
            f".lbl-other{{fill:{p.muted};font-size:11px;font-weight:500}}"
        )

    base = f"text{{font-family:{sans}}}"
    if mode == "light":
        return base + rules(LIGHT_THEME)
    if mode == "dark":
        return base + rules(DARK_THEME)
    return base + rules(LIGHT_THEME) + f"@media (prefers-color-scheme:dark){{{rules(DARK_THEME)}}}"


def _marker_svg(point: BenchmarkPoint, highlighted: bool) -> str:
    cx = _x_px(point.throughput)
    cy = _y_px(point.auroc)
    if point.library == "pysad":
        r = 5.0 if highlighted else 3.6
        opacity = "0.95" if highlighted else "0.38"
        return (
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" class="m-pysad" opacity="{opacity}"/>'
        )
    if point.library == "river":
        d = 5.4
        pts = f"{cx:.1f},{cy - d:.1f} {cx + d:.1f},{cy:.1f} {cx:.1f},{cy + d:.1f} {cx - d:.1f},{cy:.1f}"
        return f'<polygon points="{pts}" class="m-river" opacity="0.92"/>'
    if point.library == "capymoa":
        s = 4.5
        return (
            f'<rect x="{cx - s:.1f}" y="{cy - s:.1f}" '
            f'width="{2 * s:.1f}" height="{2 * s:.1f}" rx="1.2" class="m-capymoa" opacity="0.92"/>'
        )
    return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.5" class="m-pyod" opacity="0.9"/>'


def render_svg(points: list[BenchmarkPoint], mode: str = "adaptive") -> str:
    """Render the throughput-vs-AUROC comparison SVG for a given color mode."""
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
        f'width="{WIDTH}" height="{HEIGHT}" role="img" aria-labelledby="t d">',
        '<title id="t">PySAD vs River, CapyMOA, and PyOD: throughput and AUROC on 17 ODDS datasets</title>',
        '<desc id="d">Scatter plot comparing streaming multivariate anomaly detectors across PySAD, '
        "River 0.26.1, CapyMOA 0.15.1, and PyOD 3.6.7 wrapped with ReferenceWindowModel on the same "
        "hardware across 17 ODDS datasets. X-axis shows log-scale throughput in points per second; "
        "Y-axis shows mean AUROC.</desc>",
        f"<style>{_theme_css(mode)}</style>",
    ]

    # Header title and directional cue.
    parts.append(
        f'<text x="{MARGIN_LEFT:.0f}" y="22" class="title">'
        "Streaming multivariate anomaly detection across 17 ODDS datasets</text>"
    )
    parts.append(
        f'<text x="{WIDTH - MARGIN_RIGHT:.0f}" y="22" text-anchor="end" class="cue">'
        "Faster &amp; more accurate &#x2197;</text>"
    )

    # Legend strip.
    legend_y = 44.0
    legend_items = [
        ("pysad", "PySAD (26 detectors)", MARGIN_LEFT),
        ("river", "River 0.26.1", MARGIN_LEFT + 178),
        ("capymoa", "CapyMOA 0.15.1 (JVM)", MARGIN_LEFT + 300),
        ("pyod", "PyOD 3.6.7 (via ReferenceWindowModel)", MARGIN_LEFT + 482),
    ]
    for lib, text, lx in legend_items:
        if lib == "pysad":
            parts.append(
                f'<circle cx="{lx + 5:.1f}" cy="{legend_y - 4:.1f}" r="4.8" class="m-pysad"/>'
            )
        elif lib == "river":
            cx, cy, d = lx + 5, legend_y - 4, 5.2
            pts = f"{cx:.1f},{cy - d:.1f} {cx + d:.1f},{cy:.1f} {cx:.1f},{cy + d:.1f} {cx - d:.1f},{cy:.1f}"
            parts.append(f'<polygon points="{pts}" class="m-river"/>')
        elif lib == "capymoa":
            parts.append(
                f'<rect x="{lx + 0.5:.1f}" y="{legend_y - 8.5:.1f}" width="9" height="9" rx="1.2" class="m-capymoa"/>'
            )
        else:
            parts.append(
                f'<circle cx="{lx + 5:.1f}" cy="{legend_y - 4:.1f}" r="4.3" class="m-pyod"/>'
            )
        parts.append(
            f'<text x="{lx + 16:.1f}" y="{legend_y:.1f}" class="legend-text">{html.escape(text)}</text>'
        )

    # Grid lines and axis ticks.
    plot_x2 = WIDTH - MARGIN_RIGHT
    plot_y2 = HEIGHT - MARGIN_BOTTOM
    for auroc in (0.50, 0.60, 0.70, 0.80):
        yp = _y_px(auroc)
        cls = "chance" if auroc == 0.50 else "grid"
        parts.append(
            f'<line x1="{MARGIN_LEFT:.1f}" y1="{yp:.1f}" x2="{plot_x2:.1f}" y2="{yp:.1f}" class="{cls}"/>'
        )
        parts.append(
            f'<text x="{MARGIN_LEFT - 10:.1f}" y="{yp + 4:.1f}" text-anchor="end" class="tick">{auroc:.2f}</text>'
        )

    chance_y = _y_px(0.50)
    parts.append(
        f'<text x="{MARGIN_LEFT + 8:.1f}" y="{chance_y - 6:.1f}" class="chance-text">chance (0.50)</text>'
    )

    x_ticks = [
        (10.0, "10"),
        (100.0, "100"),
        (1_000.0, "1k"),
        (10_000.0, "10k"),
        (100_000.0, "100k"),
    ]
    for tps, tick_label in x_ticks:
        xp = _x_px(tps)
        parts.append(
            f'<line x1="{xp:.1f}" y1="{MARGIN_TOP:.1f}" x2="{xp:.1f}" y2="{plot_y2:.1f}" class="grid"/>'
        )
        parts.append(
            f'<text x="{xp:.1f}" y="{plot_y2 + 18:.1f}" text-anchor="middle" class="tick">{tick_label}</text>'
        )

    # Axis border lines (bottom + left).
    parts.append(
        f'<line x1="{MARGIN_LEFT:.1f}" y1="{plot_y2:.1f}" x2="{plot_x2:.1f}" y2="{plot_y2:.1f}" class="grid"/>'
    )
    parts.append(
        f'<line x1="{MARGIN_LEFT:.1f}" y1="{MARGIN_TOP:.1f}" x2="{MARGIN_LEFT:.1f}" y2="{plot_y2:.1f}" class="grid"/>'
    )

    # Axis titles.
    parts.append(
        f'<text x="{(MARGIN_LEFT + plot_x2) / 2:.1f}" y="{HEIGHT - 12:.1f}" '
        'text-anchor="middle" class="axis-label">Throughput (points / second, log scale)</text>'
    )
    parts.append(
        f'<text x="20" y="{(MARGIN_TOP + plot_y2) / 2:.1f}" text-anchor="middle" '
        f'transform="rotate(-90 20 {(MARGIN_TOP + plot_y2) / 2:.1f})" class="axis-label">Mean AUROC</text>'
    )

    # PySAD Pareto frontier polyline.
    frontier = _pysad_pareto_frontier(points)
    if len(frontier) >= 2:
        pts_attr = " ".join(f"{_x_px(p.throughput):.1f},{_y_px(p.auroc):.1f}" for p in frontier)
        parts.append(f'<polyline points="{pts_attr}" class="frontier"/>')

    # Draw unhighlighted PySAD markers underneath, then other libraries, then highlighted PySAD markers.
    highlighted_keys = set(HIGHLIGHT_LABELS)
    for p in points:
        if p.library == "pysad" and (p.library, p.raw_name) not in highlighted_keys:
            parts.append(_marker_svg(p, highlighted=False))
    for p in points:
        if p.library != "pysad":
            parts.append(_marker_svg(p, highlighted=True))
    for p in points:
        if p.library == "pysad" and (p.library, p.raw_name) in highlighted_keys:
            parts.append(_marker_svg(p, highlighted=True))

    # Draw direct labels and subtle leader lines when offset far from marker.
    for p in points:
        spec = HIGHLIGHT_LABELS.get((p.library, p.raw_name))
        if spec is None:
            continue
        px = _x_px(p.throughput)
        py = _y_px(p.auroc)
        tx = px + spec.dx
        ty = py + spec.dy
        if math.hypot(spec.dx, spec.dy) > 17.0:
            lx_end = tx - (3.0 if spec.anchor == "start" else -3.0)
            ly_end = ty - 3.5
            parts.append(
                f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{lx_end:.1f}" y2="{ly_end:.1f}" class="leader"/>'
            )
        lbl_cls = "lbl-pysad" if p.library == "pysad" else "lbl-other"
        parts.append(
            f'<text x="{tx:.1f}" y="{ty:.1f}" text-anchor="{spec.anchor}" class="{lbl_cls}">'
            f"{html.escape(spec.text)}</text>"
        )

    parts.append("</svg>\n")
    return "\n".join(parts)


def main() -> None:
    """Regenerate the adaptive, light, and dark comparison SVGs in docs/images/."""
    points = load_comparison_points()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    variants = {
        "comparison.svg": "adaptive",
        "comparison-light.svg": "light",
        "comparison-dark.svg": "dark",
    }
    for filename, mode in variants.items():
        path = OUT_DIR / filename
        path.write_text(render_svg(points, mode=mode), encoding="utf-8")
        print(f"wrote {path} ({len(points)} detectors)")


if __name__ == "__main__":
    main()
