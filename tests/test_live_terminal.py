import re
from io import StringIO

import numpy as np
import pytest

from pysad.live.terminal import TerminalConfig, TerminalDashboard

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _render_single_step(
    *,
    width: int = 100,
    detector_name: str = "LODA",
    calibrator_name: str = "Conformal",
    calibrated_score: float = 0.42,
    threshold: float = 0.95,
) -> str:
    buf = StringIO()
    dashboard = TerminalDashboard(config=TerminalConfig(width=width), output=buf)
    dashboard.state.detector_name = detector_name
    dashboard.state.calibrator_name = calibrator_name
    dashboard.state.threshold = threshold
    dashboard.update(
        feature_vector=np.array([0.1, -0.2]),
        score=1.0,
        calibrated_score=calibrated_score,
        alert=calibrated_score > threshold,
    )
    dashboard.render()
    return _ANSI_RE.sub("", buf.getvalue())


def test_render_shows_score_line_when_not_alert() -> None:
    rendered = _render_single_step(calibrated_score=0.42, threshold=0.95)
    assert "Score:" in rendered
    assert "0.4200" in rendered
    assert "ALERT" not in rendered


def test_render_shows_alert_badge_when_above_threshold() -> None:
    rendered = _render_single_step(calibrated_score=0.98, threshold=0.95)
    assert "Score:" in rendered
    assert "0.9800" in rendered
    assert "ALERT" in rendered


@pytest.mark.parametrize(
    ("width", "detector_name"),
    [(120, "LODA"), (60, "OnlineIsolationForest")],
)
def test_render_header_box_aligned(width: int, detector_name: str) -> None:
    rendered = _render_single_step(width=width, detector_name=detector_name)
    header_lines = rendered.splitlines()[:3]
    assert len(header_lines) == 3
    assert header_lines[0].startswith("╔") and header_lines[0].endswith("╗")
    assert header_lines[1].startswith("║") and header_lines[1].endswith("║")
    assert header_lines[2].startswith("╚") and header_lines[2].endswith("╝")
    lengths = {len(line) for line in header_lines}
    assert len(lengths) == 1
    assert next(iter(lengths)) >= width
    if width == 120:
        assert len(header_lines[0]) == 120
