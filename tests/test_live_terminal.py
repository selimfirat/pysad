import re
from io import StringIO

import numpy as np
import pytest

from pysad.live.terminal import (
    TerminalConfig,
    TerminalDashboard,
    compute_conformal_bounds,
    compute_feature_bounds,
    create_data_stream,
    get_calibrator,
    get_detector,
)
from pysad.transform.probability_calibration import (
    ConformalProbabilityCalibrator,
    ConformalTSFMCalibrator,
    SPOTProbabilityCalibrator,
)

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


def test_csv_stream_uses_custom_delimiter_and_no_header(tmp_path):
    path = tmp_path / "stream.csv"
    path.write_text("1;2;3\n4;5;6\n", encoding="utf-8")

    rows = list(create_data_stream(str(path), delimiter=";", has_header=False))

    assert [list(x) for x, _ in rows] == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


def test_csv_stream_skips_header_row_by_default(tmp_path):
    path = tmp_path / "stream.csv"
    path.write_text("a,b,c\n1,2,3\n4,5,6\n", encoding="utf-8")

    rows = list(create_data_stream(str(path)))

    assert [list(x) for x, _ in rows] == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


def test_csv_stream_respects_max_points(tmp_path):
    path = tmp_path / "stream.csv"
    path.write_text("h1,h2\n1,2\n3,4\n5,6\n", encoding="utf-8")

    rows = list(create_data_stream(str(path), max_points=2))

    assert [list(x) for x, _ in rows] == [[1.0, 2.0], [3.0, 4.0]]


def test_terminal_dashboard_ribbon_labels_trimming_and_cleanup():
    buf = StringIO()
    cfg = TerminalConfig(width=60, history_length=3, show_labels=True, show_ribbon=True)
    dash = TerminalDashboard(config=cfg, output=buf)
    dash.start_timer()
    dash.set_metadata("LODA", "Conformal", threshold=0.8, feature_names=["cpu", "mem"])
    for step in range(5):
        dash.update(
            feature_vector=np.array([0.1 * step, 1.0 - 0.1 * step]),
            score=float(step),
            calibrated_score=min(0.99, 0.2 * step),
            lower_bound=0.1 * step,
            upper_bound=min(1.0, 0.2 * step + 0.3),
            alert=step == 4,
            label=1 if step == 4 else 0,
        )
    dash.cleanup()

    cleaned = _ANSI_RE.sub("", buf.getvalue())
    assert len(dash.state.scores) == 3
    assert "Ribbon:" in cleaned
    assert "Labels:" in cleaned
    assert "cpu" in cleaned


def test_terminal_sparkline_and_conformal_bounds_variants():
    buf = StringIO()
    dash = TerminalDashboard(output=buf)
    assert dash._render_sparkline([], 5) == "     "
    assert dash._render_sparkline([1.0], 4) == "   ●"
    assert isinstance(dash._render_sparkline([2.0, 2.0, 2.0], 5, reference=2.0), str)
    assert "▄" in dash._render_sparkline([1.0, 2.0, 3.0], 5, reference=2.0)

    conformal = ConformalProbabilityCalibrator(window_size=20)
    assert compute_conformal_bounds(conformal, 1.0) == (0.0, 1.0)
    for v in range(1, 15):
        conformal.fit_transform_partial(float(v))
    lo, hi = compute_conformal_bounds(conformal, 5.0)
    assert lo <= hi

    spot = SPOTProbabilityCalibrator(num_init=3, depth=5)
    assert compute_conformal_bounds(spot, 1.0) == (0.0, 1.0)
    for v in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        spot.fit_transform_partial(v)
    s_lo, s_hi = compute_conformal_bounds(spot, 0.3)
    assert s_lo <= s_hi

    tsfm = ConformalTSFMCalibrator(alpha=0.1, gamma=0.05, window_size=10)
    tsfm.fit_transform_partial(0.5)
    t_lo, t_hi = compute_conformal_bounds(tsfm, 0.5)
    assert t_lo <= t_hi

    from types import SimpleNamespace

    assert compute_conformal_bounds(SimpleNamespace(threshold_=0.75), 0.5) == (0.0, 0.75)
    assert compute_conformal_bounds(object(), 0.5) == (0.0, 1.0)


def test_terminal_stream_bounds_and_factories(tmp_path):
    csv_path = tmp_path / "labeled.csv"
    csv_path.write_text("f1,f2,label\n1.0,2.0,0\n3.0,4.0,1\n", encoding="utf-8")
    rows = list(create_data_stream(str(csv_path), label_column=2))
    assert rows[0][1] == 0 and rows[1][1] == 1

    fmins, fmaxs = compute_feature_bounds(str(csv_path), label_column=2)
    np.testing.assert_allclose(fmins, [1.0, 2.0])
    np.testing.assert_allclose(fmaxs, [3.0, 4.0])

    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("f1,f2\n", encoding="utf-8")
    efmins, efmaxs = compute_feature_bounds(str(empty_csv))
    np.testing.assert_allclose(efmins, [-10.0])
    np.testing.assert_allclose(efmaxs, [10.0])

    with pytest.raises(ValueError, match="Unknown data source"):
        list(create_data_stream("nonexistent_source"))

    det = get_detector("half_space_trees", feature_mins=fmins, feature_maxes=fmaxs, random_state=0)
    assert det.__class__.__name__ == "HalfSpaceTrees"
    with pytest.raises(ValueError, match="Unknown detector"):
        get_detector("unknown_detector")

    cal = get_calibrator("conformal", window_size=10)
    assert cal.__class__.__name__ == "ConformalProbabilityCalibrator"
    with pytest.raises(ValueError, match="Unknown calibrator"):
        get_calibrator("unknown_calibrator")
