"""Tests for the Hugging Face Space interactive playground engine and Gradio app."""

from __future__ import annotations

import ast
import importlib.util

import numpy as np
import pytest

from spaces.playground import engine


def test_synthetic_stream_and_all_detectors_with_both_calibrators():
    """All supported detectors run cleanly under both Conformal and SPOT calibration."""
    for calibrator in engine.CALIBRATOR_CHOICES:
        res1 = engine.run_playground(
            stream_name=engine.STREAM_SYNTHETIC,
            detectors=["LODA", "HalfSpaceTrees", "RobustRandomCutForest"],
            calibrator_name=calibrator,
            threshold=0.985,
            max_points=600,
        )
        assert len(res1.traces) == 3
        for tr in res1.traces:
            assert len(tr.raw_scores) == 600
            assert len(tr.calibrated_scores) == 600
            assert np.all(np.isfinite(tr.calibrated_scores))
            assert (
                0.0 <= float(tr.calibrated_scores.min()) <= float(tr.calibrated_scores.max()) <= 1.0
            )
            assert tr.us_per_point > 0.0
            assert tr.auroc is not None and tr.auroc > 0.5

        res2 = engine.run_playground(
            stream_name=engine.STREAM_SYNTHETIC,
            detectors=["SeasonalHybridESD", "MedianAbsoluteDeviation", "xStream"],
            calibrator_name=calibrator,
            threshold=0.95,
            max_points=600,
        )
        assert len(res2.traces) == 3
        assert "(sensor_1)" in res2.traces[0].display_name
        assert "(sensor_1)" in res2.traces[1].display_name


def test_spot_calibrator_handles_extreme_slider_thresholds():
    """SPOT calibration remains valid across every threshold slider boundary."""
    for thresh in (0.900, 0.950, 0.985, 0.999):
        res = engine.run_playground(
            stream_name=engine.STREAM_SYNTHETIC,
            detectors=["LODA"],
            calibrator_name=engine.CALIBRATOR_SPOT,
            threshold=thresh,
            max_points=500,
        )
        assert len(res.traces[0].calibrated_scores) == 500
        assert np.all(np.isfinite(res.traces[0].calibrated_scores))


def test_live_anomaly_injection_is_detected_and_attributed_by_loda():
    """Injecting a spike raises alerts at that step and ranks the target feature highest."""
    target_step = 420
    res = engine.run_playground(
        stream_name=engine.STREAM_SYNTHETIC,
        detectors=["LODA", "HalfSpaceTrees"],
        calibrator_name=engine.CALIBRATOR_CONFORMAL,
        threshold=0.98,
        max_points=600,
        injected=[{"index": target_step, "feature_idx": 3, "magnitude_sigma": 7.0}],
        inspect_step=target_step,
    )
    assert res.injected_mask[target_step]
    assert res.explanation.step == target_step
    assert res.traces[0].injected_total == 1
    assert res.traces[0].injected_caught == 1

    # Feature 3 received the +7 sigma injection, so LODA.explain_partial() must flag it top.
    top_idx = int(np.argmax(res.explanation.contributions))
    assert res.stream.feature_names[top_idx] == "sensor_4"


def test_nyc_taxi_univariate_stream_and_graceful_explanation():
    """Bundled NYC Taxi series loads offline and handles univariate explainability cleanly."""
    res = engine.run_playground(
        stream_name=engine.STREAM_NYC_TAXI,
        detectors=["LODA", "SeasonalHybridESD", "MedianAbsoluteDeviation"],
        calibrator_name=engine.CALIBRATOR_CONFORMAL,
        threshold=0.985,
        max_points=600,
    )
    assert res.stream.is_univariate
    assert res.stream.timestamps is not None
    assert len(res.explanation.contributions) == 1
    assert np.all(np.isfinite(res.explanation.contributions))
    assert "Univariate stream" in res.explanation.note


def test_csv_upload_parser_handles_labels_text_and_missing_values(tmp_path):
    """CSV parser extracts numeric features, optional binary label column, and fills gaps."""
    csv_path = tmp_path / "metrics.csv"
    lines = ["timestamp,cpu,mem,label"]
    for i in range(40):
        cpu_val = "" if i == 15 else f"{0.2 + 0.01 * i:.3f}"
        label_val = "1" if i == 30 else "0"
        lines.append(f"2026-01-01T00:{i:02d}:00,{cpu_val},{50 + i},{label_val}")
    csv_path.write_text("\n".join(lines), encoding="utf-8")

    stream = engine.parse_csv_stream(csv_path)
    assert stream.num_points == 40
    assert stream.feature_names == ["cpu", "mem"]
    assert stream.y is not None and int(stream.y.sum()) == 1
    assert np.all(np.isfinite(stream.X))


def test_generated_pipeline_code_is_valid_python():
    """Generated pipeline code parses cleanly and uses current v0.8+ PySAD APIs."""
    for cal in engine.CALIBRATOR_CHOICES:
        for det in (["LODA"], ["HalfSpaceTrees", "LODA"], ["SeasonalHybridESD"]):
            code = engine.generate_pipeline_code(
                stream_name=engine.STREAM_SYNTHETIC,
                detectors=det,
                calibrator_name=cal,
                threshold=0.985,
                calibrator_window=350,
                warmup=200,
            )
            ast.parse(code)
            assert "fit_score_partial" in code
            assert " | " in code


def test_gradio_ui_builds_and_evaluates_if_installed():
    """When Gradio is available locally, build_demo() and evaluate_ui() execute without error."""
    if importlib.util.find_spec("gradio") is None or importlib.util.find_spec("plotly") is None:
        pytest.skip("gradio/plotly are only required in the Space runtime")

    from spaces.playground import app

    demo = app.build_demo()
    assert demo is not None
    out = app.evaluate_ui(
        stream_name=engine.STREAM_SYNTHETIC,
        csv_file=None,
        num_points=500,
        detectors=["LODA", "HalfSpaceTrees"],
        calibrator_name=engine.CALIBRATOR_CONFORMAL,
        threshold=0.985,
        calibrator_window=200,
        primary_feature="sensor_1",
        injected=[],
        inspect_step=None,
    )
    status_note, rows, stream_fig, explain_fig, explain_note, code, *_ = out
    assert "4 synthetic sensors" in status_note
    assert len(rows) == 2
    assert len(stream_fig.data) >= 3
    assert len(explain_fig.data) == 1
    assert "LODA.explain_partial()" in explain_note
    ast.parse(code)
