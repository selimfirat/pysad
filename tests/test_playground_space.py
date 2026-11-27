"""Tests for the Hugging Face Space interactive playground engine and Gradio app."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pytest

from spaces.playground import engine


def test_default_detectors_match_odds_sota_trio():
    """Playground defaults to top ODDS benchmark detectors while keeping LODA explainability."""
    assert engine.DEFAULT_DETECTORS == ["LocalOutlierProbability", "RSHash", "LODA"]


def test_synthetic_stream_and_all_detectors_with_both_calibrators():
    """All supported detectors run cleanly under Conformal and SPOT calibration."""
    res1 = engine.run_playground(
        stream_name=engine.STREAM_SYNTHETIC,
        detectors=["LocalOutlierProbability", "RSHash", "LODA"],
        calibrator_name=engine.CALIBRATOR_CONFORMAL,
        threshold=0.985,
        max_points=300,
    )
    assert len(res1.traces) == 3
    for tr in res1.traces:
        assert len(tr.raw_scores) == 300
        assert len(tr.calibrated_scores) == 300
        assert np.all(np.isfinite(tr.calibrated_scores))
        assert 0.0 <= float(tr.calibrated_scores.min()) <= float(tr.calibrated_scores.max()) <= 1.0
        assert tr.us_per_point > 0.0
        assert tr.auroc is not None and tr.auroc > 0.5

    res2 = engine.run_playground(
        stream_name=engine.STREAM_SYNTHETIC,
        detectors=["SmartSifter", "RobustRandomCutForest", "HalfSpaceTrees"],
        calibrator_name=engine.CALIBRATOR_CONFORMAL,
        threshold=0.95,
        max_points=300,
    )
    assert len(res2.traces) == 3
    for tr in res2.traces:
        assert len(tr.calibrated_scores) == 300
        assert np.all(np.isfinite(tr.calibrated_scores))
        assert tr.us_per_point > 0.0

    res3 = engine.run_playground(
        stream_name=engine.STREAM_SYNTHETIC,
        detectors=["KNNCAD", "SeasonalHybridESD", "MedianAbsoluteDeviation"],
        calibrator_name=engine.CALIBRATOR_CONFORMAL,
        threshold=0.95,
        max_points=300,
    )
    assert len(res3.traces) == 3
    assert "(sensor_1)" in res3.traces[0].display_name
    assert "(sensor_1)" in res3.traces[1].display_name
    assert "(sensor_1)" in res3.traces[2].display_name

    res4 = engine.run_playground(
        stream_name=engine.STREAM_SYNTHETIC,
        detectors=["xStream"],
        calibrator_name=engine.CALIBRATOR_CONFORMAL,
        threshold=0.95,
        max_points=300,
    )
    assert len(res4.traces) == 1
    assert np.all(np.isfinite(res4.traces[0].calibrated_scores))

    res_spot = engine.run_playground(
        stream_name=engine.STREAM_SYNTHETIC,
        detectors=["LODA", "MedianAbsoluteDeviation"],
        calibrator_name=engine.CALIBRATOR_SPOT,
        threshold=0.985,
        max_points=300,
    )
    assert len(res_spot.traces) == 2
    for tr in res_spot.traces:
        assert len(tr.calibrated_scores) == 300
        assert np.all(np.isfinite(tr.calibrated_scores))
        assert 0.0 <= float(tr.calibrated_scores.min()) <= float(tr.calibrated_scores.max()) <= 1.0


def test_spot_calibrator_handles_extreme_slider_thresholds():
    """SPOT calibration remains valid across every threshold slider boundary."""
    for thresh in (0.900, 0.950, 0.985, 0.999):
        res = engine.run_playground(
            stream_name=engine.STREAM_SYNTHETIC,
            detectors=["LODA"],
            calibrator_name=engine.CALIBRATOR_SPOT,
            threshold=thresh,
            max_points=300,
        )
        assert len(res.traces[0].calibrated_scores) == 300
        assert np.all(np.isfinite(res.traces[0].calibrated_scores))


def test_live_anomaly_injection_is_detected_and_attributed_by_loda():
    """Injecting a spike raises alerts at that step and ranks the target feature highest."""
    target_step = 250
    res = engine.run_playground(
        stream_name=engine.STREAM_SYNTHETIC,
        detectors=["LODA", "HalfSpaceTrees"],
        calibrator_name=engine.CALIBRATOR_CONFORMAL,
        threshold=0.98,
        max_points=350,
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
        max_points=300,
    )
    assert res.stream.is_univariate
    assert res.stream.timestamps is not None
    assert len(res.explanation.contributions) == 1
    assert np.all(np.isfinite(res.explanation.contributions))
    assert "Univariate stream" in res.explanation.note


def test_csv_upload_parser_handles_labels_text_and_missing_values():
    """CSV parser extracts numeric features, optional binary label column, and fills gaps."""
    lines = ["timestamp,cpu,mem,label"]
    for i in range(40):
        cpu_val = "" if i == 15 else f"{0.2 + 0.01 * i:.3f}"
        label_val = "1" if i == 30 else "0"
        lines.append(f"2026-01-01T00:{i:02d}:00,{cpu_val},{50 + i},{label_val}")
    csv_bytes = "\n".join(lines).encode("utf-8")

    stream = engine.parse_csv_stream(csv_bytes)
    assert stream.num_points == 40
    assert stream.feature_names == ["cpu", "mem"]
    assert stream.y is not None and int(stream.y.sum()) == 1
    assert np.all(np.isfinite(stream.X))

    loaded = engine.load_stream(engine.STREAM_CSV, csv_data=csv_bytes)
    assert loaded.num_points == 40
    assert loaded.feature_names == ["cpu", "mem"]


def test_generated_pipeline_code_is_valid_python():
    """Generated pipeline code parses cleanly for every detector and uses v0.8+ PySAD APIs."""
    for cal in engine.CALIBRATOR_CHOICES:
        for det in engine.DETECTOR_CHOICES:
            code = engine.generate_pipeline_code(
                stream_name=engine.STREAM_SYNTHETIC,
                detectors=[det, "LODA"] if det != "LODA" else ["LODA"],
                calibrator_name=cal,
                threshold=0.985,
                calibrator_window=350,
                warmup=200,
            )
            ast.parse(code)
            assert "fit_score_partial" in code
            assert " | " in code


def test_generated_pipeline_code_runs_end_to_end():
    """Generated snippet executes cleanly, including the LODA attribution block on alerts."""
    import numpy as np

    code = engine.generate_pipeline_code(
        stream_name=engine.STREAM_SYNTHETIC,
        detectors=["LODA"],
        calibrator_name=engine.CALIBRATOR_CONFORMAL,
        threshold=0.985,
        calibrator_window=350,
        warmup=100,
    )
    ast.parse(code)
    ns = {"np": np}
    exec(code, ns)
    assert isinstance(ns["alerts"], list)


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
        num_points=300,
        detectors=engine.DEFAULT_DETECTORS,
        calibrator_name=engine.CALIBRATOR_CONFORMAL,
        threshold=0.985,
        calibrator_window=120,
        primary_feature="sensor_1",
        injected=[],
        inspect_step=None,
    )
    status_note, rows, stream_fig, explain_fig, explain_note, code, *_ = out
    assert "4 synthetic sensors" in status_note
    assert len(rows) == 3
    assert len(stream_fig.data) >= 4
    assert len(explain_fig.data) == 1
    assert "LODA.explain_partial()" in explain_note
    ast.parse(code)


def test_arrhythmia_stream_and_json_bridge():
    """Bundled Arrhythmia stream loads offline and produces a valid JSON payload for Pyodide."""
    import json

    raw = engine.run_playground_json(
        stream_name=engine.STREAM_ARRHYTHMIA,
        detectors=engine.DEFAULT_DETECTORS,
        calibrator_name=engine.CALIBRATOR_CONFORMAL,
        threshold=0.985,
        max_points=452,
    )
    payload = json.loads(raw)
    assert payload["stream"]["num_points"] == 452
    assert payload["stream"]["num_features"] == 274
    assert len(payload["stream"]["plotted_features"]) <= 4
    assert len(payload["traces"]) == 3
    assert payload["traces"][0]["auroc"] is not None
    assert len(payload["explanation"]["feature_names"]) <= 12
    ast.parse(payload["pipeline_code"])


def test_static_space_bundle_metadata():
    """Static HF Space files exist, declare sdk: static, and default to ODDS SOTA detectors."""
    space_dir = Path(__file__).resolve().parents[1] / "spaces" / "playground"
    readme = (space_dir / "README.md").read_text(encoding="utf-8")
    index_html = (space_dir / "index.html").read_text(encoding="utf-8")
    assert "sdk: static" in readme
    assert (space_dir / "sw.js").is_file()
    assert "./sw.js" in index_html
    worker_js = (space_dir / "worker.js").read_text(encoding="utf-8")
    assert "./worker.js" in index_html
    assert "./engine.py" in worker_js
    assert "./data/nab/realKnownCause/nyc_taxi.csv" in worker_js
    assert "./data/arrhythmia.mat" in worker_js
    assert "./initial_snapshot.json" in index_html
    assert "plotly-basic-" in index_html
    assert "unpackArchive" in worker_js
    for default_detector in ("LocalOutlierProbability", "RSHash", "LODA"):
        assert f'value="{default_detector}" checked' in index_html


def test_service_worker_caches_essential_assets():
    """Service Worker precaches the assets needed for offline playback."""
    import re

    sw_js = (Path(__file__).resolve().parents[1] / "spaces/playground/sw.js").read_text(
        encoding="utf-8"
    )
    assert "PRECACHE_URLs" in sw_js
    assert "./worker.js" in sw_js
    assert "./engine.py" in sw_js
    assert "./manifest.json" in sw_js
    assert "./initial_snapshot.json" in sw_js
    assert "./data/arrhythmia.mat" in sw_js
    assert "caches.open" in sw_js
    assert "cache.addAll" in sw_js
    assert "fetch" in sw_js
    assert any(
        urlparse(url).hostname == "cdn.jsdelivr.net"
        for url in re.findall(r'"(https://[^"]+)"', sw_js)
    )


def test_initial_snapshot_json_omits_host_timings():
    """Precomputed snapshot provides deterministic scores/plots without host timing leakage."""
    import json

    payload = json.loads(engine.build_initial_snapshot_json())
    assert payload["precomputed"] is True
    assert [tr["name"] for tr in payload["traces"]] == engine.DEFAULT_DETECTORS
    for tr in payload["traces"]:
        assert tr["us_per_point"] is None
        assert tr["points_per_second"] is None
        assert tr["auroc"] is not None and tr["auroc"] > 0.5


def test_playground_files_have_no_em_or_en_dashes():
    """Playground assets avoid em dashes and en dashes in prose, titles, and placeholders."""
    space_dir = Path(__file__).resolve().parents[1] / "spaces" / "playground"
    forbidden = ("\u2014", "&mdash;", "\u2013", "&ndash;")
    for filename in ("index.html", "engine.py", "app.py", "README.md"):
        content = (space_dir / filename).read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in content, f"Found {token!r} in {filename}"


@pytest.mark.parametrize("calibrator", engine.CALIBRATOR_CHOICES)
def test_live_batches_match_prequential_run(calibrator):
    """Live batches return the same scores and alerts as a full run."""
    options = dict(
        stream_name=engine.STREAM_SYNTHETIC,
        detectors=["LODA", "HalfSpaceTrees"],
        calibrator_name=calibrator,
        max_points=300,
        injected=[{"index": 250, "feature_idx": 0, "magnitude_sigma": 7.0}],
    )
    expected = engine.run_playground(**options)
    session = engine.LiveSession(**options)
    rows = []
    while True:
        batch = session.next_batch(8)
        rows.extend(batch["points"])
        if batch["done"]:
            break
    assert len(rows) == 300
    assert [row["step"] for row in rows] == list(range(300))
    np.testing.assert_allclose([row["value"] for row in rows], expected.stream.X[:, 0])
    for i, trace in enumerate(expected.traces):
        np.testing.assert_allclose([row["scores"][i] for row in rows], trace.calibrated_scores)
        assert [row["alerts"][i] for row in rows] == trace.alerts.tolist()
    assert session.next_batch(8) == {"points": [], "done": True}
    restarted = engine.LiveSession(**options)
    assert restarted.next_batch(8)["points"] == rows[:8]


def test_share_link_settings_round_trip_reproduces_identical_result():
    """Settings stored in a share link round-trip through the engine to the same scores and alerts."""
    settings = dict(
        stream_name=engine.STREAM_SYNTHETIC,
        detectors=["RSHash", "HalfSpaceTrees", "LODA"],
        calibrator_name=engine.CALIBRATOR_SPOT,
        threshold=0.970,
        calibrator_window=400,
        max_points=500,
        injected=[{"index": 250, "feature_idx": 1, "magnitude_sigma": 7.0}],
        inspect_step=250,
        primary_feature_name="sensor_2",
    )
    first = engine.run_playground(**settings)

    # Serialize the settings exactly as the share link does (updateUrlParams), then parse
    # them back exactly as the page does (parseUrlParams), and rebuild the engine options.
    params = {
        "stream": settings["stream_name"],
        "detectors": ",".join(settings["detectors"]),
        "calibrator": settings["calibrator_name"],
        "threshold": f"{settings['threshold']:.3f}",
        "calWindow": str(settings["calibrator_window"]),
        "points": str(settings["max_points"]),
        "feature": settings["primary_feature_name"],
        "injected": json.dumps(settings["injected"]),
        "inspect": str(settings["inspect_step"]),
    }
    restored_injected = []
    parsed_injected = json.loads(params["injected"] or "[]")
    if isinstance(parsed_injected, list):
        for item in parsed_injected:
            if (
                item
                and isinstance(item.get("index"), int)
                and item["index"] >= 0
                and isinstance(item.get("feature_idx"), int)
                and item["feature_idx"] >= 0
                and isinstance(item.get("magnitude_sigma"), (int, float))
                and np.isfinite(item["magnitude_sigma"])
            ):
                restored_injected.append(item)

    restored = dict(
        stream_name=params["stream"],
        detectors=params["detectors"].split(","),
        calibrator_name=params["calibrator"],
        threshold=float(params["threshold"]),
        calibrator_window=int(params["calWindow"]),
        max_points=int(params["points"]),
        injected=restored_injected,
        inspect_step=int(params["inspect"]),
        primary_feature_name=params["feature"],
    )
    second = engine.run_playground(**restored)

    assert len(first.traces) == len(second.traces)
    for first_trace, second_trace in zip(first.traces, second.traces, strict=True):
        np.testing.assert_array_equal(first_trace.calibrated_scores, second_trace.calibrated_scores)
        assert first_trace.alerts.tolist() == second_trace.alerts.tolist()
        assert first_trace.auroc == second_trace.auroc
        assert first_trace.aupr == second_trace.aupr
        assert first_trace.precision == second_trace.precision
        assert first_trace.recall == second_trace.recall
        assert first_trace.injected_caught == second_trace.injected_caught
        assert first_trace.injected_total == second_trace.injected_total
    np.testing.assert_array_equal(first.explanation.contributions, second.explanation.contributions)
    assert first.explanation.step == second.explanation.step
    assert first.pipeline_code == second.pipeline_code
    assert first.status_note == second.status_note


def test_run_playground_rejects_unknown_stream():
    """An unknown stream name raises a clear ValueError instead of a silent failure."""
    with pytest.raises(ValueError, match="Unknown stream choice"):
        engine.run_playground(
            stream_name="Not a real stream",
            detectors=["LODA"],
            calibrator_name=engine.CALIBRATOR_CONFORMAL,
            max_points=200,
        )


def test_run_playground_rejects_unsupported_calibrator():
    """An unsupported calibrator name raises a clear ValueError."""
    with pytest.raises(ValueError, match="Unsupported calibrator"):
        engine.run_playground(
            stream_name=engine.STREAM_SYNTHETIC,
            detectors=["LODA"],
            calibrator_name="Not a real calibrator",
            max_points=200,
        )


def test_load_stream_csv_without_csv_data_raises():
    """Upload CSV mode without CSV data raises a clear ValueError."""
    with pytest.raises(ValueError, match="Upload a CSV file"):
        engine.load_stream(engine.STREAM_CSV, csv_data=None)


def test_parse_csv_stream_rejects_short_csv():
    """A CSV with fewer than 10 data rows raises a clear ValueError."""
    with pytest.raises(ValueError, match="at least 10 non-empty data rows"):
        engine.parse_csv_stream(b"cpu,mem\n0.1,0.2\n")


def test_parse_csv_stream_rejects_no_numeric_columns():
    """A CSV with only text columns raises a clear ValueError."""
    rows = ["name,city"]
    for i in range(12):
        rows.append(f"person_{i},city_{i}")
    csv_bytes = "\n".join(rows).encode("utf-8")
    with pytest.raises(ValueError, match="no numeric feature columns"):
        engine.parse_csv_stream(csv_bytes)
