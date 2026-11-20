"""Gradio + Plotly interactive Hugging Face Space playground for PySAD."""

from __future__ import annotations

import os
from typing import Any

import gradio as gr
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from spaces.playground import engine
except ImportError:
    import engine  # Deployed standalone at Hugging Face Space root.

_DETECTOR_COLORS = ("#0741ad", "#0f766e", "#b45309")
_FEATURE_COLORS = ("#3f6fd0", "#64748b", "#8aa8e6", "#94a3b8")
_GRADIO_MAJOR = int(gr.__version__.split(".", 1)[0])

CUSTOM_CSS = """
.pysad-header {
  padding: 1rem 1.25rem;
  border: 1px solid #e2ddcd;
  border-radius: 12px;
  background: linear-gradient(135deg, #fffef7 0%, #efebdf 100%);
  margin-bottom: 0.75rem;
}
.pysad-header h1 {
  margin: 0 0 0.25rem 0;
  color: #0e1a33;
  font-size: 1.55rem;
}
.pysad-header p {
  margin: 0;
  color: #5a6478;
  font-size: 0.95rem;
}
.pysad-links {
  margin-top: 0.5rem;
  font-size: 0.88rem;
}
.pysad-links a {
  color: #0741ad;
  font-weight: 600;
  text-decoration: none;
  margin-right: 1rem;
}
"""

METRICS_HEADERS = [
    "Detector",
    "AUROC",
    "AUPR",
    "Alerts",
    "Precision",
    "Recall",
    "Injected",
    "µs / pt",
    "pts / s",
]


def _fmt_metric(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "—"


def _metrics_rows(result: engine.PlaygroundResult) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for tr in result.traces:
        inj_str = f"{tr.injected_caught} / {tr.injected_total}" if tr.injected_total > 0 else "—"
        rows.append(
            [
                tr.display_name,
                _fmt_metric(tr.auroc),
                _fmt_metric(tr.aupr),
                int(tr.alerts.sum()),
                _fmt_metric(tr.precision),
                _fmt_metric(tr.recall),
                inj_str,
                f"{tr.us_per_point:,.1f}",
                f"{tr.points_per_second:,.0f}",
            ]
        )
    return rows


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 14, "color": "#5a6478"},
    )
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#fffef7",
        plot_bgcolor="#fffef7",
        height=360,
        margin={"l": 40, "r": 20, "t": 30, "b": 30},
    )
    return fig


def _build_stream_figure(result: engine.PlaygroundResult) -> go.Figure:
    """Render synchronized stream features and calibrated detector scores."""
    stream = result.stream
    n = stream.num_points
    x_axis = stream.timestamps if stream.timestamps is not None else list(range(n))

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.14,
        row_heights=[0.48, 0.52],
        subplot_titles=(
            f"Stream Values — {stream.name}",
            f"Calibrated Anomaly Scores — {result.calibrator_name}",
        ),
    )

    # Shade warm-up period on both panels.
    warmup_x1 = x_axis[min(result.warmup, n - 1)]
    for row_idx in (1, 2):
        fig.add_vrect(
            x0=x_axis[0],
            x1=warmup_x1,
            fillcolor="rgba(148, 163, 184, 0.14)",
            line_width=0,
            row=row_idx,
            col=1,
        )

    # Highlight up to 4 most informative / injected features on top panel.
    feature_indices = list(range(min(4, stream.num_features)))
    if stream.num_features > 4:
        # Prefer top-contributing features from LODA.explain_partial() alongside first features.
        ranked = np.argsort(result.explanation.contributions)[::-1][:4].tolist()
        feature_indices = sorted(set(ranked))

    for pos, feat_idx in enumerate(feature_indices):
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=stream.X[:, feat_idx],
                mode="lines",
                name=stream.feature_names[feat_idx],
                line={"width": 1.4, "color": _FEATURE_COLORS[pos % len(_FEATURE_COLORS)]},
                opacity=0.88,
            ),
            row=1,
            col=1,
        )

    # Overlay ground-truth anomaly markers on the primary feature.
    if stream.y is not None and stream.y.any():
        gt_idx = np.flatnonzero(stream.y == 1)
        y_vals = stream.X[gt_idx, feature_indices[0]]
        fig.add_trace(
            go.Scatter(
                x=[x_axis[i] for i in gt_idx],
                y=y_vals,
                mode="markers",
                name="Labeled anomaly",
                marker={
                    "color": "#fc3636",
                    "size": 6,
                    "symbol": "circle-open",
                    "line": {"width": 2},
                },
            ),
            row=1,
            col=1,
        )

    # Plot each detector's calibrated scores and alert markers.
    for idx, tr in enumerate(result.traces):
        color = _DETECTOR_COLORS[idx % len(_DETECTOR_COLORS)]
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=tr.calibrated_scores,
                mode="lines",
                name=tr.display_name,
                line={"width": 1.8, "color": color},
            ),
            row=2,
            col=1,
        )
        alert_idx = np.flatnonzero(tr.alerts)
        if len(alert_idx) > 0:
            fig.add_trace(
                go.Scatter(
                    x=[x_axis[i] for i in alert_idx],
                    y=tr.calibrated_scores[alert_idx],
                    mode="markers",
                    name=f"{tr.name} alerts",
                    marker={"color": "#fc3636", "size": 6, "symbol": "diamond"},
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

    # Threshold line and vertical cue lines for injected & inspected steps.
    fig.add_hline(
        y=result.threshold,
        line_dash="dash",
        line_color="#fc3636",
        line_width=1.5,
        annotation_text=f"threshold = {result.threshold:.3f}",
        annotation_position="top right",
        row=2,
        col=1,
    )

    for inj_i in np.flatnonzero(result.injected_mask):
        for row_idx in (1, 2):
            fig.add_vline(
                x=x_axis[int(inj_i)],
                line_dash="dot",
                line_color="#fc3636",
                line_width=1.4,
                row=row_idx,
                col=1,
            )

    inspect_x = x_axis[result.explanation.step]
    for row_idx in (1, 2):
        fig.add_vline(
            x=inspect_x,
            line_dash="dash",
            line_color="#0741ad",
            line_width=1.4,
            row=row_idx,
            col=1,
        )

    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Calibrated score", range=[-0.02, 1.05], row=2, col=1)
    fig.update_xaxes(
        title_text="Stream position" if stream.timestamps is None else "Timestamp", row=2, col=1
    )
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#fffef7",
        plot_bgcolor="#ffffff",
        font={"family": "Inter, ui-sans-serif, sans-serif", "color": "#0e1a33", "size": 12},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.09, "x": 0.0},
        height=560,
        margin={"l": 56, "r": 24, "t": 96, "b": 48},
    )
    return fig


def _build_explanation_figure(result: engine.PlaygroundResult) -> go.Figure:
    """Render sorted horizontal bar chart of feature contributions at the inspected step."""
    exp = result.explanation
    order = np.argsort(exp.contributions)
    # Keep top 12 features on wide datasets (e.g. 30-feature Credit Card Fraud) for clarity.
    if len(order) > 12:
        order = order[-12:]
    names = [exp.feature_names[i] for i in order]
    vals = [float(exp.contributions[i]) for i in order]
    colors = ["#fc3636" if i >= len(order) - 2 and v > 0 else "#0741ad" for i, v in enumerate(vals)]
    x_min = min(min(vals, default=0.0), 0.0)
    x_max = max(max(vals, default=0.0), 0.0)
    span = max(x_max - x_min, 1.0)

    fig = go.Figure(
        go.Bar(
            x=vals,
            y=names,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.2f}" for v in vals],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=f"Feature Attribution at Step {exp.step} (LODA raw score: {exp.loda_raw_score:.2f})",
        xaxis={
            "title": "Contribution (LODA.explain_partial t-statistic)",
            "range": [x_min - 0.18 * span, x_max + 0.15 * span],
            "zeroline": True,
            "zerolinecolor": "#cbd5e1",
        },
        template="plotly_white",
        paper_bgcolor="#fffef7",
        plot_bgcolor="#ffffff",
        font={"family": "Inter, ui-sans-serif, sans-serif", "color": "#0e1a33", "size": 12},
        height=290,
        margin={"l": 110, "r": 48, "t": 46, "b": 42},
    )
    return fig


def _format_injected_summary(injected: list[dict[str, Any]], feature_names: list[str]) -> str:
    if not injected:
        return "Injected anomalies: none"
    parts = [
        f"t={item['index']} ({feature_names[int(item['feature_idx'])]}, {item['magnitude_sigma']:+.1f}σ)"
        for item in injected
        if int(item["feature_idx"]) < len(feature_names)
    ]
    return "Injected anomalies: " + "; ".join(parts)


def evaluate_ui(
    stream_name: str,
    csv_file: bytes | str | None,
    num_points: int,
    detectors: list[str],
    calibrator_name: str,
    threshold: float,
    calibrator_window: int,
    primary_feature: str | None,
    injected: list[dict[str, Any]],
    inspect_step: int | None,
) -> tuple[Any, ...]:
    """Run the playground engine and return all updated Gradio component values."""
    try:
        result = engine.run_playground(
            stream_name=stream_name,
            detectors=detectors,
            calibrator_name=calibrator_name,
            threshold=float(threshold),
            calibrator_window=int(calibrator_window),
            max_points=int(num_points),
            csv_data=csv_file,
            injected=injected,
            inspect_step=inspect_step,
            primary_feature_name=primary_feature,
        )
    except Exception as exc:
        msg = str(exc)
        empty = _empty_figure(msg)
        code = engine.generate_pipeline_code(
            stream_name=stream_name,
            detectors=detectors or ["LODA"],
            calibrator_name=calibrator_name,
            threshold=float(threshold),
            calibrator_window=int(calibrator_window),
            warmup=min(200, int(num_points) // 6),
        )
        return (
            f"Status: {msg}",
            [],
            empty,
            _empty_figure("Select or upload a valid stream to view feature explanations."),
            msg,
            code,
            gr.update(),
            gr.update(),
            gr.update(),
            _format_injected_summary(injected, ["feature_0"]),
            gr.update(visible=(stream_name == engine.STREAM_CSV)),
        )

    feat_choices = result.stream.feature_names
    selected_feat = primary_feature if primary_feature in feat_choices else feat_choices[0]
    max_step = max(1, result.stream.num_points - 1)
    default_inj_step = min(max_step, max(result.warmup + 25, result.stream.num_points // 2))

    return (
        result.status_note,
        _metrics_rows(result),
        _build_stream_figure(result),
        _build_explanation_figure(result),
        result.explanation.note,
        result.pipeline_code,
        gr.update(choices=feat_choices, value=selected_feat),
        gr.update(maximum=max_step, value=min(max_step, default_inj_step)),
        gr.update(maximum=max_step, value=result.explanation.step),
        _format_injected_summary(injected, feat_choices),
        gr.update(visible=(stream_name == engine.STREAM_CSV)),
    )


def on_stream_change(
    stream_name: str,
    csv_file: bytes | str | None,
    num_points: int,
    detectors: list[str],
    calibrator_name: str,
    threshold: float,
    calibrator_window: int,
) -> tuple[Any, ...]:
    """Reset injected anomalies and feature selection when switching datasets."""
    empty_injected: list[dict[str, Any]] = []
    outputs = evaluate_ui(
        stream_name=stream_name,
        csv_file=csv_file,
        num_points=num_points,
        detectors=detectors,
        calibrator_name=calibrator_name,
        threshold=threshold,
        calibrator_window=calibrator_window,
        primary_feature=None,
        injected=empty_injected,
        inspect_step=None,
    )
    return (empty_injected, *outputs)


def on_inject_anomaly(
    stream_name: str,
    csv_file: bytes | str | None,
    num_points: int,
    detectors: list[str],
    calibrator_name: str,
    threshold: float,
    calibrator_window: int,
    primary_feature: str | None,
    injected: list[dict[str, Any]],
    inject_step: int,
    inject_sigma: float,
) -> tuple[Any, ...]:
    """Inject a live anomaly into the active stream and focus explanation on it."""
    try:
        stream = engine.load_stream(stream_name, max_points=int(num_points), csv_data=csv_file)
        feat_names = stream.feature_names
    except Exception:
        feat_names = ["feature_0"]
    feat_idx = feat_names.index(primary_feature) if primary_feature in feat_names else 0
    step_clamped = int(max(0, min(int(inject_step), int(num_points) - 1)))

    updated_injected = [
        *injected,
        {
            "index": step_clamped,
            "feature_idx": feat_idx,
            "magnitude_sigma": float(inject_sigma),
        },
    ]
    outputs = evaluate_ui(
        stream_name=stream_name,
        csv_file=csv_file,
        num_points=num_points,
        detectors=detectors,
        calibrator_name=calibrator_name,
        threshold=threshold,
        calibrator_window=calibrator_window,
        primary_feature=primary_feature,
        injected=updated_injected,
        inspect_step=step_clamped,
    )
    return (updated_injected, *outputs)


def on_clear_injected(
    stream_name: str,
    csv_file: bytes | str | None,
    num_points: int,
    detectors: list[str],
    calibrator_name: str,
    threshold: float,
    calibrator_window: int,
    primary_feature: str | None,
) -> tuple[Any, ...]:
    """Clear all injected anomalies and re-evaluate."""
    empty_injected: list[dict[str, Any]] = []
    outputs = evaluate_ui(
        stream_name=stream_name,
        csv_file=csv_file,
        num_points=num_points,
        detectors=detectors,
        calibrator_name=calibrator_name,
        threshold=threshold,
        calibrator_window=calibrator_window,
        primary_feature=primary_feature,
        injected=empty_injected,
        inspect_step=None,
    )
    return (empty_injected, *outputs)


def _build_theme() -> gr.themes.Soft:
    return gr.themes.Soft(
        primary_hue="blue",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
    )


def build_demo() -> gr.Blocks:
    """Create and wire the Gradio Blocks application."""
    blocks_kwargs: dict[str, Any] = {
        "title": "PySAD — Streaming Anomaly Detection Playground",
    }
    if _GRADIO_MAJOR < 6:
        blocks_kwargs["theme"] = _build_theme()
        blocks_kwargs["css"] = CUSTOM_CSS

    with gr.Blocks(**blocks_kwargs) as demo:
        injected_state = gr.State([])

        gr.HTML(
            """
            <div class="pysad-header">
              <h1>PySAD — Streaming Anomaly Detection Playground</h1>
              <p>Score and learn from every point in real time, compare detectors side by side, tune calibrated alerts, inject live anomalies, and inspect feature explanations.</p>
              <div class="pysad-links">
                <a href="https://github.com/selimfirat/pysad" target="_blank">GitHub</a>
                <a href="https://pysad.readthedocs.io/en/stable/" target="_blank">Documentation</a>
                <a href="https://pysad.readthedocs.io/en/stable/benchmark.html" target="_blank">Benchmarks</a>
                <a href="https://pypi.org/project/pysad/" target="_blank">PyPI</a>
              </div>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=4, min_width=320):
                stream_dd = gr.Dropdown(
                    label="1. Data Stream",
                    choices=engine.STREAM_CHOICES,
                    value=engine.STREAM_SYNTHETIC,
                )
                csv_file = gr.File(
                    label="Upload CSV (numeric columns + optional binary label column)",
                    file_types=[".csv"],
                    type="binary",
                    visible=False,
                )
                num_points = gr.Slider(
                    label="Stream points",
                    minimum=400,
                    maximum=2500,
                    step=100,
                    value=1200,
                )

                detectors_cb = gr.CheckboxGroup(
                    label="2. Streaming Detectors (compare 1–3 side by side)",
                    choices=engine.DETECTOR_CHOICES,
                    value=engine.DEFAULT_DETECTORS,
                )

                calibrator_rb = gr.Radio(
                    label="3. Score Calibrator",
                    choices=engine.CALIBRATOR_CHOICES,
                    value=engine.CALIBRATOR_CONFORMAL,
                )
                threshold_sl = gr.Slider(
                    label="Alert threshold (calibrated [0, 1])",
                    minimum=0.900,
                    maximum=0.999,
                    step=0.001,
                    value=0.985,
                )
                cal_window_sl = gr.Slider(
                    label="Conformal window size",
                    minimum=100,
                    maximum=1000,
                    step=50,
                    value=350,
                )

                with gr.Accordion("4. Live Anomaly Injection & Feature Explanation", open=True):
                    feature_dd = gr.Dropdown(
                        label="Target feature (for injection & univariate detectors)",
                        choices=["sensor_1", "sensor_2", "sensor_3", "sensor_4"],
                        value="sensor_1",
                    )
                    inject_step_sl = gr.Slider(
                        label="Inject at stream step t",
                        minimum=60,
                        maximum=1199,
                        step=1,
                        value=720,
                    )
                    inject_sigma_sl = gr.Slider(
                        label="Injection magnitude (standard deviations σ)",
                        minimum=-8.0,
                        maximum=8.0,
                        step=0.5,
                        value=5.5,
                    )
                    with gr.Row():
                        inject_btn = gr.Button("Inject Anomaly", variant="primary")
                        clear_btn = gr.Button("Clear Injected", variant="secondary")
                    injected_md = gr.Markdown("Injected anomalies: none")

                    inspect_step_sl = gr.Slider(
                        label="Explain step t (LODA.explain_partial)",
                        minimum=60,
                        maximum=1199,
                        step=1,
                        value=288,
                    )

            with gr.Column(scale=8, min_width=540):
                status_md = gr.Markdown("Loading stream...")
                metrics_df = gr.Dataframe(
                    label="Live Prequential Metrics & Throughput",
                    headers=METRICS_HEADERS,
                    column_widths=["25%", "8%", "8%", "8%", "10%", "8%", "11%", "11%", "11%"],
                    interactive=False,
                )
                stream_plot = gr.Plot(label="Stream & Calibrated Alert Scores")
                explain_plot = gr.Plot(label="Per-Feature Attribution (LODA.explain_partial)")
                explain_md = gr.Markdown("")
                code_box = gr.Code(
                    label="Reproducible PySAD Pipeline Code",
                    language="python",
                )

        ui_outputs = [
            status_md,
            metrics_df,
            stream_plot,
            explain_plot,
            explain_md,
            code_box,
            feature_dd,
            inject_step_sl,
            inspect_step_sl,
            injected_md,
            csv_file,
        ]
        eval_inputs = [
            stream_dd,
            csv_file,
            num_points,
            detectors_cb,
            calibrator_rb,
            threshold_sl,
            cal_window_sl,
            feature_dd,
            injected_state,
            inspect_step_sl,
        ]

        demo.load(
            fn=evaluate_ui,
            inputs=[
                stream_dd,
                csv_file,
                num_points,
                detectors_cb,
                calibrator_rb,
                threshold_sl,
                cal_window_sl,
                feature_dd,
                injected_state,
                gr.State(None),
            ],
            outputs=ui_outputs,
        )

        for trigger in (stream_dd, csv_file):
            trigger.change(
                fn=on_stream_change,
                inputs=[
                    stream_dd,
                    csv_file,
                    num_points,
                    detectors_cb,
                    calibrator_rb,
                    threshold_sl,
                    cal_window_sl,
                ],
                outputs=[injected_state, *ui_outputs],
            )

        num_points.release(
            fn=on_stream_change,
            inputs=[
                stream_dd,
                csv_file,
                num_points,
                detectors_cb,
                calibrator_rb,
                threshold_sl,
                cal_window_sl,
            ],
            outputs=[injected_state, *ui_outputs],
        )

        for control in (detectors_cb, calibrator_rb):
            control.change(fn=evaluate_ui, inputs=eval_inputs, outputs=ui_outputs)

        for slider in (threshold_sl, cal_window_sl, inspect_step_sl):
            slider.release(fn=evaluate_ui, inputs=eval_inputs, outputs=ui_outputs)

        inject_btn.click(
            fn=on_inject_anomaly,
            inputs=[
                stream_dd,
                csv_file,
                num_points,
                detectors_cb,
                calibrator_rb,
                threshold_sl,
                cal_window_sl,
                feature_dd,
                injected_state,
                inject_step_sl,
                inject_sigma_sl,
            ],
            outputs=[injected_state, *ui_outputs],
        )

        clear_btn.click(
            fn=on_clear_injected,
            inputs=[
                stream_dd,
                csv_file,
                num_points,
                detectors_cb,
                calibrator_rb,
                threshold_sl,
                cal_window_sl,
                feature_dd,
            ],
            outputs=[injected_state, *ui_outputs],
        )

    return demo


if __name__ == "__main__":
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    launch_kwargs: dict[str, Any] = {"server_name": server_name, "server_port": server_port}
    if _GRADIO_MAJOR >= 6:
        launch_kwargs["theme"] = _build_theme()
        launch_kwargs["css"] = CUSTOM_CSS
    build_demo().launch(**launch_kwargs)
