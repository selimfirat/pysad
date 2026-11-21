"""HTTP/SSE server for the pysad-live web dashboard."""

from __future__ import annotations

import asyncio
import json
import signal
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from typing import Any
from urllib.parse import urlparse

from .terminal import (
    compute_conformal_bounds,
    compute_feature_bounds,
    create_data_stream,
    get_calibrator,
    get_detector,
)


# Global state for the server
class ServerState:
    def __init__(self) -> None:
        self.clients: set[asyncio.Queue] = set()
        self.running = False
        self.stream_thread: threading.Thread | None = None
        self.detector = None
        self.calibrator = None
        self.args: Any = None
        self.step = 0
        self.latest_data: dict[str, Any] = {}
        self.history: list[dict[str, Any]] = []
        self.max_history = 500


STATE = ServerState()


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PySAD Live Anomaly Detection Dashboard</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            background: #0d1b22;
            color: #e0e8eb;
            min-height: 100vh;
            padding: 20px;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 16px 24px;
            background: #142833;
            border: 1px solid #1e3a4a;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .header h1 { font-size: 1.5rem; font-weight: 600; color: #00d4aa; }
        .status { display: flex; gap: 24px; font-size: 0.85rem; }
        .status-item { display: flex; align-items: center; gap: 8px; }
        .status-label { color: #7a919c; }
        .status-value { color: #00d4aa; font-weight: 500; }
        .status-value.alert { color: #ff4444; }
        .status-value.warning { color: #ffaa00; }

        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        @media (max-width: 1200px) { .grid { grid-template-columns: 1fr; } }

        .panel {
            background: #142833;
            border: 1px solid #1e3a4a;
            border-radius: 8px;
            overflow: hidden;
        }
        .panel-header {
            padding: 12px 16px;
            border-bottom: 1px solid #1e3a4a;
            background: #0d1b22;
            font-weight: 600;
            color: #7a919c;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .panel-content { padding: 16px; }

        .chart-container { position: relative; height: 300px; }
        .chart { width: 100%; height: 100%; }

        .features-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; }
        .feature-card {
            background: #0d1b22;
            border: 1px solid #1e3a4a;
            border-radius: 6px;
            padding: 12px;
        }
        .feature-name { color: #7a919c; font-size: 0.75rem; text-transform: uppercase; margin-bottom: 4px; }
        .feature-value { font-size: 1.5rem; font-weight: 600; }
        .feature-sparkline { height: 40px; margin-top: 8px; }

        .score-display {
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .score-main {
            display: flex;
            align-items: center;
            gap: 16px;
            padding: 20px;
            background: #0d1b22;
            border: 1px solid #1e3a4a;
            border-radius: 8px;
        }
        .score-value { font-size: 3rem; font-weight: 700; color: #00d4aa; }
        .score-value.alert { color: #ff4444; }
        .score-bar { flex: 1; height: 24px; background: #0d1b22; border: 1px solid #1e3a4a; border-radius: 12px; overflow: hidden; position: relative; }
        .score-bar-fill { height: 100%; background: linear-gradient(90deg, #00d4aa, #00a388); transition: width 0.1s; }
        .score-bar-fill.alert { background: linear-gradient(90deg, #ff4444, #cc0000); }
        .score-bar-threshold { position: absolute; top: 0; bottom: 0; width: 2px; background: #ffaa00; z-index: 1; }
        .score-info { display: flex; gap: 24px; color: #7a919c; font-size: 0.85rem; }

        .ribbon-chart { height: 150px; }

        .alerts-panel { max-height: 200px; overflow-y: auto; }
        .alert-item { display: flex; justify-content: space-between; padding: 8px 12px; border-bottom: 1px solid #1e3a4a; font-size: 0.8rem; }
        .alert-item:last-child { border-bottom: none; }
        .alert-time { color: #7a919c; }
        .alert-score { color: #00d4aa; }
        .alert-score.triggered { color: #ff4444; font-weight: 600; }
        .alert-label { color: #ffaa00; }

        .controls { display: flex; gap: 12px; margin-top: 20px; }
        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 6px;
            font-weight: 500;
            cursor: pointer;
            transition: background 0.2s;
        }
        .btn-primary { background: #00d4aa; color: #0d1b22; }
        .btn-primary:hover { background: #00a388; }
        .btn-danger { background: #ff4444; color: white; }
        .btn-danger:hover { background: #cc0000; }
        .btn-secondary { background: #1e3a4a; color: #e0e8eb; border: 1px solid #2a5a6a; }
        .btn-secondary:hover { background: #2a5a6a; }

        .legend { display: flex; gap: 16px; margin-top: 12px; font-size: 0.75rem; color: #7a919c; }
        .legend-item { display: flex; align-items: center; gap: 6px; }
        .legend-color { width: 12px; height: 12px; border-radius: 2px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>PySAD Live Anomaly Detection</h1>
        <div class="status">
            <div class="status-item">
                <span class="status-label">Detector:</span>
                <span class="status-value" id="detector-name">-</span>
            </div>
            <div class="status-item">
                <span class="status-label">Calibrator:</span>
                <span class="status-value" id="calibrator-name">-</span>
            </div>
            <div class="status-item">
                <span class="status-label">Step:</span>
                <span class="status-value" id="step-count">0</span>
            </div>
            <div class="status-item">
                <span class="status-label">Rate:</span>
                <span class="status-value" id="point-rate">0 pts/s</span>
            </div>
            <div class="status-item">
                <span class="status-label">Alerts:</span>
                <span class="status-value" id="alert-count">0</span>
            </div>
            <div class="status-item">
                <span class="status-label">Status:</span>
                <span class="status-value" id="connection-status">Connecting...</span>
            </div>
        </div>
    </div>

    <div class="grid">
        <div class="panel">
            <div class="panel-header">Feature Streams</div>
            <div class="panel-content">
                <div class="features-grid" id="features-grid"></div>
            </div>
        </div>

        <div class="panel">
            <div class="panel-header">Anomaly Score & Conformal Ribbon</div>
            <div class="panel-content">
                <div class="score-display">
                    <div class="score-main">
                        <div class="score-value" id="score-value">0.0000</div>
                        <div class="score-bar" style="flex: 1; max-width: 400px;">
                            <div class="score-bar-fill" id="score-bar-fill" style="width: 0%"></div>
                            <div class="score-bar-threshold" id="score-bar-threshold" style="left: 95%"></div>
                        </div>
                        <div class="score-info">
                            <span>Threshold: <span id="threshold-value">0.95</span></span>
                            <span>Calibrated: <span id="calibrated-value">0.0000</span></span>
                        </div>
                    </div>
                    <canvas class="chart ribbon-chart" id="ribbon-chart"></canvas>
                </div>
                <div class="legend">
                    <div class="legend-item"><div class="legend-color" style="background: #00d4aa;"></div> Score</div>
                    <div class="legend-item"><div class="legend-color" style="background: #1e3a4a;"></div> Ribbon (95% CI)</div>
                    <div class="legend-item"><div class="legend-color" style="background: #ffaa00;"></div> Threshold</div>
                    <div class="legend-item"><div class="legend-color" style="background: #ff4444;"></div> Alert</div>
                </div>
            </div>
        </div>

        <div class="panel" style="grid-column: span 2;">
            <div class="panel-header">Recent Alerts</div>
            <div class="panel-content alerts-panel" id="alerts-panel"></div>
        </div>
    </div>

    <div class="controls">
        <button class="btn btn-danger" id="btn-pause">Pause</button>
        <button class="btn btn-primary" id="btn-resume">Resume</button>
        <button class="btn btn-secondary" id="btn-clear">Clear History</button>
    </div>

    <script>
        // Configuration
        const MAX_HISTORY = 500;
        const RIBBON_CONFIDENCE = 0.95;

        // State
        let history = {
            steps: [],
            scores: [],
            calibrated: [],
            lower: [],
            upper: [],
            alerts: [],
            labels: [],
            features: [],
            featureNames: []
        };
        let paused = false;
        let eventSource = null;
        let alertCount = 0;
        let startTime = Date.now();
        let lastStep = 0;

        // DOM elements
        const elements = {
            detectorName: document.getElementById('detector-name'),
            calibratorName: document.getElementById('calibrator-name'),
            stepCount: document.getElementById('step-count'),
            pointRate: document.getElementById('point-rate'),
            alertCount: document.getElementById('alert-count'),
            connectionStatus: document.getElementById('connection-status'),
            scoreValue: document.getElementById('score-value'),
            scoreBarFill: document.getElementById('score-bar-fill'),
            scoreBarThreshold: document.getElementById('score-bar-threshold'),
            thresholdValue: document.getElementById('threshold-value'),
            calibratedValue: document.getElementById('calibrated-value'),
            featuresGrid: document.getElementById('features-grid'),
            alertsPanel: document.getElementById('alerts-panel'),
            ribbonChart: document.getElementById('ribbon-chart'),
            btnPause: document.getElementById('btn-pause'),
            btnResume: document.getElementById('btn-resume'),
            btnClear: document.getElementById('btn-clear'),
        };

        // Ribbon chart context
        const ribbonCtx = elements.ribbonChart.getContext('2d');
        function resizeRibbonChart() {
            const rect = elements.ribbonChart.getBoundingClientRect();
            elements.ribbonChart.width = rect.width * window.devicePixelRatio;
            elements.ribbonChart.height = rect.height * window.devicePixelRatio;
            ribbonCtx.scale(window.devicePixelRatio, window.devicePixelRatio);
        }
        window.addEventListener('resize', resizeRibbonChart);
        resizeRibbonChart();

        // Connect to SSE endpoint
        function connect() {
            if (eventSource) {
                eventSource.close();
            }
            eventSource = new EventSource('/events');
            elements.connectionStatus.textContent = 'Connecting...';
            elements.connectionStatus.className = 'status-value warning';

            eventSource.onopen = () => {
                elements.connectionStatus.textContent = 'Connected';
                elements.connectionStatus.className = 'status-value';
            };

            eventSource.onmessage = (event) => {
                if (paused) return;
                const data = JSON.parse(event.data);
                handleData(data);
            };

            eventSource.onerror = () => {
                elements.connectionStatus.textContent = 'Disconnected - Reconnecting...';
                elements.connectionStatus.className = 'status-value alert';
                setTimeout(connect, 3000);
            };
        }

        function handleData(data) {
            // Update metadata
            if (data.detector_name) elements.detectorName.textContent = data.detector_name;
            if (data.calibrator_name) elements.calibratorName.textContent = data.calibrator_name;
            if (data.threshold) {
                elements.thresholdValue.textContent = data.threshold.toFixed(3);
                const thresholdPct = data.threshold * 100;
                elements.scoreBarThreshold.style.left = thresholdPct + '%';
            }

            // Update step and rate
            const step = data.step;
            elements.stepCount.textContent = step;
            const elapsed = (Date.now() - startTime) / 1000;
            const rate = elapsed > 0 ? (step / elapsed).toFixed(1) : 0;
            elements.pointRate.textContent = rate + ' pts/s';

            // Update score
            const calibrated = data.calibrated_score;
            elements.scoreValue.textContent = calibrated.toFixed(4);
            elements.calibratedValue.textContent = calibrated.toFixed(4);
            const scorePct = Math.min(100, calibrated * 100);
            elements.scoreBarFill.style.width = scorePct + '%';

            const isAlert = data.alert;
            if (isAlert) {
                elements.scoreValue.classList.add('alert');
                elements.scoreBarFill.classList.add('alert');
                alertCount++;
                elements.alertCount.textContent = alertCount;
            } else {
                elements.scoreValue.classList.remove('alert');
                elements.scoreBarFill.classList.remove('alert');
            }

            // Update history
            history.steps.push(step);
            history.scores.push(data.score);
            history.calibrated.push(calibrated);
            history.lower.push(data.lower_bound);
            history.upper.push(data.upper_bound);
            history.alerts.push(isAlert);
            history.labels.push(data.label ?? null);
            history.features.push(data.features);

            // Trim history
            for (const key of Object.keys(history)) {
                if (history[key].length > MAX_HISTORY) {
                    history[key] = history[key].slice(-MAX_HISTORY);
                }
            }

            // Update feature cards
            updateFeatures(data.features, data.feature_names);

            // Update ribbon chart
            drawRibbonChart();

            // Update alerts panel
            if (isAlert || data.label === 1) {
                addAlertItem(data);
            }
        }

        function updateFeatures(features, featureNames) {
            if (!elements.featuresGrid.children.length) {
                // Create feature cards
                elements.featuresGrid.innerHTML = '';
                features.forEach((val, idx) => {
                    const name = featureNames?.[idx] || `Feature ${idx + 1}`;
                    const card = document.createElement('div');
                    card.className = 'feature-card';
                    card.innerHTML = `
                        <div class="feature-name">${name}</div>
                        <div class="feature-value" id="feat-val-${idx}">${val.toFixed(3)}</div>
                        <canvas class="feature-sparkline" id="feat-spark-${idx}" width="200" height="40"></canvas>
                    `;
                    elements.featuresGrid.appendChild(card);
                });
            }
            features.forEach((val, idx) => {
                const valEl = document.getElementById(`feat-val-${idx}`);
                if (valEl) valEl.textContent = val.toFixed(3);
                drawSparkline(`feat-spark-${idx}`, history.features.map(f => f[idx]));
            });
        }

        function drawSparkline(canvasId, values) {
            const canvas = document.getElementById(canvasId);
            if (!canvas) return;
            const ctx = canvas.getContext('2d');
            const rect = canvas.getBoundingClientRect();
            canvas.width = rect.width * window.devicePixelRatio;
            canvas.height = rect.height * window.devicePixelRatio;
            ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

            const w = canvas.width / window.devicePixelRatio;
            const h = canvas.height / window.devicePixelRatio;

            if (values.length < 2) return;

            const min = Math.min(...values);
            const max = Math.max(...values);
            const range = max - min || 1;

            ctx.clearRect(0, 0, w, h);
            ctx.strokeStyle = '#00d4aa';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            values.forEach((v, i) => {
                const x = (i / (values.length - 1)) * w;
                const y = h - ((v - min) / range) * h;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });
            ctx.stroke();
        }

        function drawRibbonChart() {
            const canvas = elements.ribbonChart;
            const ctx = ribbonCtx;
            const rect = canvas.getBoundingClientRect();
            const w = rect.width;
            const h = rect.height;

            if (history.calibrated.length < 2) return;

            ctx.clearRect(0, 0, w, h);

            const values = history.calibrated;
            const lower = history.lower;
            const upper = history.upper;
            const alerts = history.alerts;

            const allValues = [...values, ...lower, ...upper];
            const min = Math.min(...allValues);
            const max = Math.max(...allValues);
            const range = max - min || 1;

            const n = values.length;
            const stepX = w / (n - 1);

            // Draw ribbon (confidence band)
            ctx.fillStyle = 'rgba(30, 58, 74, 0.6)';
            ctx.beginPath();
            ctx.moveTo(0, h);
            for (let i = 0; i < n; i++) {
                const x = i * stepX;
                const y = h - ((lower[i] - min) / range) * h;
                ctx.lineTo(x, y);
            }
            for (let i = n - 1; i >= 0; i--) {
                const x = i * stepX;
                const y = h - ((upper[i] - min) / range) * h;
                ctx.lineTo(x, y);
            }
            ctx.closePath();
            ctx.fill();

            // Draw score line
            ctx.strokeStyle = '#00d4aa';
            ctx.lineWidth = 2;
            ctx.beginPath();
            values.forEach((v, i) => {
                const x = i * stepX;
                const y = h - ((v - min) / range) * h;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });
            ctx.stroke();

            // Draw threshold line
            const threshold = parseFloat(elements.thresholdValue.textContent) || 0.95;
            const threshY = h - ((threshold - min) / range) * h;
            ctx.strokeStyle = '#ffaa00';
            ctx.lineWidth = 1;
            ctx.setLineDash([5, 5]);
            ctx.beginPath();
            ctx.moveTo(0, threshY);
            ctx.lineTo(w, threshY);
            ctx.stroke();
            ctx.setLineDash([]);

            // Draw alert points
            alerts.forEach((alert, i) => {
                if (alert) {
                    const x = i * stepX;
                    const y = h - ((values[i] - min) / range) * h;
                    ctx.fillStyle = '#ff4444';
                    ctx.beginPath();
                    ctx.arc(x, y, 4, 0, Math.PI * 2);
                    ctx.fill();
                }
            });
        }

        function addAlertItem(data) {
            const item = document.createElement('div');
            item.className = 'alert-item';
            const timeStr = new Date().toLocaleTimeString();
            const alertClass = data.alert ? 'triggered' : '';
            const labelText = data.label === 1 ? ' ✓ True Anomaly' : (data.label === 0 ? ' ✗ Normal' : '');
            item.innerHTML = `
                <span class="alert-time">${timeStr} | Step ${data.step}</span>
                <span class="alert-score ${alertClass}">Score: ${data.score.toFixed(3)} | Cal: ${data.calibrated_score.toFixed(3)}</span>
                <span class="alert-label">${data.alert ? '🔴 ALERT' : '🟢 Normal'}${labelText}</span>
            `;
            elements.alertsPanel.insertBefore(item, elements.alertsPanel.firstChild);

            // Limit alerts shown
            while (elements.alertsPanel.children.length > 50) {
                elements.alertsPanel.removeChild(elements.alertsPanel.lastChild);
            }
        }

        // Controls
        elements.btnPause.addEventListener('click', () => { paused = true; });
        elements.btnResume.addEventListener('click', () => { paused = false; });
        elements.btnClear.addEventListener('click', () => {
            history = {
                steps: [], scores: [], calibrated: [], lower: [], upper: [],
                alerts: [], labels: [], features: [], featureNames: []
            };
            elements.alertsPanel.innerHTML = '';
            alertCount = 0;
            elements.alertCount.textContent = '0';
            startTime = Date.now();
        });

        // Initialize
        connect();
    </script>
</body>
</html>
"""


class SSEHandler(SimpleHTTPRequestHandler):
    """HTTP request handler with SSE support."""

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/events":
            self.handle_sse()
        elif path == "/" or path == "/index.html":
            self.handle_dashboard()
        elif path == "/api/status":
            self.handle_status()
        elif path == "/api/history":
            self.handle_history()
        else:
            self.send_error(404)

    def handle_dashboard(self) -> None:
        """Serve the dashboard HTML."""
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(DASHBOARD_HTML.encode("utf-8"))

    def handle_sse(self) -> None:
        """Handle Server-Sent Events connection."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        # Create a queue for this client
        queue: asyncio.Queue = asyncio.Queue()
        STATE.clients.add(queue)

        try:
            # Send initial metadata
            if STATE.detector and STATE.calibrator:
                meta = {
                    "detector_name": getattr(
                        STATE.detector, "__class__", type(STATE.detector)
                    ).__name__,
                    "calibrator_name": getattr(
                        STATE.calibrator, "__class__", type(STATE.calibrator)
                    ).__name__,
                    "threshold": STATE.args.threshold if STATE.args else 0.95,
                }
                self.wfile.write(f"data: {json.dumps(meta)}\n\n".encode())

            # Send history
            for entry in STATE.history:
                self.wfile.write(f"data: {json.dumps(entry)}\n\n".encode())
            self.wfile.flush()

            # Keep connection alive and send updates
            while STATE.running:
                try:
                    # Wait for data with timeout
                    data = asyncio.run_coroutine_threadsafe(
                        asyncio.wait_for(queue.get(), timeout=1.0), asyncio.get_event_loop()
                    ).result(timeout=1.5)
                    self.wfile.write(f"data: {json.dumps(data)}\n\n".encode())
                    self.wfile.flush()
                except (TimeoutError, Exception):
                    # Send heartbeat
                    self.wfile.write(b": heartbeat\n\n")
                    self.wfile.flush()
                    continue

        except (ConnectionError, BrokenPipeError, OSError):
            pass
        finally:
            STATE.clients.discard(queue)

    def handle_status(self) -> None:
        """Return current status as JSON."""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        status = {
            "running": STATE.running,
            "step": STATE.step,
            "clients": len(STATE.clients),
            "latest": STATE.latest_data,
        }
        self.wfile.write(json.dumps(status).encode())

    def handle_history(self) -> None:
        """Return history as JSON."""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(STATE.history).encode())

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default log messages."""
        pass


def broadcast(data: dict[str, Any]) -> None:
    """Broadcast data to all connected SSE clients."""
    for queue in STATE.clients:
        try:
            queue.put_nowait(data)
        except asyncio.QueueFull:
            pass


_CALIBRATORS_WITH_WINDOW = {
    "conformal",
    "conformal_tsfm",
    "gaussian_tail",
    "anomaly_likelihood",
    "group_conditional_conformal",
}


def stream_worker(args: Any) -> None:
    """Background worker that streams data and broadcasts updates."""
    import json

    detector_kwargs = json.loads(args.detector_kwargs)
    calibrator_kwargs = json.loads(args.calibrator_kwargs)

    if "random_state" not in detector_kwargs:
        detector_kwargs["random_state"] = args.detector_seed

    calibrator_name = args.calibrator.lower().replace("-", "_")
    if calibrator_name in _CALIBRATORS_WITH_WINDOW and "window_size" not in calibrator_kwargs:
        calibrator_kwargs["window_size"] = args.calibrator_window

    # Prepare CSV kwargs
    csv_kwargs = {}
    if args.csv_target:
        csv_kwargs["target_column"] = args.csv_target
    if args.csv_timestamp:
        csv_kwargs["timestamp_column"] = args.csv_timestamp

    # Compute feature bounds for detectors that need them
    feature_mins = None
    feature_maxes = None
    if args.detector.lower().replace("-", "_") in ("half_space_trees", "rs_hash"):
        feature_mins, feature_maxes = compute_feature_bounds(
            args.source,
            max_points=args.max_points,
            **csv_kwargs,
        )

    # Create detector and calibrator
    detector = get_detector(
        args.detector,
        feature_mins=feature_mins,
        feature_maxes=feature_maxes,
        **detector_kwargs,
    )
    calibrator = get_calibrator(args.calibrator, **calibrator_kwargs)

    STATE.detector = detector
    STATE.calibrator = calibrator
    STATE.args = args

    stream = create_data_stream(args.source, max_points=args.max_points, **csv_kwargs)

    STATE.running = True
    start_time = time.time()

    for x, label in stream:
        if not STATE.running:
            break

        score = detector.fit_score_partial(x)
        calibrated = calibrator.fit_transform_partial(score)
        lower, upper = compute_conformal_bounds(calibrator, score)
        alert = calibrated > args.threshold

        STATE.step += 1

        data = {
            "step": STATE.step,
            "features": x.tolist(),
            "feature_names": [f"sensor_{i + 1}" for i in range(len(x))],
            "score": float(score),
            "calibrated_score": float(calibrated),
            "lower_bound": lower,
            "upper_bound": upper,
            "alert": alert,
            "label": label,
            "threshold": args.threshold,
            "detector_name": args.detector,
            "calibrator_name": args.calibrator,
            "timestamp": time.time(),
        }

        STATE.latest_data = data
        STATE.history.append(data)
        if len(STATE.history) > STATE.max_history:
            STATE.history = STATE.history[-STATE.max_history :]

        broadcast(data)

        # Control rate
        elapsed = time.time() - start_time
        expected = STATE.step * args.refresh_rate
        if elapsed < expected:
            time.sleep(expected - elapsed)

    STATE.running = False


def run_server(args: Any) -> int:
    """Run the web dashboard server."""
    # Start streaming in background thread
    stream_thread = threading.Thread(target=stream_worker, args=(args,), daemon=True)
    stream_thread.start()
    STATE.stream_thread = stream_thread

    # Create HTTP server
    server = HTTPServer((args.host, args.port), SSEHandler)

    print(f"Starting PySAD Live Dashboard on http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop")

    def signal_handler(signum: int, frame: Any) -> None:
        print("\nShutting down...")
        STATE.running = False
        server.shutdown()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        STATE.running = False
        server.server_close()
        print("Server stopped.")

    return 0
