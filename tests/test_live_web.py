import json
import threading
import time
import urllib.request

from pysad.live import server


def _args(port: int) -> object:
    import argparse

    return argparse.Namespace(
        source="synthetic",
        max_points=2,
        csv_target=None,
        csv_timestamp=None,
        detector="loda",
        detector_seed=42,
        detector_kwargs="{}",
        calibrator="conformal",
        calibrator_window=300,
        calibrator_kwargs="{}",
        threshold=0.95,
        refresh_rate=0.01,
        host="127.0.0.1",
        port=port,
    )


def test_web_dashboard_serves_status_and_html() -> None:
    args = _args(0)
    httpd = server.HTTPServer((args.host, args.port), server.SSEHandler)
    port = httpd.server_address[1]
    http_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    http_thread.start()
    stream_thread = threading.Thread(target=server.stream_worker, args=(args,), daemon=True)
    stream_thread.start()

    try:
        for _ in range(100):
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/api/status", timeout=1
                ) as response:
                    status = json.loads(response.read().decode())
                break
            except Exception:
                pass
        else:  # pragma: no cover - server failed to start
            raise AssertionError("web dashboard did not start")

        assert status["running"] is True

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=3) as response:
            html = response.read().decode()
        assert "PySAD Live Anomaly Detection" in html
    finally:
        server.STATE.running = False
        httpd.shutdown()
        httpd.server_close()


def test_sse_events_endpoint_sends_metadata() -> None:
    args = _args(0)
    httpd = server.HTTPServer((args.host, args.port), server.SSEHandler)
    port = httpd.server_address[1]
    http_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    http_thread.start()
    stream_thread = threading.Thread(target=server.stream_worker, args=(args,), daemon=True)
    stream_thread.start()

    try:
        for _ in range(100):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=1)
                break
            except Exception:
                pass
        else:  # pragma: no cover - server failed to start
            raise AssertionError("web dashboard did not start")

        # Wait for the background stream worker to attach detector and calibrator.
        for _ in range(100):
            if server.STATE.running:
                break
            time.sleep(0.05)
        else:  # pragma: no cover - stream worker failed to start
            raise AssertionError("stream worker did not start")

        # The SSE handler writes the initial metadata before entering its loop.
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/events", timeout=5) as response:
            assert response.status == 200
            found_metadata = False
            for line in response:
                line = line.decode().strip()
                if line.startswith("data: "):
                    payload = json.loads(line[len("data: ") :])
                    if "detector_name" in payload:
                        found_metadata = True
                        break
            assert found_metadata, "no SSE metadata received"
    finally:
        server.STATE.running = False
        httpd.shutdown()
        httpd.server_close()
