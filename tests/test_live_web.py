import asyncio
import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from pysad.live import server


def _args(port: int) -> object:
    import argparse

    return argparse.Namespace(
        source="synthetic",
        max_points=2,
        csv_target=None,
        csv_timestamp=None,
        csv_delimiter=",",
        csv_header=True,
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


def test_web_dashboard_history_404_and_broadcast(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("f1,f2\n0.1,0.2\n0.3,0.4\n0.5,0.6\n", encoding="utf-8")
    args = _args(0)
    args.source = str(csv_path)
    args.detector = "half_space_trees"
    args.max_points = 3

    server.STATE.history.clear()
    server.STATE.max_history = 2
    try:
        server.stream_worker(args)
        assert len(server.STATE.history) == 2

        httpd = server.HTTPServer((args.host, 0), server.SSEHandler)
        port = httpd.server_address[1]
        http_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        http_thread.start()
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/history", timeout=2) as resp:
                hist = json.loads(resp.read().decode())
            assert len(hist) == 2

            with pytest.raises(urllib.error.HTTPError) as exc:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/nonexistent", timeout=2)
            assert exc.value.code == 404
        finally:
            httpd.shutdown()
            httpd.server_close()

        q_ok: asyncio.Queue[dict[str, int]] = asyncio.Queue(maxsize=2)
        q_full: asyncio.Queue[dict[str, int]] = asyncio.Queue(maxsize=1)
        q_full.put_nowait({"step": 0})
        server.STATE.clients.update({q_ok, q_full})
        server.broadcast({"step": 1})
        assert q_ok.get_nowait() == {"step": 1}
        server.STATE.clients.discard(q_ok)
        server.STATE.clients.discard(q_full)
    finally:
        server.STATE.history.clear()
        server.STATE.max_history = 500
