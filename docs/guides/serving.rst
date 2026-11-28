.. meta::
   :description: Run PySAD with Prometheus alerts, OpenTelemetry metrics, Kafka, FastAPI, the MCP server, and disk or Redis state checkpoints.

Serve detectors in production
=============================

This guide covers Kafka consumers, FastAPI endpoints, disk and Redis checkpoints, Prometheus alerting, and OpenTelemetry metrics. External clients belong in your application environment, not in PySAD dependencies.

.. code-block:: python

   import numpy as np

   rng = np.random.default_rng(0)
   stream = rng.normal(size=(500, 4))

Kafka / Redpanda consumer
-------------------------

Install ``kafka-python`` or ``confluent-kafka``. Keep one detector per independent partition, or share one detector if partitions share normal behavior:

.. code-block:: python

   import pickle
   from kafka import KafkaConsumer

   from pysad.models import LODA
   from pysad.transform.probability_calibration import ConformalProbabilityCalibrator

   consumer = KafkaConsumer(
       "metrics",
       bootstrap_servers="localhost:9092",
       group_id="anomaly-detector",
       value_deserializer=pickle.loads,
       auto_offset_reset="earliest",
   )
   detectors, calibrators = {}, {}

   for msg in consumer:
       p = msg.partition
       if p not in detectors:
           detectors[p] = LODA(random_state=0)
           calibrators[p] = ConformalProbabilityCalibrator(window_size=1000)
       score = detectors[p].fit_score_partial(msg.value)
       prob = calibrators[p].fit_transform_partial(score)
       if prob > 0.99:
           print(f"ALERT partition={p} score={score:.4f} prob={prob:.4f}")

Deserialize payloads with a schema registry (Avro, Protobuf) or JSON with fixed feature order in production.

FastAPI streaming endpoints
---------------------------

Install ``fastapi`` and ``uvicorn``. Expose either Server-Sent Events or a WebSocket endpoint over localhost behind your proxy:

.. code-block:: python

   import json
   import numpy as np
   from fastapi import FastAPI, Request, WebSocket
   from fastapi.responses import StreamingResponse
   from pydantic import BaseModel

   from pysad.models import LODA
   from pysad.transform.probability_calibration import ConformalProbabilityCalibrator

   model = LODA(random_state=0)
   calibrator = ConformalProbabilityCalibrator(window_size=1000)
   app = FastAPI()

   class Point(BaseModel):
       values: list[float]

   async def score_stream(request: Request):
       async for line in request.stream():
           if not line:
               continue
           x = np.array(Point.model_validate_json(line).values, dtype=np.float64)
           score = model.fit_score_partial(x)
           prob = calibrator.fit_transform_partial(score)
           yield f"data: {json.dumps({'score': float(score), 'probability': float(prob)})}\n\n"

   @app.post("/score")
   async def score_endpoint(request: Request):
       return StreamingResponse(score_stream(request), media_type="text/event-stream")

   @app.websocket("/ws/score")
   async def websocket_score(websocket: WebSocket):
       await websocket.accept()
       try:
           while True:
               x = np.array(json.loads(await websocket.receive_text())["values"], dtype=np.float64)
               score = model.fit_score_partial(x)
               prob = calibrator.fit_transform_partial(score)
               await websocket.send_text(json.dumps({"score": float(score), "probability": float(prob)}))
       except Exception:
           await websocket.close()

Run with ``uvicorn serving:app --host 127.0.0.1 --port 8000``.

Disk checkpointing
------------------

Write checkpoints atomically with a temporary file, rotate old files, and tag ``pysad.__version__`` so mismatched loads warn cleanly:

.. code-block:: python

   import pickle
   import tempfile
   import warnings
   from pathlib import Path
   import pysad

   def save_checkpoint(model, calibrator, base_path: Path, step: int, keep: int = 5):
       base_path.parent.mkdir(parents=True, exist_ok=True)
       target = base_path.parent / f"{base_path.name}_{step:020d}.pkl"
       payload = {"pysad_version": pysad.__version__, "model": model, "calibrator": calibrator}
       with tempfile.NamedTemporaryFile(mode="wb", dir=base_path.parent, delete=False) as tmp:
           pickle.dump(payload, tmp)
           tmp_path = Path(tmp.name)
       tmp_path.replace(target)
       for old in sorted(base_path.parent.glob(f"{base_path.name}_*.pkl"))[:-keep]:
           old.unlink()

   def load_latest_checkpoint(base_path: Path):
       files = sorted(base_path.parent.glob(f"{base_path.name}_*.pkl"))
       if not files:
           return None
       state = pickle.loads(files[-1].read_bytes())
       if state["pysad_version"] != pysad.__version__:
           warnings.warn(f"Saved under pysad {state['pysad_version']}, running {pysad.__version__}.")
       return state["model"], state["calibrator"]

.. warning::

   Load pickles only from sources you trust. Pin the PySAD version in production requirements so pickle schemas stay identical.

Redis checkpointing
-------------------

Install ``redis``. Persist per-source detector state in hashes or TTL keys, using ``r.pipeline()`` when committing multiple keys atomically:

.. code-block:: python

   import pickle
   import redis

   r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=False)

   def save_detectors(detectors: dict[str, object], calibrators: dict[str, object]) -> None:
       pipe = r.pipeline()
       for source_id, model in detectors.items():
           pipe.hset(
               "anomaly_detectors",
               source_id,
               pickle.dumps({"model": model, "calibrator": calibrators[source_id]}),
           )
       pipe.execute()

   def load_detector(source_id: str):
       raw = r.hget("anomaly_detectors", source_id)
       return None if raw is None else pickle.loads(raw)

Prometheus and OpenTelemetry metrics
------------------------------------

Expose counters on localhost using ``prometheus_client`` (or OpenTelemetry's ``PrometheusMetricReader``) in your scoring loop or in wrappers around ``pysad-live --json-output`` and ``pysad.mcp``:

.. code-block:: python

   from prometheus_client import Counter, start_http_server

   from pysad.models import LODA
   from pysad.transform.probability_calibration import ConformalProbabilityCalibrator

   model = LODA(random_state=0)
   calibrator = ConformalProbabilityCalibrator(window_size=1000)
   points = Counter("pysad_points_total", "Points scored")
   alerts = Counter("pysad_alerts_total", "Points above the alert threshold")
   start_http_server(port=8001, addr="127.0.0.1")

   for index, x in enumerate(stream):
       score = model.fit_score_partial(x)
       probability = calibrator.fit_transform_partial(score)
       points.inc()
       if index >= 200 and probability >= 0.99:
           alerts.inc()

Keep metric labels fixed (never use timestamps or record IDs as labels). Alert via Prometheus when ``sum by (job, instance) (increase(pysad_alerts_total[5m])) > 0`` holds for one minute. For OpenTelemetry, create counters via ``MeterProvider(metric_readers=[PrometheusMetricReader()])``; see :doc:`mcp` for wrapping ``create_mcp_server().call_tool``.

Next steps
----------

- :doc:`production` covers warm-up from history, per-source detectors, and latency measurement.
- :doc:`alerts` explains thresholds and probability calibration.
- :doc:`evaluate` shows how to measure detector performance before deploying.
- :doc:`streaming_engines` shows recipes for Polars, Bytewax, PyFlink, River, CapyMOA, scikit-learn, and scikit-multiflow.
- :doc:`mcp` covers Model Context Protocol server tools, production hardening, and monitoring.
