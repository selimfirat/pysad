.. meta::
   :description: Run PySAD with Prometheus alerts, OpenTelemetry metrics, Polars batches, Bytewax, PyFlink, Kafka, FastAPI, the MCP server, and state checkpoints.

Serve detectors in production
=============================

This guide covers alert metrics, file batches, stream processors, HTTP endpoints, and state checkpoints.
All recipes call the public PySAD API. External clients belong in your application environment, not in PySAD dependencies.

The examples use synthetic data. Replace it with your own stream and infrastructure.

.. code-block:: python

   import numpy as np

   rng = np.random.default_rng(0)
   stream = rng.normal(size=(500, 4))

Kafka / Redpanda consumer
-------------------------

Install the Kafka client:

.. code-block:: bash

   pip install kafka-python

Or for the faster ``confluent-kafka`` client:

.. code-block:: bash

   pip install confluent-kafka

Consumer loop with a detector per partition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If partitions map to independent sources, keep one detector per partition:

.. code-block:: python

   import pickle
   import signal
   import sys
   from kafka import KafkaConsumer

   from pysad.models import LODA
   from pysad.transform.probability_calibration import ConformalProbabilityCalibrator

   consumer = KafkaConsumer(
       "metrics",
       bootstrap_servers="localhost:9092",
       group_id="anomaly-detector",
       value_deserializer=lambda m: pickle.loads(m),
       auto_offset_reset="earliest",
       enable_auto_commit=False,
   )

   detectors = {}
   calibrators = {}
   checkpoint_interval = 10_000
   points_since_checkpoint = 0

   def save_checkpoints(signum=None, frame=None):
       for partition, model in detectors.items():
           with open(f"detector_partition_{partition}.pkl", "wb") as f:
               pickle.dump({"model": model, "calibrator": calibrators[partition]}, f)
       sys.exit(0)

   signal.signal(signal.SIGTERM, save_checkpoints)
   signal.signal(signal.SIGINT, save_checkpoints)

   for message in consumer:
       partition = message.partition
       x = message.value  # shape (num_features,)

       if partition not in detectors:
           detectors[partition] = LODA(random_state=0)
           calibrators[partition] = ConformalProbabilityCalibrator(window_size=1000)

       score = detectors[partition].fit_score_partial(x)
       prob = calibrators[partition].fit_transform_partial(score)

       # Emit alert if prob > 0.99
       if prob > 0.99:
           print(f"ALERT partition={partition} score={score:.4f} prob={prob:.4f}")

       points_since_checkpoint += 1
       if points_since_checkpoint >= checkpoint_interval:
           save_checkpoints()
           points_since_checkpoint = 0

       consumer.commit()

.. note::

   Deserialize the message payload to a NumPy array of shape ``(num_features,)`` before scoring. The example uses ``pickle`` for simplicity; in production use a schema registry (Avro, Protobuf) or JSON with a fixed feature order.

Consumer loop with a single shared detector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If all partitions share the same normal behavior, use one detector:

.. code-block:: python

   from kafka import KafkaConsumer

   from pysad.models import LODA
   from pysad.transform.probability_calibration import ConformalProbabilityCalibrator

   consumer = KafkaConsumer(
       "metrics",
       bootstrap_servers="localhost:9092",
       group_id="anomaly-detector",
       value_deserializer=lambda m: pickle.loads(m),
   )

   model = LODA(random_state=0)
   calibrator = ConformalProbabilityCalibrator(window_size=1000)

   for message in consumer:
       x = message.value
       score = model.fit_score_partial(x)
       prob = calibrator.fit_transform_partial(score)

       if prob > 0.99:
           print(f"ALERT offset={message.offset} score={score:.4f} prob={prob:.4f}")

FastAPI streaming endpoint
--------------------------

Install FastAPI and Uvicorn:

.. code-block:: bash

   pip install fastapi uvicorn

Streaming endpoint with Server-Sent Events
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This endpoint accepts a stream of JSON points and returns anomaly scores as Server-Sent Events:

.. code-block:: python

   import asyncio
   import json
   import pickle
   from contextlib import asynccontextmanager
   from pathlib import Path

   import numpy as np
   from fastapi import FastAPI, Request
   from fastapi.responses import StreamingResponse
   from pydantic import BaseModel

   from pysad.models import LODA
   from pysad.transform.probability_calibration import ConformalProbabilityCalibrator

   CHECKPOINT_DIR = Path("checkpoints")
   CHECKPOINT_DIR.mkdir(exist_ok=True)

   class Point(BaseModel):
       values: list[float]
       timestamp: float | None = None

   points_seen = 0
   model: LODA | None = None
   calibrator: ConformalProbabilityCalibrator | None = None

   @asynccontextmanager
   async def lifespan(app: FastAPI):
       global model, calibrator
       model = LODA(random_state=0)
       calibrator = ConformalProbabilityCalibrator(window_size=1000)

       # Load latest checkpoint if it exists
       checkpoints = sorted(CHECKPOINT_DIR.glob("state_*.pkl"))
       if checkpoints:
           with open(checkpoints[-1], "rb") as f:
               state = pickle.load(f)
           model, calibrator = state["model"], state["calibrator"]
           print(f"Loaded checkpoint: {checkpoints[-1]}")

       yield

       # Save on shutdown
       with open(CHECKPOINT_DIR / f"state_{int(asyncio.get_event_loop().time())}.pkl", "wb") as f:
           pickle.dump({"model": model, "calibrator": calibrator}, f)

   app = FastAPI(lifespan=lifespan)

   async def score_stream(request: Request):
       global points_seen
       async for line in request.stream().__aiter__():
           if not line:
               continue
           point = Point.model_validate_json(line)
           x = np.array(point.values, dtype=np.float64)

           score = model.fit_score_partial(x)
           prob = calibrator.fit_transform_partial(score)

           yield f"data: {json.dumps({'score': float(score), 'probability': float(prob)})}\n\n"

           # Checkpoint every 1000 points
           points_seen += 1
           if points_seen % 1000 == 0:
               with open(CHECKPOINT_DIR / f"state_{points_seen}.pkl", "wb") as f:
                   pickle.dump({"model": model, "calibrator": calibrator}, f)

   @app.post("/score")
   async def score_endpoint(request: Request):
       return StreamingResponse(score_stream(request), media_type="text/event-stream")

Run with:

.. code-block:: bash

   uvicorn serving:app --host 0.0.0.0 --port 8000

Client example:

.. code-block:: python

   import httpx
   import numpy as np

   async def send_points():
       async with httpx.AsyncClient() as client:
           async with client.stream("POST", "http://localhost:8000/score", json=None) as response:
               async for line in response.aiter_lines():
                   if line.startswith("data: "):
                       print(line[6:])

   # Or send points one by one
   for x in np.random.randn(100, 4):
       async with httpx.AsyncClient() as client:
           r = await client.post("http://localhost:8000/score", json={"values": x.tolist()})
           print(r.text)

.. note::

   Keep a counter in the application to trigger checkpoints. PySAD detectors do not expose a shared instance counter.

WebSocket endpoint for bidirectional streaming
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For lower latency, use a WebSocket:

.. code-block:: python

   import json
   import pickle
   from pathlib import Path

   import numpy as np
   from fastapi import FastAPI, WebSocket

   from pysad.models import LODA
   from pysad.transform.probability_calibration import ConformalProbabilityCalibrator

   CHECKPOINT_DIR = Path("checkpoints")
   CHECKPOINT_DIR.mkdir(exist_ok=True)

   model = LODA(random_state=0)
   calibrator = ConformalProbabilityCalibrator(window_size=1000)

   app = FastAPI()

   @app.websocket("/ws/score")
   async def websocket_score(websocket: WebSocket):
       await websocket.accept()
       points_seen = 0
       try:
           while True:
               data = await websocket.receive_text()
               point = json.loads(data)
               x = np.array(point["values"], dtype=np.float64)

               score = model.fit_score_partial(x)
               prob = calibrator.fit_transform_partial(score)

               await websocket.send_text(json.dumps({"score": float(score), "probability": float(prob)}))

               points_seen += 1
               if points_seen % 1000 == 0:
                   with open(CHECKPOINT_DIR / f"state_{points_seen}.pkl", "wb") as f:
                       pickle.dump({"model": model, "calibrator": calibrator}, f)
       except Exception:
           await websocket.close()

Disk checkpointing
------------------

The :doc:`production` guide shows basic pickling. This section adds robustness for production use.

Atomic writes with a temporary file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Avoid partial writes by writing to a temporary file and renaming:

.. code-block:: python

   import pickle
   import tempfile
   from pathlib import Path

   def save_state(model, calibrator, path: Path):
       path.parent.mkdir(parents=True, exist_ok=True)
       with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as tmp:
           pickle.dump({"model": model, "calibrator": calibrator}, tmp)
           tmp_path = Path(tmp.name)
       tmp_path.replace(path)  # Atomic on POSIX

   def load_state(path: Path):
       with open(path, "rb") as f:
           return pickle.load(f)

Checkpoint rotation
~~~~~~~~~~~~~~~~~~~

Reuse ``save_state`` above and keep the last five checkpoints. Pass the application counter as ``points_seen``:

.. code-block:: python

   def save_state_rotating(model, calibrator, base_path: Path, points_seen: int, keep: int = 5):
       path = base_path.parent / f"{base_path.name}_{points_seen:020d}.pkl"
       save_state(model, calibrator, path)

       # Remove old checkpoints.
       checkpoints = sorted(base_path.parent.glob(f"{base_path.name}_*.pkl"))
       for old in checkpoints[:-keep]:
           old.unlink()

   def load_latest_state(base_path: Path):
       checkpoints = sorted(base_path.parent.glob(f"{base_path.name}_*.pkl"))
       if not checkpoints:
           return None
       with open(checkpoints[-1], "rb") as f:
           return pickle.load(f)

Versioned checkpoints
~~~~~~~~~~~~~~~~~~~~~

Include the PySAD version in the checkpoint to detect incompatibility:

.. code-block:: python

   import pickle
   import pysad

   def save_state_versioned(model, calibrator, path: Path):
       state = {
           "pysad_version": pysad.__version__,
           "model": model,
           "calibrator": calibrator,
       }
       with open(path, "wb") as f:
           pickle.dump(state, f)

   def load_state_versioned(path: Path):
       with open(path, "rb") as f:
           state = pickle.load(f)
       if state["pysad_version"] != pysad.__version__:
           import warnings
           warnings.warn(
               f"Checkpoint saved with pysad {state['pysad_version']}, "
               f"current version is {pysad.__version__}. Scores may differ."
           )
       return state["model"], state["calibrator"]

.. warning::

   Load pickles only from sources you trust. A pickle can execute arbitrary code when loaded. Pin the PySAD version in your requirements (e.g. ``pysad==0.6.0``) to ensure the same pickle format.

Redis checkpointing
-------------------

Install the Redis client:

.. code-block:: bash

   pip install redis

Basic Redis checkpointing
~~~~~~~~~~~~~~~~~~~~~~~~~

Store model state in Redis with a TTL for automatic expiry:

.. code-block:: python

   import pickle
   import redis

   from pysad.models import LODA
   from pysad.transform.probability_calibration import ConformalProbabilityCalibrator

   r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=False)

   CHECKPOINT_KEY = "anomaly_detector:state"
   CHECKPOINT_TTL = 86400  # 24 hours

   def save_to_redis(model, calibrator):
       state = pickle.dumps({"model": model, "calibrator": calibrator})
       r.setex(CHECKPOINT_KEY, CHECKPOINT_TTL, state)

   def load_from_redis():
       data = r.get(CHECKPOINT_KEY)
       if data is None:
           return None
       return pickle.loads(data)

   # Initialize or restore
   state = load_from_redis()
   if state:
       model, calibrator = state["model"], state["calibrator"]
       print("Restored from Redis")
   else:
       model = LODA(random_state=0)
       calibrator = ConformalProbabilityCalibrator(window_size=1000)

   # In your scoring loop
   for points_seen, x in enumerate(stream, start=1):
       score = model.fit_score_partial(x)
       prob = calibrator.fit_transform_partial(score)

       if points_seen % 1000 == 0:
           save_to_redis(model, calibrator)

Per-source checkpoints in Redis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When running one detector per source (e.g., per server), use a Redis hash:

.. code-block:: python

   import pickle
   import redis

   r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=False)

   DETECTORS_KEY = "anomaly_detectors"

   def save_detector(source_id: str, model, calibrator):
       state = pickle.dumps({"model": model, "calibrator": calibrator})
       r.hset(DETECTORS_KEY, source_id, state)

   def load_detector(source_id: str):
       data = r.hget(DETECTORS_KEY, source_id)
       if data is None:
           return None
       return pickle.loads(data)

   def delete_detector(source_id: str):
       r.hdel(DETECTORS_KEY, source_id)

   # In your scoring loop
   detectors = {}
   calibrators = {}
   points_seen = {}

   for source_id, x in stream_with_source_ids():
       if source_id not in detectors:
           state = load_detector(source_id)
           if state:
               detectors[source_id] = state["model"]
               calibrators[source_id] = state["calibrator"]
           else:
               detectors[source_id] = LODA(random_state=0)
               calibrators[source_id] = ConformalProbabilityCalibrator(window_size=1000)

       model = detectors[source_id]
       calibrator = calibrators[source_id]

       score = model.fit_score_partial(x)
       prob = calibrator.fit_transform_partial(score)

       points_seen[source_id] = points_seen.get(source_id, 0) + 1
       if points_seen[source_id] % 1000 == 0:
           save_detector(source_id, model, calibrator)

Redis with atomic operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use Redis transactions for consistency when updating multiple keys:

.. code-block:: python

   import pickle
   import redis

   r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=False)

   def save_all_atomic(detectors: dict[str, tuple], calibrators: dict[str, object]):
       pipe = r.pipeline()
       for source_id, model in detectors.items():
           state = pickle.dumps({"model": model, "calibrator": calibrators[source_id]})
           pipe.hset("anomaly_detectors", source_id, state)
       pipe.execute()

Prometheus alerting
-------------------

Prometheus collects metrics from an HTTP endpoint. A counter records a total that increases with each event.
Run this recipe in one scoring process. Keep port 8001 reachable from your Prometheus server.
Use the `Prometheus Python client <https://github.com/prometheus/client_python>`_ in your application environment.

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

The recipe uses the synthetic ``stream`` from the start of this guide.
For continuous collection, replace it with your live input loop and keep the process running.
The first 200 points train the model and calibrator without alerts.
See :doc:`alerts` for threshold selection. The calibrated value measures a score rank, not the probability of a fault.
Keep metric labels fixed. Do not use timestamps, record IDs, or unbounded source IDs as labels.

Save this Prometheus configuration as ``prometheus.yml``. Save the rule below as ``pysad_alerts.yml`` beside it.
Start Prometheus with ``prometheus --config.file=prometheus.yml``.
If Prometheus runs on another host, change the target and the HTTP bind address.

.. code-block:: yaml

   global:
     scrape_interval: 15s
     evaluation_interval: 15s
   rule_files:
     - pysad_alerts.yml
   scrape_configs:
     - job_name: pysad
       static_configs:
         - targets: ["127.0.0.1:8001"]
   alerting:
     alertmanagers:
       - static_configs:
           - targets: ["127.0.0.1:9093"]

This `alert rule <https://prometheus.io/docs/prometheus/latest/configuration/alerting_rules/>`_
fires when at least one threshold crossing occurs within five minutes.
The condition must stay true for one minute. Alertmanager on port 9093 routes notifications through your configured receivers.
Prometheus can evaluate the rule without Alertmanager, but it cannot deliver notifications through it.

.. code-block:: yaml

   groups:
     - name: pysad
       rules:
         - alert: PySADAnomaly
           expr: sum by (job, instance) (increase(pysad_alerts_total[5m])) > 0
           for: 1m
           labels:
             severity: warning
           annotations:
             summary: "PySAD detected an anomaly on {{ $labels.instance }}"

OpenTelemetry metrics
---------------------

OpenTelemetry can export the same counters through a Prometheus endpoint.
Use this recipe instead of the Prometheus client recipe in a separate process.
It uses ``opentelemetry-api``, ``opentelemetry-sdk``, ``opentelemetry-exporter-prometheus``, and ``prometheus-client`` in your application environment.
The `OpenTelemetry Prometheus exporter <https://opentelemetry-python.readthedocs.io/en/latest/exporter/prometheus/prometheus.html>`_
connects the meter to the HTTP endpoint.

.. code-block:: python

   from opentelemetry.exporter.prometheus import PrometheusMetricReader
   from opentelemetry.sdk.metrics import MeterProvider
   from prometheus_client import start_http_server

   from pysad.models import LODA
   from pysad.transform.probability_calibration import ConformalProbabilityCalibrator

   provider = MeterProvider(metric_readers=[PrometheusMetricReader()])
   meter = provider.get_meter("pysad.serving")
   points = meter.create_counter("pysad_points", description="Points scored")
   alerts = meter.create_counter("pysad_alerts", description="Points above the alert threshold")
   model = LODA(random_state=0)
   calibrator = ConformalProbabilityCalibrator(window_size=1000)
   start_http_server(port=8001, addr="127.0.0.1")

   try:
       for index, x in enumerate(stream):
           score = model.fit_score_partial(x)
           probability = calibrator.fit_transform_partial(score)
           points.add(1)
           if index >= 200 and probability >= 0.99:
               alerts.add(1)
   finally:
       provider.shutdown()

The exporter adds ``_total`` to these counter names.
Use the same Prometheus configuration and alert rule above.
Keep the process running with a live input loop so Prometheus can collect the counters.

Polars streaming batches
------------------------

A lazy Polars query can yield batches without collecting the full result into one DataFrame.
This recipe reads a CSV with numeric columns ``cpu``, ``memory``, and ``latency`` in stream order.
Keep the detector outside the batch loop so it learns across batch boundaries.

.. code-block:: python

   import polars as pl

   from pysad.models import LODA
   from pysad.utils import PolarsStreamer

   model = LODA(random_state=0)
   streamer = PolarsStreamer()
   query = pl.scan_csv("metrics.csv").select("cpu", "memory", "latency")
   for batch in query.collect_batches(chunk_size=1024, maintain_order=True):
       for x in streamer.iter(batch):
           score = model.fit_score_partial(x)
           print(score)

`collect_batches <https://docs.pola.rs/api/python/stable/reference/lazyframe/api/polars.LazyFrame.collect_batches.html>`_
is an unstable Polars API. Use a Polars version that supports it in your application environment.
Consume the batch iterator to completion.
Keep feature order fixed and handle null values before scoring.
For an existing DataFrame, see :doc:`stream_data`.

Bytewax keyed state
-------------------

Bytewax routes each string key to its own state.
Use ``stateful_map`` to keep one detector per source.
This recipe needs Bytewax in your application environment.
Save it as ``bytewax_serving.py`` and run ``python -m bytewax.run bytewax_serving:flow``.

.. code-block:: python

   import bytewax.operators as op
   import numpy as np
   from bytewax.connectors.stdio import StdOutSink
   from bytewax.dataflow import Dataflow
   from bytewax.testing import TestingSource

   from pysad.models import LODA

   def score_point(model: LODA | None, values: list[float]) -> tuple[LODA, float]:
       if model is None:
           model = LODA(random_state=0)
       score = model.fit_score_partial(np.asarray(values, dtype=float))
       return model, score

   flow = Dataflow("pysad_serving")
   source = op.input(
       "source",
       flow,
       TestingSource([("server-a", [0.1, 0.2]), ("server-b", [0.3, 0.4])]),
   )
   scores = op.stateful_map("score", source, score_point)
   op.output("scores", scores, StdOutSink())

The `stateful_map operator <https://github.com/bytewax/bytewax/blob/main/pysrc/bytewax/operators/__init__.py>`_
returns the updated state and one output value. Bytewax keeps the key on the output.
Replace ``TestingSource`` and ``StdOutSink`` with your application connectors.
For restart recovery, configure Bytewax recovery storage and a replayable input source.
The standard output sink does not provide exactly-once delivery.
Limit active keys or remove idle state to bound the total number of detectors.

PyFlink keyed state
-------------------

PyFlink provides a Python DataStream API for Apache Flink.
Use Flink-managed keyed state so checkpoints include each detector.
This recipe needs a matching Flink and PyFlink installation, Java, and PySAD on every Python worker.

Save the recipe as ``flink_serving.py`` and run it with your configured Flink Python environment.
For example, use ``flink run --python flink_serving.py``.
The `Flink state guide <https://nightlies.apache.org/flink/flink-docs-stable/docs/dev/datastream/fault-tolerance/state/>`_
covers keyed state and expiration.

.. code-block:: python

   import numpy as np
   from pyflink.common import Types
   from pyflink.datastream import KeyedProcessFunction, StreamExecutionEnvironment
   from pyflink.datastream.state import ValueStateDescriptor

   from pysad.models import LODA

   class ScorePoint(KeyedProcessFunction):
       def open(self, runtime_context):
           descriptor = ValueStateDescriptor("detector", Types.PICKLED_BYTE_ARRAY())
           self.model_state = runtime_context.get_state(descriptor)

       def process_element(self, value, ctx):
           model = self.model_state.value()
           if model is None:
               model = LODA(random_state=0)
           score = model.fit_score_partial(np.asarray(value[1], dtype=float))
           self.model_state.update(model)
           yield value[0], score

   env = StreamExecutionEnvironment.get_execution_environment()
   env.enable_checkpointing(10_000)
   source = env.from_collection(
       collection=[("server-a", [0.1, 0.2]), ("server-b", [0.3, 0.4])],
       type_info=Types.TUPLE([Types.STRING(), Types.LIST(Types.DOUBLE())]),
   )
   scores = source.key_by(lambda value: value[0], key_type=Types.STRING()).process(
       ScorePoint(),
       output_type=Types.TUPLE([Types.STRING(), Types.DOUBLE()]),
   )
   scores.print()
   env.execute("PySAD serving")

The recipe updates state before emitting each score.
``Types.PICKLED_BYTE_ARRAY()`` serializes the Python model for Flink state.
Use trusted checkpoints and the same PySAD version on all workers.
For a live job, replace the collection source and print sink with your connectors.
Configure durable checkpoint storage and source replay before relying on restart recovery.
The print sink does not provide exactly-once delivery.
Set state expiration or limit active keys to bound the total number of detectors.
See :doc:`alerts` to calibrate scores before applying an alert threshold.

Zero-dependency keyed state
---------------------------

This recipe follows the PyFlink keyed-state pattern without adding any dependency. It keeps one detector per source key in a plain Python dict and checkpoints the dict to disk. It needs only the Python standard library and NumPy, which is already a PySAD dependency. Use it when you want per-source state without running a Flink cluster.

Save the recipe as ``keyed_serving.py``:

.. code-block:: python

   import pickle
   from pathlib import Path

   import numpy as np

   from pysad.models import LODA
   from pysad.transform.probability_calibration import ConformalProbabilityCalibrator

   STATE_PATH = Path("keyed_state.pkl")

   def load_states():
       if not STATE_PATH.exists():
           return {}
       with open(STATE_PATH, "rb") as f:
           return pickle.load(f)

   def save_states(states):
       tmp_path = STATE_PATH.with_suffix(".tmp")
       with open(tmp_path, "wb") as f:
           pickle.dump(states, f)
       tmp_path.replace(STATE_PATH)

   def get_entry(states, source_id):
       if source_id not in states:
           states[source_id] = {
               "model": LODA(random_state=0),
               "calibrator": ConformalProbabilityCalibrator(window_size=1000),
           }
       return states[source_id]

   states = load_states()
   points_seen = 0
   for source_id, x in stream_with_source_ids():
       entry = get_entry(states, source_id)
       score = entry["model"].fit_score_partial(np.asarray(x, dtype=float))
       prob = entry["calibrator"].fit_transform_partial(score)

       points_seen += 1
       if points_seen % 1000 == 0:
           save_states(states)

   save_states(states)

.. note::

   Load pickles only from sources you trust. A pickle can execute arbitrary code when loaded.
   Pin the PySAD version in your requirements (e.g. ``pysad==0.6.0``) to ensure the same pickle format.

   This recipe runs in a single process. For a distributed Flink job, use the PyFlink keyed state recipe above, which needs pyflink, Java, and PySAD on every worker.

   Limit active keys or remove idle state to bound the total number of detectors. See :doc:`alerts` to calibrate scores before applying an alert threshold.

Zero-dependency streaming recipes
---------------------------------

These recipes run PySAD detectors inside River, CapyMOA, scikit-learn, and scikit-multiflow streams. PySAD does not add any of these libraries as a dependency. Install the library you use in your own application environment. The CapyMOA and scikit-multiflow adapters never import those libraries, so they run with PySAD alone.

River
~~~~~

River learns from one point at a time, like PySAD. :class:`~pysad.integrations.PySADToRiverAdapter` exposes a PySAD detector through River's ``learn_one`` and ``score_one`` methods. Install River in your application environment. PySAD does not declare it.

.. code-block:: python

   from functools import partial

   from river import preprocessing

   from pysad.integrations import PySADToRiverAdapter
   from pysad.models import LODA
   from pysad.utils import ArrayStreamer

   import numpy as np

   rng = np.random.default_rng(0)
   stream = rng.normal(size=(400, 3))

   pipeline = preprocessing.StandardScaler() | PySADToRiverAdapter(partial(LODA, random_state=0))

   scores = []
   for x in ArrayStreamer().iter(stream):
       features = dict(enumerate(x))
       score = pipeline.score_one(features)
       pipeline.learn_one(features)
       scores.append(score)

Call ``score_one`` before ``learn_one`` for each point. This is the prequential order: score before learning. The adapter keeps only the wrapped model, so memory stays bounded as the stream grows.

The first dictionary fixes the feature order. Later dictionaries may list keys in any order, but missing or new keys raise ``ValueError``. Set the order explicitly with ``feature_names``, for example ``feature_names=("cpu", "latency")``. Access the wrapped model as ``model_``.

CapyMOA
~~~~~~~

CapyMOA evaluates streaming detectors on a stream of instances. :class:`~pysad.integrations.PySADToCapyMoaAdapter` exposes a PySAD detector through CapyMoa's interface: ``partial_fit``, ``score``, ``predict``, ``predict_proba`` and ``reset``. This adapter never imports CapyMOA, so it runs with PySAD alone. If CapyMOA is installed, pass a CapyMOA ``Instance`` directly. Its ``x`` attribute holds the feature vector.

.. code-block:: python

   import numpy as np

   from pysad.integrations import PySADToCapyMoaAdapter
   from pysad.models import LODA

   rng = np.random.default_rng(0)
   stream = rng.normal(size=(400, 3))

   adapter = PySADToCapyMoaAdapter(LODA(random_state=0))

   scores = []
   for x in stream:
       score = adapter.score(x)
       adapter.partial_fit(x)
       scores.append(score)

Call ``score`` before ``partial_fit`` for each instance. Higher scores mean more anomalous instances. ``predict`` returns binary labels 0 or 1 and needs a decision-capable model, such as one piped into :class:`~pysad.transform.postprocessing.ThresholdFilter`. ``predict_proba`` returns ``[P(normal), P(anomaly)]`` and needs a probability-calibrated model. ``reset`` is a no-op. Build a fresh adapter around a fresh model to start over.

scikit-learn
~~~~~~~~~~~~

scikit-learn is a PySAD dependency, so no extra install is needed. :class:`~pysad.integrations.PySADToScikitLearnAdapter` exposes a fitted PySAD model through scikit-learn's outlier interface. Fit the PySAD model on a warm-up batch first. The adapter's ``fit`` method does not learn from ``X``. It keeps the fitted model.

.. code-block:: python

   import numpy as np

   from pysad.integrations import PySADToScikitLearnAdapter
   from pysad.models import LODA
   from pysad.utils import ArrayStreamer

   rng = np.random.default_rng(0)
   stream = rng.normal(size=(400, 3))

   warmup = stream[:200]
   model = LODA(random_state=0)
   for x in ArrayStreamer().iter(warmup):
       model.fit_partial(x)

   adapter = PySADToScikitLearnAdapter(model, threshold=2.5)

   X_test = stream[200:]
   scores = -adapter.score_samples(X_test)
   labels = adapter.predict(X_test)

``score_samples`` returns negative PySAD scores, so lower values mean more anomalous instances. Negate it to recover PySAD scores, where higher means more anomalous. ``predict`` returns -1 for outliers and 1 for inliers, the scikit-learn convention. Set ``threshold`` in PySAD score units before calling ``decision_function`` or ``predict``.

scikit-multiflow
~~~~~~~~~~~~~~~~

scikit-multiflow evaluates streaming estimators one batch at a time. :class:`~pysad.integrations.PySADToScikitMultiflowAdapter` exposes a PySAD model through scikit-multiflow's interface: ``partial_fit``, ``predict``, ``predict_proba`` and ``score``. This adapter never imports scikit-multiflow, so it runs with PySAD alone. If scikit-multiflow is installed, plug the adapter into its streams and evaluators.

.. code-block:: python

   import numpy as np

   from pysad.integrations import PySADToScikitMultiflowAdapter
   from pysad.models import LODA
   from pysad.transform.postprocessing import ThresholdFilter

   rng = np.random.default_rng(0)
   stream = rng.normal(size=(400, 3))

   model = LODA(random_state=0) | ThresholdFilter(threshold=0.9)
   adapter = PySADToScikitMultiflowAdapter(model)

   X_batch = np.atleast_2d(stream[:50])
   labels = adapter.predict(X_batch)

Call ``partial_fit(X, y)`` to learn each row of a batch. PySAD models are unsupervised and ignore ``y``. ``predict`` returns binary labels 0 for normal and 1 for anomalous, and needs a decision-capable model. ``predict_proba`` returns ``[P(normal), P(anomaly)]`` per row and needs a probability-calibrated model. ``score(X, y)`` returns the fraction of predicted labels that equal ``y``.

Model Context Protocol (MCP) server
-----------------------------------

The MCP server exposes PySAD detectors as tools for AI assistants and LLM clients. It runs over stdio transport, so it needs no network port or host. It requires the optional ``mcp`` dependency:

.. code-block:: bash

   pip install pysad[mcp]

Start the server:

.. code-block:: bash

   pysad-mcp

Or run it as a module:

.. code-block:: bash

   python -m pysad.mcp.server

Connect an MCP client such as Claude Desktop, Cursor, or VS Code. See :doc:`mcp` for the full client configuration and the list of tools.

Score points through the MCP server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Call tools programmatically with :func:`~pysad.mcp.server.create_mcp_server`. Session-based tools (for example ``list_detectors``) return ``(content, structured)`` where ``structured`` holds the parsed result. Auto-discovered detector tools (for example ``pysad_loda``) return a list whose first item holds the result as JSON text:

.. code-block:: python

   import asyncio
   import json

   from pysad.mcp.server import create_mcp_server

   async def main():
       server = create_mcp_server()

       _, info = await server.call_tool("list_detectors", {})
       print("Detectors:", [d["name"] for d in info["detectors"][:5]])

       result = await server.call_tool("pysad_loda", {
           "data": [[0.1, 0.2], [0.2, 0.3], [10.0, 20.0]],
           "fit": True,
           "random_state": 0,
       })
       scores = json.loads(result[0].text)["scores"]
       print("Scores:", scores)

   asyncio.run(main())

Each standalone ``call_tool`` call uses a fresh context, so detector and calibrator state does not carry over between calls. Session-based tools such as ``initialize_detector`` and ``score_point`` keep state within one live MCP client session. Configure a client such as Claude Desktop to use them; see :doc:`mcp`.

For HTTP or WebSocket serving, use the FastAPI recipes earlier in this guide instead.

Prometheus metrics export
-------------------------

Prometheus scrapes metrics from an HTTP endpoint. The ``prometheus_client`` package belongs in your application environment, not in PySAD dependencies. The examples below wrap pysad-live and pysad.mcp so their points and alerts are exposed as counters on port 8001. Use the ``prometheus.yml`` and ``pysad_alerts.yml`` from the Prometheus alerting recipe to scrape and alert on these counters.

pysad-live
~~~~~~~~~~

pysad-live prints one JSON line per point when you run it with ``--json-output``. Pipe that output into a small exporter process that increments Prometheus counters:

.. code-block:: python

   import json
   import subprocess

   from prometheus_client import Counter, start_http_server

   start_http_server(port=8001, addr="127.0.0.1")
   points = Counter("pysad_points_total", "Points scored")
   alerts = Counter("pysad_alerts_total", "Points above the alert threshold")

   proc = subprocess.Popen(
       [
           "pysad-live",
           "--source",
           "synthetic",
           "--detector",
           "loda",
           "--calibrator",
           "conformal",
           "--json-output",
       ],
       stdout=subprocess.PIPE,
       text=True,
   )
   for line in proc.stdout:
       point = json.loads(line)
       points.inc()
       if point.get("alert"):
           alerts.inc()

Run pysad-live in one terminal and this exporter in another. Prometheus scrapes ``http://127.0.0.1:8001/metrics``.

pysad.mcp
~~~~~~~~~

The MCP server runs over stdio, so run it in your own process and wrap ``call_tool`` to count calls. Start a Prometheus HTTP server in the same process:

.. code-block:: python

   import asyncio

   from prometheus_client import Counter, start_http_server

   from pysad.mcp.server import create_mcp_server

   start_http_server(port=8001, addr="127.0.0.1")
   calls = Counter("pysad_mcp_calls_total", "MCP tool calls", ["tool"])

   async def call_tool_counted(server, name, arguments):
       calls.labels(tool=name).inc()
       return await server.call_tool(name, arguments)

   async def main():
       server = create_mcp_server()
       _, info = await call_tool_counted(server, "list_detectors", {})
       print("Detectors:", [d["name"] for d in info["detectors"][:5]])

   asyncio.run(main())

Next steps
----------

- :doc:`production` covers warm-up from history, per-source detectors, and latency measurement.
- :doc:`alerts` explains thresholds and probability calibration.
- :doc:`evaluate` shows how to measure detector performance before deploying.
