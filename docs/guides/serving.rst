.. meta::
   :description: Serve PySAD streaming anomaly detectors in production with Kafka consumers, FastAPI endpoints, and Redis or disk state checkpointing.

Serve detectors in production
=============================

This guide shows how to run PySAD detectors behind common production patterns: consuming from Kafka or Redpanda, exposing a FastAPI streaming endpoint, and checkpointing model state to disk or Redis.

The examples use synthetic data. Replace it with your own stream and infrastructure.

.. code-block:: python

   import numpy as np

   rng = np.random.default_rng(0)
   X_history = rng.normal(size=(1000, 4))
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
       async for line in request.stream().__aiter__():
           if not line:
               continue
           point = Point.model_validate_json(line)
           x = np.array(point.values, dtype=np.float64)

           score = model.fit_score_partial(x)
           prob = calibrator.fit_transform_partial(score)

           yield f"data: {json.dumps({'score': float(score), 'probability': float(prob)})}\n\n"

           # Checkpoint every 1000 points
           if model.n_instances_seen % 1000 == 0:
               with open(CHECKPOINT_DIR / f"state_{model.n_instances_seen}.pkl", "wb") as f:
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

   The ``model.n_instances_seen`` attribute is available on all PySAD detectors after the first ``fit_partial`` call. Use it to trigger periodic checkpoints.

WebSocket endpoint for bidirectional streaming
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For lower latency, use a WebSocket:

.. code-block:: python

   import json
   import pickle
   from pathlib import Path

   import numpy as np
   from fastapi import FastAPI, WebSocket
   from fastapi.responses import HTMLResponse

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
       try:
           while True:
               data = await websocket.receive_text()
               point = json.loads(data)
               x = np.array(point["values"], dtype=np.float64)

               score = model.fit_score_partial(x)
               prob = calibrator.fit_transform_partial(score)

               await websocket.send_text(json.dumps({"score": float(score), "probability": float(prob)}))

               if model.n_instances_seen % 1000 == 0:
                   with open(CHECKPOINT_DIR / f"state_{model.n_instances_seen}.pkl", "wb") as f:
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

Keep only the last N checkpoints to bound disk usage:

.. code-block:: python

   import glob
   import os
   import pickle
   import tempfile
   from pathlib import Path

   def save_state_rotating(model, calibrator, base_path: Path, keep: int = 5):
       base_path.parent.mkdir(parents=True, exist_ok=True)
       timestamp = int(model.n_instances_seen)
       path = base_path.parent / f"{base_path.name}_{timestamp}.pkl"

       with tempfile.NamedTemporaryFile(mode="wb", dir=base_path.parent, delete=False) as tmp:
           pickle.dump({"model": model, "calibrator": calibrator}, tmp)
           tmp_path = Path(tmp.name)
       tmp_path.replace(path)

       # Remove old checkpoints
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
   for x in stream:
       score = model.fit_score_partial(x)
       prob = calibrator.fit_transform_partial(score)

       if model.n_instances_seen % 1000 == 0:
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

       if model.n_instances_seen % 1000 == 0:
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

Next steps
----------

- :doc:`production` covers warm-up from history, per-source detectors, and latency measurement.
- :doc:`alerts` explains thresholds and probability calibration.
- :doc:`evaluate` shows how to measure detector performance before deploying.
