Run PySAD in streaming engines and ML libraries
===============================================

Integrate PySAD detectors into streaming engines and ML libraries without adding core dependencies. Install the third-party framework you use in your own application environment; the CapyMOA and scikit-multiflow adapters never import those libraries.

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

Next steps
----------

- :doc:`serving` covers Kafka/Redpanda consumers, FastAPI endpoints, Disk/Redis checkpointing, Prometheus, OpenTelemetry, and MCP serving.
- :doc:`stream_data` covers reading CSV files, DataFrames, and message queues.
- :doc:`alerts` explains thresholds and probability calibration.
