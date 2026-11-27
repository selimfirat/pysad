.. meta::
   :description: Run PySAD detectors in River pipelines, score before learning, detect drift, and keep state bounded.

Run a PySAD detector in River
=============================

Run PySAD detectors inside River pipelines, score each point before learning, detect drift in the score stream, and keep model state bounded.

River learns from one point at a time, like PySAD. :class:`~pysad.integrations.PySADToRiverAdapter` exposes a PySAD detector through River's ``learn_one`` and ``score_one`` methods, so a PySAD model can be a stage of a River pipeline. :class:`~pysad.integrations.RiverDriftDetector` exposes a PySAD drift meter through River's drift detector interface.

The short adapter example lives in :doc:`pipelines`. This guide shows three loops: prequential evaluation, drift detection, and bounded-memory state handling.

The examples use :func:`~pysad.utils.generate_stream`, so they need no download.

Score before you learn
----------------------

Prequential evaluation scores each point before the model learns from it. The adapter keeps this order: call ``score_one`` first, then ``learn_one``. A metric records the label and the score:

.. code-block:: python

   from functools import partial

   from river import preprocessing

   from pysad.evaluation import AUROCMetric
   from pysad.integrations import PySADToRiverAdapter
   from pysad.models import LODA
   from pysad.utils import generate_stream

   X, y = generate_stream(num_instances=800, num_features=2, random_state=0)

   pipeline = preprocessing.StandardScaler() | PySADToRiverAdapter(partial(LODA, random_state=0))
   metric = AUROCMetric()

   for x, label in zip(X, y):
       features = dict(enumerate(x))
       score = pipeline.score_one(features)
       metric.update(int(label), score)
       pipeline.learn_one(features)

   print(f"AUROC: {metric.get():.3f}")

.. code-block:: text

   AUROC: 0.793

The scaler transforms each point with its current state, then the detector scores it. Only then do the scaler and the detector learn from the point. This is the prequential order: score before learn.

Detect drift in the score stream
--------------------------------

Run a drift detector on the scores of the PySAD model to spot when the stream changes. :class:`~pysad.integrations.RiverDriftDetector` wraps a PySAD drift meter, such as :class:`~pysad.statistics.PageHinkleyMeter`, through River's drift detector interface:

.. code-block:: python

   from functools import partial

   import numpy as np

   from pysad.integrations import PySADToRiverAdapter, RiverDriftDetector
   from pysad.models import LODA
   from pysad.statistics import PageHinkleyMeter

   rng = np.random.default_rng(0)
   X = np.concatenate([rng.normal(0, 1, (200, 2)), rng.normal(3, 1, (200, 2))])

   detector = PySADToRiverAdapter(partial(LODA, random_state=0))
   drift = RiverDriftDetector(partial(PageHinkleyMeter, threshold=5.0, min_instances=1))

   for i, x in enumerate(X):
       features = dict(enumerate(x))
       score = detector.score_one(features)
       drift.update(score)
       detector.learn_one(features)
       if drift.drift_detected:
           print(f"Drift detected at index {i} after score {score:.2f}")
           break

.. code-block:: text

   Drift detected at index 21 after score 1.22

The meter keeps a bounded summary of recent scores, so drift detection uses constant memory.

Keep state bounded
------------------

The adapter keeps only the wrapped model, never the whole stream. Each model bounds its own memory with a window or summary statistics. River gives you three ways to handle the adapter state:

- ``clone()`` creates an untrained copy. Inferred feature names are reset; explicit ``feature_names`` are retained.
- ``clone(include_attributes=True)`` copies the learned model and feature mapping.
- ``pickle`` saves and restores the full state for later runs.

.. code-block:: python

   import pickle

   from river import preprocessing

   from pysad.integrations import PySADToRiverAdapter
   from pysad.models import StandardAbsoluteDeviation

   pipeline = preprocessing.StandardScaler() | PySADToRiverAdapter(StandardAbsoluteDeviation)
   for value in [1.0, 3.0, 2.0, 10.0]:
       pipeline.learn_one({"value": value})

   print("original:", round(pipeline.score_one({"value": 10.0}), 4))

   fresh = pipeline.clone()
   print("fresh clone:", round(fresh.score_one({"value": 10.0}), 4))

   restored = pipeline.clone(include_attributes=True)
   print("restored clone:", round(restored.score_one({"value": 10.0}), 4))

   loaded = pickle.loads(pickle.dumps(pipeline))
   print("pickled:", round(loaded.score_one({"value": 10.0}), 4))

.. code-block:: text

   original: 1.4247
   fresh clone: 0.0
   restored clone: 1.4247
   pickled: 1.4247

A fresh clone starts learning from scratch, which is useful after a permanent change in the stream. A clone with attributes or a pickled pipeline resumes from the saved state. Save the state at a fixed interval, for example every 10,000 points, and when the process stops.

Load a pickle only with the PySAD version that saved it. The pickle format can change between releases. Pin the version, for example ``pysad==0.6.0``, in your requirements.

.. warning::

   Load pickles only from sources that you trust. A pickle can run code when you load it.

Next steps
----------

- :doc:`evaluate` shows how to measure a detector before you deploy it.
- :doc:`pipelines` shows the short adapter example and how to combine preprocessing, projections, postprocessing and ensembles.
- :doc:`production` shows how to run one detector per source and control the time per point.
- :doc:`../benchmarks/run` shows how to compare PySAD with River through the adapter.
- :doc:`/api/integrations` lists the adapter and drift detector classes.
