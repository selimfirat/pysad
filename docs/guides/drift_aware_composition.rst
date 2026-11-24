.. meta::
   :description: Compose a drift-aware streaming anomaly detector with DriftAdaptiveDetector, StreamPipeline, and the pipe operator.

Detect drift and keep learning
==============================

A drift-aware stream detector resets its model when the data distribution changes, then keeps scoring new points. This guide shows how to combine :class:`~pysad.models.DriftAdaptiveDetector`, :class:`~pysad.models.StreamPipeline`, and the ``|`` pipe operator with public APIs.

The examples on this page use the ``arrhythmia`` benchmark dataset:

.. code-block:: python

   from pysad.utils import ArrayStreamer, Data

   X, y = Data().get_data("arrhythmia.mat")
   streamer = ArrayStreamer()

Reset a detector on drift
-------------------------

:class:`~pysad.models.DriftAdaptiveDetector` wraps a detector and a drift meter. It scores each point, updates the meter, and fits the point. If the meter signals drift, it creates a fresh detector and meter before fitting the alerting point once:

.. code-block:: python

   from pysad.models import DriftAdaptiveDetector, LODA
   from pysad.statistics import PageHinkleyMeter

   adaptive = DriftAdaptiveDetector(
       model_factory=lambda: LODA(random_state=0),
       drift_meter=PageHinkleyMeter(delta=0.005, threshold=25.0),
   )

   for x in streamer.iter(X):
       score = adaptive.fit_score_partial(x)

The factory creates a fresh detector after every reset. Pass a factory with a fixed ``random_state`` for reproducible resets. ``score_partial`` does not learn or update the meter.

Compose stages into one model
-----------------------------

:class:`~pysad.models.StreamPipeline` chains preprocessors, a detector and postprocessors into one model with the same interface as a detector:

.. code-block:: python

   from pysad.models import LODA, StreamPipeline
   from pysad.transform.postprocessing import ZScorePostprocessor
   from pysad.transform.preprocessing import InstanceUnitNormScaler

   pipeline = StreamPipeline(
       model=LODA(random_state=0),
       preprocessors=[InstanceUnitNormScaler()],
       postprocessors=[ZScorePostprocessor(window_size=100)],
   )

   for x in streamer.iter(X):
       score = pipeline.fit_score_partial(x)

Each preprocessor calls ``fit_transform_partial``, the detector calls ``fit_score_partial``, and each postprocessor calls ``fit_transform_partial``. ``score_partial`` scores a point with the current state of each stage and fits no stage.

Use the pipe operator
---------------------

The ``|`` operator builds a :class:`~pysad.models.StreamPipeline`. A transformer on the left becomes a preprocessor, and a postprocessor on the right becomes a postprocessor:

.. code-block:: python

   from pysad.models import LODA, StreamPipeline
   from pysad.transform.postprocessing import ZScorePostprocessor
   from pysad.transform.preprocessing import InstanceUnitNormScaler

   pipeline = (
       InstanceUnitNormScaler()
       | LODA(random_state=0)
       | ZScorePostprocessor(window_size=100)
   )

   for x in streamer.iter(X):
       score = pipeline.fit_score_partial(x)

The pipe operator keeps the stages in order: left stages run first.

Build a drift-aware pipeline
----------------------------

Compose a scaler, a drift-adaptive detector, and a postprocessor into one model:

.. code-block:: python

   from pysad.models import DriftAdaptiveDetector, LODA
   from pysad.statistics import PageHinkleyMeter
   from pysad.transform.postprocessing import ZScorePostprocessor
   from pysad.transform.preprocessing import InstanceUnitNormScaler

   detector = DriftAdaptiveDetector(
       model_factory=lambda: LODA(random_state=0),
       drift_meter=PageHinkleyMeter(delta=0.005, threshold=25.0),
   )

   pipeline = (
       InstanceUnitNormScaler()
       | detector
       | ZScorePostprocessor(window_size=100)
   )

   for x in streamer.iter(X):
       score = pipeline.fit_score_partial(x)

When the meter signals drift, the wrapper resets the detector. The scaler keeps its state, so the new detector still receives scaled points.

Raise alerts from the pipeline
------------------------------

End the pipe with a decision-capable postprocessor to get binary predictions:

.. code-block:: python

   from pysad.models import DriftAdaptiveDetector, LODA
   from pysad.statistics import PageHinkleyMeter
   from pysad.transform.postprocessing import ThresholdFilter
   from pysad.transform.preprocessing import InstanceUnitNormScaler

   detector = DriftAdaptiveDetector(
       model_factory=lambda: LODA(random_state=0),
       drift_meter=PageHinkleyMeter(delta=0.005, threshold=25.0),
   )

   pipeline = (
       InstanceUnitNormScaler()
       | detector
       | ThresholdFilter(threshold=0.99)
   )

   for x in streamer.iter(X):
       alert = pipeline.fit_predict_partial(x)

Next steps
----------

- :doc:`alerts` calibrates the output of a pipeline and raises threshold alerts.
- :doc:`evaluate` measures whether a stage improves the result.
- :doc:`pipelines` shows preprocessing, projections, and ensembles in more detail.
