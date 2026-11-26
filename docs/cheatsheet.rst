.. meta::
   :description: One-page quick reference for PySAD: install, score a stream, measure, alert, chain stages, and stream data.

PySAD quick reference
=====================

This page lists the most common PySAD workflows. Run the code blocks in order. Each block works as a standalone script. See :doc:`getting_started` for a guided tour and :doc:`api` for every parameter.

Install
-------

.. code-block:: console

   $ pip install pysad

Core loop
---------

Score each point before the model learns from it:

.. code-block:: python

   import numpy as np

   from pysad.models import LODA
   from pysad.utils import ArrayStreamer

   rng = np.random.default_rng(42)
   X = rng.normal(size=(1000, 4))

   model = LODA(random_state=0)
   scores = []
   for x in ArrayStreamer().iter(X):
       scores.append(model.fit_score_partial(x))

:meth:`~pysad.core.BaseModel.fit_score_partial` scores ``x`` first, then learns from it. Scores are Python floats. Higher means more anomalous.

Measure
-------

.. code-block:: python

   from pysad.evaluation import AUROCMetric

   y = np.zeros(1000, dtype=int)
   y[::50] = 1

   model = LODA(random_state=0)
   metric = AUROCMetric()
   for x, label in ArrayStreamer().iter(X, y):
       metric.update(label, model.fit_score_partial(x))

   print(f"AUROC: {metric.get():.3f}")

Scoring before learning is prequential evaluation: each score is measured on a point the model has not seen yet.

Turn scores into alerts
-----------------------

Raw scores have no fixed scale. Calibrate each score against recent scores, then alert at a fixed rank:

.. code-block:: python

   from pysad.transform.probability_calibration import ConformalProbabilityCalibrator

   pipeline = LODA(random_state=0) | ConformalProbabilityCalibrator(window_size=500)

   for t, x in enumerate(ArrayStreamer().iter(X)):
       value = pipeline.fit_score_partial(x)
       if t >= 200 and value >= 0.99:
           print(f"alert at {t}")

A value of 0.99 means that 99% of the scores in the window are lower. Use :class:`~pysad.transform.postprocessing.ThresholdFilter` or :class:`~pysad.transform.postprocessing.QuantileFilter` for binary 0/1 labels instead of print statements.

Chain stages
------------

Chain a scaler, a detector and a calibrator into one model with ``|``:

.. code-block:: python

   from pysad.transform.preprocessing import StandardScaler

   pipeline = StandardScaler() | LODA(random_state=0) | ConformalProbabilityCalibrator(window_size=500)

Combine several detectors with ``&``:

.. code-block:: python

   from pysad.models import HalfSpaceTrees

   ensemble = LODA(random_state=0) & HalfSpaceTrees(random_state=0)

A pipeline or ensemble is itself a model: call :meth:`~pysad.core.BaseModel.fit_score_partial` on it as you would on a single detector.

Stream your data
----------------

.. code-block:: python

   from pysad.utils import PandasStreamer, CSVStreamer, Data

   streamer = PandasStreamer()
   for x in streamer.iter(df):
       score = model.fit_score_partial(x)

   X_cardio, y_cardio = Data().get_data("cardio.mat")

:class:`~pysad.utils.ArrayStreamer`, :class:`~pysad.utils.PandasStreamer`, :class:`~pysad.utils.PolarsStreamer` and :class:`~pysad.utils.CSVStreamer` replay arrays, DataFrames and CSV files one row at a time. :class:`~pysad.utils.Data` downloads labeled benchmark datasets on first use and caches them in ``~/.cache/pysad``.

Pick a detector
---------------

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Your data
     - Start with
     - Why
   * - One series with cycles
     - :class:`~pysad.models.SeasonalHybridESD`
     - Removes daily and weekly cycles first
   * - One series without cycles
     - :class:`~pysad.models.MedianAbsoluteDeviation`
     - Follows drift with recent values only
   * - Several features
     - :class:`~pysad.models.LODA`
     - Fast, parameter-light, supports explanations
   * - Many features
     - :class:`~pysad.models.xStream`
     - Hashes points into fewer features
   * - Network traffic or logs
     - :class:`~pysad.models.KitNet`
     - Built for live intrusion detection
   * - Abrupt distribution shifts
     - :class:`~pysad.models.DriftAdaptiveDetector`
     - Resets any detector on detected drift

See :doc:`detectors` for every detector, its input, its memory and its paper.

See also
--------

- :doc:`getting_started`: run your first detector in ten minutes.
- :doc:`guides/alerts`: choose thresholds and warm-up windows.
- :doc:`guides/pipelines`: build preprocessing and postprocessing chains.
- :doc:`guides/stream_data`: connect CSV files, DataFrames and message queues.
- :doc:`api`: every class and parameter.
