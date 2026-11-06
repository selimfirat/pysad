Examples
========

These scripts come from the `examples folder <https://github.com/selimfirat/pysad/tree/master/examples>`_ on GitHub. The automatic tests run each script end to end on the ``master`` branch. To run a script, download it. Then run ``python <script>.py``. The scripts download the benchmark datasets that they need.

To try PySAD in the browser without an installation, open the `NYC taxi demo in JupyterLite <https://selimfirat.github.io/pysad/lab/index.html?path=nyc_taxi_alerts.ipynb>`_ or the `quick start notebook in Colab <https://colab.research.google.com/github/selimfirat/pysad/blob/master/examples/quickstart.ipynb>`_.

Score a benchmark dataset
-------------------------

This script is the shortest complete loop. It streams a dataset, scores each point and measures AUROC. AUROC is a number that shows how well the scores separate anomalies from normal points.

.. literalinclude:: ../examples/example_usage_short.py
   :language: python

It prints:

.. code-block:: text

   AUROC: 0.758

Scale features online
---------------------

This script scales each incoming feature vector with an instance standard scaler (zero mean, unit variance per instance) before feeding it to a detector.

.. literalinclude:: ../examples/example_preprocessing.py
   :language: python

It prints:

.. code-block:: text

   AUROC: 0.789

Give a detector the recent history
----------------------------------

This script turns each point of a univariate NAB series into a window of its 24 most recent points. LODA then scores these multivariate shingles. See :doc:`guides/pipelines`.

.. literalinclude:: ../examples/example_shingles.py
   :language: python

It prints:

.. code-block:: text

   AUROC: 0.627

Build a full pipeline
---------------------

This script shows how the stages of a pipeline connect. It scales each point, scores it with xStream and smooths the score. It does not tune the stages, so its AUROC is lower than the plain loop above. Measure each stage on your data, as :doc:`guides/evaluate` shows. :doc:`guides/pipelines` explains each stage.

.. literalinclude:: ../examples/example_usage.py
   :language: python

It prints:

.. code-block:: text

   AUROC: 0.543

Chain the stages into one model
-------------------------------

This script puts a scaler, xStream, a running z-score and a calibrator into one :class:`~pysad.models.StreamPipeline`. Then it scores the stream with the pipeline as with a single detector. See :doc:`guides/pipelines`.

.. literalinclude:: ../examples/example_stream_pipeline.py
   :language: python

It prints:

.. code-block:: text

   AUROC: 0.742

Project high-dimensional data
-----------------------------

This script projects each point into fewer features before scoring it with an online anomaly detector.

.. literalinclude:: ../examples/example_projection.py
   :language: python

It prints:

.. code-block:: text

   AUROC: 0.749

Evaluate in windows
-------------------

A single score over the whole stream hides how a detector's separation changes over time. This script scores each point once, then reports the AUROC of consecutive windows of 100 points next to the AUROC over the whole stream. The windowed score averages the windows, so a stretch where the scores collapse does not dominate it.

.. literalinclude:: ../examples/example_windowed_metric.py
   :language: python

It prints:

.. code-block:: text

   AUROC: 0.638
   Windowed AUROC: 0.755

Evaluate with several metrics
-----------------------------

This script runs :func:`~pysad.evaluation.evaluate_stream` once and gets three metrics. The function scores each point before the detector learns from it.

.. literalinclude:: ../examples/example_evaluate_stream.py
   :language: python

It prints:

.. code-block:: text

   AUROC: 0.768
   AUPR: 0.489
   Windowed AUROC: 0.756

Raise alerts with a calibrator
------------------------------

This script uses a calibrator to convert each score into a value between 0 and 1. It raises an alert when this value is above a threshold. See :doc:`guides/alerts`.

.. literalinclude:: ../examples/example_probability_calibration.py
   :language: python

It prints:

.. code-block:: text

   27 alerts, 17 of them true anomalies

Raise alerts at a chosen risk
-----------------------------

This script fits a Generalized Pareto distribution to the highest scores with SPOT (streaming peaks-over-threshold). You choose the risk ``q``, and the script raises an alert when the calibrated score is above ``1 - q``, so a normal point raises an alert with probability ``q``. See :doc:`guides/alerts`.

.. literalinclude:: ../examples/example_spot_calibration.py
   :language: python

It prints:

.. code-block:: text

   32 alerts, 27 of them true anomalies
   alarm threshold on the raw scores: 3.013

Postprocess scores
------------------

This script standardizes the streaming anomaly scores of a detector using a running z-score window.

.. literalinclude:: ../examples/example_postprocessing.py
   :language: python

It prints:

.. code-block:: text

   AUROC: 0.778

Combine detectors
-----------------

This script normalizes the scores of two detectors. Then it averages them with an ensembler, which combines the scores of several detectors.

.. literalinclude:: ../examples/example_ensemble.py
   :language: python

It prints:

.. code-block:: text

   AUROC: 0.811

Run a PyOD model
----------------

This script runs a batch Isolation Forest on a sliding window (the most recent points). A batch detector needs all of its data before it can score. See :doc:`guides/pyod`.

.. literalinclude:: ../examples/example_pyod_integration.py
   :language: python

It prints:

.. code-block:: text

   AUROC: 0.807

Find anomalies in a seasonal series
-----------------------------------

This script compares Seasonal ESD and Seasonal Hybrid ESD on a series with a daily cycle. See :doc:`guides/seasonal`.

.. literalinclude:: ../examples/example_seasonal_esd.py
   :language: python

It prints:

.. code-block:: text

   SeasonalESD flagged [150, 260, 340], true anomalies [150, 260, 340]
   SeasonalHybridESD flagged [150, 260, 340], true anomalies [150, 260, 340]

Both variants find the three spikes and flag no normal values. :doc:`guides/seasonal` explains when to use each variant.

Remove a trend and a cycle with OneShotSTL
------------------------------------------

This script removes the trend and the daily cycle of a series with :class:`~pysad.transform.preprocessing.OneShotSTL`. Then :class:`~pysad.models.StandardAbsoluteDeviation` scores the residuals. The level of the series jumps halfway through. See :doc:`guides/seasonal`.

.. literalinclude:: ../examples/example_one_shot_stl.py
   :language: python

It prints:

.. code-block:: text

   AUROC: 0.990

Stream a pandas DataFrame
-------------------------

This script scores the rows of a DataFrame with a time index. Then it prints the timestamps with the highest scores.

.. literalinclude:: ../examples/example_pandas_streamer.py
   :language: python

It prints:

.. code-block:: text

   Highest-scoring anomaly alerts:
   2025-01-01T11:10:00 score=5.6249
   2025-01-01T12:00:00 score=5.5371
   2025-01-01T10:20:00 score=5.0840
   2025-01-01T11:58:00 score=4.1992
   2025-01-01T11:05:00 score=4.1908

Use a density detector
----------------------

This script scores each point with Inqmad. Inqmad estimates how dense the stream is near each point. It scales the features with its first points, so it takes the raw points.

.. literalinclude:: ../examples/example_usage_inqmad.py
   :language: python

It prints:

.. code-block:: text

   AUROC: 0.739

Use a Gaussian mixture detector
-------------------------------

This script scores each point with SmartSifter. SmartSifter learns a mixture of Gaussians online and gives each point its negative log-likelihood under the mixture. The ``discount`` parameter sets how fast the mixture forgets old points.

.. literalinclude:: ../examples/example_usage_smart_sifter.py
   :language: python

It prints:

.. code-block:: text

   AUROC: 0.672

Find a change point
-------------------

This script finds a level shift in a noisy series with ChangeFinder. ChangeFinder learns an autoregressive model of the series online and scores each value by its negative log-likelihood under the model. A second model learns the moving average of these scores, and its smoothed scores are the change point scores.

.. literalinclude:: ../examples/example_change_finder.py
   :language: python

It prints:

.. code-block:: text

   Highest change point score at 604, level shift at 600

Use a random space tree detector
--------------------------------

This script scores each point with RS-Forest. RS-Forest splits the feature space with random trees and gives each point the negative density of the previous window in the nodes that hold it. Its first window sets the feature ranges of the trees.

.. literalinclude:: ../examples/example_rs_forest.py
   :language: python

It prints:

.. code-block:: text

   AUROC: 0.651

Use a local density detector
----------------------------

This script scores each point with IncrementalLOF. IncrementalLOF compares the density near each point with the density near its nearest neighbors in a window of recent points. It updates only the neighbors that each new point changes.

.. literalinclude:: ../examples/example_incremental_lof.py
   :language: python

It prints:

.. code-block:: text

   AUROC: 0.731

Use a matrix sketch detector
----------------------------

This script scores each point with ADeMS. ADeMS keeps a Frequent Directions sketch, a small matrix that approximates the earlier points. The score of a point is the part of its direction that the top singular vectors of the sketch do not reconstruct.

.. literalinclude:: ../examples/example_adems.py
   :language: python

It prints:

.. code-block:: text

   AUROC: 0.770

Track running statistics
------------------------

This script tracks the mean and the variance of a stream. Running statistics update with each new value.

.. literalinclude:: ../examples/example_statistics.py
   :language: python

It prints:

.. code-block:: text

   mean: -0.048, standard deviation: 0.977
