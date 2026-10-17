Examples
========

These scripts come from the `examples folder <https://github.com/selimfirat/pysad/tree/master/examples>`_ on GitHub. The automatic tests run each script end to end on the ``master`` branch. To run a script, download it. Then run ``python <script>.py``. The scripts download the benchmark datasets that they need.

To try PySAD in the browser without an installation, open the `quick start notebook in Colab <https://colab.research.google.com/github/selimfirat/pysad/blob/master/examples/quickstart.ipynb>`_.

Score a benchmark dataset
-------------------------

This script is the shortest complete loop. It streams a dataset, scores each point and measures AUROC. AUROC is a number that shows how well the scores separate anomalies from normal points.

.. literalinclude:: ../examples/example_usage_short.py
   :language: python

It prints:

.. code-block:: text

   AUROC: 0.758

Build a full pipeline
---------------------

This script scales each point, scores it with xStream and smooths the score. :doc:`guides/pipelines` explains each stage.

.. literalinclude:: ../examples/example_usage.py
   :language: python

It prints:

.. code-block:: text

   AUROC: 0.649

Raise alerts with a calibrator
------------------------------

This script uses a calibrator to convert each score into a value between 0 and 1. It raises an alert when this value is above a threshold. See :doc:`guides/alerts`.

.. literalinclude:: ../examples/example_probability_calibration.py
   :language: python

It prints:

.. code-block:: text

   29 alerts at points [102, 141, 187, 189, 190, 198, 204, 207, 218, 252, 253, 257, 293, 297, 303, 312, 313, 316, 354, 356, 370, 379, 388, 395, 403, 424, 427, 430, 449]

Combine detectors
-----------------

This script normalizes the scores of two detectors. Then it averages them with an ensembler, which combines the scores of several detectors.

.. literalinclude:: ../examples/example_ensemble.py
   :language: python

It prints:

.. code-block:: text

   AUROC: 0.801

Run a PyOD model
----------------

This script runs a batch Isolation Forest on a sliding window (the most recent points). A batch detector needs all of its data before it can score. See :doc:`guides/pyod`.

.. literalinclude:: ../examples/example_pyod_integration.py
   :language: python

It prints:

.. code-block:: text

   AUROC: 0.819

Find anomalies in a seasonal series
-----------------------------------

This script compares Seasonal ESD and Seasonal Hybrid ESD on a series with a daily cycle. See :doc:`guides/seasonal`.

.. literalinclude:: ../examples/example_seasonal_esd.py
   :language: python

It prints:

.. code-block:: text

   SeasonalESD flagged [150, 260, 340, 364], true anomalies [150, 260, 340]
   SeasonalHybridESD flagged [150, 260, 265, 266, 267, 340, 364], true anomalies [150, 260, 340]

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

This script scales each point and scores it with Inqmad. Inqmad estimates how dense the stream is near each point.

.. literalinclude:: ../examples/example_usage_inqmad.py
   :language: python

It prints:

.. code-block:: text

   AUROC: 0.803

Track running statistics
------------------------

This script tracks the mean and the variance of a stream. Running statistics update with each new value.

.. literalinclude:: ../examples/example_statistics.py
   :language: python

It prints:

.. code-block:: text

   mean: -0.039, standard deviation: 1.006
