Examples
========

These scripts come from the `examples folder <https://github.com/selimfirat/pysad/tree/master/examples>`_ on GitHub. The automatic tests run each script end to end on the ``master`` branch. To run a script, download it. Then run ``python <script>.py``. The scripts download the benchmark datasets that they need.

To try PySAD in the browser without an installation, open the `quick start notebook in Colab <https://colab.research.google.com/github/selimfirat/pysad/blob/master/examples/quickstart.ipynb>`_.

Score a benchmark dataset
-------------------------

This script is the shortest complete loop. It streams a dataset, scores each point and measures AUROC. AUROC is a number that shows how well the scores separate anomalies from normal points.

.. literalinclude:: ../examples/example_usage_short.py
   :language: python

Build a full pipeline
---------------------

This script scales each point, scores it with xStream and smooths the score. :doc:`guides/pipelines` explains each stage.

.. literalinclude:: ../examples/example_usage.py
   :language: python

Raise alerts with a calibrator
------------------------------

This script uses a calibrator to convert each score into a value between 0 and 1. It raises an alert when this value is above a threshold. See :doc:`guides/alerts`.

.. literalinclude:: ../examples/example_probability_calibration.py
   :language: python

Combine detectors
-----------------

This script normalizes the scores of two detectors. Then it averages them with an ensembler, which combines the scores of several detectors.

.. literalinclude:: ../examples/example_ensemble.py
   :language: python

Run a PyOD model
----------------

This script runs a batch Isolation Forest on a sliding window (the most recent points). A batch detector needs all of its data before it can score. See :doc:`guides/pyod`.

.. literalinclude:: ../examples/example_pyod_integration.py
   :language: python

Find anomalies in a seasonal series
-----------------------------------

This script compares Seasonal ESD and Seasonal Hybrid ESD on a series with a daily cycle. See :doc:`guides/seasonal`.

.. literalinclude:: ../examples/example_seasonal_esd.py
   :language: python

Stream a pandas DataFrame
-------------------------

This script scores the rows of a DataFrame with a time index. Then it prints the timestamps with the highest scores.

.. literalinclude:: ../examples/example_pandas_streamer.py
   :language: python

Use a density detector
----------------------

This script scales each point and scores it with Inqmad. Inqmad estimates how dense the stream is near each point.

.. literalinclude:: ../examples/example_usage_inqmad.py
   :language: python

Track running statistics
------------------------

This script tracks the mean and the variance of a stream. Running statistics update with each new value.

.. literalinclude:: ../examples/example_statistics.py
   :language: python
