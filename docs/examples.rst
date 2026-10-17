Examples
========

These scripts come from the `examples folder <https://github.com/selimfirat/pysad/tree/master/examples>`_ on GitHub. The automatic tests run each one end to end on the ``master`` branch. To run one, download it and run ``python <script>.py``. The scripts download the benchmark datasets that they need.

To try PySAD in the browser without an installation, open the `quick start notebook in Colab <https://colab.research.google.com/github/selimfirat/pysad/blob/master/examples/quickstart.ipynb>`_.

Score a benchmark dataset
-------------------------

The shortest complete loop: stream a dataset, score each point and measure AUROC.

.. literalinclude:: ../examples/example_usage_short.py
   :language: python

Build a full pipeline
---------------------

Preprocess each point, score it with xStream and smooth the score. :doc:`guides/pipelines` explains each stage.

.. literalinclude:: ../examples/example_usage.py
   :language: python

Raise alerts with a calibrator
------------------------------

Convert scores into values between 0 and 1 and alert above a threshold. See :doc:`guides/alerts`.

.. literalinclude:: ../examples/example_probability_calibration.py
   :language: python

Combine detectors
-----------------

Normalize the scores of two detectors and average them with an ensembler.

.. literalinclude:: ../examples/example_ensemble.py
   :language: python

Run a PyOD model
----------------

Run a batch Isolation Forest on a sliding window. See :doc:`guides/pyod`.

.. literalinclude:: ../examples/example_pyod_integration.py
   :language: python

Find anomalies in a seasonal series
-----------------------------------

Compare Seasonal ESD and Seasonal Hybrid ESD on a series with a daily cycle. See :doc:`guides/seasonal`.

.. literalinclude:: ../examples/example_seasonal_esd.py
   :language: python

Stream a pandas DataFrame
-------------------------

Score the rows of a DataFrame with a time index and print the timestamps with the highest scores.

.. literalinclude:: ../examples/example_pandas_streamer.py
   :language: python

Use a density detector
----------------------

Scale each point and score it with Inqmad, which estimates how dense the stream is near each point.

.. literalinclude:: ../examples/example_usage_inqmad.py
   :language: python

Track running statistics
------------------------

Track the mean and the variance of a stream.

.. literalinclude:: ../examples/example_statistics.py
   :language: python
