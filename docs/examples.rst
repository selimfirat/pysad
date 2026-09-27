Examples
========

Quick Start
^^^^^^^^^^^^^^^^^^

To try PySAD without installing anything, open the `quick start notebook in Colab <https://colab.research.google.com/github/selimfirat/pysad/blob/master/examples/quickstart.ipynb>`_.

Here's a simple example showing how to use PySAD for anomaly detection on streaming data:

.. literalinclude:: ../examples/example_usage_short.py
   :language: python


Example Full Usage
^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../examples/example_usage.py
   :language: python


Example Statistics Usage
^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../examples/example_statistics.py
   :language: python


Example Ensembler Usage
^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../examples/example_ensemble.py
   :language: python


Example Probability Calibrator Usage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../examples/example_probability_calibration.py
   :language: python

Example PyOD Integration
^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../examples/example_pyod_integration.py
   :language: python

Example Seasonal ESD Usage
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../examples/example_seasonal_esd.py
   :language: python

Pandas DataFrame Streaming
^^^^^^^^^^^^^^^^^^^^^^^^^^

``PandasStreamer`` iterates over a pandas DataFrame row by row, making it
possible to score tabular time-series data with PySAD streaming models.

Pandas is an optional dependency. Install it with:

.. code-block:: bash

   pip install pysad[pandas]

The following example creates a small DataFrame with a ``DatetimeIndex`` and
injected CPU and latency spikes, then prints the timestamps with the highest
anomaly scores.

.. literalinclude:: ../examples/example_pandas_streamer.py
   :language: python
