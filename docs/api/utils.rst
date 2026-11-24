Utilities
=========

.. module:: pysad.utils

This module holds the streaming input helpers, the dataset loaders, and the bounded
window utilities of PySAD. A streamer yields one instance at a time from an array, a
CSV file, or a DataFrame. A dataset loader downloads and returns a labeled benchmark
dataset. A bounded window utility keeps the last N values of a stream in constant
memory.

Streaming input helpers
-----------------------

A streamer turns a container of instances into a stream of feature vectors of shape
``(num_features,)``, optionally paired with labels. Call ``iter`` to loop over the
instances one at a time. Pass ``shuffle=True`` with a ``random_state`` for a
reproducible shuffled order. Keep the file order when time matters.

:class:`~pysad.utils.ArrayStreamer` streams the rows of a NumPy array.
:class:`~pysad.utils.CSVStreamer` reads a CSV file one record at a time without
loading the full file. Select numeric features and an optional label column by index.
:class:`~pysad.utils.PandasStreamer` streams the rows of a pandas DataFrame as NumPy
arrays. :class:`~pysad.utils.PolarsStreamer` streams the rows of a Polars DataFrame
as NumPy arrays.

See :doc:`/guides/stream_data` for end-to-end examples.

Dataset loaders
---------------

:class:`~pysad.utils.Data` downloads and loads labeled anomaly detection datasets. It
caches each download in ``~/.cache/pysad`` (overridable with ``PYSAD_DATA_DIR``) and
verifies a SHA-256 checksum for every file. :meth:`~pysad.utils.Data.datasets` lists
the available datasets by collection, and :meth:`~pysad.utils.Data.get_data` returns
the features and labels of one dataset. :meth:`~pysad.utils.Data.get_iterator`
streams one instance at a time, and :meth:`~pysad.utils.Data.get_stream` returns
shuffled arrays.

The collections include the multivariate ODDS datasets, the univariate NAB and UCR
series, KDD Cup 99, SMD, SMAP and MSL telemetry, credit-card fraud, INSECTS, and
TSB-drift. See :doc:`/guides/stream_data` for per-collection examples.

Bounded window utilities
------------------------

:class:`~pysad.utils.Window` keeps the last ``window_size`` values of a stream,
oldest first. Adding a value takes constant time, independent of the window size.
Use :meth:`~pysad.utils.Window.update` to add one value,
:meth:`~pysad.utils.Window.update_batch` to add several, and
:meth:`~pysad.utils.Window.get` or :meth:`~pysad.utils.Window.get_array` to read the
window.

:class:`~pysad.utils.RingBuffer` is the fixed-capacity circular buffer behind
:class:`~pysad.utils.Window`. It stores scalar items or 1-D vectors of a fixed shape
and returns views of its stored items without copying.

Helpers
-------

:func:`~pysad.utils.fix_seed` seeds Python's ``random`` module and NumPy's global
random state for reproducible streams. :func:`~pysad.utils.get_minmax_array` and
:func:`~pysad.utils.get_minmax_scalar` return feature boundaries.
:func:`~pysad.utils.check_feature_bounds` and
:func:`~pysad.utils.check_univariate` validate inputs before they reach a detector.
:func:`~pysad.utils.generate_stream` creates a labeled synthetic stream with point
anomalies, bursts and drifts.

The bounded-memory sketches :class:`~pysad.utils.CountMinSketch` and
:class:`~pysad.utils.HyperLogLog` estimate item frequencies and distinct counts over
a stream. The plotting helpers :func:`~pysad.utils.plot_stream` and
:func:`~pysad.utils.plot_streaming_ribbon` draw the features, scores and alerts of a
stream.

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: ../generated
   :kind: classes
   :tail: CountMinSketch, Data, HyperLogLog, RingBuffer, Window

Functions
---------

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :kind: functions
