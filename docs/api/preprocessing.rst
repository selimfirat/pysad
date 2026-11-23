Preprocessing
=============

.. module:: pysad.transform.preprocessing

This module holds transformers that change each point before the detector gets it. Each transformer learns from one instance at a time and keeps bounded memory. Compose transformers with the ``|`` operator. See :doc:`/guides/pipelines`.

Scalers
-------

:class:`~pysad.transform.preprocessing.StandardScaler` scales each feature to zero mean and unit variance with one running meter per feature. Memory grows with the number of features, not with the stream length. Before the first ``fit_partial`` call, points pass through unchanged.

:class:`~pysad.transform.preprocessing.MinMaxScaler` scales each feature with the running minimum and maximum into the unit interval. Values outside the observed range scale below zero or above one.

:class:`~pysad.transform.preprocessing.InstanceStandardScaler` scales each feature vector to zero mean and unit standard deviation across its own features. It learns nothing from earlier instances.

:class:`~pysad.transform.preprocessing.InstanceUnitNormScaler` scales each feature vector to unit norm. A zero-norm vector passes through unchanged.

:class:`~pysad.transform.preprocessing.IdentityScaler` returns each feature vector unchanged, for pipelines that need a scaler slot.

Encoders and parsers
--------------------

:class:`~pysad.transform.preprocessing.FunctionTransformer` wraps a Python callable as a transformer, so custom feature extractors, domain parsers, or frozen embedding functions fit inside a ``|`` pipeline without subclassing. Pass ``fit_func=None`` to leave the wrapped callable untouched. That is how a frozen encoder is composed into PySAD: the encoder runs inside the callable, and PySAD never updates it. The callable must return one numeric vector of the same width on every call.

:class:`~pysad.transform.preprocessing.DrainLogParser` parses each log line into an event ID with Drain's fixed-depth tree :cite:`he2017drain`. It keeps one template cluster per tree path, so memory stays bounded by ``depth`` and ``max_child``. It returns the event ID of the matched template, or ``-1.0`` when no template matches.

Composition
-----------

Use the ``|`` operator to chain transformers left to right. Each stage receives the output of the previous stage. A stage must accept the output type of the stage before it: :class:`~pysad.transform.preprocessing.DrainLogParser` reads strings, while :class:`~pysad.transform.preprocessing.FunctionTransformer` returns numeric vectors.

Frozen encoder
..............

Wrap a pretrained encoder in :class:`~pysad.transform.preprocessing.FunctionTransformer` with ``fit_func=None``. PySAD runs the callable but never updates it. This example uses a NumPy stand-in, so it needs no model download and no new dependency:

.. code-block:: python

   import numpy as np

   from pysad.models import LODA
   from pysad.transform.preprocessing import FunctionTransformer


   def encode_window(context):
       """Return summary features as a stand-in for frozen embeddings.

       Args:
           context (np.ndarray): The context window, oldest value first.

       Returns:
           np.ndarray: A vector with the window mean and standard deviation.
       """
       return np.array([context.mean(), context.std()], dtype=np.float64)


   pipeline = (
       FunctionTransformer(encode_window, fit_func=None, output_dims=2)
       | LODA(random_state=0)
   )

   values = np.sin(np.arange(120) * 2 * np.pi / 24).reshape(-1, 1)
   for point in values:
       score = pipeline.fit_score_partial(point)

Log parser
..........

Parse log lines with :class:`~pysad.transform.preprocessing.DrainLogParser`. The first occurrence of a template gets a new event ID, and later occurrences reuse it:

.. code-block:: python

   from pysad.transform.preprocessing import DrainLogParser

   lines = ["ok 1", "ok 2", "error 99999999", "ok 1"]
   parser = DrainLogParser()

   for line in lines:
       print(parser.fit_transform_partial(line))

Composed with a detector, the log parser turns text into a numeric stream:

.. code-block:: python

   from pysad.models import LODA
   from pysad.transform.preprocessing import DrainLogParser

   pipeline = DrainLogParser() | LODA(random_state=0)

   for line in lines:
       score = pipeline.fit_score_partial(line)

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: ../generated
   :tail: ModifiedSTLResidualTransformer, OneShotSTL, SeasonalTrendDecomposer, ShingleTransformer
