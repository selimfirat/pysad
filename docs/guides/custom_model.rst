Write your own detector
=======================

A PySAD detector is a subclass of :class:`~pysad.core.BaseModel` with two methods. After you write them, your detector works with every streamer, calibrator, ensembler and metric in PySAD.

Implement the two methods
-------------------------

This detector keeps an exponentially weighted mean and variance of each feature. These statistics give more weight to recent points. The detector scores a point by its largest z-score over the features. A z-score is the distance from the mean in standard deviations:

.. code-block:: python

   import numpy as np

   from pysad.core import BaseModel


   class EWMAZScore(BaseModel):
       """Scores each point by its distance from an exponentially weighted mean.

       Args:
           alpha (float): The weight of the newest point, between 0 and 1 (Default=0.05).
       """

       def __init__(self, alpha=0.05):
           self.alpha = alpha
           self.mean_ = None
           self.var_ = None

       def fit_partial(self, X, y=None):
           if self.mean_ is None:
               self.mean_ = X.astype(float)
               self.var_ = np.ones_like(self.mean_)
               return self

           diff = X - self.mean_
           self.mean_ += self.alpha * diff
           self.var_ = (1 - self.alpha) * (self.var_ + self.alpha * diff**2)
           return self

       def score_partial(self, X):
           if self.mean_ is None:
               return 0.0

           z = (X - self.mean_) / np.sqrt(self.var_ + 1e-12)
           return np.abs(z).max()

Follow these rules:

- ``fit_partial(X, y=None)`` takes one point of shape ``(num_features,)``, updates the state and returns ``self``. A detector that does not use labels ignores ``y``.
- ``score_partial(X)`` returns a score for one point. A higher score means a more anomalous point. It must not change the state.
- Keep the state at a fixed maximum size. Store a window or summary statistics, not every point.
- If the detector uses random numbers, take a ``random_state`` parameter and resolve it with :func:`sklearn.utils.check_random_state`.

You get the rest from :class:`~pysad.core.BaseModel`:

- ``fit_score_partial`` calls ``fit_partial`` and then ``score_partial``.
- ``fit``, ``score`` and ``fit_score`` process a 2-D array row by row.
- ``score_partial`` and ``fit_score_partial`` return a Python ``float``, even if your method returns a NumPy scalar or a one-element array.

Run your detector on a stream
-----------------------------

Use the detector like any other PySAD detector. The ``cardio`` dataset has all its anomalies at the end, so this example shuffles the points with a fixed seed:

.. code-block:: python

   from pysad.evaluation import AUROCMetric
   from pysad.utils import ArrayStreamer, Data

   X, y = Data().get_data("cardio.mat")

   model = EWMAZScore(alpha=0.01)
   metric = AUROCMetric()

   for x, label in ArrayStreamer(shuffle=True, random_state=0).iter(X, y):
       metric.update(label, model.fit_score_partial(x))

   print(f"AUROC: {metric.get():.3f}")

.. code-block:: text

   AUROC: 0.930

Other components
----------------

The other component types follow the same pattern. Subclass the base class from :mod:`pysad.core` and implement its abstract methods:

.. list-table::
   :header-rows: 1

   * - Component
     - Base class
     - Methods
   * - Preprocessor or projector
     - :class:`~pysad.core.BaseTransformer`
     - ``fit_partial``, ``transform_partial``
   * - Postprocessor, ensembler or calibrator
     - :class:`~pysad.core.BasePostprocessor`
     - ``fit_partial``, ``transform_partial``
   * - Metric
     - :class:`~pysad.core.BaseMetric`
     - ``update``, ``get``

Add your detector to PySAD
--------------------------

PySAD accepts published detectors with 50 or more citations. :doc:`/inclusion_criteria` gives all the criteria. To add your detector to PySAD, do these steps:

#. Implement the detector as its paper describes, in a new module in ``pysad/models``.
#. Cite the paper in the class docstring with ``:cite:`` and add the BibTeX entry to ``docs/refs.bib``.
#. Document each difference from the paper in the docstring.
#. Add the class to ``pysad/models/__init__.py``, to ``__all__`` and to ``docs/api/models.rst``.
#. Add the class to the shared tests in ``tests/models``: ``test_models.py``, ``test_score_types.py``, ``test_random_state.py`` and ``test_properties.py``. To find the correct place, search these files for an existing detector, such as ``LODA``.
#. If your detector has required arguments, add them to ``MODEL_PARAMS`` in ``test_pickling.py``. This file tests every class in ``pysad.models.__all__``. Together, the shared tests cover the score type, pickling, ``random_state`` and general properties of every detector.
#. Run the tests for your detector:

   .. code-block:: console

      $ pytest tests/models -k EWMAZScore

:doc:`/contributing` explains how to set up the development environment and open a pull request.

Next steps
----------

- :doc:`evaluate` measures your detector.
- :doc:`/contributing` explains how to open a pull request.
