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

- ``fit_score_partial`` calls ``score_partial`` and then ``fit_partial``. Thus ``score_partial`` must work before the first ``fit_partial``, for example with a score of 0.
- ``fit``, ``score`` and ``fit_score`` process a 2-D array row by row.
- ``score_partial`` and ``fit_score_partial`` return a Python ``float``, even if your method returns a NumPy scalar or a one-element array.

Expose binary decisions (optional)
----------------------------------

Score-only detectors need no extra code: users can pipe them into :class:`~pysad.transform.postprocessing.ThresholdFilter`, :class:`~pysad.transform.postprocessing.QuantileFilter`, or :class:`~pysad.transform.postprocessing.NonparametricDynamicThresholdPostprocessor` to call ``predict_partial`` and ``fit_predict_partial``.

If your detector's paper defines its own binary decision, opt in on your :class:`~pysad.core.BaseModel` subclass:

- Binary-output detectors whose ``score_partial`` returns ``0.0`` or ``1.0`` set ``_output_kind = "decision"``.
- Continuous-score detectors with a paper-defined threshold set ``_has_decision_rule = True`` and override ``_decide_from_score(self, score) -> int`` to return ``0`` or ``1`` without mutating state.

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

Benchmark your custom model
---------------------------

Benchmark a detector that is not part of PySAD with the benchmark runner. Save the detector in a file and pass ``--custom path/to/file.py:ClassName``. The runner loads the class and scores it on the datasets whose part matches its input kind, the same way it scores the built-in detectors. See :doc:`/benchmarks/run` for the full usage, including the constructor arguments the runner fills in automatically.

The ``EWMAZScore`` detector defined earlier in this guide works with the runner. Save it as ``ewma_zscore.py`` and run:

.. code-block:: console

   $ python benchmarks/benchmark.py --custom ewma_zscore.py:EWMAZScore

A multivariate model runs on the ODDS datasets and the streams with drift. A univariate model runs on the NAB and UCR series. Custom models are not stored in the result cache, so each run scores them again.

Pass ``--latex`` together with ``--custom`` to also write ``docs/benchmark/summary.tex``:

.. code-block:: console

   $ python benchmarks/benchmark.py --custom ewma_zscore.py:EWMAZScore --latex

The table lists each dataset and the mean AUROC of each detector on it, with your custom model as one column next to the built-in detectors it was compared against.

A ready-to-run custom model script is in ``examples/example_custom_model_benchmark.py``. It defines ``EWMAZScore`` with ``_input_kind = "multivariate"`` and scores it through the runner:

.. code-block:: console

   $ python examples/example_custom_model_benchmark.py

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
   * - Postprocessor or calibrator
     - :class:`~pysad.core.BasePostprocessor`
     - ``fit_partial``, ``transform_partial``
   * - Score ensembler
     - :class:`~pysad.core.BaseScoreEnsembler`
     - ``fit_partial``, ``transform_partial``
   * - Metric
     - :class:`~pysad.core.BaseMetric`
     - ``update``, ``get``

To wrap a Python callable without subclassing :class:`~pysad.core.BaseTransformer`, use :class:`~pysad.transform.preprocessing.FunctionTransformer`. Custom probability calibrators set ``_output_kind = "probability"``, and thresholding postprocessors set ``_output_kind = "decision"`` (or ``_has_decision_rule = True`` with ``_decide_from_score``).

Add your detector to PySAD
--------------------------

PySAD accepts published detectors whose paper has 20 or more citations, or appeared in 2025 or later at a CORE A* conference (for example, ICLR, ICCV, ICML, or NeurIPS). It also accepts a streaming detector with a top-three score on a public streaming benchmark such as NAB. :doc:`/inclusion_criteria` gives all the criteria. To add your detector to PySAD, create a single module in ``pysad/models``:

#. Implement the detector as its paper describes, in a new module in ``pysad/models``.
#. Cite the paper in the class docstring with ``:cite:`` and define ``__bibtex__`` on the class or at module scope (do not edit ``docs/refs.bib``).
#. Document each difference from the paper in the docstring.

   A detector from the benchmark route cites the benchmark and its reference implementation instead of a paper, and documents each difference from that implementation.
#. Run ``python scripts/sync_exports.py`` (also run by pre-commit) to regenerate ``pysad/models/__init__.py``, ``__all__``, and the ``README.md`` detector table. The API reference (``docs/api/models.rst``), detector table (``docs/detectors.rst``), bibliography, benchmark registry (``benchmarks/benchmark.py``), and shared test suites in ``tests/models`` discover the new class automatically.
#. Add an entry to ``short_stream_params`` in ``tests/models/conftest.py`` (or ``_short_stream_params`` on the class) only if your detector requires non-standard constructor arguments without defaults or smaller window sizes for short test streams (shared across all seven suites in ``tests/models/``).
#. Run the tests for your detector:

   .. code-block:: console

      $ pytest tests/models -k EWMAZScore

:doc:`/contributing` explains how to set up the development environment and open a pull request.

Next steps
----------

- :doc:`evaluate` measures your detector.
- :doc:`/contributing` explains how to open a pull request.
