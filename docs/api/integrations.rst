Integrations
============

.. module:: pysad.integrations

This module holds bidirectional integrations between PySAD, PyOD, scikit-learn, and River: wrappers that run batch detectors and River forecasters on a stream, and adapters that expose PySAD detectors inside River and scikit-learn pipelines. See :doc:`/guides/pyod` and :doc:`/guides/pipelines`.

The stream wrappers and the forecasting adapter need no library beyond NumPy. The River adapters require River, and the scikit-learn adapter requires scikit-learn; both are optional install-time dependencies.

Stream wrappers for external models
-----------------------------------

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: ../generated
   :exclude: PySADToRiverAdapter, RiverDriftDetector, PySADToScikitLearnAdapter, ForecastingAnomalyDetector, ForecasterProtocol, TSFMResidualAdapter
   :tail: ReferenceWindowModel, OneFitModel, PredictiveAnomalyDetector

River and scikit-learn pipeline adapters
----------------------------------------

.. autoclass:: PySADToRiverAdapter
   :exclude-members: clone, mutate

.. autoclass:: RiverDriftDetector
   :exclude-members: clone, mutate

.. autoclass:: PySADToScikitLearnAdapter
   :members: fit, score_samples, decision_function, predict

Forecasting adapters
--------------------

.. autoclass:: ForecastingAnomalyDetector

.. autoclass:: ForecasterProtocol

TSFM residual adapter
---------------------

.. autoclass:: TSFMResidualAdapter

See :ref:`tsfm-residual-composition` for a runnable recipe with no model download or new dependency.

Compare residual pipelines
~~~~~~~~~~~~~~~~~~~~~~~~~~

Run the optional comparison from the repository root::

    python benchmarks/bench_residual_adapter.py --compare --output comparison.csv

The comparison chains ``ForecastingAnomalyDetector``, ``TSFMResidualAdapter``, and
``ConformalTSFMCalibrator``. It also runs ``LODA`` and
``StreamingAutoregressiveDetector`` on the same seasonal stream with labeled spikes.
Each method scores before learning. Labels only enter AUROC.
The report excludes warm-up values from AUROC and timing.
It reports AUROC, median milliseconds per point, and score repeatability.
If the scored labels contain only one class, AUROC is NaN.
These synthetic results do not measure performance on real datasets.

The default forecaster uses a moving average. It is a stand-in, not a TSFM.
To compare an existing TSFM, pass a local factory::

    python benchmarks/bench_residual_adapter.py --compare --forecaster my_forecaster.py:create

The file must export a zero-argument ``create()`` function.
Each call must return a fresh object with ``forecast(horizon)`` and ``learn_one(y)`` methods.
``forecast`` returns a sequence whose first value predicts the next observation.
``learn_one`` receives the current observation after scoring.
The factory must reset model state and random seeds for repeatable runs.
The benchmark adds no dependencies or model downloads.
If your factory needs a forecasting library, use your existing environment.
Timing excludes factory setup and includes forecasting, scoring, and learning.
