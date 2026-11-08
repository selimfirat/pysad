PySAD vs River
==============

`River <https://riverml.xyz/>`_ does machine learning on streams: classification, regression, clustering, forecasting, drift detection and anomaly detection. Like PySAD, it learns from one point at a time. This page compares PySAD with River 0.26.1.

Functions
---------

.. list-table::
   :header-rows: 1
   :stub-columns: 1

   * - Function
     - PySAD
     - River
   * - Points
     - NumPy arrays of shape ``(num_features,)``
     - Dictionaries from feature names to values
   * - Learn and score
     - ``fit_partial``, ``score_partial`` and ``fit_score_partial``
     - ``learn_one`` and ``score_one``
   * - Streaming detectors
     - 20, see :doc:`../detectors`
     - 7: ``HalfSpaceTrees``, ``LODA``, ``LocalOutlierFactor``, ``OneClassSVM``, ``GaussianScorer``, ``StandardAbsoluteDeviation`` and ``PredictiveAnomalyDetection``
   * - Batch detectors
     - Every PyOD detector, through :class:`~pysad.models.integrations.ReferenceWindowModel` and :class:`~pysad.models.integrations.OneFitModel`
     - None
   * - Transformers
     - Scalers, shingles and seasonal decomposition
     - Scalers, imputers, encoders and feature extraction
   * - Projectors
     - Gaussian, sparse and StreamHash random projections
     - Gaussian and sparse random projections
   * - Score ensemblers
     - 5 rules, for example the average and the maximum of the scores
     - None
   * - Score postprocessors
     - Running averages, maximums, medians and z-scores of the scores, and the dynamic threshold of Telemanom
     - None
   * - Scores to labels
     - :class:`~pysad.transform.postprocessing.ThresholdFilter` and :class:`~pysad.transform.postprocessing.QuantileFilter`, also after a probability calibrator
     - ``ThresholdFilter`` and ``QuantileFilter``
   * - Probability calibrators
     - Gaussian tail, conformal and SPOT
     - None
   * - Prequential evaluation
     - ``evaluate_stream``, with AUROC, AUPR, precision and recall, also on a sliding window
     - ``progressive_val_score``, with an approximate ROC AUC

Both libraries have Half-Space Trees, LODA, an incremental local outlier factor and the standard absolute deviation. PySAD has more detectors for anomaly detection, and also the tools around a detector: ensemblers, postprocessors and calibrators. River has the other tasks of machine learning on streams. Use River if anomaly detection is one step of a larger River pipeline.

:class:`~pysad.transform.postprocessing.QuantileFilter` gives the same labels as River's ``QuantileFilter`` from the seventh score on. It does not decide whether the detector learns from a point with the label 1. The loop in :doc:`../guides/alerts` makes that choice.

Results
-------

The table compares the multivariate detectors of PySAD with the streaming detectors of River. Each detector runs once on each of the |num_datasets| ODDS datasets of the :doc:`benchmark <../benchmark>`, with the same points and seed 0.

Each detector scores each point before it learns from the point, as :doc:`../benchmarks/methodology` describes. The metrics leave out the scores of the warm-up points, the first 256 points of a stream. The PySAD detectors use the settings of the benchmark. The River detectors use their default parameters. Their points are dictionaries of the features, and the times do not include the conversion of the points to dictionaries.

The table has one row for each detector, from the best mean rank to the worst. The ranks compare only the detectors in this table. The columns are the same as in the summary of :doc:`../benchmarks/multivariate`, and the best value of each column is in bold. The names of the PySAD detectors link to their documentation, and the names of the other detectors start with their module. The last column gives the version of the library that gave the results. For PySAD, the version also gives the commit of the run, because the run uses the code of the repository.

.. csv-table::
   :file: ../benchmark/comparison_river.csv
   :header-rows: 1
   :class: benchmark-table best-columns

All the runs used one computer, so compare the times only with each other.

River's ``HalfSpaceTrees`` gives every point the score 0 until its first window of 250 points is full. Thus, on the datasets with fewer than about 500 points, it gets few or no informative scores after the warm-up.

River recommends that you scale the data of ``OneClassSVM`` with ``StandardScaler`` and add features with ``RBFSampler``. This comparison uses the defaults of all the detectors, so ``OneClassSVM`` gets the points without these steps.

To run the comparison, install the ``compare`` dependency group, which needs Python 3.11 or newer. The group includes CapyMOA, which needs Java, as :doc:`capymoa` describes. Then run the script from the root of the repository:

.. code-block:: console

   $ pip install -e . --group compare
   $ python benchmarks/compare.py

The script runs the libraries that are installed. With this group, it writes ``docs/benchmark/comparison_river.csv``, ``docs/benchmark/comparison_pyod.csv`` and ``docs/benchmark/comparison_capymoa.csv``. StreamAD and Alibi Detect need NumPy 1, so they have their own group, as :doc:`streamad` describes. With the same versions of the libraries, a new run gives the same AUROC and AUPR. Only the times change.
