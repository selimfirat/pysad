PySAD vs PyOD
=============

`PyOD <https://github.com/yzhao062/pyod>`_ does anomaly detection on batch data. A PyOD detector trains on a whole dataset, then scores new points with what it learned. It does not learn from one point at a time. This page compares PySAD with PyOD 3.6.7.

PySAD does not compete with PyOD. It runs PyOD detectors on streams: :class:`~pysad.models.integrations.ReferenceWindowModel` trains a PyOD detector on a sliding window of recent points, and :class:`~pysad.models.integrations.OneFitModel` trains it once on the first points. :doc:`../guides/pyod` shows how.

Functions
---------

.. list-table::
   :header-rows: 1
   :stub-columns: 1

   * - Function
     - PySAD
     - PyOD
   * - Data
     - A stream, one point at a time
     - A dataset, all points at once
   * - Learn and score
     - ``fit_partial``, ``score_partial`` and ``fit_score_partial``
     - ``fit``, ``decision_function`` and ``predict``
   * - Detectors
     - 20 streaming detectors, see :doc:`../detectors`, and every PyOD detector through a sliding window
     - About 60 batch detectors, also deep learning, graph and time series detectors
   * - Memory
     - Bounded: a window or summary statistics
     - Depends on the detector and the size of the training data
   * - Transformers
     - Scalers, shingles and seasonal decomposition, one point at a time
     - None
   * - Projectors
     - Gaussian, sparse and StreamHash random projections
     - None
   * - Score ensemblers
     - 5 rules, one point at a time
     - The average, maximum, median, AOM and MOA of a score matrix, and ensemble detectors such as LSCP and SUOD
   * - Score postprocessors
     - Running averages, maximums, medians and z-scores of the scores
     - None
   * - Probability calibrators
     - Gaussian tail, conformal and SPOT, updated with each score
     - ``predict_proba`` with the ``linear`` or ``unify`` method, from the training scores
   * - Evaluation
     - Prequential AUROC, AUPR, precision and recall, also on a sliding window
     - ROC AUC and precision at rank n on a test set

Both libraries have LODA, an isolation forest, a local outlier factor and the median absolute deviation. In PySAD, these detectors learn from one point at a time: :class:`~pysad.models.LODA`, :class:`~pysad.models.IForestASD`, :class:`~pysad.models.OnlineIsolationForest`, :class:`~pysad.models.IncrementalLOF` and :class:`~pysad.models.MedianAbsoluteDeviation`. Use PyOD if all your data is available before you train a detector. Use PySAD if the points arrive one at a time and the detector must follow the changes of the stream.

Results
-------

The table compares the multivariate detectors of PySAD with batch detectors of PyOD. Each detector runs once on each of the |num_datasets| ODDS datasets of the :doc:`benchmark <../benchmark>`, with the same points and seed 0.

Each detector scores each point before it learns from the point, as :doc:`../benchmarks/methodology` describes. The metrics leave out the scores of the warm-up points, the first 256 points of a stream. The PySAD detectors use the settings of the benchmark. The PyOD detectors run in :class:`~pysad.models.integrations.ReferenceWindowModel` with their default parameters. Each detector first trains on the warm-up points. Then, after each 100 points, it trains a new detector on the last 1,000 points.

The table has one row for each detector, from the best mean rank to the worst. The ranks compare only the detectors in this table. The columns are the same as in the summary of :doc:`../benchmarks/multivariate`, and the best value of each column is in bold. The names of the PySAD detectors link to their documentation, and the names of the other detectors start with their module. The last column gives the version of the library that gave the results. For PySAD, the version also gives the commit of the run, because the run uses the code of the repository.

.. csv-table::
   :file: ../benchmark/comparison_pyod.csv
   :header-rows: 1
   :class: benchmark-table best-columns

All the runs used one computer, so compare the times only with each other. The time of a PyOD detector includes its training runs. A point that starts a training run takes much longer than the other points.

To run the comparison, follow the steps at the end of :doc:`river`. The same script writes both tables.
