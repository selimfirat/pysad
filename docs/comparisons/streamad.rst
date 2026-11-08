PySAD vs StreamAD
=================

`StreamAD <https://github.com/Fengrui-Liu/StreamAD>`_ does anomaly detection on streams. Like PySAD, it learns from one point at a time. Its last release, 0.3.1, is from May 2023. It needs NumPy 1, pandas 1 and statsmodels 0.13.5. This page compares PySAD with StreamAD 0.3.1.

Functions
---------

.. list-table::
   :header-rows: 1
   :stub-columns: 1

   * - Function
     - PySAD
     - StreamAD
   * - Points
     - NumPy arrays of shape ``(num_features,)``
     - NumPy arrays of shape ``(num_features,)``
   * - Learn and score
     - ``fit_partial``, ``score_partial`` and ``fit_score_partial``
     - ``fit_score``, which learns from a point before it scores the point, unless ``score_first=True``
   * - Streaming detectors
     - See :doc:`../detectors`
     - 6 multivariate: ``HSTreeDetector``, ``LodaDetector``, ``OCSVMDetector``, ``RShashDetector``, ``RrcfDetector`` and ``xStreamDetector``. 7 univariate: ``KNNDetector``, ``MadDetector``, ``SArimaDetector``, ``SRDetector``, ``SpotDetector``, ``ZScoreDetector`` and ``ZSpotDetector``
   * - Batch detectors
     - Every PyOD detector, through :class:`~pysad.models.integrations.ReferenceWindowModel` and :class:`~pysad.models.integrations.OneFitModel`
     - None
   * - Transformers
     - Scalers, shingles and seasonal decomposition
     - ``detrend=True`` in each detector, which subtracts the mean of the last points
   * - Projectors
     - Gaussian, sparse and StreamHash random projections
     - None
   * - Score ensemblers
     - 5 rules, for example the average and the maximum of the scores
     - ``WeightEnsemble``, a weighted average, and ``VoteEnsemble``, a majority vote
   * - Score postprocessors
     - Running averages, maximums, medians and z-scores of the scores
     - ``ZScoreCalibrator``, which maps each score to [0, 1] with a running z-score
   * - Scores to labels
     - Probability calibrators, then a threshold on the probability
     - ``TDigestCalibrator``, which gives 1 to a score outside running percentiles of the scores
   * - Probability calibrators
     - Gaussian tail, conformal and SPOT
     - None
   * - Running statistics
     - Meters with ``update`` and ``remove``, see :doc:`../api/statistics`
     - ``StreamStatistic``: the maximum, minimum, mean, standard deviation, sum and variance, of all points or of a window
   * - Stream simulators
     - ``ArrayStreamer`` and ``PandasStreamer``
     - ``StreamGenerator``
   * - Evaluation
     - ``evaluate_stream``, with AUROC, AUPR, precision and recall, also on a sliding window
     - Precision, recall and F-score of labels, point by point, by anomaly series, or as in NAB

Both libraries have Half-Space Trees, LODA, RS-Hash, the robust random cut forest, xStream, KNN-CAD, the spectral residual and the median absolute deviation. StreamAD's ``SpotDetector`` applies SPOT to the values of a univariate stream. PySAD applies SPOT to the scores of a detector, in :class:`~pysad.transform.probability_calibration.SPOTProbabilityCalibrator`. PySAD has more detectors, also the batch detectors of PyOD, and probability calibrators.

Results
-------

The table compares the multivariate detectors of PySAD with the multivariate detectors of StreamAD. Each detector runs once on each of the |num_datasets| ODDS datasets of the :doc:`benchmark <../benchmark>`, with the same points and seed 0.

Each detector scores each point before it learns from the point, as :doc:`../benchmarks/methodology` describes. The metrics leave out the scores of the warm-up points, the first 256 points of a stream. The PySAD detectors use the settings of the benchmark. The StreamAD detectors use their default parameters, with ``score_first=True``. ``OCSVMDetector`` and ``RrcfDetector`` fail with ``score_first=True``, so they use the default, which learns from each point before it scores the point. The StreamAD detectors take no seed. They draw from NumPy's global random state, which the comparison seeds with 0. A StreamAD detector gives no score until its first window of 50 points is full. The comparison uses the score 0 for these points.

The table has one row for each detector, from the best mean rank to the worst. The ranks compare only the detectors in this table. The columns are the same as in the summary of :doc:`../benchmarks/multivariate`, and the best value of each column is in bold. The names of the PySAD detectors link to their documentation, and the names of the other detectors start with their module. The last column gives the version of the library that gave the results. For PySAD, the version also gives the commit of the run, because the run uses the code of the repository.

.. csv-table::
   :file: ../benchmark/comparison_streamad.csv
   :header-rows: 1
   :class: benchmark-table best-columns

All the runs used one computer, so compare the times only with each other.

StreamAD's ``HSTreeDetector`` returns the mass of a point, as in the paper of Half-Space Trees, and ``RShashDetector`` and ``xStreamDetector`` return counts or frequencies. All three are higher for normal points, so these detectors get an AUROC below 0.5.

To run this comparison, install the ``compare-numpy1`` dependency group, which needs Python 3.11 or older. The ``accumulation-tree`` package, which StreamAD needs, has no wheels, so you also need a C compiler. Then run the script from the root of the repository:

.. code-block:: console

   $ pip install -e . --group compare-numpy1
   $ python benchmarks/compare.py

The script writes ``docs/benchmark/comparison_streamad.csv`` and ``docs/benchmark/comparison_alibi_detect.csv``.
