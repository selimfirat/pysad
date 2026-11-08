PySAD vs Alibi Detect
=====================

`Alibi Detect <https://github.com/SeldonIO/alibi-detect>`_ does outlier, adversarial and drift detection, on tabular data, images, text and time series. Most of its outlier detectors learn from a batch of points, then score new batches. Its ``Mahalanobis`` detector learns online: it updates its mean and covariance with each batch that it scores. This page compares PySAD with Alibi Detect 0.13.0.

Functions
---------

.. list-table::
   :header-rows: 1
   :stub-columns: 1

   * - Function
     - PySAD
     - Alibi Detect
   * - Data
     - A stream, one point at a time
     - Batches of points
   * - Learn and score
     - ``fit_partial``, ``score_partial`` and ``fit_score_partial``
     - ``fit``, ``score`` and ``predict``. ``Mahalanobis`` learns in ``score``
   * - Outlier detectors
     - Streaming detectors, see :doc:`../detectors`, and every PyOD detector through a sliding window
     - 10: ``IForest``, ``Mahalanobis`` and ``SpectralResidual``, and 7 that need TensorFlow or Prophet, for example ``OutlierVAE`` and ``OutlierProphet``
   * - Drift detectors
     - None
     - Detectors that compare a batch with reference data, and online detectors, for example ``MMDDriftOnline`` and ``CVMDriftOnline``
   * - Adversarial detectors
     - None
     - ``AdversarialAE`` and ``ModelDistillation``
   * - Transformers and projectors
     - Scalers, shingles, seasonal decomposition and random projections, one point at a time
     - None
   * - Score ensemblers
     - 5 rules, one point at a time
     - None
   * - Score postprocessors
     - Running averages, maximums, medians and z-scores of the scores
     - None
   * - Scores to labels
     - Probability calibrators, then a threshold on the probability
     - ``infer_threshold``, which sets the threshold to a percentile of the scores of a batch, then ``predict``
   * - Evaluation
     - Prequential AUROC, AUPR, precision and recall, also on a sliding window
     - None

Both libraries have an isolation forest and the spectral residual. In PySAD, the isolation forests learn from one point at a time: :class:`~pysad.models.IForestASD` and :class:`~pysad.models.OnlineIsolationForest`. Alibi Detect's ``SpectralResidual`` scores a whole time series, and PySAD's :class:`~pysad.models.SpectralResidual` scores each new value. Use Alibi Detect if you need drift or adversarial detection, or deep learning detectors for images and text. Use PySAD if the points arrive one at a time and the detector must follow the changes of the stream.

Results
-------

The table compares the multivariate detectors of PySAD with ``Mahalanobis``, the outlier detector of Alibi Detect that learns online. Each detector runs once on each of the |num_datasets| ODDS datasets of the :doc:`benchmark <../benchmark>`, with the same points and seed 0.

Each detector scores each point before it learns from the point, as :doc:`../benchmarks/methodology` describes. The metrics leave out the scores of the warm-up points, the first 256 points of a stream. The PySAD detectors use the settings of the benchmark. ``Mahalanobis`` uses its default parameters, and it gets each point as a batch of one point. Each call of ``score`` scores the point and adds it to the mean and covariance.

The table has one row for each detector, from the best mean rank to the worst. The ranks compare only the detectors in this table. The columns are the same as in the summary of :doc:`../benchmarks/multivariate`, and the best value of each column is in bold. The names of the PySAD detectors link to their documentation, and the names of the other detectors start with their module. The last column gives the version of the library that gave the results. For PySAD, the version also gives the commit of the run, because the run uses the code of the repository.

.. csv-table::
   :file: ../benchmark/comparison_alibi_detect.csv
   :header-rows: 1
   :class: benchmark-table best-columns

All the runs used one computer, so compare the times only with each other.

To run the comparison, follow the steps at the end of :doc:`streamad`. The same script writes this table.
