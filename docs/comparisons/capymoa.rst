PySAD vs CapyMOA
================

`CapyMOA <https://github.com/adaptive-machine-learning/CapyMOA>`_ does machine learning on streams: classification, regression, clustering, drift detection and anomaly detection. It runs the learners of MOA, a Java library, from Python, and it adds learners in Python. Like PySAD, it learns from one point at a time. This page compares PySAD with CapyMOA 0.15.1.

Functions
---------

.. list-table::
   :header-rows: 1
   :stub-columns: 1

   * - Function
     - PySAD
     - CapyMOA
   * - Needs
     - Python
     - Python and Java 9 or newer
   * - Points
     - NumPy arrays of shape ``(num_features,)``
     - ``Instance`` objects of a ``Schema``, for example from ``Instance.from_array``
   * - Learn and score
     - ``fit_partial``, ``score_partial`` and ``fit_score_partial``
     - ``train`` and ``score_instance``
   * - Streaming detectors
     - See :doc:`../detectors`
     - 10: ``AdaptiveIsolationForest``, ``Autoencoder``, ``HalfSpaceTrees``, ``IForestASD``, ``Loda``, ``OnlineIsolationForest``, ``RSHash``, ``RobustRandomCutForest``, ``StreamRHF`` and ``StreamingIsolationForest``
   * - Batch detectors
     - Every PyOD detector, through :class:`~pysad.integrations.ReferenceWindowModel` and :class:`~pysad.integrations.OneFitModel`
     - None
   * - Transformers
     - Scalers, shingles and seasonal decomposition
     - The filters of MOA through ``MOATransformer``, for example normalization and standardization
   * - Projectors
     - Gaussian, sparse and StreamHash random projections
     - Random projections and the hashing trick, as filters of MOA
   * - Score ensemblers
     - 5 rules, for example the average and the maximum of the scores
     - None
   * - Score postprocessors
     - Running averages, maximums, medians and z-scores of the scores
     - None
   * - Probability calibrators
     - Gaussian tail, conformal and SPOT
     - None
   * - Prequential evaluation
     - ``evaluate_stream``, with AUROC, AUPR, precision and recall, also on a sliding window
     - ``prequential_evaluation_anomaly``, with the AUC of MOA, also on windows of points

Both libraries have Half-Space Trees, LODA, RS-Hash, the robust random cut forest, the Online Isolation Forest and IForestASD, an isolation forest on a sliding window. CapyMOA also has two other isolation forests for streams, ``AdaptiveIsolationForest`` and ``StreamingIsolationForest``, and StreamRHF, a forest of random histograms, and an autoencoder. PySAD has more detectors, and also the tools around a detector: ensemblers, postprocessors and calibrators. CapyMOA has the other tasks of machine learning on streams, with the learners of MOA. Use CapyMOA if anomaly detection is one step of a larger CapyMOA or MOA pipeline.

Results
-------

The table compares the multivariate detectors of PySAD with the streaming detectors of CapyMOA. Each detector runs once on each of the |num_datasets| ODDS datasets of the :doc:`benchmark <../benchmark>`, with the same points and seed 0.

Each detector scores each point before it learns from the point, as :doc:`../benchmarks/methodology` describes. The metrics leave out the scores of the warm-up points, the first 256 points of a stream. The PySAD detectors use the settings of the benchmark. The CapyMOA detectors use their default parameters and seed 0. Their points are ``Instance`` objects, and the times do not include the conversion of the points to ``Instance`` objects. The table leaves out ``Autoencoder``, because it needs PyTorch.

The table has one row for each detector, from the best mean rank to the worst. The ranks compare only the detectors in this table. The columns are the ones of the summary of :doc:`../benchmarks/multivariate`, plus a Mean AUPR column, and the best value of each column is in bold. The names of the PySAD detectors link to their documentation, and the names of the other detectors start with their module. The last column gives the version of the library that gave the results. For PySAD, the version also gives the commit of the run, because the run uses the code of the repository.

.. csv-table::
   :file: ../benchmark/comparison_capymoa.csv
   :header-rows: 1
   :class: benchmark-table best-columns

All the runs used one computer, so compare the times only with each other.

CapyMOA's ``HalfSpaceTrees`` gives every point the score 0 until its first window of 250 points is full, which yields an AUROC of 0.500 on small datasets after the warm-up.

CapyMOA needs Java 9 or newer. To check your version, run ``java -version``. If Java is missing, install a Java runtime, for example with ``sudo apt-get install default-jre-headless`` on Debian or Ubuntu, or with ``brew install openjdk`` on macOS. If you have more than one Java, set ``JAVA_HOME`` to the one that CapyMOA must use. Then follow the steps at the end of :doc:`river`. The same script writes this table.
