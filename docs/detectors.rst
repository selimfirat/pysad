Detectors
=========

This page lists the input that each detector takes, what it needs before the stream starts and the memory that it keeps. For the parameters, follow the link to the API reference. To choose a detector, see :doc:`guides/choose_detector`. To compare the results of the multivariate detectors, see the :doc:`benchmark <benchmark>`.

In the memory column, "fixed" means that the memory has an upper limit that does not depend on the length of the stream. A window is the most recent points, up to a fixed number.

Multivariate detectors
----------------------

These detectors take points with one or more features. A feature is one measured value of a point.

.. list-table::
   :header-rows: 1
   :widths: 28 30 20 22

   * - Detector
     - Method
     - Needs before the stream
     - Memory
   * - :class:`~pysad.models.xStream`
     - Half-space chains on streamhash projections :cite:`xstream`
     - Nothing
     - Fixed: the chain counts of two windows, and the points of the first window until it is full
   * - :class:`~pysad.models.LODA`
     - Histograms on sparse random projections :cite:`pevny2016loda`
     - Nothing
     - Fixed: ``num_random_cuts`` histograms of ``num_bins`` bins
   * - :class:`~pysad.models.HalfSpaceTrees`
     - Mass profiles of random half-space trees :cite:`tan2011fast`
     - The range of each feature
     - Fixed: ``num_trees`` full trees of depth ``max_depth``, about 11 MB with the defaults
   * - :class:`~pysad.models.RobustRandomCutForest`
     - Collusive displacement in random cut trees :cite:`guha2016robust`
     - Nothing
     - Fixed: ``num_trees`` trees of ``tree_size`` points
   * - :class:`~pysad.models.RSForest`
     - Density of random space trees, scored against the previous window :cite:`wu2014rsforest`
     - Nothing
     - Fixed: ``num_trees`` full trees of depth ``max_depth``, about 32 MB with the defaults
   * - :class:`~pysad.models.RSHash`
     - Time-decayed counts of random subspace grids :cite:`sathe2016subspace`
     - The range of each feature
     - Fixed: ``num_hash_fns`` hash tables of ``hash_range`` slots
   * - :class:`~pysad.models.MStream`
     - Decayed count-min sketch counts of each feature and of the whole point, in the current time tick against the earlier ticks :cite:`bhatia2021mstream`
     - Nothing
     - Fixed: two counts for each of ``num_buckets`` buckets of every feature and of ``num_hash_fns`` sketch rows
   * - :class:`~pysad.models.IForestASD`
     - Isolation Forest on a sliding window :cite:`ding2013anomaly`
     - Nothing (initial points optional)
     - Fixed: two windows of ``window_size`` points
   * - :class:`~pysad.models.OnlineIsolationForest`
     - Online Isolation Forest :cite:`leveni2024online`
     - Nothing
     - Fixed: ``num_trees`` trees and a sliding window of ``window_size`` points
   * - :class:`~pysad.models.SmartSifter`
     - Gaussian mixture learned by sequentially discounting EM :cite:`yamanishi2000smartsifter`
     - Nothing
     - Fixed: ``num_components`` Gaussians, quadratic in the number of features
   * - :class:`~pysad.models.KitNet`
     - Ensemble of autoencoders :cite:`mirsky2018kitsune`
     - Nothing
     - Fixed. Learns only during its grace periods (1,100 points by default)
   * - :class:`~pysad.models.ExactStorm`
     - Neighbor counts in a sliding window :cite:`angiulli2007detecting`
     - Nothing
     - Fixed: a window of ``window_size`` points
   * - :class:`~pysad.models.ApproxStorm`
     - The neighbor counts of :class:`~pysad.models.ExactStorm`, estimated from a sample of safe inliers :cite:`angiulli2007detecting`
     - Nothing
     - Fixed: a window of ``window_size`` points
   * - :class:`~pysad.models.MCOD`
     - The neighbor counts of :class:`~pysad.models.ExactStorm`, faster with micro-clusters :cite:`kontaki2011continuous`
     - Nothing
     - Fixed: a window of ``window_size`` points and the centers of its micro-clusters
   * - :class:`~pysad.models.NETS`
     - The neighbor counts of :class:`~pysad.models.ExactStorm`, faster with a grid of cell counts :cite:`yoon2019nets`
     - Nothing
     - Fixed: a window of ``window_size`` points and the counts of its non-empty cells
   * - :class:`~pysad.models.CPOD`
     - The neighbor counts of :class:`~pysad.models.ExactStorm`, faster with core points that index the window by distance rings :cite:`tran2020cpod`
     - Nothing
     - Fixed: a window of ``window_size`` points, its core points and the window points within 2R of each core point
   * - :class:`~pysad.models.D3`
     - Negative number of window points within a radius, estimated from a kernel density over a chain sample of the window :cite:`subramaniam2006online`
     - Nothing
     - Fixed: a window of ``window_size`` points and a sample of ``sample_fraction`` times ``window_size`` points
   * - :class:`~pysad.models.ThreshLEAP`
     - The number of neighbors below k that a probe finds when it tests the window newest first and stops at k neighbors :cite:`cao2014scalable`
     - Nothing
     - Fixed: a window of ``window_size`` points
   * - :class:`~pysad.models.IncrementalLOF`
     - Local outlier factor in a sliding window :cite:`pokrajac2007incremental`
     - Nothing
     - Fixed: a window of ``window_size`` points and their ``num_neighbors`` nearest neighbors
   * - :class:`~pysad.models.OnlineOneClassSVM`
     - Online one-class support vector machine (NORMA) with a Gaussian kernel :cite:`kivinen2004online`
     - Nothing
     - Fixed: the support vectors among the last ``window_size`` points
   * - :class:`~pysad.models.DILOF`
     - The local outlier factor of :class:`~pysad.models.IncrementalLOF` in a window that keeps a density-preserving summary of old points :cite:`na2018dilof`
     - Nothing
     - Fixed: a window of at most ``window_size`` points and their ``num_neighbors`` nearest neighbors
   * - :class:`~pysad.models.KELOS`
     - How much lower the kernel density at a point is than the densities at its nearest micro-cluster centers in the window (KLOME) :cite:`qin2019scalable`
     - Nothing
     - Fixed: a window of ``window_size`` points and the sums of its micro-clusters
   * - :class:`~pysad.models.AnyOut`
     - Distance to the mean, or Gaussian density, of the closest cluster at the level of a ClusTree that the time budget ``depth`` reaches :cite:`assent2012anyout`
     - Nothing
     - Fixed: a tree of at most ``max_height`` levels, with at most 3 to the power ``max_height`` leaf clusters
   * - :class:`~pysad.models.ADeMS`
     - Residual of each point after its projection on a Frequent Directions sketch of the normal points :cite:`huang2015streaming`
     - Nothing (initial points to use a threshold)
     - Fixed: a sketch of ``sketch_size`` columns, linear in the number of features
   * - :class:`~pysad.models.LocalOutlierProbability`
     - Local outlier probability :cite:`kriegel2009loop`, as in PyNomaly :cite:`constantinou2018pynomaly`
     - Initial points (``initial_X``)
     - Fixed: the initial points. Does not learn from the stream
   * - :class:`~pysad.models.Inqmad`
     - Quantum measurement density estimation :cite:`gallego2022inqmad`
     - The number of features, the number of random Fourier features (``dim_x``) and the kernel bandwidth (``gamma``)
     - Fixed: a ``dim_x`` by ``dim_x`` matrix
   * - :class:`~pysad.models.TEDA`
     - Eccentricity from the recursive mean and variance :cite:`angelov2014anomaly`
     - Nothing
     - Fixed: the count, the mean and the variance

Univariate detectors
--------------------

These detectors take points with exactly one feature. Give each value as an array of shape ``(1,)``. :class:`~pysad.models.HTM` also takes the time of each value: give each point as the array ``[timestamp, value]``.

.. list-table::
   :header-rows: 1
   :widths: 31 27 20 22

   * - Detector
     - Method
     - Needs before the stream
     - Memory
   * - :class:`~pysad.models.SeasonalHybridESD`
     - STL (seasonal-trend decomposition) residuals and a robust generalized ESD (extreme Studentized deviate) test :cite:`hochenbaum2017automatic`
     - The period and the window size
     - Fixed: a window of ``window_size`` values
   * - :class:`~pysad.models.SeasonalESD`
     - STL residuals and a generalized ESD test :cite:`hochenbaum2017automatic`
     - The period and the window size
     - Fixed: a window of ``window_size`` values
   * - :class:`~pysad.models.MedianAbsoluteDeviation`
     - Distance from the median of the last ``window_size`` values, in scaled median absolute deviations of those values (1.4826 × MAD) :cite:`hochenbaum2017automatic`
     - Nothing
     - Fixed: a window of ``window_size`` values
   * - :class:`~pysad.models.StandardAbsoluteDeviation`
     - Distance from the running mean or recent median in standard deviations :cite:`hochenbaum2017automatic`
     - ``window_size`` when using ``subtracted_statistic="median"``
     - Fixed; median mode retains only the last ``window_size`` values while variance remains an all-history running statistic
   * - :class:`~pysad.models.HoltWinters`
     - Violations of a confidence band of seasonal deviations around a Holt-Winters forecast, counted in a moving window :cite:`brutlag2000aberrant`
     - The period
     - Fixed: a seasonal coefficient and a deviation for each of the ``period`` points of the cycle, and the last ``window_size`` violations
   * - :class:`~pysad.models.ChangeFinder`
     - Logarithmic loss under an autoregressive model that sequentially discounting learning fits, and a second such model on the smoothed losses for change points :cite:`takeuchi2006unifying`
     - Nothing
     - Fixed: ``order`` values and ``window_size`` scores
   * - :class:`~pysad.models.HTM`
     - The fraction of active columns that the temporal memory of Hierarchical Temporal Memory did not predict, turned into an anomaly likelihood :cite:`ahmad2017unsupervised`
     - The range of the values (``min_val`` and ``max_val``), and a timestamp in seconds since the Unix epoch with each value
     - Grows slowly: at most 128 segments of 32 synapses for each of the 65,536 cells, and a window of 8,640 scores
   * - :class:`~pysad.models.KNNCAD`
     - Conformalized k-nearest neighbors :cite:`burnaev2016conformalized`
     - A probationary period: at least 48 values that it learns from before it scores
     - Fixed: the last 19 values and about 2 × ``probationary_period`` training and calibration windows of 19 values
   * - :class:`~pysad.models.MatrixProfile`
     - Distance from the newest subsequence to its nearest earlier subsequence, the matrix profile of STAMPI :cite:`yeh2016matrix`
     - The subsequence length (``window_length``)
     - Fixed: the last ``history_size`` values
   * - :class:`~pysad.models.DAMP`
     - Approximate distance from the newest subsequence to its nearest earlier subsequence, exact for each new left-discord, with a backward search that stops early :cite:`lu2022matrix`
     - The subsequence length (``window_length``)
     - Fixed: the last ``history_size + window_length - 1`` values
   * - :class:`~pysad.models.SpectralResidual`
     - Saliency of the newest value in the spectral residual of a sliding window, with estimated values after it :cite:`ren2019time`
     - The window size
     - Fixed: a window of ``window_size`` values
   * - :class:`~pysad.models.RelativeEntropy`
     - Relative entropy of windowed histograms :cite:`wang2011statistical`
     - The range of the values
     - Grows slowly: a window and one histogram for each distinct state of the stream
   * - :class:`~pysad.models.SAND`
     - Distance of the latest subsequence to weighted k-Shape clusters of past subsequences :cite:`boniol2021sand`
     - The subsequence length
     - Grows slowly: a batch of ``batch_size`` values and one ``centroid_length`` by ``centroid_length`` matrix per cluster, with at most ``num_clusters`` new clusters per batch

Integrations
------------

These wrappers run a batch detector from `PyOD <https://github.com/yzhao062/pyod>`_ on a stream. A batch detector trains on a complete dataset. For instructions, see :doc:`guides/pyod`.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Wrapper
     - Behavior
   * - :class:`~pysad.models.integrations.ReferenceWindowModel`
     - Trains the detector on a window of recent points and trains it again every ``sliding_size`` points :cite:`xstream`.
   * - :class:`~pysad.models.integrations.OneFitModel`
     - Trains the detector once on initial points and then only scores.

:class:`~pysad.models.integrations.PredictiveAnomalyDetector` scores each value of a univariate stream by its deviation from the one-step forecast of a `River <https://riverml.xyz/>`_ forecaster, the plug-in method of EGADS :cite:`laptev2015generic`. The forecaster learns each value after the model scores it.

Baselines for testing
---------------------

These models do not detect anomalies. Use them to test a pipeline or as a reference in an evaluation.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Model
     - Score
   * - :class:`~pysad.models.NullModel`
     - Always 0.5.
   * - :class:`~pysad.models.RandomModel`
     - A random value between 0 and 1.
   * - :class:`~pysad.models.PerfectModel`
     - The label of the point (the known answer: anomaly or normal). It gives the best possible result for a metric.
