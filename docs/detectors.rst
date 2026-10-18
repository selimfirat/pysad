Detectors
=========

PySAD has 16 streaming detectors. This page lists the input that each detector takes, what it needs before the stream starts and the memory that it keeps. For the parameters, follow the link to the API reference. To choose a detector, see :doc:`guides/choose_detector`.

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
     - Fixed: ``num_trees`` full trees of depth ``max_depth``, about 1.6 million nodes with the defaults
   * - :class:`~pysad.models.RobustRandomCutForest`
     - Collusive displacement in random cut trees :cite:`guha2016robust`
     - Nothing
     - Fixed: ``num_trees`` trees of ``tree_size`` points
   * - :class:`~pysad.models.RSHash`
     - Time-decayed counts of random subspace grids :cite:`sathe2016subspace`
     - The range of each feature
     - Grows: one count for each grid cell that a point visits
   * - :class:`~pysad.models.IForestASD`
     - Isolation Forest on a sliding window :cite:`ding2013anomaly`
     - Nothing (initial points optional)
     - Fixed: two windows of ``window_size`` points
   * - :class:`~pysad.models.KitNet`
     - Ensemble of autoencoders :cite:`mirsky2018kitsune`
     - Nothing
     - Fixed. Learns only during its grace periods (100,000 points by default)
   * - :class:`~pysad.models.ExactStorm`
     - Neighbor counts in a sliding window :cite:`angiulli2007detecting`
     - Nothing
     - Fixed: a window of ``window_size`` points
   * - :class:`~pysad.models.LocalOutlierProbability`
     - Local outlier probability :cite:`kriegel2009loop`, with PyNomaly :cite:`constantinou2018pynomaly`
     - Initial points (``initial_X``)
     - Fixed: the initial points. Does not learn from the stream
   * - :class:`~pysad.models.Inqmad`
     - Quantum measurement density estimation :cite:`gallego2022inqmad`
     - The number of features, the number of random Fourier features (``dim_x``) and the kernel bandwidth (``gamma``)
     - Fixed: a ``dim_x`` by ``dim_x`` matrix

Univariate detectors
--------------------

These detectors take points with exactly one feature. Give each value as an array of shape ``(1,)``.

.. list-table::
   :header-rows: 1
   :widths: 31 27 20 22

   * - Detector
     - Method
     - Needs before the stream
     - Memory
   * - :class:`~pysad.models.SeasonalHybridESD`
     - STL (seasonal-trend decomposition) residuals and a robust generalized ESD (extreme Studentized deviate) test :cite:`hochenbaum2017automatic`
     - The period, the window size and the maximum number of anomalies per window
     - Fixed: a window of ``window_size`` values
   * - :class:`~pysad.models.SeasonalESD`
     - STL residuals and a generalized ESD test :cite:`hochenbaum2017automatic`
     - The period, the window size and the maximum number of anomalies per window
     - Fixed: a window of ``window_size`` values
   * - :class:`~pysad.models.MedianAbsoluteDeviation`
     - Distance from the median of the last ``window_size`` values, in scaled median absolute deviations of those values (1.4826 × MAD) :cite:`hochenbaum2017automatic`
     - Nothing
     - Fixed: a window of ``window_size`` values
   * - :class:`~pysad.models.StandardAbsoluteDeviation`
     - Distance from the running mean in standard deviations :cite:`hochenbaum2017automatic`
     - Nothing
     - Fixed with the default mean. Grows with ``subtracted_statistic="median"``
   * - :class:`~pysad.models.KNNCAD`
     - Conformalized k-nearest neighbors :cite:`burnaev2016conformalized`
     - A probationary period: at least 48 values that it learns from before it scores
     - Grows: keeps every value in a buffer, but scores only against its recent training and calibration windows
   * - :class:`~pysad.models.RelativeEntropy`
     - Relative entropy of windowed histograms :cite:`wang2011statistical`
     - The range of the values
     - Grows slowly: a window and one histogram for each distinct state of the stream

PyOD integrations
-----------------

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
