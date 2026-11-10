Examples
========

These scripts come from the `examples folder <https://github.com/selimfirat/pysad/tree/master/examples>`_ on GitHub. The automatic tests run each script end to end on the ``master`` branch. To run a script, download it. Then run ``python <script>.py``. The scripts download the benchmark datasets that they need.

To try PySAD in the browser without an installation, open the `NYC taxi demo in JupyterLite <https://selimfirat.github.io/pysad/lab/index.html?path=nyc_taxi_alerts.ipynb>`_ or the `quick start notebook in Colab <https://colab.research.google.com/github/selimfirat/pysad/blob/master/examples/quickstart.ipynb>`_.

Score a benchmark dataset
-------------------------

This script is the shortest complete loop. It streams a dataset, scores each point and measures AUROC. AUROC is a number that shows how well the scores separate anomalies from normal points.

.. literalinclude:: ../examples/example_usage_short.py
   :language: python

It prints:

.. code-block:: text

   AUROC: 0.758

Score a stream with LODA
------------------------

This script scores each point of cardio with LODA. LODA estimates the density of a point from sparse random projections. It prints the AUROC of the scores against the labels.

.. literalinclude:: ../examples/example_loda.py
   :language: python

It prints:

.. code-block:: text

   AUROC: 0.946

Score a stream with HalfSpaceTrees
----------------------------------

This script scores each point of cardio with HalfSpaceTrees. It compares each point to the density of the previous window, using the min and max of each feature. It prints the AUROC of the scores against the labels.

.. literalinclude:: ../examples/example_half_space_trees.py
   :language: python

It prints:

.. code-block:: text

   AUROC: 0.928

Score a stream with xStream
---------------------------

This script scores each point of cardio with xStream. xStream bins StreamHash projections into half-space chains. It prints the AUROC of the scores against the labels.

.. literalinclude:: ../examples/example_xstream.py
   :language: python

It prints:

.. code-block:: text

   AUROC: 0.845

Score a stream with KitNet
--------------------------

This script scores each point of cardio with KitNet. KitNet is an ensemble of small autoencoders made for network traffic. It prints the AUROC of the scores against the labels.

.. literalinclude:: ../examples/example_kitnet.py
   :language: python

It prints:

.. code-block:: text

   AUROC: 0.878

Score a synthetic stream
------------------------

This script makes a labeled stream with :func:`~pysad.utils.generate_stream` and scores it. The stream has point anomalies, bursts of anomalies and a slow drift. It needs no download, so it suits quick tests.

.. literalinclude:: ../examples/example_generate_stream.py
   :language: python

It prints:

.. code-block:: text

   AUROC: 0.867

Integrate a one-dimensional stream
----------------------------------

This script uses one feature per point. It scores each point before the update and calibrates the raw score.
See :doc:`guides/one_dimensional_integration` for input shapes, feature bounds, and the py-outlier wrapper mismatch.

.. literalinclude:: ../examples/example_one_dimensional_integration.py
   :language: python

Scale features online
---------------------

This script scales each incoming feature vector with an instance standard scaler (zero mean, unit variance per instance) before feeding it to a detector.

.. literalinclude:: ../examples/example_preprocessing.py
   :language: python

It prints:

.. code-block:: text

   AUROC: 0.789

Give a detector the recent history
----------------------------------

This script turns each point of a univariate NAB series into a window of its 24 most recent points. LODA then scores these multivariate shingles. See :doc:`guides/pipelines`.

.. literalinclude:: ../examples/example_shingles.py
   :language: python

It prints:

.. code-block:: text

   AUROC: 0.627

Build a full pipeline
---------------------

This script shows how the stages of a pipeline connect. It scales each point, scores it with xStream and smooths the score. It does not tune the stages, so its AUROC is lower than the plain loop above. Measure each stage on your data, as :doc:`guides/evaluate` shows. :doc:`guides/pipelines` explains each stage.

.. literalinclude:: ../examples/example_usage.py
   :language: python

It prints:

.. code-block:: text

   AUROC: 0.543

Chain the stages into one model
-------------------------------

This script puts a scaler, xStream, a running z-score and a calibrator into one :class:`~pysad.models.StreamPipeline`. Then it scores the stream with the pipeline as with a single detector. See :doc:`guides/pipelines`.

.. literalinclude:: ../examples/example_stream_pipeline.py
   :language: python

It prints:

.. code-block:: text

   AUROC: 0.742

Use a River pipeline
--------------------

This script puts a PySAD detector after a River scaler. It scores each dictionary before learning from it. See :doc:`guides/pipelines` for the fixed feature mapping.

.. literalinclude:: ../examples/example_river_pipeline.py
   :language: python

Use a scikit-learn pipeline
----------------------------------------

Fit the scaler and PySAD model on historical data before wrapping them.
Batch scoring leaves the model unchanged.
Lower ``score_samples`` values mean more anomalous instances.
The threshold uses PySAD score units. ``predict`` returns -1 above the threshold and 1 at or below it.

.. literalinclude:: ../examples/example_sklearn_pipeline.py
   :language: python

Project high-dimensional data
-----------------------------

This script projects each point into fewer features before scoring it with an online anomaly detector.

.. literalinclude:: ../examples/example_projection.py
   :language: python

It prints:

.. code-block:: text

   AUROC: 0.749

Evaluate in windows
-------------------

A single score over the whole stream hides how a detector's separation changes over time. This script scores each point once, then reports the AUROC of consecutive windows of 100 points next to the AUROC over the whole stream. The windowed score averages the windows, so a stretch where the scores collapse does not dominate it.

.. literalinclude:: ../examples/example_windowed_metric.py
   :language: python

It prints:

.. code-block:: text

   AUROC: 0.638
   Windowed AUROC: 0.755

Evaluate with several metrics
-----------------------------

This script runs :func:`~pysad.evaluation.evaluate_stream` once and gets three metrics. The function scores each point before the detector learns from it.

.. literalinclude:: ../examples/example_evaluate_stream.py
   :language: python

It prints:

.. code-block:: text

   AUROC: 0.768
   AUPR: 0.489
   Windowed AUROC: 0.756

Raise alerts with a calibrator
------------------------------

This script uses a calibrator to convert each score into a value between 0 and 1. It raises an alert when this value is above a threshold. See :doc:`guides/alerts`.

.. literalinclude:: ../examples/example_probability_calibration.py
   :language: python

It prints:

.. code-block:: text

   27 alerts, 17 of them true anomalies

Label scores with a fixed threshold
-----------------------------------

This script calibrates each score to a value between 0 and 1. Then it turns the value into a 0/1 label with a fixed threshold, and measures the precision and the recall of the labels. See :doc:`guides/alerts`.

.. literalinclude:: ../examples/example_threshold_filter.py
   :language: python

It prints:

.. code-block:: text

   Precision: 0.656, recall: 0.318

Raise alerts at a chosen risk
-----------------------------

This script fits a Generalized Pareto distribution to the highest scores with SPOT (streaming peaks-over-threshold). You choose the risk ``q``, and the script raises an alert when the calibrated score is above ``1 - q``, so a normal point raises an alert with probability ``q``. See :doc:`guides/alerts`.

.. literalinclude:: ../examples/example_spot_calibration.py
   :language: python

It prints:

.. code-block:: text

   32 alerts, 27 of them true anomalies
   alarm threshold on the raw scores: 3.013

Raise alerts with the anomaly likelihood
----------------------------------------

This script converts raw scores between 0 and 1 into the anomaly likelihood of HTM. The likelihood compares the average of the recent scores with the distribution of these averages over a longer window. So a sustained increase of the scores raises an alert, and most single noisy scores do not. See :doc:`guides/alerts`.

.. literalinclude:: ../examples/example_anomaly_likelihood.py
   :language: python

It prints:

.. code-block:: text

   22 raw scores are at least 0.99
   likelihood alerts at [549, 2620, 4001], the anomaly starts at 4000

Plot the alerts
---------------

This script calibrates the scores of LODA on a synthetic stream. Then it plots the features, the calibrated scores with the threshold, and the alerts against the labels. It needs matplotlib, which PySAD does not install. See :doc:`guides/alerts`.

.. literalinclude:: ../examples/example_plot_stream.py
   :language: python

Flag anomalies with a dynamic threshold
---------------------------------------

This script scores the telemetry value of a SMAP channel with the median absolute deviation. Then it applies the nonparametric dynamic threshold of Telemanom, which outputs 0 for a nominal point and the score of the anomalous sequence for a flagged point. See :doc:`guides/alerts`.

.. literalinclude:: ../examples/example_dynamic_threshold.py
   :language: python

It prints:

.. code-block:: text

   Flagged points: 37, of which in an anomaly range: 25
   AUROC: 0.646

Label scores with a running quantile
------------------------------------

This script scores the cardio dataset with LODA. Then it labels each score 1 if the score is at least the 0.9 quantile of the previous scores, and 0 otherwise. See :doc:`guides/alerts`.

.. literalinclude:: ../examples/example_quantile_filter.py
   :language: python

It prints:

.. code-block:: text

   Precision: 0.381, recall: 0.636

Postprocess scores
------------------

This script standardizes the streaming anomaly scores of a detector using a running z-score window.

.. literalinclude:: ../examples/example_postprocessing.py
   :language: python

It prints:

.. code-block:: text

   AUROC: 0.778

Combine detectors
-----------------

This script normalizes the scores of two detectors. Then it averages them with an ensembler, which combines the scores of several detectors.

.. literalinclude:: ../examples/example_ensemble.py
   :language: python

It prints:

.. code-block:: text

   AUROC: 0.811

Explain an anomaly
------------------

This script shifts two features of a point and asks :class:`~pysad.models.LODA` which features make the point anomalous. For each feature, :meth:`~pysad.models.LODA.explain_partial` compares the projections that use the feature with the projections that do not.

.. literalinclude:: ../examples/example_loda_explanation.py
   :language: python

It prints:

.. code-block:: text

   Score: 3.46
   Features by contribution: [2, 5, 0, 1, 4, 3, 7, 6]

Run a PyOD model
----------------

This script runs a batch Isolation Forest on a sliding window (the most recent points). A batch detector needs all of its data before it can score. See :doc:`guides/pyod`.

.. literalinclude:: ../examples/example_pyod_integration.py
   :language: python

It prints:

.. code-block:: text

   AUROC: 0.807

Fit a PyOD model once
---------------------

This script fits a batch KNN on the first 100 points, then only scores the rest of the stream. The model does not learn from the stream. See :doc:`guides/pyod`.

.. literalinclude:: ../examples/example_one_fit_model.py
   :language: python

It prints:

.. code-block:: text

   AUROC: 0.857

Score forecast errors of a River forecaster
-------------------------------------------

This script scores a seasonal series with spikes with :class:`~pysad.models.integrations.PredictiveAnomalyDetector`. A River ``SNARIMAX`` forecaster predicts each value from the last 24 values, and the score is the absolute error of the forecast. River needs Python 3.11 or newer.

.. literalinclude:: ../examples/example_predictive_anomaly_detector.py
   :language: python

It prints:

.. code-block:: text

   AUROC: 1.000

Find anomalies in a seasonal series
-----------------------------------

This script compares Seasonal ESD and Seasonal Hybrid ESD on a series with a daily cycle. See :doc:`guides/seasonal`.

.. literalinclude:: ../examples/example_seasonal_esd.py
   :language: python

It prints:

.. code-block:: text

   SeasonalESD flagged [150, 260, 340], true anomalies [150, 260, 340]
   SeasonalHybridESD flagged [150, 260, 340], true anomalies [150, 260, 340]

Both variants find the three spikes and flag no normal values. :doc:`guides/seasonal` explains when to use each variant.

Detect an outage with Holt-Winters forecasting
----------------------------------------------

This script finds a 12-hour drop in a series with a daily cycle with :class:`~pysad.models.HoltWinters`. The model forecasts each value from a baseline, a trend and a seasonal cycle. A value is a violation if it falls outside a band of two seasonal deviations around its forecast. A failure is at least 7 violations among the last 9 values.

.. literalinclude:: ../examples/example_holt_winters.py
   :language: python

It prints:

.. code-block:: text

   Failures from 606 to 625, outage at [600, 612)

The first failure is the 7th value of the outage. The failures go on after the outage ends, because the baseline learned part of the drop, so the normal values are above their forecasts until the baseline adapts again.

Find an unusual shape with the matrix profile
---------------------------------------------

This script scores a sine series with :class:`~pysad.models.MatrixProfile`. One cycle has twice the frequency, but each of its values is in the normal range. The subsequence that ends at a value gets a high score if no earlier subsequence has a similar shape.

.. literalinclude:: ../examples/example_matrix_profile.py
   :language: python

It prints:

.. code-block:: text

   Highest score at 1237, anomalous cycle at [1200, 1250)

Find an unusual shape in a long series with DAMP
------------------------------------------------

This script scores a sine series of 20,000 values with :class:`~pysad.models.DAMP`. As in the previous example, one cycle has twice the frequency. The backward search of DAMP can reach every earlier value, but it stops as soon as it finds a neighbor closer than the best discord score so far. So most values cost only a search over the last cycles, and the anomalous cycle still gets its exact score.

.. literalinclude:: ../examples/example_damp.py
   :language: python

It prints:

.. code-block:: text

   Most unusual subsequence at [15001, 15051), anomalous cycle at [15000, 15050)

Score spikes with the spectral residual
---------------------------------------

This script scores a seasonal series with spikes with :class:`~pysad.models.SpectralResidual`. For each value, the model appends estimated values after the window that ends with it. It then scores the value by its saliency in the spectral residual of the window, relative to the saliency of the values before it.

.. literalinclude:: ../examples/example_spectral_residual.py
   :language: python

It prints:

.. code-block:: text

   AUROC: 0.856

Find unusual subsequences with SAND
-----------------------------------

This script scores a periodic series with SAND. SAND clusters the subsequences of each batch of values and scores the latest subsequence by its distance to the clusters. Four subsequences of the series have a different shape. ``subsequence_length`` is the length of these anomalies.

.. literalinclude:: ../examples/example_sand.py
   :language: python

It prints:

.. code-block:: text

   AUROC after the first batch: 0.939

Remove a trend and a cycle with OneShotSTL
------------------------------------------

This script removes the trend and the daily cycle of a series with :class:`~pysad.transform.preprocessing.OneShotSTL`. Then :class:`~pysad.models.StandardAbsoluteDeviation` scores the residuals. The level of the series jumps halfway through. See :doc:`guides/seasonal`.

.. literalinclude:: ../examples/example_one_shot_stl.py
   :language: python

It prints:

.. code-block:: text

   AUROC: 0.990

Stream a pandas DataFrame
-------------------------

This script scores the rows of a DataFrame with a time index. Then it prints the timestamps with the highest scores.

.. literalinclude:: ../examples/example_pandas_streamer.py
   :language: python

It prints:

.. code-block:: text

   Highest-scoring anomaly alerts:
   2025-01-01T11:10:00 score=5.6249
   2025-01-01T12:00:00 score=5.5371
   2025-01-01T10:20:00 score=5.0840
   2025-01-01T11:58:00 score=4.1992
   2025-01-01T11:05:00 score=4.1908

Stream a Polars DataFrame
-------------------------

This script scores the rows of a Polars DataFrame. It keeps the timestamps in a column outside the features. Then it prints the timestamps with the highest scores.

.. literalinclude:: ../examples/example_polars_streamer.py
   :language: python

It prints:

.. code-block:: text

   Highest-scoring anomaly alerts:
   2025-01-01T11:10:00 score=5.8440
   2025-01-01T12:00:00 score=5.8106
   2025-01-01T10:20:00 score=4.8580
   2025-01-01T11:58:00 score=4.2457
   2025-01-01T11:05:00 score=4.2420

Use a density detector
----------------------

This script scores each point with Inqmad. Inqmad estimates how dense the stream is near each point. It scales the features with its first points, so it takes the raw points.

.. literalinclude:: ../examples/example_usage_inqmad.py
   :language: python

It prints:

.. code-block:: text

   AUROC: 0.739

Use a Gaussian mixture detector
-------------------------------

This script scores each point with SmartSifter. SmartSifter learns a mixture of Gaussians online and gives each point its negative log-likelihood under the mixture. The ``discount`` parameter sets how fast the mixture forgets old points.

.. literalinclude:: ../examples/example_usage_smart_sifter.py
   :language: python

It prints:

.. code-block:: text

   AUROC: 0.672

Detect anomalies with HTM
-------------------------

This script runs HTM, the detector of Numenta in NAB, on the New York City taxi series of NAB. HTM takes each point with its timestamp and learns the sequence of the values and of the times of day. It scores each point by the anomaly likelihood of the fraction of its active columns that it did not predict.

.. literalinclude:: ../examples/example_htm.py
   :language: python

It prints:

.. code-block:: text

   15 alerts, 3 of them in the 5 anomaly windows

Find a change point
-------------------

This script finds a level shift in a noisy series with ChangeFinder. ChangeFinder learns an autoregressive model of the series online and scores each value by its negative log-likelihood under the model. A second model learns the moving average of these scores, and its smoothed scores are the change point scores.

.. literalinclude:: ../examples/example_change_finder.py
   :language: python

It prints:

.. code-block:: text

   Highest change point score at 604, level shift at 600

Use an eccentricity detector
----------------------------

This script scores each point with TEDA. TEDA keeps only the running mean and variance of the stream. It scores each point by its squared distance to the mean, divided by the variance. A score above :math:`m^2` is the paper's alert rule for :math:`m` standard deviations.

.. literalinclude:: ../examples/example_usage_teda.py
   :language: python

It prints:

.. code-block:: text

   AUROC: 0.791
   Alerts: 2

Use a random space tree detector
--------------------------------

This script scores each point with RS-Forest. RS-Forest splits the feature space with random trees and gives each point the negative density of the previous window in the nodes that hold it. Its first window sets the feature ranges of the trees.

.. literalinclude:: ../examples/example_rs_forest.py
   :language: python

It prints:

.. code-block:: text

   AUROC: 0.651

Load credit-card fraud transactions
-----------------------------------

This script loads OpenML 1597 through the shared dataset cache and streams the first five transactions in their original time order. ``Class`` is separate from the 30 features. See :doc:`guides/stream_data` for the source and dataset license.

.. literalinclude:: ../examples/example_creditcard.py
   :language: python

Use a multi-aspect stream detector
----------------------------------

This script scores the http connections of KDD Cup 1999 in time order with MStream. MStream counts the records in buckets of each feature and of the whole record. A record scores high when its buckets hold many more records in the current time tick than in an average earlier tick, as in a burst of similar connections.

.. literalinclude:: ../examples/example_mstream.py
   :language: python

It prints:

.. code-block:: text

   AUROC: 0.992

Detect a burst of graph edges
------------------------------

This script scores a small, reproducible edge stream with :class:`~pysad.models.MIDAS`. Each row is ``[source, destination]``, where the two integers identify the nodes of a directed edge. Keep the same ID for a node throughout the stream; these values are categorical identifiers.

``tick_size=4`` groups every four consecutive edges into one tick. PySAD uses edge counts to define ticks, so no timestamp column is needed. The first 20 ticks contain each of four edges once. The final tick repeats the edge ``[0, 1]`` four times, creating a burst. The fixed ``random_state`` makes the sketch hashes reproducible.

.. literalinclude:: ../examples/example_midas.py
   :language: python

It prints:

.. code-block:: text

   Maximum regular score: 0.000
   Last burst edge score: 7.500

Higher scores mean more anomalous edges. As the repeated edge accumulates within the final tick, its score rises above the regular scores. All edges in the first tick score zero because MIDAS needs an earlier tick for comparison.

Use a local density detector
----------------------------

This script scores each point with IncrementalLOF. IncrementalLOF compares the density near each point with the density near its nearest neighbors in a window of recent points. It updates only the neighbors that each new point changes.

.. literalinclude:: ../examples/example_incremental_lof.py
   :language: python

It prints:

.. code-block:: text

   AUROC: 0.731

Use a kernel detector
---------------------

This script scores each point with OnlineOneClassSVM. OnlineOneClassSVM keeps a weighted sum of Gaussian kernels on recent points that fell outside its boundary, and scores a point by how far it falls below the boundary. The weights of old points decay with each new point.

.. literalinclude:: ../examples/example_online_one_class_svm.py
   :language: python

It prints:

.. code-block:: text

   AUROC: 0.727

Summarize old points with DILOF
-------------------------------

This script scores each point with DILOF. DILOF computes the same local outlier factor as IncrementalLOF. When its window is full, it replaces the oldest half of the window with a smaller summary that keeps the density of these points, so it uses less memory than a window of the same history.

.. literalinclude:: ../examples/example_dilof.py
   :language: python

It prints:

.. code-block:: text

   AUROC: 0.726

Use kernel density local outliers with KELOS
--------------------------------------------

This script scores each point with KELOS. KELOS groups the points of its window into small clusters and uses the cluster centers as the centers of a kernel density estimate. The score of a point is how much lower the density at the point is than the densities at its nearest cluster centers.

.. literalinclude:: ../examples/example_kelos.py
   :language: python

It prints:

.. code-block:: text

   AUROC: 0.750

Count the neighbors of a sensor reading with D3
-----------------------------------------------

This script scores the readings of one temperature sensor with D3. D3 keeps a random sample of the last 1,000 readings and puts a kernel on each sampled reading. These kernels give a density estimate of the window. The score of a reading is the negative estimated number of window readings within a radius of it, so a reading with few neighbors gets a high score.

.. literalinclude:: ../examples/example_d3.py
   :language: python

It prints:

.. code-block:: text

   AUROC: 0.876

Use anytime outliers with AnyOut
--------------------------------

This script scores each point with AnyOut. AnyOut keeps the stream in a tree of clusters, where each level describes the data in more detail than the level above it. A point descends the tree until its time budget ends, and its score is the distance to the mean of the closest cluster at that level. Here the budget is 4 levels.

.. literalinclude:: ../examples/example_anyout.py
   :language: python

It prints:

.. code-block:: text

   AUROC: 0.763

Use a matrix sketch detector
----------------------------

This script scores each point with ADeMS. ADeMS keeps a Frequent Directions sketch, a small matrix that approximates the earlier points. The score of a point is the part of its direction that the top singular vectors of the sketch do not reconstruct.

.. literalinclude:: ../examples/example_adems.py
   :language: python

It prints:

.. code-block:: text

   AUROC: 0.770

Track running statistics
------------------------

This script tracks the mean and the variance of a stream. Running statistics update with each new value.

.. literalinclude:: ../examples/example_statistics.py
   :language: python

It prints:

.. code-block:: text

   mean: -0.048, standard deviation: 0.977

Detect a change in the mean
---------------------------

This script streams 500 values with mean 0 followed by 500 values with mean 1. It updates :class:`~pysad.statistics.PageHinkleyMeter` directly and raises an alarm when its two-sided cumulative statistic exceeds a fixed threshold. The ``delta`` parameter discounts small deviations; the threshold controls when the accumulated evidence triggers an alarm.

.. literalinclude:: ../examples/example_page_hinkley.py
   :language: python

It prints:

.. code-block:: text

   First change detected at index 549 (zero-based)

The mean changes at index 500; the alarm follows once enough evidence accumulates. This meter uses the full stream history and cannot remove observations, so it cannot be wrapped in :class:`~pysad.statistics.RunningStatistic`.

Detect a drift with Hoeffding's bounds
--------------------------------------

This script streams 1,000 bits with an error rate of 0.2 followed by 1,000 bits with an error rate of 0.5. It updates :class:`~pysad.statistics.HDDMAMeter` and :class:`~pysad.statistics.HDDMWMeter` directly and stops at the first drift. A warning reports a possible drift with a weaker test. Both meters assume that each value is in [0, 1], so they need no threshold.

.. literalinclude:: ../examples/example_hddm.py
   :language: python

It prints:

.. code-block:: text

   HDDMAMeter: warning at index 1010, drift at index 1011 (zero-based)
   HDDMWMeter: warning at index 1010, drift at index 1014 (zero-based)

The rate changes at index 1000. :class:`~pysad.statistics.HDDMAMeter` compares plain averages, and :class:`~pysad.statistics.HDDMWMeter` compares exponentially weighted averages. After a drift, the next update forgets the earlier values.

Detect a rise in the error rate
-------------------------------

This script streams 1,000 bits with an error rate of 0.2 followed by 1,000 bits with an error rate of 0.5. A bit is 1 for an error and 0 for a correct prediction. It updates :class:`~pysad.statistics.DDMMeter` directly and stops at the first drift. A warning reports a possible drift with a weaker test. If the error rate falls again, the warning ends without a drift, so the script prints the start of the last warning zone.

.. literalinclude:: ../examples/example_ddm.py
   :language: python

It prints:

.. code-block:: text

   Warning from index 1026, drift at index 1067 (zero-based)

The rate changes at index 1000. On this stream, the meter detects the drift later than the HDDM meters, because it compares the error rate since the last drift with its lowest value. After a drift, the next update forgets the earlier values.

Detect a fall in the accuracy of a sliding window
-------------------------------------------------

This script streams the same bits as the DDM example: 1,000 bits with an error rate of 0.2 followed by 1,000 bits with an error rate of 0.5. It updates :class:`~pysad.statistics.FHDDMMeter` directly and stops at the first drift. The meter keeps the accuracy of the last 100 values and compares it with the largest accuracy of a full window since the last drift. A drift is a difference of at least the Hoeffding bound.

.. literalinclude:: ../examples/example_fhddm.py
   :language: python

It prints:

.. code-block:: text

   Drift at index 1054 (zero-based), error rate in the window 0.40

The rate changes at index 1000. The meter has no warning level. On this stream, it detects the drift earlier than :class:`~pysad.statistics.DDMMeter`, because the window forgets the values before the change. After a drift, the next update empties the window.

Detect a rise in the error rate from the distance between errors
-----------------------------------------------------------------

This script streams 2,000 bits with an error rate of 0.05 followed by 1,000 bits with an error rate of 0.3. It updates :class:`~pysad.statistics.EDDMMeter` directly and prints each drift with the first warning before it. The meter tracks the mean and the standard deviation of the distances between consecutive errors, and compares them with their largest value since the last drift.

.. literalinclude:: ../examples/example_eddm.py
   :language: python

It prints:

.. code-block:: text

   Warning at index 850, drift at index 1060 (zero-based), mean distance between errors 21.2
   Warning at index 1839, drift at index 2014 (zero-based), mean distance between errors 14.5

The rate changes at index 2000. The first drift is a false alarm: the rate does not change there, but by chance the distances got shorter than at their largest. After a drift, the next update forgets the earlier values.

Detect a change in the distribution
-----------------------------------

This script streams 1000 values with standard deviation 1 followed by 1000 values with standard deviation 10. Both parts have mean 0, so a meter that tracks the mean does not see the change. :class:`~pysad.statistics.KSWINMeter` compares the newest values of its sliding window with a random sample of the older values by the Kolmogorov-Smirnov test, and sets ``drift_detected`` when the two distributions differ.

.. literalinclude:: ../examples/example_kswin.py
   :language: python

It prints:

.. code-block:: text

   Drift detected at index 1102 (zero-based), distance 0.60

The spread changes at index 1000. After a drift, the window keeps only its newest values, so the next test compares values of the new distribution only.

MemStream
---------

MemStream trains a denoising autoencoder on a small normal calibration block, then scores each new point against its bounded memory of accepted encodings. The calibration block must be normal; the labels are used here only to select that block. Monitoring starts after the calibration segment.

.. literalinclude:: ../examples/example_memstream.py
   :language: python
