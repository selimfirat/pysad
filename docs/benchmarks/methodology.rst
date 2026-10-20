How the benchmark works
=======================

This page explains the steps of a run, the metrics and the limits of the results. :doc:`/benchmark` lists the parts of the benchmark and their result pages, and :doc:`datasets` describes the data.

Steps of a run
--------------

For each detector and each dataset, the benchmark does these steps once, with seed 0:

#. It loads the data with :class:`~pysad.utils.Data`.
#. It keeps some of the points, as the list after these steps shows. Then it scales each feature to the range from 0 to 1, with the minimum and the maximum of these points.
#. It makes the detector with its default parameters and ``random_state=0``. Some detectors also get the arguments that they require, as the list below shows.
#. It gives each point to ``fit_score_partial``, which scores the point and then trains the detector on it.
#. It computes the metrics. The first 256 points are the warm-up, and their scores do not count, because the detector has little or no model when it scores them. A stream with fewer than 1,024 points has a warm-up of a quarter of its points.

The parts keep different points, and they use different metrics:

``odds``
   The first 5,000 points of an ODDS dataset, after a shuffle with ``random_state=0``. Many ODDS files put all the anomalies at the end, so the order of the file is not a fair stream. The metrics are AUROC and AUPR.

``drift``
   Every k-th point of a stream, in time order. k is the smallest step that leaves at most 10,000 points, so the stream keeps all its drifts. The metrics are AUROC and AUPR.

``univariate``, ``nab`` and ``shingles``
   All the points of a NAB series, in time order. The ``shingles`` part turns each value into a shingle of the last 10 values, and the first 9 shingles repeat the first value. The metrics are the NAB score, and AUROC on the series with anomalies.

``univariate`` on UCR
   All the points of a UCR series with at most 20,000 points, in time order. The metrics are the solved series, and AUROC after the training part.

Step 2 uses the minimum and the maximum of the data. A live stream does not know them in advance. For a live stream, take them from historical data, or use known physical limits.

These detectors get more arguments in step 3:

- :class:`~pysad.models.HalfSpaceTrees` and :class:`~pysad.models.RSHash` get the range from 0 to 1 for each feature.
- :class:`~pysad.models.Inqmad` gets the number of features.
- :class:`~pysad.models.LocalOutlierProbability` fits on the warm-up points.
- :class:`~pysad.models.KNNCAD` learns during the probationary period of NAB, as in NAB.
- :class:`~pysad.models.RelativeEntropy` gets the range from 0 to 1.
- :class:`~pysad.models.SeasonalESD` and :class:`~pysad.models.SeasonalHybridESD` get the period of the series and a window of 3 periods and one point. The R package of their paper needs more than 2 periods. With an odd window, the detectors give the same anomalies as the R package.

Metrics
-------

AUROC
   The probability that a random anomaly gets a higher score than a random normal point. A random detector gets 0.5, and a perfect detector gets 1. :doc:`/guides/evaluate` explains AUROC and AUPR in more detail.

AUPR
   The average precision over all the score thresholds. A random detector gets the fraction of anomalies in the dataset, and a perfect detector gets 1.

NAB score
   NAB marks a window of points around each anomaly, and its score counts each window as one event. The first alert in a window detects it, and an early alert counts more than a late one. A window without an alert is a miss. An alert outside the windows is a false alert, and a false alert soon after a window costs less than others. NAB uses one alert threshold for all the series of a detector, the threshold that gives the highest score. Then it scales the score: a detector without alerts gets 0, and a perfect detector gets 100. The score leaves out the probationary period of each series, which is its first 15% of points, at most 750.

Solved series
   A detector solves a UCR series when its highest score after the training part is at most 100 points from the anomaly. The authors of the archive use this rule.

NAB has three profiles. The standard profile weights a false alert at 0.11 of a miss. The profile "low false alerts" doubles the weight of a false alert, and the profile "low misses" doubles the weight of a miss.

One threshold for all the series needs scores with the same scale on each series. :class:`~pysad.models.KNNCAD` and :class:`~pysad.models.RelativeEntropy` give scores from 0 to 1, as the detectors of NAB do. The benchmark gives the scores of the other detectors to :class:`~pysad.transform.probability_calibration.ConformalProbabilityCalibrator`, whose window holds all the earlier scores of the series. With its default window of 300 scores, many points tie at the highest value.

The benchmark uses the scoring code of NAB. With it, :class:`~pysad.models.KNNCAD` gets NAB scores of 58.0, 43.4 and 64.8, the scores that the scoreboard of NAB lists for its KNN-CAD detector. :class:`~pysad.models.RelativeEntropy` gets higher scores than on the scoreboard, because its histogram follows the paper.

What the results do not show
----------------------------

- Tuned parameters can give very different results. For example, :class:`~pysad.models.KitNet` trains on its first 1,100 points by default. 7 ODDS datasets have fewer points, so on them, KitNet never ends its training. :doc:`/guides/choose_detector` names the parameters with the largest effect.
- The points of the ODDS datasets are independent, so these streams have no drift and no seasonality. The streams with drift show how a detector follows changes over time, but they are only 8 streams. Their thinning also makes the changes faster in points.
- The NAB scoreboard has no results for MAD, SAD and the ESD detectors. Their NAB scores depend on the calibration, which NAB does not use. The periods come from the data, because NAB gives no periods.
- The UCR results come from the short series only. A longer series can be harder for a detector.
- Each detector runs with one seed, and the shingles have one size. Another seed or another shingle size can give other results.
- A run on another platform can give slightly different results for :class:`~pysad.models.Inqmad`, and for :class:`~pysad.models.LODA` on series with many equal scores.
- :class:`~pysad.models.integrations.ReferenceWindowModel` runs a PyOD model, so its results depend on that model. :class:`~pysad.models.IForestASD` is one example of it.
