Evaluation methodology
======================

This page explains the steps of a run, the metrics and the limits of the results. :doc:`datasets` describes the data, and :doc:`/benchmark` shows the results.

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
- :class:`~pysad.models.HTM` gets each point with its timestamp, the range from 0 to 1, the probationary period of NAB, and the seed 0. It does not run on the UCR series, which have no timestamps.
- :class:`~pysad.models.KNNCAD` learns during the probationary period of NAB, as in NAB.
- :class:`~pysad.models.RelativeEntropy` gets the range from 0 to 1.
- :class:`~pysad.models.SAND` gets a fixed subsequence length of 10, a batch size of 256, and ``random_state=0`` on both NAB and UCR. See :doc:`univariate` for its initialization and scoring protocol.
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

Alerts of a calibrator
   For each run of the ``odds`` part, the benchmark gives all the scores to each calibrator. Then it counts the alerts after the warm-up.

NAB has three profiles. The standard profile weights a false alert at 0.11 of a miss. The profile "low false alerts" doubles the weight of a false alert, and the profile "low misses" doubles the weight of a miss.

One threshold for all the series needs scores with the same scale on each series. :class:`~pysad.models.HTM`, :class:`~pysad.models.KNNCAD` and :class:`~pysad.models.RelativeEntropy` give scores from 0 to 1, as the detectors of NAB do. The benchmark gives the scores of the other detectors to :class:`~pysad.transform.probability_calibration.ConformalProbabilityCalibrator`, whose window holds all the earlier scores of the series. With its default window of 300 scores, many points tie at the highest value.

The benchmark uses the scoring code of NAB. With it, :class:`~pysad.models.KNNCAD` gets NAB scores of 58.0, 43.4 and 64.8, the scores that the scoreboard of NAB lists for its KNN-CAD detector. :class:`~pysad.models.RelativeEntropy` gets higher scores than on the scoreboard, because its histogram follows the paper. :class:`~pysad.models.HTM` gets 64.2, 56.8 and 68.3. The scoreboard lists 64.6, 56.7 and 69.2 for the numentaTM detector of NAB, which uses the same kind of temporal memory, and 70.1, 63.1 and 74.3 for the numenta detector, which uses the backtracking temporal memory of NuPIC.

What the results do not show
----------------------------

- Tuned parameters can give very different results. For example, :class:`~pysad.models.KitNet` trains on its first 1,100 points by default. 7 ODDS datasets have fewer points, so on them, KitNet never ends its training. :doc:`/guides/choose_detector` names the parameters with the largest effect.
- :class:`~pysad.models.RSForest` scores every point 0.0 until its first window of 512 points is full. On streams with fewer points than 512, such as arrhythmia, glass, ionosphere, lympho, vertebral and wbc, every score after the warm-up is 0.0. Its AUROC is then 0.5 and its AUPR is the fraction of anomalies.
- The points of the ODDS datasets are independent, so these streams have no drift and no seasonality. The streams with drift show how a detector follows changes over time, but they are only |num_drift_streams| streams. Their thinning also makes the changes faster in points.
- The NAB scoreboard has no results for MAD, SAD and the ESD detectors. Their NAB scores depend on the calibration, which NAB does not use. The periods come from the data, because NAB gives no periods.
- The UCR results come from the short series only. A longer series can be harder for a detector.
- Each detector runs with one seed, and the shingles have one size. Another seed or another shingle size can give other results. A change that draws the same random numbers in another order has the same effect, and on a small dataset it can move the AUROC a lot. For example, `pull request 397 <https://github.com/selimfirat/pysad/pull/397>`__ made :class:`~pysad.models.RobustRandomCutForest` cut all its trees in one batch. The model stayed the same, but the cuts of the trees came in another order. Its AUROC on glass with seed 0 went from 0.715 to 0.638, but over seeds 0 to 19 its mean AUROC on glass is 0.675 before the change and 0.686 after it.
- A run on another platform can give slightly different results for :class:`~pysad.models.Inqmad`, and for :class:`~pysad.models.LODA` on series with many equal scores.
- On a series with many equal points, :class:`~pysad.models.Inqmad` gives these points scores that are equal up to rounding, so the rounding decides their order and the AUROC. For example, on ``nab/realAWSCloudwatch/ec2_disk_write_bytes_1ef3de.csv`` with shingles, 3,217 of the 4,474 scored points, 322 of them anomalies, are copies of the first shingle. A change in rounding moved its AUROC from 0.529 to 0.617, but with the scores rounded to 9 decimals, the AUROC is 0.530 in both cases.
- Commit `777fcad <https://github.com/selimfirat/pysad/commit/777fcad1c236b677d245bbb1dd41b769eeb23173>`__ rounds benchmark scores to 9 significant digits before metrics. This change keeps near-equal scores tied across platforms. For :class:`~pysad.models.Inqmad` without shingles, it changes AUROC from 0.5438 to 0.4143 on ``nab/artificialWithAnomaly/art_increase_spike_density.csv``. This series contains only two values. With seed 0 and the 256-point warm-up, 3,776 points count in AUROC. Rounding gives 1,985 points one score, including 197 anomalies. Their raw scores span less than 5e-10. The lower AUROC reflects the intended ties. The detector does not change.
- :class:`~pysad.models.LocalOutlierProbability` gives probabilities against its initial reference points. Far from these points, most probabilities approach 1. Commit `777fcad <https://github.com/selimfirat/pysad/commit/777fcad1c236b677d245bbb1dd41b769eeb23173>`__ introduced rounding to 9 significant digits before metric calculation. This benchmark change keeps equal scores equal across platforms. It also merges probabilities close to 1. On ``nab/realAWSCloudwatch/ec2_cpu_utilization_ac20cd.csv`` with shingles, 3,588 of the 3,776 scored points round to 1. All 403 anomalies occur among these points. The same raw scores give AUROC 0.774957 before rounding and 0.527868 after rounding. The detector did not change. Commit ``ca9b381e`` already includes this rounding rule. The drop from 0.775 to 0.528 reflects the intended benchmark rule, rather than a later detector regression.
- On a series of integer counts, many points of a window of :class:`~pysad.models.IncrementalLOF` are copies, so many LOF values are exactly 1 in exact arithmetic. In floating point, they differ in the last bits, and before `pull request 544 <https://github.com/selimfirat/pysad/pull/544>`__, these differences decided their order. For example, on ``nab/realTweets/Twitter_volume_GOOG.csv``, 13,584 of the 15,586 scored points, 1,145 of them anomalies, have an LOF of 1 within 1e-14. With the rounding to 9 significant digits, these points tie, and the AUROC went from 0.543 to 0.521. The rounding moved the AUROC of 5 other series by 0.005 to 0.019, but the mean AUROC on NAB only from 0.519 to 0.518.
- :class:`~pysad.integrations.ReferenceWindowModel` runs a PyOD model, so its results depend on that model. :class:`~pysad.models.IForestASD` is one example of it.
- The anomalies also enter the window of a calibrator. On a dataset with many anomalies, fewer normal points pass the threshold, so the false alert rate can be lower than expected. The window of :class:`~pysad.transform.probability_calibration.ConformalProbabilityCalibrator` holds 300 scores by default, so its highest value is 299/300, and a threshold of 0.999 never raises an alert.
