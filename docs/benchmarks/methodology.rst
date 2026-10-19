Methodology and datasets
========================

This page explains how the :doc:`multivariate detector benchmark </benchmark>` evaluates each detector, which datasets it uses and what the results do and do not show. To reproduce or update the numbers, see :doc:`run`.

Evaluation protocol
-------------------

The benchmark follows the prequential (predict, then train) rules of :doc:`/guides/evaluate`. For each dataset, each detector and each seed from 0 to 4, it does these steps:

#. It loads the dataset with :class:`~pysad.utils.Data`. Then it shuffles the points with ``ArrayStreamer(shuffle=True, random_state=seed)``. Many of the files put all the anomalies at the end, so the order of the file is not a fair stream.
#. It keeps the first 5,000 points of the shuffled stream. The first 256 of these points are the warm-up. A stream with fewer than 1,024 points has a warm-up of a quarter of its points.
#. It scales each feature to the range from 0 to 1, with the minimum and the maximum from :func:`~pysad.utils.get_minmax_array`. :class:`~pysad.models.HalfSpaceTrees` and :class:`~pysad.models.RSHash` require the range of each feature in advance, and after the scaling, this range is from 0 to 1.
#. It makes the detector with its default parameters and ``random_state=seed``. Some detectors also get the arguments that they require. :class:`~pysad.models.HalfSpaceTrees` and :class:`~pysad.models.RSHash` get the range from 0 to 1 for each feature. :class:`~pysad.models.Inqmad` gets the number of features. :class:`~pysad.models.LocalOutlierProbability` fits on the warm-up points.
#. It gives each point to ``fit_score_partial``, which scores the point and then trains the detector on it.
#. It computes AUROC and AUPR over the scores of the points after the warm-up. The detector learns from the warm-up points, but it has little or no model when it scores them. The xStream paper and the Numenta Anomaly Benchmark leave out such points in the same way.

Step 3 uses the minimum and the maximum of the whole stream. A live stream does not know them in advance. For a live stream, take them from historical data, or use known physical limits.

Datasets
--------

The datasets come from the `ODDS <http://odds.cs.stonybrook.edu/>`_ repository (Outlier Detection DataSets). The column "Points used" gives the number of points after the limit of 5,000. The column "Points evaluated" gives the number of points after the warm-up.

.. csv-table::
   :file: ../benchmark/datasets.csv
   :header-rows: 1
   :stub-columns: 1
   :class: benchmark-table

What the results do not show
----------------------------

- The univariate detectors take one value per point, so the benchmark does not run them. These detectors are :class:`~pysad.models.MedianAbsoluteDeviation`, :class:`~pysad.models.StandardAbsoluteDeviation`, :class:`~pysad.models.KNNCAD`, :class:`~pysad.models.RelativeEntropy`, :class:`~pysad.models.SeasonalESD` and :class:`~pysad.models.SeasonalHybridESD`.
- :class:`~pysad.models.integrations.ReferenceWindowModel` runs a PyOD model, so its results depend on that model. :class:`~pysad.models.IForestASD` is one example of it.
- The points of these datasets are independent, and the benchmark puts them in a random order. The streams have no drift and no seasonality. Thus the results do not show how a detector follows changes over time.
- Tuned parameters can give very different results. For example, :class:`~pysad.models.KitNet` trains on its first 1,100 points by default. 7 of these datasets have fewer points, so on them, KitNet never ends its training. :doc:`/guides/choose_detector` names the parameters with the largest effect.
