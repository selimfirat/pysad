Methodology and datasets
========================

This page explains how the :doc:`multivariate detector benchmark </benchmark>` evaluates each detector, which datasets it uses and what the results do and do not show. To reproduce or update the numbers, see :doc:`run`.

Evaluation protocol
-------------------

The benchmark follows the prequential (predict, then train) rules of :doc:`/guides/evaluate`. For each dataset, each detector and each seed from 0 to 4, it does these steps:

#. It loads the dataset with :class:`~pysad.utils.Data`. Then it shuffles the points with ``random_state=seed``. Many of the files put all the anomalies at the end, so the order of the file is not a fair stream.
#. It keeps the first 5,000 points of the shuffled stream. The first 256 of these points are the warm-up. A stream with fewer than 1,024 points has a warm-up of a quarter of its points.
#. It scales each feature to the range from 0 to 1, with the minimum and the maximum of these points. :class:`~pysad.models.HalfSpaceTrees` and :class:`~pysad.models.RSHash` require the range of each feature in advance, and after the scaling, this range is from 0 to 1.
#. It makes the detector with its default parameters and ``random_state=seed``. Some detectors also get the arguments that they require. :class:`~pysad.models.HalfSpaceTrees` and :class:`~pysad.models.RSHash` get the range from 0 to 1 for each feature. :class:`~pysad.models.Inqmad` gets the number of features. :class:`~pysad.models.LocalOutlierProbability` fits on the warm-up points.
#. It gives each point to ``fit_score_partial``, which scores the point and then trains the detector on it.
#. It computes AUROC and AUPR over the scores of the points after the warm-up. The detector learns from the warm-up points, but it has little or no model when it scores them. The xStream paper and the Numenta Anomaly Benchmark leave out such points in the same way.

:meth:`Data.get_stream <pysad.utils.Data.get_stream>` does steps 1 to 3. This line gives the stream of ``cardio`` for seed 0, exactly as the benchmark uses it:

.. code-block:: python

   from pysad.utils import Data

   X, y = Data().get_stream("cardio.mat", random_state=0, max_points=5000, scale=True)

Step 3 uses the minimum and the maximum of the whole stream. A live stream does not know them in advance. For a live stream, take them from historical data, or use known physical limits.

Datasets
--------

The datasets come from the `ODDS <http://odds.cs.stonybrook.edu/>`_ repository (Outlier Detection DataSets). The column "Points used" gives the number of points after the limit of 5,000. The column "Points evaluated" gives the number of points after the warm-up.

.. csv-table::
   :file: ../benchmark/datasets.csv
   :header-rows: 1
   :stub-columns: 1
   :class: benchmark-table

Streams with drift
------------------

The streams with drift come from three sources, and :class:`~pysad.utils.Data` loads each one in its time order, for example with ``Data().get_data("insects/abrupt.csv")``:

- ``Data.datasets("kddcup99")`` gives the http connections of logged in users from the 10% file of KDD Cup 99, as scikit-learn's ``fetch_kddcup99`` does. Its 3 features are the logarithms of the duration and of the bytes in each direction. The attacks have the label 1.
- ``Data.datasets("insects")`` gives the 4 INSECTS streams of the ARCUS paper. The anomalies are the points of the southern house mosquito, 1.2% of each stream.
- ``Data.datasets("smd")`` gives the test halves of the 28 machines of SMD. The benchmark uses the first machine of each of the 3 groups of machines. The first half of each machine has no labels, so the benchmark does not use it.

For each stream and each detector, the benchmark does the steps of the protocol above with seed 0, with two changes. It keeps the time order of the points. And instead of the first 5,000 points, it keeps every k-th point, with the smallest k that leaves at most 10,000 points. Thus the stream keeps all its drifts, and the run stays short. The first 256 points are the warm-up.

The column "Points used" gives the number of points after the thinning.

.. csv-table::
   :file: ../benchmark/drift_streams.csv
   :header-rows: 1
   :stub-columns: 1
   :class: benchmark-table

What the results do not show
----------------------------

- The univariate detectors take one value per point, so this benchmark does not run them. :doc:`univariate` compares them on the series of the Numenta Anomaly Benchmark.
- :class:`~pysad.models.integrations.ReferenceWindowModel` runs a PyOD model, so its results depend on that model. :class:`~pysad.models.IForestASD` is one example of it.
- The points of the ODDS datasets are independent, and the benchmark puts them in a random order. These streams have no drift and no seasonality. The streams with drift show how a detector follows changes over time, but they are only 8 streams with one seed.
- The thinning of a stream with drift makes the changes faster in points. A detector with a window of a fixed number of points sees more time in its window than on the full stream.
- Tuned parameters can give very different results. For example, :class:`~pysad.models.KitNet` trains on its first 1,100 points by default. 7 of these datasets have fewer points, so on them, KitNet never ends its training. :doc:`/guides/choose_detector` names the parameters with the largest effect.
