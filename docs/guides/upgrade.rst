Upgrade from 0.1.x
==================

This guide lists the changes that you make to move a pipeline from PySAD 0.1.x to the current version. Many pipelines pin ``pysad==0.1.1``. Most class names and module paths did not change, so most of the work is in the constructor parameters and in the calibrator output.

Why upgrade
-----------

PySAD 0.1.1 calls ``np.float`` in the batch methods (``fit``, ``score``, ``fit_score``, ``transform`` and ``fit_transform``) and in some detectors. NumPy 1.24 removed ``np.float``, so these calls fail with an ``AttributeError`` on a current NumPy. A pipeline that pins 0.1.1 must also pin ``numpy<1.24``. Its ensemblers also import PyOD.

The current version supports Python 3.10 to 3.14. It needs only NumPy, scikit-learn and SciPy. It does not install PyOD. If you use :mod:`pysad.models.integrations` with a PyOD detector, install PyOD yourself. :class:`~pysad.models.IForestASD` now uses the Isolation Forest of scikit-learn, so it does not need PyOD.

Steps
-----

#. Remove the pin and install the current version with ``pip install -U pysad``.
#. Change the calibrator parameters and the alert rule. See `Calibrators`_.
#. Change the parameters that the current version removed or renamed. See `Other parameters`_.
#. If you use ``fit_score_partial`` or ``fit_score``, read `Score before fit`_.
#. Fit your models again. Do not load models that you pickled with 0.1.x.

Calibrators
-----------

The calibrator class names did not change. Both classes are in :mod:`pysad.transform.probability_calibration`. The current version also adds :class:`~pysad.transform.probability_calibration.SPOTProbabilityCalibrator`.

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Class
     - 0.1.x parameters
     - Current parameters
   * - :class:`~pysad.transform.probability_calibration.ConformalProbabilityCalibrator`
     - ``windowed=True, window_size=300``
     - ``window_size=300``
   * - :class:`~pysad.transform.probability_calibration.GaussianTailProbabilityCalibrator`
     - ``running_statistics=True, window_size=6400``
     - ``running_statistics=True, window_size=8000, short_window_size=10``

ConformalProbabilityCalibrator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The current version removed ``windowed``. The calibrator always keeps the last ``window_size`` scores. If you used ``windowed=False``, set ``window_size`` to the length of your stream or larger. The memory then grows with the stream, as in 0.1.x.

The output changed direction. In 0.1.x, the calibrator returned the fraction of the window scores that are higher than the score. A low value meant an anomaly. The current calibrator returns the fraction of the window scores that are lower than the score. A high value means an anomaly, as for every other PySAD output. The two values add up to ``1 - 1/n``, where ``n`` is the number of scores in the window, if no two scores are equal.

Change the alert rule:

.. code-block:: python

   # 0.1.x
   calibrator = ConformalProbabilityCalibrator(windowed=True, window_size=300)
   is_alert = calibrator.fit_transform_partial(score) < 0.05

   # current version
   calibrator = ConformalProbabilityCalibrator(window_size=300)
   is_alert = calibrator.fit_transform_partial(score) > 0.95

GaussianTailProbabilityCalibrator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The output direction did not change. A high value means an anomaly.

The current version follows the anomaly likelihood of Ahmad et al. (2017). It compares the mean of the last ``short_window_size`` scores with the mean and standard deviation of the last ``window_size`` scores. Version 0.1.x used the current score instead of the short-term mean. The default ``window_size`` changed from 6400 to 8000.

To get the 0.1.x output, set ``short_window_size=1`` and ``window_size=6400``:

.. code-block:: python

   calibrator = GaussianTailProbabilityCalibrator(window_size=6400, short_window_size=1)

With ``running_statistics=False``, the current version computes the variance over all scores. Version 0.1.x computed it over the last ``window_size`` scores. The outputs are the same until the stream is longer than ``window_size``.

Score before fit
----------------

In 0.1.x, ``fit_score_partial`` fitted the model to the point and then scored the point. The current version scores the point first and then fits the model to it. This order is called prequential evaluation. The model never scores a point that it learned from, so the scores and the AUROC values differ from 0.1.x. ``fit_score`` uses the same order.

For calibrators and other postprocessors, ``fit_transform_partial`` did not change. It fits the score first and then transforms it.

Other parameters
----------------

The current version removed or renamed these parameters and classes:

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Class
     - 0.1.x
     - Current version
   * - :class:`~pysad.models.StandardAbsoluteDeviation`
     - ``substracted_statistic``
     - ``subtracted_statistic``
   * - :class:`~pysad.models.RSHash`
     - ``sampling_points``
     - Removed
   * - ``MedianPostprocessor``
     - The class
     - Removed. Use :class:`~pysad.transform.postprocessing.RunningMedianPostprocessor`.
   * - :class:`~pysad.utils.Data`
     - ``data_base_path="data"``
     - ``data_base_path=None``, which uses ``~/.cache/pysad``
   * - :meth:`Data.get_iterator <pysad.utils.Data.get_iterator>`
     - ``seed``
     - ``random_state``

These default values changed. To keep the 0.1.x value, pass it to the constructor. The scores can still differ, because later versions also changed some algorithms to match their papers.

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Class
     - 0.1.x default
     - Current default
   * - :class:`~pysad.models.HalfSpaceTrees`
     - ``window_size=100``
     - ``window_size=250``
   * - :class:`~pysad.models.xStream`
     - ``depth=25, window_size=25``
     - ``depth=15, window_size=256``
   * - :class:`~pysad.models.RobustRandomCutForest`
     - ``num_trees=4, shingle_size=4``
     - ``num_trees=100, shingle_size=1``
   * - :class:`~pysad.models.KitNet`
     - ``grace_feature_mapping=None, grace_anomaly_detector=50000``
     - ``grace_feature_mapping=100, grace_anomaly_detector=1000``
   * - :class:`~pysad.models.ExactStorm`
     - ``max_radius=0.1``
     - ``max_radius=None``, which sets the radius from the first ``calibration_size`` points
   * - :class:`~pysad.models.MedianAbsoluteDeviation` and :class:`~pysad.models.StandardAbsoluteDeviation`
     - All points
     - ``window_size=500``

Every stochastic class now takes a ``random_state`` parameter. Pass an integer to get the same scores on each run.
