Integrate a one-dimensional stream
==================================

A one-dimensional stream has one feature per point, such as a sensor value.
This recipe uses public PySAD methods to score each point, calibrate its score, and update the detector.
It needs no wrapper class or extra dependency.

Give each point the correct shape
---------------------------------

Single-instance methods take a NumPy array with shape ``(num_features,)``.
For one feature, pass ``np.array([value], dtype=float)``, with shape ``(1,)``.
Do not pass a scalar or a row matrix with shape ``(1, 1)``.

Batch methods take shape ``(num_instances, num_features)``.
``score(X)`` returns a NumPy array with shape ``(num_instances,)`` and dtype ``np.float64``.
Convert a one-dimensional array of values with ``X = values.reshape(-1, 1)``.
Then :class:`~pysad.utils.ArrayStreamer` yields vectors with shape ``(1,)``.
Keep the feature count and order fixed throughout the stream.

Score before the update
------------------------

``score_partial(x)`` returns a Python ``float`` without learning from ``x``.
Higher scores mean more anomalous points, even when all scores are negative.
Repeated scoring without updates does not train the detector.

Call ``score_partial(x)`` before ``fit_partial(x)`` to score against earlier points only.
This order is prequential evaluation, or score-then-learn.
``fit_score_partial(x)`` combines these two operations in that order:

.. code-block:: python

   score = model.fit_score_partial(x)

The separate calls let an integration decide when to learn:

.. code-block:: python

   score = model.score_partial(x)
   model.fit_partial(x)

If you fit first, the detector scores a point that it already learned from.
If you want a fixed model, fit historical data first and then call only ``score_partial``.
See :doc:`alerts` for a loop that learns only from points without alerts.

Set feature bounds before the stream
------------------------------------

:class:`~pysad.models.HalfSpaceTrees` and :class:`~pysad.models.RSHash` require ``feature_mins`` and ``feature_maxes``.
Each array needs one entry per feature, so one-dimensional bounds have shape ``(1,)``.
These bounds describe feature values, not anomaly scores or alert thresholds.

Use known limits or representative historical data that precedes the stream.
Historical data needs shape ``(num_instances, 1)``.
Calculate its bounds with ``history.min(axis=0)`` and ``history.max(axis=0)``.
Do not calculate bounds from future evaluation points.
Choose distinct limits that cover the expected values.
The example uses known limits of 0 and 40.

HalfSpaceTrees uses the bounds to build its trees without scaling inputs.
RSHash uses the bounds to normalize each feature internally with ``(x - feature_mins) / (feature_maxes - feature_mins)``.
Neither detector clips inputs to the bounds.
If you scale the input, express the bounds in the same units.
If the expected range changes, create a detector with new bounds and fit recent history.

Calibrate the raw score
-----------------------

Raw scores have no common range across detectors.
:class:`~pysad.transform.probability_calibration.ConformalProbabilityCalibrator` maps a score to the fraction of window scores below it.
Call ``fit_transform_partial(score)`` to include the current score before the transformation.
Pass the raw score as a scalar, not the feature vector.

The calibrated value lies between 0 and 1.
A high value means that the raw score exceeds most recent scores.
It does not give the probability that the point is an anomaly.
With 64 distinct scores in the window, the largest value is ``63 / 64``.
Choose a window and threshold together, as :doc:`alerts` explains.

Run the recipe
---------------

This example generates sensor values and adds one high value.
It scores each point, calibrates that score, and then fits the point.
It skips alerts for 128 points, which cover one model window and one calibration window.
This warm-up length is an example choice, not a rule for all streams.
The model learns from every point, including points that raise alerts.

.. literalinclude:: ../../examples/example_one_dimensional_integration.py
   :language: python

Replace the generated values with your stream.
For a benchmark dataset, load it with :class:`~pysad.utils.Data` and select one column with ``X[:, :1]``.
Keep the two-dimensional batch shape before you pass it to ``ArrayStreamer``.

Understand the py-outlier wrapper mismatch
------------------------------------------

The `py-outlier wrapper at revision 9a15aba <https://github.com/askmy-stack/py-outlier/blob/9a15aba1b259d1982f65a8f12742cf5f4ae2a4c1/src/anomaly_detection/models/streaming/pysad_wrapper.py>`_
exposes ``score_one`` and ``fit_partial`` through its own interface.
Its calls to the underlying PySAD model differ from the API in this recipe:

* It calls ``HalfSpaceTrees(**params)``. The default empty parameters omit the required feature bounds.
* It reshapes each point to ``(1, -1)``. PySAD single-instance methods need a feature vector with shape ``(num_features,)``.
* It calls ``self._model.score_one(...)``. PySAD models expose ``score_partial``, not ``score_one``.

A wrapper can expose its own ``score_one`` method.
That method must call PySAD's ``score_partial`` with a feature vector.
It must supply the bounds and keep scoring separate from updates.
For River's ``score_one`` and ``learn_one`` interface, use :class:`~pysad.integrations.RiverAnomalyDetector` as :doc:`pipelines` explains.
