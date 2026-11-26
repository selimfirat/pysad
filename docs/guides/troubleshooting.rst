.. meta::
    :description: Diagnose streaming edge cases in PySAD: cold-start scores, constant streams, and NaN or infinite values and scores.

Troubleshoot streaming edge cases
=================================

Streams sometimes look wrong in ways that are expected. This guide covers the three most common cases: the cold start, a constant stream, and NaN or infinite values. Each section gives the symptom, the cause and the fix.

The examples on this page use synthetic data. Replace it with your own stream:

.. code-block:: python

    import numpy as np

    rng = np.random.default_rng(0)
    stream = rng.normal(size=(500, 2))  # live points, one row at a time

Cold start: the first scores are fixed values
---------------------------------------------

A detector learns from the stream itself, so it knows little at the start. Before the first ``fit_partial`` call, ``score_partial`` returns a fixed value, usually 0.0. Some detectors also return a fixed value until their window fills. :doc:`/concepts` lists the warm-up behavior of selected detectors.

The fixed scores do not reflect anomalousness. Three fixes help:

- Fit the detector to historical data before the stream starts. See :doc:`production`.
- Skip a warm-up period before you raise alerts, for example the first 200 points. :doc:`alerts` shows the loop.
- Evaluate consecutive windows of the stream separately, to see the warm-up in a metric. See :doc:`evaluate`.

Constant stream: every score is the same
-----------------------------------------

If the input repeats one value, the spread of the stream is zero. Detectors that divide by the spread, such as :class:`~pysad.models.StandardAbsoluteDeviation`, :class:`~pysad.models.MedianAbsoluteDeviation` and :class:`~pysad.models.LODA`, then return 0 for every point. The scores stay finite, and no point looks anomalous.

A constant stream usually means a stuck sensor or a failed feed. Check the raw input before you trust the scores. Track the variance of each feature with :class:`~pysad.statistics.RunningStatistic`:

.. code-block:: python

    from pysad.models import LODA
    from pysad.statistics import RunningStatistic, VarianceMeter

    model = LODA(random_state=0)
    spreads = [RunningStatistic(VarianceMeter, window_size=100) for _ in range(stream.shape[1])]

    for x in stream:
        for spread, value in zip(spreads, x):
            spread.update(float(value))
        score = model.fit_score_partial(x)

    if all(spread.get() == 0 for spread in spreads):
        print("Every feature is constant. Check the sensor.")

A calibrator behaves the same way on a constant stream. It gives 0.0 to each point that matches the recent scores, and 1.0 to the first point that differs. The first deviation after a long plateau therefore raises an alert at any threshold. This is expected.

NaN input: every detector raises a ValueError
---------------------------------------------

Every model rejects a point that holds NaN with a ``ValueError`` that names the model:

.. code-block:: python

    model = LODA(random_state=0)
    model.score_partial(np.array([1.0, np.nan]))
    # ValueError: LODA does not accept NaN values.

A rejected point leaves the model unchanged. One bad point cannot poison the scores of the points that follow. Replace missing values before the detector sees the point. :doc:`/faq` shows the common fix, which repeats the last known value of each feature.

NaN or infinite scores: check the input for infinite values
------------------------------------------------------------

The NaN check does not catch infinite values. A point with ``inf`` passes it, and detectors can then return infinite or NaN scores, for example when a running spread overflows. Guard the input with ``np.isfinite``, which catches NaN and infinity together:

.. code-block:: python

    model = LODA(random_state=0)
    last = np.zeros(stream.shape[1])

    for x in stream:
        if not np.isfinite(x).all():
            x = np.where(np.isfinite(x), x, last)  # replace NaN and inf with the last known value
        last = x
        score = model.fit_score_partial(x)

If your source can produce values beyond the float range, clip or rescale them before scoring, because overflow in the input can make the scores NaN. Scores are plain Python floats, so ``np.isfinite(score)`` tells you whether a score is usable.

Next steps
----------

- :doc:`production` shows warm-up from history, per-source detectors and score monitoring.
- :doc:`alerts` shows calibration and thresholds.
- :doc:`/faq` answers more short questions about input and scores.
