.. meta::
    :description: Diagnose installation problems and runtime issues in PySAD: cold-start scores, constant streams, and NaN or infinite values and scores.

Troubleshoot streaming edge cases
=================================

Streams sometimes look wrong in ways that are expected. This guide covers the three most common cases: the cold start, a constant stream, and NaN or infinite values. Each section gives the symptom, the cause and the fix.

The examples on this page use synthetic data. Replace it with your own stream:

.. code-block:: python

    import numpy as np

    rng = np.random.default_rng(0)
    stream = rng.normal(size=(500, 2))  # live points, one row at a time

Installation
------------

PySAD installs from PyPI, conda-forge, or from source. See :doc:`/installation` for the full steps. This section covers the most common installation problems.

``ModuleNotFoundError: No module named 'pysad'``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PySAD is not installed, or you started Python in a different environment than the one where you installed it.

Fix it:

- Run ``python -c "import pysad; print(pysad.__version__)"``. If it raises ``ModuleNotFoundError``, install PySAD first with ``pip install pysad`` or ``conda install -c conda-forge pysad``.
- Make sure that the Python you run is the same Python where you installed PySAD. Run ``which python`` and ``which pip`` to check that they point to the same environment.

``Python version is not supported``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PySAD supports Python 3.11 to 3.14. A version outside that range can fail at import time or behave wrongly.

Fix it:

- Run ``python --version``. If the version is below 3.11 or above 3.14, install Python 3.11 to 3.14 and create a new environment.
- Then run ``pip install pysad`` again in that environment.

``The imported version does not match the installed version``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After ``pip install -U pysad``, a running Python process can still hold the old version in memory.

Fix it:

- Restart Python after you upgrade. A running process keeps the version it started with.
- Run ``python -c "import pysad; print(pysad.__version__)"`` to confirm the installed version.

``An integration library is missing``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The PyOD and River integrations need PyOD and River installed separately. PySAD does not install them for you.

Fix it:

- For PyOD detectors, run ``pip install pyod`` or ``conda install -c conda-forge pyod``.
- For River forecasters, run ``pip install river``.
- Then restart Python and import the integration again.

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

Runtime
-------

A model raises an error while you score a stream, or the scores look wrong. This section covers the most common runtime problems.

``TypeError: a plain number is not a valid point``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A detector takes one point as an array of shape ``(num_features,)``. A plain number or a 0-D array raises a ``TypeError``. Univariate detectors take exactly one feature, so pass ``np.array([value])``.

Fix it:

- Give each point as a 1-D array, for example ``np.array([2.5])`` for one feature.
- To stream a 1-D array of values, reshape it first with ``values.reshape(-1, 1)``. See :doc:`stream_data`.

``ValueError: the point has the wrong number of features``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A detector fixes the number of features at the first point. A later point with more or fewer features raises a ``ValueError``.

Fix it:

- Make sure that every point has the same number of features as the first point.
- To change the features, start a new detector and fit it to historical data with the new features. See :doc:`production`.

``TypeError: this model has no predict method``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``predict_partial``, ``fit_predict_partial``, ``predict`` and ``fit_predict`` raise a ``TypeError`` on a model whose output is not a binary decision. A score-only model cannot predict.

Fix it:

- Check ``model.output_kind``. A score-only model has ``output_kind == "score"``. Use ``score_partial`` instead of ``predict_partial``.
- Wrap a score-only model with a threshold or a calibrator to get decisions. See :doc:`alerts`.

``Every run gives a different result``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some detectors use random numbers. Two runs with the same data can give different scores.

Fix it:

- Pass ``random_state=0``, or another integer, to each detector that uses randomness.
- Or call :func:`pysad.utils.fix_seed` at the start of the script to seed the global state that ``random_state=None`` draws from.
- To report a stable result, repeat the run with three to five seeds and give the mean. See :doc:`/concepts`.

``Loading a pickle fails or gives the wrong scores``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PySAD detectors and calibrators can be pickled, but you must load a pickle with the same PySAD version that saved it. A different version can change the saved state and give wrong scores or fail to load.

Fix it:

- Save the detector and its calibrator together, so a restart continues from the same state.
- Check ``pysad.__version__`` before you load the pickle. Use the version that saved it. See :doc:`production`.

``ReferenceWindowModel is slow at the first points``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Without ``initial_window_X``, :class:`~pysad.integrations.ReferenceWindowModel` trains a new batch model for each of the first ``window_size`` points. Those points take much longer than the later ones.

Fix it:

- Pass initial points to ``initial_window_X`` to train once instead. See :doc:`pyod`.

Next steps
----------

- :doc:`production` shows warm-up from history, per-source detectors and score monitoring.
- :doc:`alerts` shows calibration and thresholds.
- :doc:`/faq` answers more short questions about input and scores.
