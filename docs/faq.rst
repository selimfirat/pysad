Frequently asked questions
==========================

Which detector do I start with?
-------------------------------

For data with several features, start with :class:`~pysad.models.LODA`. For one metric, start with :class:`~pysad.models.SeasonalHybridESD` if it has a daily or weekly cycle, and with :class:`~pysad.models.MedianAbsoluteDeviation` if it does not. Then compare two or three detectors on your data. See :doc:`guides/choose_detector`.

What does a score mean?
-----------------------

A higher score means a more anomalous point. The scale is different for each detector, so a raw score has no meaning on its own. To get a value with the same meaning for every detector, calibrate the score. See :doc:`guides/alerts`.

Why do I get a ValueError for a univariate detector?
----------------------------------------------------

Univariate detectors take points with exactly one feature, as an array of shape ``(1,)``. Give ``np.array([value])``, not ``value``, and reshape a 1-D array with ``values.reshape(-1, 1)`` before you stream it. See :doc:`guides/stream_data`.

Why are the first scores strange?
---------------------------------

A detector learns from the stream itself, so it knows little at the start. Some detectors give 0 until a window is full. Skip a warm-up period before you raise alerts, and use an evaluation that shows the warm-up separately. See :doc:`guides/evaluate`.

Why do I get a different result on each run?
--------------------------------------------

Some detectors use random numbers. Pass ``random_state=0``, or another integer, to each of them, or call :func:`pysad.utils.fix_seed` at the start of the script. See :doc:`concepts`.

Can I use streaming detectors on a batch dataset?
-------------------------------------------------

Yes. Stream the dataset with :class:`~pysad.utils.ArrayStreamer`, or call ``fit_score`` on the full array, which processes the rows in order. A streaming detector sees each point only once, so a batch detector that sees all data can be more accurate on a fixed dataset. For batch detection, see `PyOD <https://pyod.readthedocs.io/>`_.

Can I use batch detectors on a stream?
--------------------------------------

Yes. :class:`~pysad.models.integrations.ReferenceWindowModel` trains a PyOD detector on a sliding window, and :class:`~pysad.models.integrations.OneFitModel` trains it once on initial data. See :doc:`guides/pyod`.

How do I report a bug or ask a question?
----------------------------------------

To report a bug, open an `issue <https://github.com/selimfirat/pysad/issues/new/choose>`_ with your PySAD version, your Python version and a short script that shows the problem. To ask a question, start a thread in `Discussions <https://github.com/selimfirat/pysad/discussions>`_. To contribute, see :doc:`contributing`.
