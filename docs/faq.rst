Frequently asked questions
==========================

Which detector do I start with?
-------------------------------

For data with several features, start with :class:`~pysad.models.LODA`. For a single series of values with a daily or weekly cycle, start with :class:`~pysad.models.SeasonalHybridESD`. If the series has no cycle, start with :class:`~pysad.models.MedianAbsoluteDeviation` for a short stream, or :class:`~pysad.models.StandardAbsoluteDeviation` for a long one. Then compare two or three detectors on your data. See :doc:`guides/choose_detector`.

What does a score mean?
-----------------------

A higher score means a more anomalous point. The scale is different for each detector, so a raw score has no meaning on its own. To get a value with the same meaning for every detector, calibrate the score. See :doc:`guides/alerts`.

Why does a univariate detector raise an error?
----------------------------------------------

Univariate detectors take points with exactly one feature. A point with several features raises a ``ValueError``, and some detectors raise a ``TypeError`` for a plain number. Give each value as an array of shape ``(1,)``, for example ``np.array([value])``, and reshape a 1-D array with ``values.reshape(-1, 1)`` before you stream it. See :doc:`guides/stream_data`.

Why are the first scores 0 or unstable?
---------------------------------------

A detector learns from the stream itself, so it knows little at the start. Some detectors give 0 until a window is full. Skip a warm-up period before you raise alerts, and measure the detector on each window of the stream to see the warm-up period separately. See :doc:`guides/evaluate`.

Why do I get a different result on each run?
--------------------------------------------

Some detectors use random numbers. Pass ``random_state=0``, or another integer, to each of them, or call :func:`pysad.utils.fix_seed` at the start of the script. See :doc:`concepts`.

Can I use streaming detectors on a batch dataset?
-------------------------------------------------

Yes. Stream the dataset with :class:`~pysad.utils.ArrayStreamer`, or call ``fit_score`` on the full array, which processes the rows in order. A streaming detector sees each point only once, so a batch detector that sees all data can be more accurate on a fixed dataset. For batch detection, see `PyOD <https://pyod.readthedocs.io/>`_.

Can I use batch detectors on a stream?
--------------------------------------

Yes. :class:`~pysad.models.integrations.ReferenceWindowModel` trains a PyOD detector on a sliding window, and :class:`~pysad.models.integrations.OneFitModel` trains it once on initial data. See :doc:`guides/pyod`.

Why does my result change when I shuffle the benchmark data?
------------------------------------------------------------

The order of a stream matters. In some benchmark files, all anomalies are at the end. For example, the 176 anomalies of ``cardio`` are its last points. A detector that learns from the stream then sees a long run of anomalies, and they start to look normal. ``Data().get_data`` keeps the order of the file, and ``Data().get_iterator`` shuffles by default. Use the same order for every detector that you compare.

Why is ReferenceWindowModel slow at the start?
----------------------------------------------

Without ``initial_window_X``, :class:`~pysad.models.integrations.ReferenceWindowModel` trains a new batch model for each of the first ``window_size`` points. Pass initial points to train once instead. See :doc:`guides/pyod`.

Can I save a detector and load it later?
----------------------------------------

Yes. PySAD detectors and calibrators can be pickled. Load a pickle only with the PySAD version that saved it. See :doc:`guides/production`.

Can I use labels to train a detector?
-------------------------------------

Most detectors are unsupervised and ignore the ``y`` argument of ``fit_partial``. If you know which points are normal, call ``fit_partial`` only for those points, and ``score_partial`` for every point. The detector then learns only normal behavior.

How do I report a bug or ask a question?
----------------------------------------

To report a bug, open an `issue <https://github.com/selimfirat/pysad/issues/new/choose>`_. Include your PySAD version, your Python version and a short script that shows the problem. To ask a question, start a thread in `Discussions <https://github.com/selimfirat/pysad/discussions>`_. To contribute, see :doc:`contributing`.
