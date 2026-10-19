Frequently asked questions
==========================

Find your question in one of the groups below. Each answer gives the short answer first and then links to the guide with the full steps.

.. container:: faq-groups

   - :ref:`faq-start`

     Which detector to use, hardware and other libraries

   - :ref:`faq-input`

     Single values, missing values, categories and several sources

   - :ref:`faq-scores`

     What a score means, thresholds and scores of 0

   - :ref:`faq-evaluation`

     Seeds, stream order, metrics and data without labels

   - :ref:`faq-production`

     Speed, memory, saved state and changes in the data

   - :ref:`faq-contributing`

     Bugs, new detectors and citations

.. _faq-start:

Getting started
---------------

Which detector do I start with?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For data with several features, start with :class:`~pysad.models.LODA`. For a single series of values with a daily or weekly cycle, start with :class:`~pysad.models.SeasonalHybridESD`. If the series has no cycle, start with :class:`~pysad.models.MedianAbsoluteDeviation`. Then compare two or three detectors on your data. See :doc:`guides/choose_detector`.

Do I need a GPU?
~~~~~~~~~~~~~~~~

No. PySAD runs on the CPU. Its detectors use NumPy, SciPy and scikit-learn, and no detector uses a GPU. ``pip install pysad`` installs every dependency, and PySAD has no optional extras. See :doc:`installation`.

Can I use streaming detectors on a batch dataset?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes. Stream the dataset with :class:`~pysad.utils.ArrayStreamer`, or call ``fit_score`` on the full array, which processes the rows in order. A streaming detector sees each point only once. On a fixed dataset, a batch detector that sees all the data can be more accurate. For batch detection, see `PyOD <https://pyod.readthedocs.io/>`_.

Can I use batch detectors on a stream?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes. :class:`~pysad.models.integrations.ReferenceWindowModel` trains a PyOD detector on a sliding window, and :class:`~pysad.models.integrations.OneFitModel` trains it once on initial data. See :doc:`guides/pyod`.

Does PySAD work with scikit-learn or River?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PySAD works next to them, but its classes are not scikit-learn estimators or River models. They have no ``get_params`` method, so ``clone`` and ``GridSearchCV`` do not accept them, and a scikit-learn ``Pipeline`` cannot score with them.

To use a scikit-learn transformer, fit it to historical data. Then transform each point before the detector scores it:

.. code-block:: python

   import numpy as np
   from sklearn.preprocessing import StandardScaler

   from pysad.models import LODA

   rng = np.random.default_rng(0)
   X_history = rng.normal(size=(500, 3))
   stream = rng.normal(size=(100, 3))

   scaler = StandardScaler().fit(X_history)
   model = LODA(random_state=0)

   for x in stream:
       x = scaler.transform(x.reshape(1, -1))[0]  # one row in, one row out
       score = model.fit_score_partial(x)

River models take each point as a dict. Convert the dict to a NumPy array, with the same key order for every point. For the stages that PySAD has for each point, see :doc:`guides/pipelines`.

.. _faq-input:

Input data
----------

Why does a univariate detector raise an error?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Univariate detectors take points with exactly one feature. A point with several features raises a ``ValueError``, and some detectors raise a ``TypeError`` for a plain number. Give each value as an array of shape ``(1,)``, for example ``np.array([value])``. To stream a 1-D array, reshape it first with ``values.reshape(-1, 1)``. See :doc:`guides/stream_data`.

How do I handle missing values (NaN)?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Replace each missing value before the detector gets the point. PySAD has no step that fills missing values, and the detectors handle ``NaN`` (not a number) in different ways. Some raise an error, and some give a ``NaN`` score. :class:`~pysad.models.StandardAbsoluteDeviation` gives ``NaN`` for every point after the first ``NaN``.

A common fix is to repeat the last known value of each feature:

.. code-block:: python

   import numpy as np

   from pysad.models import LODA

   stream = np.array([[1.0, 2.0], [1.1, np.nan], [0.9, 2.1]])

   model = LODA(random_state=0)
   last = np.zeros(stream.shape[1])

   for x in stream:
       x = np.where(np.isnan(x), last, x)  # fill each gap with the last known value
       last = x
       score = model.fit_score_partial(x)

See :doc:`guides/production`.

How do I use categorical features?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Convert each category to numbers before you stream the points. PySAD detectors take only arrays of numbers. For a feature with a few categories, use one-hot encoding: one feature per category, with the value 1 for the category of the point and 0 for the others. Fix the list of categories before the stream starts, because the number of features cannot change:

.. code-block:: python

   import numpy as np

   COLORS = ["red", "green", "blue"]  # every category, fixed before the stream starts


   def encode(color, size):
       return np.array([color == c for c in COLORS] + [size], dtype=float)


   print(encode("green", 2.5))

.. code-block:: text

   [0.  1.  0.  2.5]

If each category has its own normal behavior, keep one detector for each category instead, as for :ref:`several sources <faq-sources>`.

Can I add or remove a feature during the stream?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No. A detector fixes the number of features at the first point. After that, most detectors raise an error for a point with a different number of features. Some detectors give a score without an error, and this score has no meaning. To change the features, start a new detector and fit it to historical data with the new features. See :doc:`guides/production`.

.. _faq-sources:

How do I stream data from several sources?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Keep one detector for each source, for example in a dictionary with the source name as the key. Normal behavior is usually different for each source, and one shared detector mixes them. If the sources measure the same moment, for example CPU and memory of one server, combine them into one point. See :doc:`guides/production`.

Do detectors use time stamps?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No. A detector sees only the order of the points, not their time stamps. A seasonal detector counts its ``period`` in points, so it expects a fixed interval between points. If the points arrive at irregular times, resample them to a fixed interval first. The pandas ``resample`` method does this. To use the time of day as a signal, add it as a feature. See :doc:`guides/seasonal`.

Can I use labels to train a detector?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Not directly. Most detectors are unsupervised: they learn without labels and ignore the ``y`` argument of ``fit_partial``. If you know which points are normal, call ``fit_partial`` only for those points, and ``score_partial`` for every point. The detector then learns only normal behavior. Before the first fit, a detector gives a fixed score, usually 0. See :doc:`concepts`.

.. _faq-scores:

Scores and alerts
-----------------

What does a score mean?
~~~~~~~~~~~~~~~~~~~~~~~

A higher score means a more anomalous point. The scale is different for each detector, so a raw score has no meaning on its own. To get a value with the same meaning for every detector, calibrate the score. See :doc:`guides/alerts`.

How do I choose a threshold?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Calibrate the scores, and then choose the threshold from the rate of false alerts that you accept. A calibrator converts each score into the fraction of recent scores that are lower. With a threshold of 0.99, about 1 in 100 normal points raises an alert. With 0.999, about 1 in 1,000 does. For a high threshold, the calibrator window must hold enough scores, for example 5,000 or more for 0.999. See :doc:`guides/alerts`.

Why are all or most scores 0?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some detectors give 0 by design, during a warm-up or for every point that is not anomalous:

- :class:`~pysad.models.KitNet` gives 0 during its first grace period, the first 100 points with the defaults. For a short stream, lower ``grace_feature_mapping`` and ``grace_anomaly_detector``.
- :class:`~pysad.models.KNNCAD` gives 0 until it gets ``probationary_period`` points.
- :class:`~pysad.models.SeasonalESD` and :class:`~pysad.models.SeasonalHybridESD` give 0 until the window is full. After that, they give 0 for every point that their test does not flag.
- :class:`~pysad.models.RelativeEntropy` gives only 0 or 1. It gives 1 only when the values of the window have a new distribution.
- :class:`~pysad.models.LocalOutlierProbability` gives a probability, which is often exactly 0 for normal points.

If another detector gives only 0, look at the input. For example, :class:`~pysad.models.StandardAbsoluteDeviation` gives 0 when all values are the same. See :doc:`detectors`.

Why are the first scores unstable?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A detector learns from the stream itself, so it knows little at the start. Skip a warm-up period before you raise alerts, for example the first 200 points. Or fit the detector to historical data before the stream starts. To see the warm-up in a metric, measure each window of the stream separately. See :doc:`guides/evaluate`.

.. _faq-evaluation:

Evaluation and benchmarks
-------------------------

Which metric do I report?
~~~~~~~~~~~~~~~~~~~~~~~~~

Report AUROC and AUPR, because they need no threshold. AUROC measures how well the scores rank anomalies above normal points. AUPR, the area under the precision-recall curve, shows more of the difference between detectors when anomalies are rare. If you also raise alerts, add precision and recall at your threshold. See :doc:`guides/evaluate`.

Which detector is the best?
~~~~~~~~~~~~~~~~~~~~~~~~~~~

No detector is the best for all data. The :doc:`benchmark <benchmark>` gives the results of each multivariate detector with its default parameters on |num_datasets| datasets. Use them to find candidates, and then compare the candidates on your own data. See :doc:`guides/choose_detector`.

Why do I get a different result on each run?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some detectors use random numbers. Pass ``random_state=0``, or another integer, to each of them. Or call :func:`pysad.utils.fix_seed` at the start of the script. To report a stable result, repeat the run with three to five seeds and give the mean. See :doc:`concepts`.

Why does my result change when I shuffle the benchmark data?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The order of a stream matters. In some benchmark files, all anomalies are at the end. For example, the 176 anomalies of ``cardio`` are its last points. A detector that learns from the stream then sees a long run of anomalies, and they start to look normal. ``Data().get_data`` keeps the order of the file, and ``Data().get_iterator`` shuffles by default. Use the same order for every detector that you compare. See :doc:`guides/stream_data`.

How do I evaluate a detector without labels?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add synthetic anomalies to a copy of your data, or label a sample of the alerts by hand. With synthetic anomalies, you know the label of every point, so every metric works. Hand labels of the alerts give the precision, the fraction of alerts that are true anomalies. They do not give the recall, because you do not know which anomalies raised no alert. See :doc:`guides/choose_detector`.

.. _faq-production:

Running in production
---------------------

How fast is a detector, and how much memory does it use?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most detectors take between 0.005 and 1 millisecond per point. We measured this on the 21 features of ``cardio`` on an Apple silicon laptop. Some detectors take longer:

- :class:`~pysad.models.SeasonalESD` and :class:`~pysad.models.SeasonalHybridESD` take about 4 milliseconds, because they decompose the full window for each point.
- :class:`~pysad.models.integrations.ReferenceWindowModel` and :class:`~pysad.models.IForestASD` retrain at a fixed interval, so a few points take much longer than the others.

Most detectors keep a fixed amount of memory, and :doc:`detectors` gives the memory of each one. To measure the time per point on your own data, see :doc:`guides/production`.

Why is ReferenceWindowModel slow at the start?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Without ``initial_window_X``, :class:`~pysad.models.integrations.ReferenceWindowModel` trains a new batch model for each of the first ``window_size`` points. Pass initial points to train once instead. See :doc:`guides/pyod`.

Can I save a detector and load it later?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes. PySAD detectors and calibrators can be pickled. Save the detector and its calibrator together, so that a restart continues from the same state. Load a pickle only with the PySAD version that saved it. See :doc:`guides/production`.

What do I do when normal behavior changes?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a slow change, let the detector learn from every point with ``fit_score_partial``. A detector with a window or with time decay then follows the change. After a large, permanent change, for example a new software release, start a new detector. Fit it to data from after the change. See :doc:`guides/production`.

.. _faq-contributing:

Contributing and support
------------------------

How do I write my own detector?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extend :class:`~pysad.core.BaseModel` and implement ``fit_partial`` and ``score_partial``. Your detector then works with every streamer, calibrator, ensembler and metric in PySAD. See :doc:`guides/custom_model`.

Which detectors can PySAD include?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PySAD includes a detector only if its paper has 50 or more citations. If your detector does not meet this rule yet, share it as a :class:`~pysad.core.BaseModel` subclass. We can then add a link to it in the documentation. See :doc:`inclusion_criteria`.

How do I cite PySAD?
~~~~~~~~~~~~~~~~~~~~

Cite the PySAD paper, and also the paper of each detector that you use:

.. code-block:: bibtex

   @article{pysad,
     title={PySAD: A Streaming Anomaly Detection Framework in Python},
     author={Yilmaz, Selim F and Kozat, Suleyman S},
     journal={arXiv preprint arXiv:2009.02572},
     year={2020}
   }

To cite a specific release, use its Zenodo DOI. See :doc:`citing`.

How do I report a bug or ask a question?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To report a bug, open an `issue <https://github.com/selimfirat/pysad/issues/new/choose>`_. Include your PySAD version, your Python version and a short script that shows the problem. To ask a question, start a thread in `Discussions <https://github.com/selimfirat/pysad/discussions>`_. To contribute code, see :doc:`contributing`.
