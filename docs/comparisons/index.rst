Overview
========

PySAD does anomaly detection on streams. Other Python libraries cover part of this task:

- `River <https://riverml.xyz/>`_ and `CapyMOA <https://github.com/adaptive-machine-learning/CapyMOA>`_ do machine learning on streams, and anomaly detection is one of their tasks.
- `StreamAD <https://github.com/Fengrui-Liu/StreamAD>`_ does anomaly detection on streams.
- `PyOD <https://github.com/yzhao062/pyod>`_ does anomaly detection on batch data.
- `Alibi Detect <https://github.com/SeldonIO/alibi-detect>`_ does outlier, adversarial and drift detection, mostly on batch data.
- `Darts <https://github.com/unit8co/darts>`_ does time series forecasting, and it detects anomalies in a whole time series.

Table 1 of the :ref:`PySAD paper <citing-pysad>` compares PySAD with other libraries by their main functions. The table below gives the same comparison for the current versions: PySAD, River 0.26.1, CapyMOA 0.15.1, StreamAD 0.3.1, PyOD 3.6.7, Alibi Detect 0.13.0 and Darts 0.48.1. River includes creme and scikit-multiflow, two of the libraries in the paper.

.. list-table::
   :header-rows: 1
   :stub-columns: 1
   :class: feature-table

   * - Function
     - PySAD
     - River
     - CapyMOA
     - StreamAD
     - PyOD
     - Alibi Detect
     - Darts
   * - Language
     - Python
     - Python
     - Python and Java
     - Python
     - Python
     - Python
     - Python
   * - Main task
     - Anomaly detection on streams
     - Machine learning on streams
     - Machine learning on streams
     - Anomaly detection on streams
     - Anomaly detection on batch data
     - Outlier, adversarial and drift detection
     - Time series forecasting
   * - Streaming detectors [#count]_
     - ✓
     - ✓
     - ✓
     - ✓
     - ✗
     - Mahalanobis only
     - ✗
   * - Batch detectors
     - Every PyOD detector, through a sliding window
     - None
     - None
     - None
     - About 60
     - 9
     - Scorers and anomaly models, also every PyOD detector
   * - Learns from one point at a time
     - ✓
     - ✓
     - ✓
     - ✓
     - ✗
     - Mahalanobis only
     - ✗
   * - Transformers
     - ✓
     - ✓
     - ✓
     - ✗
     - ✗
     - ✗
     - Batch only
   * - Projectors
     - ✓
     - ✓
     - ✓
     - ✗
     - ✗
     - ✗
     - ✗
   * - Score ensemblers
     - ✓
     - ✗
     - ✗
     - ✓
     - Batch only
     - ✗
     - ✗
   * - Score postprocessors
     - ✓
     - ✗
     - ✗
     - ✓
     - ✗
     - ✗
     - ✗
   * - Probability calibrators
     - ✓
     - ✗
     - ✗
     - ✗
     - Batch only
     - ✗
     - ✗
   * - Running statistics
     - ✓
     - ✓
     - ✗
     - ✓
     - ✗
     - ✗
     - ✗
   * - Stream simulators
     - ✓
     - ✓
     - ✓
     - ✓
     - ✗
     - ✗
     - ✗
   * - Prequential evaluation
     - ✓
     - ✓
     - ✓
     - ✗
     - ✗
     - ✗
     - ✗

.. [#count] The PySAD detectors are on the page :doc:`../detectors`, without the baselines ``NullModel``, ``PerfectModel`` and ``RandomModel``. The River detectors are in ``river.anomaly``, without ``ThresholdFilter`` and ``QuantileFilter``, which turn scores into labels. The CapyMOA detectors are in ``capymoa.anomaly``. The StreamAD detectors are in ``streamad.model``, without the baseline ``RandomDetector``. PyOD has ``SAND``, which its paper defines for streams, but PyOD fits it on a whole series in one call. Alibi Detect's ``Mahalanobis`` updates its mean and covariance with each batch of points that it scores. Its other outlier detectors learn from a batch of points.

The rows mean:

- A transformer changes each point before a detector gets it, for example a scaler.
- A projector maps each point to fewer dimensions.
- A score ensembler combines the scores of several detectors into one score.
- A score postprocessor changes the score of each point, for example with a running average of the recent scores.
- A probability calibrator maps each score to a probability that the point is anomalous.
- A stream simulator gives the points of a dataset one at a time.
- In prequential evaluation, a detector scores each point before it learns from the point.

✓ means that the library has the function, and ✗ means that it does not. "Batch only" means that the library has the function, but it works on a whole dataset and not one point at a time.

The pages :doc:`river`, :doc:`capymoa`, :doc:`streamad`, :doc:`pyod`, :doc:`alibi_detect` and :doc:`darts` compare the functions in more detail. All of them except the Darts page also give the results of the detectors on the benchmark datasets.
