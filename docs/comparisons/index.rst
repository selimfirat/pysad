Overview
========

PySAD does anomaly detection on streams. Two other Python libraries cover part of this task. `River <https://riverml.xyz/>`_ does machine learning on streams, and anomaly detection is one of its tasks. `PyOD <https://github.com/yzhao062/pyod>`_ does anomaly detection on batch data.

Table 1 of the :doc:`PySAD paper <../citing>` compares PySAD with other libraries by their main functions. The table below gives the same comparison for the current versions: PySAD, River 0.26.1 and PyOD 3.6.7. River includes creme and scikit-multiflow, two of the libraries in the paper.

.. list-table::
   :header-rows: 1
   :stub-columns: 1
   :class: feature-table

   * - Function
     - PySAD
     - River
     - PyOD
   * - Language
     - Python
     - Python
     - Python
   * - Main task
     - Anomaly detection on streams
     - Machine learning on streams
     - Anomaly detection on batch data
   * - Streaming detectors [#count]_
     - 20
     - 7
     - 0
   * - Batch detectors
     - Every PyOD detector, through a sliding window
     - None
     - About 60
   * - Learns from one point at a time
     - ✓
     - ✓
     - ✗
   * - Transformers
     - ✓
     - ✓
     - ✗
   * - Projectors
     - ✓
     - ✓
     - ✗
   * - Score ensemblers
     - ✓
     - ✗
     - Batch only
   * - Score postprocessors
     - ✓
     - ✗
     - ✗
   * - Probability calibrators
     - ✓
     - ✗
     - Batch only
   * - Running statistics
     - ✓
     - ✓
     - ✗
   * - Stream simulators
     - ✓
     - ✓
     - ✗
   * - Prequential evaluation
     - ✓
     - ✓
     - ✗

.. [#count] PySAD has 23 models. The count leaves out the 3 baselines ``NullModel``, ``PerfectModel`` and ``RandomModel``. River has 7 detectors in ``river.anomaly``. Its ``ThresholdFilter`` and ``QuantileFilter`` turn scores into labels, and the count leaves them out. PyOD has ``SAND``, which its paper defines for streams, but PyOD fits it on a whole series in one call.

The rows mean:

- A transformer changes each point before a detector gets it, for example a scaler.
- A projector maps each point to fewer dimensions.
- A score ensembler combines the scores of several detectors into one score.
- A score postprocessor changes the score of each point, for example with a running average of the recent scores.
- A probability calibrator maps each score to a probability that the point is anomalous.
- A stream simulator gives the points of a dataset one at a time.
- In prequential evaluation, a detector scores each point before it learns from the point.

✓ means that the library has the function, and ✗ means that it does not. "Batch only" means that the library has the function, but it works on a whole dataset and not one point at a time.

The pages :doc:`river` and :doc:`pyod` compare the functions in more detail, and give the results of the detectors on the benchmark datasets.
