Features
========

Online Anomaly Detection
^^^^^^^^^^^^^^^^^^^^^^^^

`PySAD` provides methods for online/sequential anomaly detection, i.e. anomaly detection on streaming data, where model updates itself as a new instance arrives.


Resource-Efficient
^^^^^^^^^^^^^^^^^^

Streaming methods efficiently handle the limitied memory and processing time requirements of the data streams so that they can be used in near real-time. The methods can only store an instance or a small window of recent instances.


Streaming Anomaly Detection Tools
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`PySAD` contains stream simulators, evaluators, preprocessors, statistic trackers, postprocessors, probability calibrators and more. In addition to streaming models, `PySAD` also provides integrations for batch anomaly detectors of the `PyOD <https://github.com/yzhao062/pyod/>`_ so that they can be used in the streaming setting.


Comprehensiveness
^^^^^^^^^^^^^^^^^

`PySAD` serves models that are specifically designed for both univariate and multivariate data. Furthermore, one can experiment via `PySAD` in supervised, semi-supervised and unsupervised setting.


User Friendly
^^^^^^^^^^^^^

Users with any experience level can easily use `PySAD`. One can easily design experiments and combine the tools in the framework. Moreover, the existing methods in `PySAD` are easy to extend.


Free and Open Source Software (FOSS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`PySAD` is distributed under `BSD License 2.0 <https://github.com/selimfirat/pysad/blob/master/LICENSE>`_ and favors FOSS principles.
