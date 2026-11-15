Models
======

.. module:: pysad.models

This module holds the streaming anomaly detectors. :doc:`/detectors` compares them. :doc:`/guides/choose_detector` helps you choose one.

``ADeMS`` implements AnomDetect with the deterministic DetUpdate sketch. It scores raw, uncentered reconstruction error before learning each instance. Use ``initial_X`` for trusted normal training data when filtering updates with ``threshold``; without a threshold, the first stream instances bootstrap the model. See ``examples/example_adems.py``.

``FluxEV`` scores unexpected fluctuations in one equally spaced series. Set ``period`` to match the stream. After calibration, scores above ``threshold_`` indicate anomalies.
``FRAHST`` tracks an orthonormal principal subspace and adapts its rank with energy thresholds. It returns binary rank-increase alarms. Its scores detect changes in correlation. See :class:`~pysad.models.FRAHST` for initialization and idle-input rules.

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: ../generated
   :tail: NullModel, PerfectModel, RandomModel, StreamPipeline
