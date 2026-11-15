Models
======

.. module:: pysad.models

This module holds the streaming anomaly detectors. :doc:`/detectors` compares them. :doc:`/guides/choose_detector` helps you choose one.

``ADeMS`` implements AnomDetect with the deterministic DetUpdate sketch. It scores raw, uncentered reconstruction error before learning each instance. Use ``initial_X`` for trusted normal training data when filtering updates with ``threshold``; without a threshold, the first stream instances bootstrap the model. See ``examples/example_adems.py``.

``FRAHST`` tracks an orthonormal principal subspace and adapts its rank with energy thresholds. It returns binary rank-increase alarms. Its scores detect changes in correlation. See :class:`~pysad.models.FRAHST` for initialization and idle-input rules.

``DriftAdaptiveDetector`` resets a detector after Page-Hinkley alerts on its scores. Pass a factory that creates fresh detectors. The monitor adds constant memory and update cost. Scoring alone does not update the drift monitor.

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: ../generated
   :tail: DriftAdaptiveDetector, NullModel, PerfectModel, RandomModel, StreamPipeline
