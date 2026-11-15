Models
======

.. module:: pysad.models

This module holds the streaming anomaly detectors. :doc:`/detectors` compares them. :doc:`/guides/choose_detector` helps you choose one.

``ADeMS`` implements AnomDetect with the deterministic DetUpdate sketch. It scores raw, uncentered reconstruction error before learning each instance. Use ``initial_X`` for trusted normal training data when filtering updates with ``threshold``; without a threshold, the first stream instances bootstrap the model. See ``examples/example_adems.py``.

``GNGA`` implements GNG-A with a bounded prototype graph. It scores the nearest prototype distance minus the mean edge length. Positive scores mark anomalies under the paper's decision rule. The class docstring explains startup rules, memory limits, and paper ambiguities.

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: ../generated
   :tail: NullModel, PerfectModel, RandomModel, StreamPipeline
