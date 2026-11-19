Stream generators
=================

.. module:: pysad.transform.stream_generators

This module holds adversarial stream generators that wrap a normal stream to benchmark how well a detector resists poisoning and burst attacks. Each generator is a transformer: it receives a normal base point and returns either that point unchanged or an injected anomaly. After each call, ``is_anomaly`` tells whether the returned point is an anomaly.

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: ../generated

   SlowPoisonStream
   StealthBurstStream
