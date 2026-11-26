Transform
=========

.. module:: pysad.transform

This package holds the stages that sit around a detector: transformers that change each point before the detector scores it, projectors that reduce the number of features, postprocessors that smooth or threshold scores, ensemblers that combine several detectors, and calibrators that turn scores into probabilities. Each stage learns from one instance at a time and keeps bounded memory.

Import each name from the module that documents it, for example ``from pysad.transform.preprocessing import StandardScaler``.

Subpackages
-----------

.. autosummary::
   :nosignatures:

   ensemble
   postprocessing
   preprocessing
   probability_calibration
   projection
   stream_generators

Preprocessing
-------------

:doc:`preprocessing` holds the scalers, encoders, log parser, and seasonal decomposers. Compose transformers with the ``|`` operator.

Calibration
-----------

:doc:`calibration` holds the probability calibrators. A calibrator wraps a detector and converts each raw score into a value between 0 and 1. Higher values mean more anomalous. Chain a detector and a calibrator with ``|``.

Postprocessing
--------------

:doc:`postprocessing` holds score smoothers, normalizers, and threshold filters. Each postprocessor reads one score at a time and keeps bounded memory.

Ensembling
----------

:doc:`ensemble` holds ensemblers that combine the scores of several detectors into one score.

Projection
----------

:doc:`projection` holds random projectors that reduce the number of features.

Stream generators
-----------------

:doc:`stream_generators` holds adversarial stream generators that wrap a normal stream to benchmark how well a detector resists poisoning and burst attacks.
