Ensembling
==========

.. module:: pysad.transform.ensemble

This module holds ensemblers that combine the scores of several detectors into one score. See :doc:`/guides/pipelines`.

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: ../generated

   AverageScoreEnsembler
   MaximumScoreEnsembler
   MedianScoreEnsembler
   AverageOfMaximumScoreEnsembler
   MaximumOfAverageScoreEnsembler
