Postprocessing
==============

.. module:: pysad.transform.postprocessing

This module holds postprocessors that smooth or normalize scores. The ``Running`` variants use a window of recent scores. The other variants use every score since the start of the stream. Of these, ``MaxPostprocessor`` and ``MedianPostprocessor`` keep every score, so their memory grows with the stream. See :doc:`/guides/alerts`.

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: ../generated

   RunningAveragePostprocessor
   RunningMaxPostprocessor
   RunningMedianPostprocessor
   RunningZScorePostprocessor
   AveragePostprocessor
   MaxPostprocessor
   MedianPostprocessor
   ZScorePostprocessor
