Postprocessing
==============

.. module:: pysad.transform.postprocessing

This module holds postprocessors that smooth, normalize, or threshold scores, or turn them into 0/1 labels. Pass ``window_size`` to compute statistics over a sliding window of recent scores instead of the full stream. See :doc:`/guides/alerts`.

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: ../generated
