Utilities
=========

.. module:: pysad.utils

This module holds streamers, benchmark datasets and helper functions. A streamer gives the points of a dataset one at a time. See :doc:`/guides/stream_data`.

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: ../generated

   ArrayStreamer
   PandasStreamer
   Data
   Window

Functions
---------

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   check_feature_bounds
   check_univariate
   fix_seed
   get_minmax_array
   get_minmax_scalar
