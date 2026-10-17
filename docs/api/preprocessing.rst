Preprocessing
=============

.. module:: pysad.transform.preprocessing

This module holds transformers that change each point before the detector gets it. See :doc:`/guides/pipelines`.

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: ../generated

   IdentityScaler
   InstanceStandardScaler
   InstanceUnitNormScaler
   ModifiedSTLResidualTransformer
   SeasonalTrendDecomposer
