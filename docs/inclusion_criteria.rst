Inclusion criteria
==================

PySAD includes only well-established detectors. A detector must come from a paper with 50 or more citations. `scikit-learn <https://scikit-learn.org/stable/faq.html#what-are-the-inclusion-criteria-for-new-algorithms>`_ and `PyOD <https://pyod.readthedocs.io/en/latest/>`_ use similar rules.

If you published a streaming detector, you can make it work with PySAD before it meets this rule:

#. Extend :class:`~pysad.core.BaseModel`, as :doc:`guides/custom_model` shows.
#. Share your code.

We can then add a link to your detector in the documentation. To ask a question, open an `issue <https://github.com/selimfirat/pysad/issues>`_ or start a thread in `Discussions <https://github.com/selimfirat/pysad/discussions>`_.
