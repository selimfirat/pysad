Inclusion criteria
==================

PySAD includes only established detectors. A detector qualifies in one of two ways.

A paper: the detector comes from a peer-reviewed paper that defines the streaming method, and the paper meets at least one of these conditions:

#. It has 20 or more citations.
#. It appeared at a conference or in a journal with the rank A* in the `CORE ranking <https://portal.core.edu.au/conf-ranks/>`_.
#. Its official code repository on GitHub has 50 or more stars.

A benchmark result: the streaming detector has a top-three score on the scoreboard of a public streaming benchmark, for example the `NAB scoreboard <https://github.com/numenta/NAB#user-content-scoreboard>`_. Its reference implementation must process the stream one instance at a time and must be public under a license that allows the port. The port must reproduce the published score. A batch method does not qualify through this route. Wrap it with :class:`~pysad.integrations.ReferenceWindowModel` or :class:`~pysad.integrations.OneFitModel` instead. The detector then needs no paper, and its docstring cites the benchmark and the reference implementation instead.

`scikit-learn <https://scikit-learn.org/stable/faq.html#what-are-the-inclusion-criteria-for-new-algorithms>`_ and `PyOD <https://pyod.readthedocs.io/en/latest/>`_ use similar rules.

If you published a streaming detector, you can make it work with PySAD before it meets this rule:

#. Extend :class:`~pysad.core.BaseModel`, as :doc:`guides/custom_model` shows.
#. Share your code.

We can then add a link to your detector in the documentation. To ask a question, open an `issue <https://github.com/selimfirat/pysad/issues>`_ or start a thread in `Discussions <https://github.com/selimfirat/pysad/discussions>`_.
