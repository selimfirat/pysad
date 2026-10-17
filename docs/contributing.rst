.. _contributing:

Contributing
============

Code is not the only way to help. Bug reports, answers to questions, documentation and examples are all contributions. I aim to reply to new issues and pull requests within a few days.

We follow the `Python Software Foundation Code of Conduct <https://policies.python.org/python.org/code-of-conduct/>`_.

Where to start
--------------

- To report a bug, open an `issue <https://github.com/selimfirat/pysad/issues/new/choose>`_ with your PySAD version, your Python version and a short script that shows the problem.
- To request a feature, open an `issue <https://github.com/selimfirat/pysad/issues/new/choose>`_ that describes your use case.
- To ask a question or share an idea, start a thread in `Discussions <https://github.com/selimfirat/pysad/discussions>`_.
- For a first contribution, choose an issue with the `good first issue <https://github.com/selimfirat/pysad/labels/good%20first%20issue>`_ label.

Set up the development environment
----------------------------------

You need pip 25.1 or newer for the ``--group`` option.

.. code-block:: console

   $ git clone https://github.com/selimfirat/pysad.git
   $ cd pysad
   $ pip install --upgrade pip
   $ pip install -e . --group dev
   $ pre-commit install

The ``dev`` group installs the pinned runtime dependencies, pytest, Sphinx and pre-commit. ``pre-commit install`` lints and formats your changes on every commit.

Check your change
-----------------

Before you open a pull request, run these commands from the repository root:

.. code-block:: console

   $ pre-commit run --all-files
   $ pytest --cov=pysad
   $ pytest -m examples
   $ bash build_docs.sh

The commands run the linters and the formatter, the unit tests, every script in ``examples/`` and the documentation build. `pre-commit.ci <https://pre-commit.ci>`_ also runs the hooks on every pull request and pushes a commit with the fixes that it can make. Pull before you push again.

Development style
-----------------

PySAD follows `trunk-based development <https://trunkbaseddevelopment.com/>`_. Each change goes to ``master`` through a short-lived branch, for example ``fix/123-short-name``, and a pull request. CI tests and lints every pull request, so ``master`` is always ready to release. Keep each pull request small and about one topic.

Pull request checklist
----------------------

- The change fits the aim of PySAD: anomaly detection on streaming data.
- ``pre-commit run --all-files`` and all tests pass, also in CI.
- No open pull request or issue covers the same work.
- A new feature has tests and an example or documentation that shows how to use it.
- A new detector cites its paper and meets the :doc:`inclusion_criteria`. :doc:`guides/custom_model` lists the steps.
