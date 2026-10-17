.. _contributing:

Contributing
============

Code is not the only way to help. We aim to reply to new issues and pull requests within a few days. The project follows the `Python Software Foundation Code of Conduct <https://policies.python.org/python.org/code-of-conduct/>`_.

Help without code
-----------------

- Answer questions in `Discussions <https://github.com/selimfirat/pysad/discussions>`_.
- Report a bug that you found.
- Tell us about a part of the documentation that is not clear.
- Share a project that uses PySAD in `Discussions <https://github.com/selimfirat/pysad/discussions>`_.

Report a bug or ask a question
------------------------------

- If you found a bug, open a `bug report <https://github.com/selimfirat/pysad/issues/new/choose>`_. Include your PySAD version, your Python version and a short script that shows the problem.
- If you want a new feature or detector, open a `feature request <https://github.com/selimfirat/pysad/issues/new/choose>`_. A new detector must meet the :doc:`inclusion_criteria`.
- If you have a question, start a thread in `Discussions <https://github.com/selimfirat/pysad/discussions>`_.
- If you found a security problem, do not open a public issue. Follow the `security policy <https://github.com/selimfirat/pysad/security/policy>`_.

For a first contribution, choose an issue with the `good first issue <https://github.com/selimfirat/pysad/labels/good%20first%20issue>`_ label.

Open a pull request
-------------------

PySAD uses trunk-based development: each change goes to the ``master`` branch through a short-lived branch and a pull request. Each pull request fixes one issue.

You need git, Python 3.10 or newer, and pip 25.1 or newer.

Set up once
~~~~~~~~~~~

#. Fork `the repository <https://github.com/selimfirat/pysad>`_ on GitHub.
#. Clone your fork. Replace ``<user>`` with your GitHub user name:

   .. code-block:: console

      $ git clone https://github.com/<user>/pysad.git

#. Go into the repository root. Run all the other commands from there:

   .. code-block:: console

      $ cd pysad

#. Upgrade pip. The ``--group`` option in the next step needs pip 25.1 or newer:

   .. code-block:: console

      $ pip install --upgrade pip

#. Install PySAD in editable mode with the ``dev`` dependency group. The group pins the runtime dependencies and adds pytest, Sphinx and pre-commit:

   .. code-block:: console

      $ pip install -e . --group dev

   If you use uv, run ``uv sync --group dev`` instead.

#. Install the git hook of pre-commit. pre-commit is a tool that runs the linters and the formatter on each commit:

   .. code-block:: console

      $ pre-commit install

Make the change
~~~~~~~~~~~~~~~

#. If no issue describes your change, open one first.
#. Make a branch named ``<type>/<issue>-<slug>``, for example ``fix/123-loda-nan-scores``:

   .. code-block:: console

      $ git switch -c fix/123-loda-nan-scores

#. Change the code. If you add a detector, follow the steps in :doc:`guides/custom_model`.
#. Add tests that check scores, shapes and return types, not private state. If a test uses random numbers, give it a fixed seed.

`AGENTS.md <https://github.com/selimfirat/pysad/blob/master/AGENTS.md>`_ in the repository root lists all the rules for code, tests and naming.

Check the change
~~~~~~~~~~~~~~~~

#. Run the linters, the formatter, mypy and the file checks:

   .. code-block:: console

      $ pre-commit run --all-files

#. Run the unit tests. Add ``-n auto`` to run them in parallel:

   .. code-block:: console

      $ pytest

#. Run every script in ``examples/``. The plain ``pytest`` command skips them:

   .. code-block:: console

      $ pytest -m examples

#. If you changed a docstring or a file in ``docs/``, build the documentation. CI does not build it for each pull request:

   .. code-block:: console

      $ bash build_docs.sh

   Open ``docs/_build/html/index.html`` to read the result.

Send the change
~~~~~~~~~~~~~~~

#. Push your branch to your fork.
#. Open a pull request into ``master``. If the change is not ready for review, open it as a draft.
#. Write the title as a conventional commit: ``<type>: <summary>``, for example ``fix: return finite LODA scores for constant features``.

   Use one of these types: ``feat``, ``fix``, ``docs``, ``build``, ``ci``, ``refactor``, ``test``, ``style`` or ``chore``. For a breaking change, put ``!`` before the colon, for example ``feat!:``. A workflow labels the pull request from the type, and the release notes group the pull requests by these labels.

#. Wait for CI. CI (continuous integration) is the set of checks that GitHub runs on each pull request. `pre-commit.ci <https://pre-commit.ci>`_ also pushes a commit with the fixes that it can make. If it does, pull before you push again.

Review checklist
----------------

We apply this checklist to each pull request:

- The change fits the aim of PySAD: anomaly detection on streaming data.
- The pull request fixes one issue, and no other open pull request covers the same work.
- All CI checks pass.
- New behavior has tests. A new feature also has an example or documentation.
- A new public class is in the ``__init__.py`` and ``__all__`` of its subpackage, and in the page of its module in ``docs/api/``.
- A new detector meets the :doc:`inclusion_criteria` and follows the steps in :doc:`guides/custom_model`.
- If the change touches a detector, the description gives AUROC and time per instance before and after the change. Measure them with ``fit_score_partial`` on the ODDS ``arrhythmia``, ``cardio`` and ``vowels`` datasets with seeds 0 to 4.
- The change does not keep old behavior through deprecation aliases or compatibility code.

AI tools
--------

You can use AI tools to write issues, pull requests, reviews and code. These rules apply:

- Put the text that an AI tool writes in a quote block or a code block.
- Add your own comment that tells why the AI text is relevant.
- Read and understand all the code that you submit.
- Do not let an AI agent open an issue or a non-draft pull request, or comment, before you approve the exact content.

We close contributions that an AI agent generated with no human review.
