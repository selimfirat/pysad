.. _pyopensci_review:

pyOpenSci review checklist
==========================

pyOpenSci reviews scientific Python packages. This page lists each review requirement and links it to the evidence in this repository. Reviewers and users can check PySAD against the list before they open a review.

The repository is `selimfirat/pysad <https://github.com/selimfirat/pysad>`_. File links below point to the ``master`` branch.

Review requirements
-------------------

.. list-table:: pyOpenSci requirements and repository evidence
   :header-rows: 1
   :widths: 35 10 55

   * - Requirement
     - Status
     - Evidence

   * - The package installs with pip.
     - Met
     - :doc:`installation` shows ``pip install pysad`` and the conda-forge option.

   * - The documentation explains the package.
     - Met
     - `README.md <https://github.com/selimfirat/pysad/blob/master/README.md>`_ and the `documentation <https://pysad.readthedocs.io/>`_.

   * - The public API is documented.
     - Met
     - :doc:`api` lists every public class, parameter and method.

   * - Package metadata is complete.
     - Met
     - `pyproject.toml <https://github.com/selimfirat/pysad/blob/master/pyproject.toml>`_ sets the name, version, description, license, keywords, classifiers and project URLs.

   * - The package uses an open source license.
     - Met
     - `LICENSE <https://github.com/selimfirat/pysad/blob/master/LICENSE>`_ is the BSD 3-Clause License. See also :doc:`license`.

   * - Code style is checked.
     - Met
     - `ruff <https://github.com/selimfirat/pysad/blob/master/.pre-commit-config.yaml>`_ runs through `pre-commit <https://github.com/selimfirat/pysad/blob/master/.github/workflows/lint-workflows.yml>`_.

   * - Type hints are checked.
     - Met
     - `mypy <https://github.com/selimfirat/pysad/blob/master/pyproject.toml>`_ checks the whole ``pysad`` package, and ``py.typed`` marks it as typed.

   * - Tests run in continuous integration.
     - Met
     - `tests/ <https://github.com/selimfirat/pysad/tree/master/tests>`_ runs under `pytest <https://github.com/selimfirat/pysad/blob/master/.github/workflows/coverage.yml>`_, and `coveralls <https://coveralls.io/github/selimfirat/pysad>`_ reports coverage.

   * - Examples run in continuous integration.
     - Met
     - `examples/ <https://github.com/selimfirat/pysad/tree/master/examples>`_ runs under `.github/workflows/examples.yml`.

   * - Dependency versions are pinned.
     - Met
     - `pyproject.toml <https://github.com/selimfirat/pysad/blob/master/pyproject.toml>`_ pins the dev, test and docs dependency groups so tests and docs run against known versions.

   * - Results stay reproducible.
     - Met
     - `pysad.utils.fix_seed` fixes the random seed, and shared model tests use `short_stream_params` with fixed seeds.

   * - The software is citable.
     - Met
     - `CITATION.cff <https://github.com/selimfirat/pysad/blob/master/CITATION.cff>`_ and the Zenodo DOI badge in the README point to the archived release.

   * - Each detector cites its paper.
     - Met
     - :doc:`detectors` lists every detector with its paper, and :doc:`references` holds the bibliography.

   * - Contributing rules are public.
     - Met
     - :doc:`contributing` explains setup, branches, checks and pull requests. `.github/CONTRIBUTING.md` repeats the summary.

   * - A code of conduct applies.
     - Met
     - `.github/CODE_OF_CONDUCT.md` adopts the Python Software Foundation Code of Conduct.

   * - Issue templates exist.
     - Met
     - `.github/ISSUE_TEMPLATE` holds bug report and feature request templates.

   * - A security policy exists.
     - Met
     - `.github/SECURITY.md` explains how to report security problems privately.

   * - Dependencies update automatically.
     - Met
     - `.github/dependabot.yml` opens pull requests for dependency updates.

   * - Stale issues are closed.
     - Met
     - `.github/workflows/stale.yml` labels and closes issues with no recent activity.

   * - Releases are published.
      - Met
      - `.github/workflows/publish.yml` builds, signs and publishes releases to PyPI and GitHub Releases, and GitHub Releases lists the changelog.

How to use this list
--------------------

Open each evidence link and make sure that the file matches the requirement. If a file is missing or out of date, open an issue or a pull request that fixes it. Each pull request fixes one issue.

Investigation summary
---------------------

This checklist was checked against the repository on the master branch. Every evidence file listed above exists, so all twenty requirements are met.

One out-of-date reference was corrected: the "Releases are published" row previously pointed to `.github/workflows/release.yml`, which does not exist. Releases are built, signed and published by `.github/workflows/publish.yml`. The link now points to the correct workflow.
